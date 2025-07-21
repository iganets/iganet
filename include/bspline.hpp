/**
   @file include/bspline.hpp

   @brief Multivariate B-splines

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <algorithm>
#include <exception>
#include <filesystem>
#include <functional>
#include <regex>

#include <core.hpp>
#include <options.hpp>
#include <patch.hpp>

#include <utils/blocktensor.hpp>
#include <utils/container.hpp>
#include <utils/fqn.hpp>
#include <utils/index_sequence.hpp>
#include <utils/integer_pow.hpp>
#include <utils/linalg.hpp>
#include <utils/serialize.hpp>
#include <utils/tensorarray.hpp>
#include <utils/vslice.hpp>

/// @brief Sequence of expression (parametric coordinates)
///
/// For each item in this sequence corresponding expressions will be
/// generated for function spaces, boundary spaces, etc.
#define GENERATE_EXPR_SEQ (curl)(div)(grad)(hess)(jac)(lapl)

/// @brief Sequence of expression (physical coordinates)
///
/// For each item in this sequence corresponding expressions will be
/// generated for function spaces, boundary spaces, etc.
#define GENERATE_IEXPR_SEQ (icurl)(idiv)(igrad)(ihess)(ijac)(ilapl)

namespace iganet {
using namespace literals;
using utils::operator+;

//  clang-format off
/// @brief Enumerator for specifying the initialization of B-spline coefficients
enum class init : short_t {
  none = 0,  /*!< leave coefficient values uninitialized */
  zeros = 1, /*!< set coefficient values to zero */
  ones = 2,  /*!< set coefficient values to one */
  linear =
      3, /*!< set coefficient values to \f$0,1,\dots \#\text{coeffs}-1\f$ */
  random = 4,   /*!< set coefficient values to random numbers */
  greville = 5, /*!< set coefficient values to the Greville abscissae */
  linspace = 6  /*!< set coefficient values to \f$0,1,\dots\f$ pattern (mostly
                   for testing) */
};
//  clang-format on

/// @brief Enumerator for specifying the derivative of B-spline evaluation
///
/// **Examples**
///
/// * 3d Laplace operator `dx^2+dy^2+dz^2`
/// * 2d convection operator with time derivative dt+dx+dy`
enum class deriv : short_t {
  func = 0, /*!< function value */

  dx = 1,    /*!< first derivative in x-direction  */
  dy = 10,   /*!< first derivative in y-direction  */
  dz = 100,  /*!< first derivative in z-direction  */
  dt = 1000, /*!< first derivative in t-direction  */
};

/// @brief Adds two enumerators for specifying the derivative of B-spline
/// evaluation
///
/// @param[in] lhs First derivative enumerator
///
/// @param[in] rhs Second derivative enumerator
///
/// @result Sum of the two enumerators
inline constexpr auto operator+(deriv lhs, deriv rhs) {
  return deriv(static_cast<short_t>(lhs) + static_cast<short_t>(rhs));
}

/// @brief Raises an enumerator for specifying the derivative of B-spline
/// evaluation to a higher exponent
///
/// @param[in] lhs Derivative enumerator
///
/// @param[in] rhs Exponent
///
/// @result Derivative enumerator raised to the exponent
inline constexpr auto operator^(deriv lhs, short_t rhs) {
  return deriv(static_cast<short_t>(lhs) * static_cast<short_t>(rhs));
}

  /// @brief SplineCore base class
  class SplineCore_ {};

  /// @brief UniformSplineCore base class
  class UniformSplineCore_ : public SplineCore_ {};

  /// @brief NonUniformSplineCore base class
  class NonUniformSplineCore_ : public SplineCore_ {};

  /// @brief Concept to identify template parameters that are derived from iganet::SplineCore_
  template<typename T>
  concept SplineCoreType = std::is_base_of_v<SplineCore_, T>;

  /// @brief Concept to identify template parameters that are derived from iganet::UniformSplineCore_
  template<typename T>
  concept UniformSplineCoreType = std::is_base_of_v<UniformSplineCore_, T>;

  /// @brief Concept to identify template parameters that are derived from iganet::NonUniformSplineCore_
  template<typename T>
  concept NonUniformSplineCoreType = std::is_base_of_v<NonUniformSplineCore_, T>;
  
/// @brief Tensor-product uniform B-spline (core functionality)
///
/// This class implements the core functionality of all B-spline
/// classes and serves as base class for (non-)uniform B-splines.
///
/// Mathematically, this class defines a mapping
///
/// \f[
/// \mathbf{f}:\hat\Omega \mapsto \Omega
/// \f]
///
/// from the \f$d_\text{par}\f$-dimensional *parametric space*
/// \f$\hat\Omega=[0,1]^{d_\text{par}}\f$ to the
/// \f$d_\text{geo}\f$-dimensional *geometric space*
/// \f$\Omega\subset\mathbb{R}^{d_\text{geo}}\f$.
///
/// This mapping is defined by tensor-product B-spline basis
/// functions
///
/// \f[
/// B_I(\boldsymbol{\xi}) = \bigotimes_{d=1}^{d_\text{par}} B_{i_d,p_d}(\xi_d)
/// \f]
///
/// and the control points
///
/// \f[
/// \mathbf{c}_I = \mathbf{c}_{i_1,i_2,\dots, i_{d_\text{par}}} \in
/// \mathbb{R}^{d_\text{geo}}.
/// \f]
///
/// Here, \f$i_d\f$ are the local numbers of the univariate
/// B-splines \f$\left(B_{i_d,p_d}\right)_{i_d=1}^{n_d}\f$ in the
/// \f$d\f$-th parametric dimension, \f$p_d\f$ is the respective
/// *degree*, and \f$n_d\f$ is the number of univariate B-splines in
/// the \f$d\f$-th direction. Moreover, \f$0\le \xi_{i_d}\le 1\f$ is
/// the parametric value at which the B-spline is evaluated. The
/// multivariate B-spline function is defined as follows
///
/// \f[
/// \mathbf{f}(\boldsymbol{\xi}) = \sum_{I=1}^N B_I(\boldsymbol{\xi})
/// \mathbf{c}_I \f]
///
/// Here and below we adopt the vector notation \f$\boldsymbol{\xi}
/// = \left(\xi_1,\xi_2,\dots,\xi_{d_\text{par}}\right)^\top\f$ and
/// combine multiple local indices
/// \f$i_1,i_2,\dots,i_{d_\text{par}}\f$ of univariate B-spline
/// basis functions into the global index \f$1\le I \le N\f$ with
/// \f$N=n_1\cdot n_2\cdot\dots\cdot n_{d_\text{par}}\f$ denoting
/// the total number of multivariate B-splines.
///
/// This class implements B-spline functions and their derivatives
/// for 1, 2, 3, and 4 parametric dimensions. The univariate
/// B-splines are uniquely determined by their knot vectors
///
/// \f[
/// \left(t_{i_d}\right)_{i_d=1}^{n_d+p_d+1}
/// \f]
///
/// with \f$0\le t_{i_d}\le 1\f$ and \f$t_{i_d}\le t_{i_d+1}\f$ for
/// all \f$i_d\f$, that is, the knot vectors are given by a
/// non-decreasing sequence of values in the interval \f$[0,1]\f$
/// with the possibility that knot values are repeated.
///
/// This class implements the evaluation of B-splines and their
/// derivatives as explained in Chapters 2 and 3 from \cite Lyche:2011.
///
/// @note C++ uses 0-based indexing so that all of the above
/// formulas need to be shifted by -1. Moreover, all vectors,
/// matrices, and tensors are implemented as `torch::Tensor`
/// objects and hence adopt Torch's local-to-global mapping. It is
/// therefore imperative to always use Torch's indexing
/// functionality to extract sub-tensors.
template <typename real_t, short_t GeoDim, short_t... Degrees>
class UniformBSplineCore
    : public UniformSplineCore_,
      public utils::Serializable,
      public BSplinePatch<real_t, GeoDim, sizeof...(Degrees)> {
  /// @brief Enable access to private members
  template <typename BSplineCore> requires SplineCoreType<BSplineCore> friend class BSplineCommon;

protected:
  /// @brief Dimension of the parametric space
  /// \f$\hat\Omega=[0,1]^{d_\text{par}}\f$
  static constexpr const short_t parDim_ = sizeof...(Degrees);

  /// @brief Dimension of the geometric space
  /// \f$\Omega\subset\mathbb{R}^{d_\text{geo}}\f$
  static constexpr const short_t geoDim_ = GeoDim;

  /// @brief Array storing the degrees
  /// \f$\left(p_d\right)_{d=1}^{d_\text{par}}\f$
  static constexpr const std::array<short_t, parDim_> degrees_ = {Degrees...};

  /// @brief Array storing the sizes of the knot vectors
  /// \f$\left(n_d+p_d+1\right)_{d=1}^{d_\text{par}}\f$
  std::array<int64_t, parDim_> nknots_;

  /// @brief Array storing the sizes of the coefficients of the
  /// control net \f$\left(n_d\right)_{d=1}^{d_\text{par}}\f$
  std::array<int64_t, parDim_> ncoeffs_;

  /// @brief Array storing the sizes of the coefficients of the
  /// control net \f$\left(n_d\right)_{d=1}^{d_\text{par}}\f$ in
  /// reverse order (needed for coeffs_view)
  std::array<int64_t, parDim_> ncoeffs_reverse_;

  /// @brief Array storing the knot vectors
  /// \f$\left(\left(t_{i_d}\right)_{i_d=1}^{n_d+p_d+1}\right)_{d=1}^{d_\text{par}}\f$
  utils::TensorArray<parDim_> knots_;

  /// @brief Array storing the coefficients of the control net
  /// \f$\left(\mathbf{c}_{i_d}\right)_{i_d=1}^{n_d}\f$,
  /// \f$\mathbf{c}_{i_d}\in\mathbb{R}^{d_\text{geo}}\f$
  utils::TensorArray<geoDim_> coeffs_;

  /// @brief Options
  Options<real_t> options_;

public:
  /// @brief Value type
  using value_type = real_t;

  /// @brief Deduces the type of the template template parameter `BSpline`
  /// when exposed to the class template parameters `real_t` and
  /// `GeoDim`, and the `Degrees` parameter pack. The optional
  /// template parameter `degree_elevate` can be used to
  /// (de-)elevate the degrees by an additive constant
  template <template <typename, short_t, short_t...> class BSpline,
            std::make_signed<short_t>::type degree_elevate = 0>
  using derived_type = BSpline<real_t, GeoDim, (Degrees + degree_elevate)...>;

  /// @brief Deduces the self-type possibly degrees (de-)elevated by
  /// the additive constant `degree_elevate`
  template <std::make_signed<short_t>::type degree_elevate = 0>
  using self_type = derived_type<UniformBSplineCore, degree_elevate>;

  /// @brief Deduces the derived self-type when exposed to different
  /// class template parameters `real_t` and `GeoDim`, and the
  /// `Degrees` parameter pack
  template <typename other_t, short_t GeoDim_, short_t... Degrees_>
  using derived_self_type = UniformBSplineCore<other_t, GeoDim_, Degrees_...>;

  /// @brief Deduces the derived self-type when exposed to a
  /// different class template parameter `real_t`
  template <typename other_t>
  using real_derived_self_type =
      UniformBSplineCore<other_t, GeoDim, Degrees...>;

  /// @brief Returns the `device` property
  inline torch::Device device() const noexcept override {
    return options_.device();
  }

  /// @brief Returns the `device_index` property
  inline int32_t device_index() const noexcept override {
    return options_.device_index();
  }

  /// @brief Returns the `dtype` property
  inline torch::Dtype dtype() const noexcept override {
    return options_.dtype();
  }

  /// @brief Returns the `layout` property
  inline torch::Layout layout() const noexcept override {
    return options_.layout();
  }

  /// @brief Returns the `requires_grad` property
  inline bool requires_grad() const noexcept override {
    return options_.requires_grad();
  }

  /// @brief Returns the `pinned_memory` property
  inline bool pinned_memory() const noexcept override {
    return options_.pinned_memory();
  }

  /// @brief Returns true if the layout is sparse
  inline bool is_sparse() const noexcept override {
    return options_.is_sparse();
  }

  /// @brief Returns true if the B-spline is uniform
  inline static constexpr bool is_uniform() noexcept { return true; }

  /// @brief Returns true if the B-spline is non-uniform
  inline static constexpr bool is_nonuniform() noexcept { return false; }

  /// @brief Sets the B-spline object's `requires_grad` property
  ///
  /// @note: It is only necessary to set `requires_grad` to true if
  /// gradients with respect to B-spline entities, e.g., the control
  /// points should be computed. For computing the gradients with
  /// respect to the sampling points the B-spline's `requires_grad`
  /// property can be false.
  inline UniformBSplineCore &
  set_requires_grad(bool requires_grad) noexcept override {
    if (options_.requires_grad() == requires_grad)
      return *this;

    for (short_t i = 0; i < parDim_; ++i)
      knots_[i].set_requires_grad(requires_grad);

    for (short_t i = 0; i < geoDim_; ++i)
      coeffs_[i].set_requires_grad(requires_grad);

    Options<real_t> tmp(options_.requires_grad(requires_grad));
    options_.~Options<real_t>();
    new (&options_) Options<real_t>(tmp);

    return *this;
  }

  /// @brief Returns a constant reference to the B-spline object's options
  inline const Options<real_t> &options() const noexcept { return options_; }

  /// @brief Default constructor
  ///
  /// @param[in] options Options configuration
  UniformBSplineCore(Options<real_t> options = Options<real_t>{})
      : options_(options) {
    nknots_.fill(0);
    ncoeffs_.fill(0);
    ncoeffs_reverse_.fill(0);
  }

  /// @brief Constructor for equidistant knot vectors
  ///
  /// @param[in] ncoeffs Number of coefficients per parametric dimension
  ///
  /// @param[in] init Type of initialization
  ///
  /// @param[in] options Options configuration
  UniformBSplineCore(const std::array<int64_t, parDim_> &ncoeffs,
                     enum init init = init::greville,
                     Options<real_t> options = Options<real_t>{})
      : options_(options), ncoeffs_(ncoeffs), ncoeffs_reverse_(ncoeffs) {
    // Reverse ncoeffs
    std::reverse(ncoeffs_reverse_.begin(), ncoeffs_reverse_.end());

    // Initialize knot vectors
    init_knots();

    // Initialize coefficients
    init_coeffs(init);
  }

  /// @brief Constructor for equidistant knot vectors
  ///
  /// @param[in] ncoeffs Number of coefficients per parametric dimension
  ///
  /// @param[in] coeffs Vectors of coefficients per parametric dimension
  ///
  /// @param[in] clone  If true, coefficients will be cloned. Otherwise,
  /// coefficients will be aliased
  ///
  /// @param[in] options Options configuration
  ///
  /// @note It is not checked whether vectors of coefficients are
  /// compatible with the given Options object if clone is false.
  UniformBSplineCore(const std::array<int64_t, parDim_> &ncoeffs,
                     const utils::TensorArray<geoDim_> &coeffs,
                     bool clone = false,
                     Options<real_t> options = Options<real_t>{})
      : options_(options), ncoeffs_(ncoeffs), ncoeffs_reverse_(ncoeffs) {
    // Reverse ncoeffs
    std::reverse(ncoeffs_reverse_.begin(), ncoeffs_reverse_.end());

    // Initialize knot vectors
    init_knots();

    // Check compatibility
    for (short_t i = 0; i < geoDim_; ++i)
      if (coeffs[i].numel() != ncumcoeffs())
        throw std::runtime_error("Invalid number of coefficients");

    // Copy/clone coefficients
    if (clone)
      for (short_t i = 0; i < geoDim_; ++i)
        coeffs_[i] = coeffs[i]
                         .clone()
                         .to(options.requires_grad(false))
                         .requires_grad_(options.requires_grad());
    else
      for (short_t i = 0; i < geoDim_; ++i)
        coeffs_[i] = coeffs[i];
  }

  /// @brief Constructor for equidistant knot vectors
  ///
  /// @param[in] ncoeffs Number of coefficients per parametric dimension
  ///
  /// @param[in] coeffs Vectors of coefficients per parametric dimension
  ///
  /// @param[in] options Options configuration
  ///
  /// @note It is not checked whether vectors of coefficients are
  /// compatible with the given Options object if clone is false.
  UniformBSplineCore(const std::array<int64_t, parDim_> &ncoeffs,
                     utils::TensorArray<geoDim_> &&coeffs,
                     Options<real_t> options = Options<real_t>{})
      : options_(options), ncoeffs_(ncoeffs), ncoeffs_reverse_(ncoeffs),
        coeffs_(std::move(coeffs)) {
    // Reverse ncoeffs
    std::reverse(ncoeffs_reverse_.begin(), ncoeffs_reverse_.end());

    // Initialize knot vectors
    init_knots();
  }

  /// @brief Copy constructor
  ///
  /// @param[in] other Uniform B-spline object to copy
  ///
  /// @param[in] options Options configuration
  template <typename other_t>
  UniformBSplineCore(
      const UniformBSplineCore<other_t, GeoDim, Degrees...> &other,
      Options<real_t> options = Options<real_t>{})
      : options_(options), ncoeffs_(other.ncoeffs()),
        ncoeffs_reverse_(ncoeffs_), nknots_(other.nknots()) {
    // Reverse ncoeffs
    std::reverse(ncoeffs_reverse_.begin(), ncoeffs_reverse_.end());

    // Clone coefficients
    for (short_t i = 0; i < geoDim_; ++i)
      coeffs_[i] = other.coeffs(i)
                       .clone()
                       .to(options.requires_grad(false))
                       .requires_grad_(options.requires_grad());

    // Clone knot vectors
    for (short_t i = 0; i < parDim_; ++i)
      knots_[i] = other.knots(i)
                      .clone()
                      .to(options.requires_grad(false))
                      .requires_grad_(options.requires_grad());
  }

  /// @brief Returns the parametric dimension
  ///
  /// @result Number of parametric dimensions
  inline static constexpr short_t parDim() noexcept { return parDim_; }

  /// @brief Returns the geometric dimension
  ///
  /// @result Number of geometric dimensions
  inline static constexpr short_t geoDim() noexcept { return geoDim_; }

  /// @brief Returns a constant reference to the array of degrees
  ///
  /// @result Array of degrees for all parametric dimensions
  inline static constexpr const std::array<short_t, parDim_> &
  degrees() noexcept {
    return degrees_;
  }

  /// @brief Returns a constant reference to the degree in the
  /// \f$i\f$-th dimension
  ///
  /// @param[in] i Parametric dimension
  ///
  /// @result Degree for the given parametric dimension
  inline static constexpr const short_t &degree(short_t i) noexcept {
    assert(i >= 0 && i < parDim_);
    return degrees_[i];
  }

  /// @brief Returns a constant reference to the array of knot
  /// vectors
  ///
  /// @result Array of knot vectors
  inline const utils::TensorArray<parDim_> &knots() const noexcept {
    return knots_;
  }

  /// @brief Returns a constant reference to the knot vector in the
  /// \f$i\f$-th dimension
  ///
  /// @param[in] i Parametric dimension
  ///
  /// @result Knot vector for the given parametric dimension
  inline const torch::Tensor &knots(short_t i) const noexcept {
    assert(i >= 0 && i < parDim_);
    return knots_[i];
  }

  /// @brief Returns a non-constant reference to the array of knot
  /// vectors
  ///
  /// @result Array of knot vectors
  inline utils::TensorArray<parDim_> &knots() noexcept { return knots_; }

  /// @brief Returns a non-constant reference to the knot vector in
  /// the \f$i\f$-th dimension
  ///
  /// @param[in] i Parametric dimension
  ///
  /// @result Knot vector for the given parametric dimension
  inline torch::Tensor &knots(short_t i) noexcept {
    assert(i >= 0 && i < parDim_);
    return knots_[i];
  }

  /// @brief Returns a constant reference to the array of knot
  /// vector dimensions
  ///
  /// @result Array of knot vector dimensions
  inline const std::array<int64_t, parDim_> &nknots() const noexcept {
    return nknots_;
  }

  /// @brief Returns the dimension of the knot vector in the
  /// \f$i\f$-th dimension
  ///
  /// @param[in] i Parametric dimension
  ///
  /// @result Knot vector dimension for the given parametric dimension
  inline int64_t nknots(short_t i) const noexcept {
    assert(i >= 0 && i < parDim_);
    return nknots_[i];
  }

  /// @brief Returns a constant reference to the array of
  /// coefficient vectors
  ///
  /// @result Array of coefficient vectors
  inline const utils::TensorArray<geoDim_> &coeffs() const noexcept {
    return coeffs_;
  }

  /// @brief Returns a constant reference to the coefficient vector
  /// in the \f$i\f$-th dimension
  ///
  /// @param[in] i Geometric dimension
  ///
  /// @result Coefficient vector for the given geometric dimension
  inline const torch::Tensor &coeffs(short_t i) const noexcept {
    assert(i >= 0 && i < geoDim_);
    return coeffs_[i];
  }

  /// @brief Returns a non-constant reference to the array of
  /// coefficient vectors
  ///
  /// @result Array of coefficient vectord
  inline utils::TensorArray<geoDim_> &coeffs() noexcept { return coeffs_; }

  /// @brief Returns a non-constant reference to the coefficient
  /// vector in the \f$i\f$-th dimension
  ///
  /// @param[in] i Geometric dimension
  ///
  /// @result Coefficient vector for the given geometric dimension
  inline torch::Tensor &coeffs(short_t i) noexcept {
    assert(i >= 0 && i < geoDim_);
    return coeffs_[i];
  }

  /// @brief Returns an array of views to the coefficient vectors
  ///
  /// @result Array of views to the coefficient vectors
  inline utils::TensorArray<geoDim_> coeffs_view() const noexcept {
    utils::TensorArray<geoDim_> coeffs;
    for (short_t i = 0; i < geoDim_; ++i)
      coeffs[i] = coeffs_view(i);
    return coeffs;
  }

  /// @brief Returns a view to the coefficient vector in the
  /// \f$i\f$-th dimension
  ///
  /// @param[in] i Geometric dimension
  ///
  /// @result View of the coefficient vector for the given geometric dimension
  inline const auto coeffs_view(short_t i) const noexcept {
    assert(i >= 0 && i < geoDim_);
    if constexpr (parDim_ > 1)
      if (coeffs_[i].dim() > 1)
        return coeffs_[i].view(utils::to_ArrayRef(ncoeffs_reverse_) + (-1_i64));
      else
        return coeffs_[i].view(utils::to_ArrayRef(ncoeffs_reverse_));
    else
      return coeffs_[i];
  }

  /// @brief Returns the total number of coefficients
  ///
  /// @result Total number of coefficients
  inline int64_t ncumcoeffs() const noexcept {
    int64_t s = 1;

    for (short_t i = 0; i < parDim_; ++i)
      s *= ncoeffs(i);

    return s;
  }

  /// @brief Returns a constant reference to the array of
  /// coefficient vector dimensions
  ///
  /// @result Array of coefficient vector dimensions
  inline const std::array<int64_t, parDim_> &ncoeffs() const noexcept {
    return ncoeffs_;
  }

  /// @brief Returns the total number of coefficients in the
  /// \f$i\f$-th direction
  ///
  /// @param[in] i Parametric dimension
  ///
  /// @result Total number of coefficients in given parametric dimension
  inline int64_t ncoeffs(short_t i) const noexcept {
    assert(i >= 0 && i < parDim_);
    return ncoeffs_[i];
  }

private:
  /// @brief Returns all coefficients as a single tensor
  ///
  /// @result Tensor of coefficients
  template <std::size_t... Is>
  inline torch::Tensor as_tensor_(std::index_sequence<Is...>) const noexcept {
    return torch::cat({coeffs_[Is]...});
  }

public:
  /// @brief Returns all coefficients as a single tensor
  ///
  /// @result Tensor of coefficients
  inline torch::Tensor as_tensor() const noexcept override {
    return as_tensor_(std::make_index_sequence<geoDim_>{});
  }

private:
  /// @brief Sets all coefficients from a single tensor
  ///
  /// @result Updates spline object
  template <std::size_t... Is>
  inline UniformBSplineCore &
  from_tensor_(std::index_sequence<Is...>,
               const torch::Tensor &tensor) noexcept {
    ((coeffs_[Is] = tensor.index(
          {torch::indexing::Slice(Is * ncumcoeffs(), (Is + 1) * ncumcoeffs()),
           "..."})),
     ...);
    return *this;
  }

public:
  /// @brief Sets all coefficients from a single tensor
  ///
  /// @param[in] tensor Tensor from which to extract the coefficients
  ///
  /// @result Updated spline object
  inline UniformBSplineCore &
  from_tensor(const torch::Tensor &tensor) noexcept override {
    return from_tensor_(std::make_index_sequence<geoDim_>{}, tensor);
  }

  /// @brief Returns the size of the single tensor representation of
  /// all coefficients
  //
  /// @result Size of the tensor
  inline int64_t as_tensor_size() const noexcept override {
    return geoDim_ * ncumcoeffs();
  }

  /// @brief Returns the Greville abscissae
  ///
  /// The Greville abscissae are defined as
  ///
  /// \f[
  ///   g_{i_d} = \frac{\xi_{i_d+1} + \xi_{i_d+2} + \dots +
  ///   \xi_{i_d+p_d+1}}{p_d-1}
  /// \f]
  ///
  ///
  /// @param[in] interior If true only interior Greville abscissae are
  /// considered
  ///
  /// @result Array of Greville abscissae
  inline auto greville(bool interior = false) const {
    if constexpr (parDim_ == 0) {
      return torch::zeros(1, options_);
    } else {
      utils::TensorArray<parDim_> coeffs;

      // Fill coefficients with the tensor-product of Greville
      // abscissae values per univariate dimension
      for (short_t i = 0; i < parDim_; ++i) {
        coeffs[i] = torch::ones(1, options_);

        for (short_t j = 0; j < parDim_; ++j) {
          if (i == j) {

            int64_t offset = interior ? 1 : 0;
            int64_t count = ncoeffs_[j] - (interior ? 2 : 0);

            // idx_base: (count, 1)
            auto idx_base = torch::arange(count, options_.requires_grad(false).template dtype<int64_t>()).unsqueeze(1);

            // offsets: (1, degree)
            auto offsets = torch::arange(1, degrees_[j] + 1, options_.requires_grad(false).template dtype<int64_t>()).unsqueeze(0);

            // indices: (count, degree)
            auto indices = idx_base + offset + offsets;

            // Gather relevant knot values: shape (count, degree)
            auto gathered = knots_[j].index_select(0, indices.flatten()).view({count, degrees_[j]});
            
            // Compute mean along degree dimension (dim=1)
            auto greville_ = gathered.mean(1);
            
            coeffs[i] = torch::kron(greville_, coeffs[i]);
          } else
            coeffs[i] =
              torch::kron(torch::ones(ncoeffs_[j], options_), coeffs[i]);
        }

        // Enable gradient calculation for non-leaf tensor
        if (options_.requires_grad())
          coeffs[i].retain_grad();
      }

      return coeffs;
    }
  }

  /// @brief Returns the value of the B-spline object from
  /// precomputed basis function
  ///
  /// This function implements steps 2-3 of algorithm \ref
  /// BSplineEvaluation for univariate B-splines
  /// (i.e. \f$d_\text{par}=1\f$)
  ///
  /// @param[in] basfunc Value(s) of the multivariate B-spline basis
  ///                    functions evaluated at the point(s) `xi`
  ///
  /// @param[in] coeff_indices Indices where to evaluate the coefficients
  ///
  /// @param[in] numeval Number of evaluation points
  ///
  /// @param[in] sizes Dimension of the result
  ///
  /// @result Value(s) of the univariate B-spline object
  ///
  ///
  /// @note This function does not work of the basis functions are
  /// evaluated with memory_optimized flag to true @{
  inline utils::BlockTensor<torch::Tensor, 1, geoDim_>
  eval_from_precomputed(const torch::Tensor &basfunc,
                        const torch::Tensor &coeff_indices, int64_t numeval,
                        torch::IntArrayRef sizes) const override {

    utils::BlockTensor<torch::Tensor, 1, geoDim_> result;

    for (short_t i = 0; i < geoDim_; ++i)
      result.set(
          i, utils::dotproduct(
                 basfunc,
                 coeffs(i).index_select(0, coeff_indices).view({-1, numeval}))
                 .view(sizes));
    return result;
  }

  inline utils::BlockTensor<torch::Tensor, 1, geoDim_>
  eval_from_precomputed(const utils::TensorArray<parDim_> &basfunc,
                        const torch::Tensor &coeff_indices, int64_t numeval,
                        torch::IntArrayRef sizes) const override {

    utils::BlockTensor<torch::Tensor, 1, geoDim_> result;

    if constexpr (parDim_ == 0) {
      for (short_t i = 0; i < geoDim_; ++i)
        result.set(i, coeffs_[i]);
    }

    else {
      // Lambda expression to evaluate the spline function
      std::function<torch::Tensor(short_t, short_t)> eval_;

      eval_ = [&, this](short_t i, short_t dim) {
        if (dim == 0) {
          return torch::matmul(coeffs(i)
                                   .index_select(0, coeff_indices)
                                   .view({numeval, -1, degrees_[0] + 1}),
                               basfunc[0].view({numeval, -1, 1}));
        } else {
          return torch::matmul(
              (eval_(i, dim - 1)).view({numeval, -1, degrees_[dim] + 1}),
              basfunc[dim].view({numeval, -1, 1}));
        }
      };

      for (short_t i = 0; i < geoDim_; ++i)
        result.set(i, (eval_(i, parDim_ - 1)).view(sizes));
    }
    return result;
  }
  /// @}

  /// @brief Returns the value of the B-spline object in the point `xi`
  ///
  /// This implementation follows the procedure described in
  /// Chapters 2 and 3 of \cite Lyche:2011.
  ///
  /// @anchor BSplineEvaluation **Algorithm: B-spline evaluation**
  ///
  /// 1. Determine the indices
  ///    \f$(i_d)_{d=1}^{d_\text{par}}\f$ of the knot spans such that
  ///
  ///    \f[
  ///      \boldsymbol{\xi} = \left(\xi_1, \dots, \xi_{d_\text{par}}\right)^\top
  ///      \in \bigotimes_{d=1}^{d_\text{par}} [t_{i_d}, t_{i_d+1}).
  ///    \f]
  ///
  /// 2. Evaluate the vectors of univariate B-spline basis functions (or
  ///    their derivatives) that are non-zero at \f$\boldsymbol{\xi}\f$
  ///
  ///    \f[
  ///      D^{r_d}\mathbf{B}_d =
  ///      \left( D^{r_d} B_{i_d-p_d,p_d}, \dots, D^{r_d} B_{i_d,p_d}
  ///      \right)^\top,
  ///    \f]
  ///
  ///    where \f$ p_d \f$ is the degree of the \f$d\f$-th
  ///    univariate B-spline and \f$ r_d \f$ denotes the requested
  ///    derivative in the \f$d\f$-direction.
  ///
  /// 3. Multiply the tensor-product of the above row vectors by the
  ///    column vector of control points
  ///
  ///    \f[
  ///    \left( \bigotimes_{d=1}^{d_\text{par}} D^{r_d}\mathbf{B}_d \right)
  ///    \cdot \mathbf{c}_\mathcal{J}, \f]
  ///
  ///    where \f$\mathcal{J}\f$ is the subset of global indices
  ///    that belong to the coefficients
  ///
  ///    \f[
  ///    \mathbf{c}_{i_1-p_1:i_1,\dots,i_\text{par}-p_\text{par}:i_\text{par}}
  ///    \f]
  ///
  /// @tparam deriv Composition of derivative indicators of type \ref deriv
  ///
  /// @param[in] xi Point(s) where to evaluate the multivariate B-spline object
  ///
  /// @result Value(s) of the multivariate B-spline evaluated at the point(s)
  /// `xi`
  ///
  /// @{
  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval(const torch::Tensor &xi) const {
    if constexpr (parDim_ == 1)
      return eval<deriv, memory_optimized>(utils::TensorArray1({xi}));
    else
      throw std::runtime_error("Invalid parametric dimension");
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval(const utils::TensorArray<parDim_> &xi) const {
    return eval<deriv, memory_optimized>(xi, find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns the value of the univariate B-spline object in
  /// the points `xi`
  ///
  /// This function implements steps 2-3 of algorithm \ref
  /// BSplineEvaluation for univariate B-splines
  /// (i.e. \f$d_\text{par}=1\f$)
  ///
  /// @tparam deriv Composition of derivative indicators of type \ref deriv
  ///
  /// @param[in] xi Point(s) where to evaluate the univariate B-spline object
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the univariate
  /// B-spline object
  ///
  /// @result Value(s) of the univariate B-spline evaluated at the point(s) `xi`
  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval(const utils::TensorArray<parDim_> &xi,
                   const utils::TensorArray<parDim_> &knot_indices) const {
    return eval<deriv, memory_optimized>(
        xi, knot_indices, find_coeff_indices<memory_optimized>(knot_indices));
  }

  /// @brief Returns the value of the univariate B-spline object in
  /// the points `xi`
  ///
  /// This function implements steps 2-3 of algorithm \ref
  /// BSplineEvaluation for univariate B-splines
  /// (i.e. \f$d_\text{par}=1\f$)
  ///
  /// @tparam deriv Composition of derivative indicators of type \ref deriv
  ///
  /// @param[in] xi Point(s) where to evaluate the univariate B-spline object
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the univariate
  /// B-spline object
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// univariate B-spline object
  ///
  /// @result Value(s) of the univariate B-spline evaluated at the point(s) `xi`
  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval(const utils::TensorArray<parDim_> &xi,
                   const utils::TensorArray<parDim_> &knot_indices,
                   const torch::Tensor &coeff_indices) const {

    utils::BlockTensor<torch::Tensor, 1, geoDim_> result;

    if constexpr (parDim_ == 0) {
      for (short_t i = 0; i < geoDim_; ++i)
        if constexpr (deriv == deriv::func)
          result.set(i, coeffs_[i]);
        else
          result.set(i, torch::zeros_like(coeffs_[i]));
      return result;
    } // parDim == 0

    else {

      // Check compatibility of arguments
      for (short_t i = 0; i < parDim_; ++i)
        assert(xi[i].sizes() == knot_indices[i].sizes());
      for (short_t i = 1; i < parDim_; ++i)
        assert(xi[0].sizes() == xi[i].sizes());

      if constexpr (memory_optimized) {
        // memory-optimized

        if (coeffs(0).dim() > 1)
          throw std::runtime_error(
              "Memory-optimized evaluation requires single-valued coefficient");

        else {
          auto basfunc =
              eval_basfunc<deriv, memory_optimized>(xi, knot_indices);

          // Lambda expression to evaluate the spline function
          std::function<torch::Tensor(short_t, short_t)> eval_;

          eval_ = [&, this](short_t i, short_t dim) {
            if (dim == 0) {
              return torch::matmul(
                  coeffs(i)
                      .index_select(0, coeff_indices)
                      .view({xi[0].numel(), -1, degrees_[0] + 1}),
                  basfunc[0].view({xi[0].numel(), -1, 1}));
            } else {
              return torch::matmul(
                  (eval_(i, dim - 1))
                      .view({xi[0].numel(), -1, degrees_[dim] + 1}),
                  basfunc[dim].view({xi[0].numel(), -1, 1}));
            }
          };

          for (short_t i = 0; i < geoDim_; ++i)
            result.set(i, (eval_(i, parDim_ - 1)).view(xi[0].sizes()));

          return result;
        } // coeffs(0).dim() > 1
      }

      else {
        // not memory-optimized

        auto basfunc = eval_basfunc<deriv, memory_optimized>(xi, knot_indices);

        if (coeffs(0).dim() > 1) {
          // coeffs has extra dimension
          auto sizes = xi[0].sizes() + (-1_i64);
          for (short_t i = 0; i < geoDim_; ++i)
            result.set(i, utils::dotproduct(basfunc.unsqueeze(-1),
                                            coeffs(i)
                                                .index_select(0, coeff_indices)
                                                .view({-1, xi[0].numel(),
                                                       coeffs(i).size(-1)}))
                              .view(sizes));
        } else {
          // coeffs does not have extra dimension
          for (short_t i = 0; i < geoDim_; ++i)
            result.set(i, utils::dotproduct(basfunc,
                                            coeffs(i)
                                                .index_select(0, coeff_indices)
                                                .view({-1, xi[0].numel()}))
                              .view(xi[0].sizes()));
        }
        return result;
      }
    }
  }

  /// @brief Returns the indices of knot spans containing `xi`
  ///
  /// This function returns the indices
  /// \f$(i_d)_{d=1}^{d_\text{par}}\f$ of the knot spans such that
  ///
  /// \f[
  ///   \boldsymbol{\xi} \in [t_{i_1}, t_{i_1+1}) \times [t_{i_2}, t_{i_2+1})
  ///   \times \dots \times [t_{i_{d_\text{par}}}, t_{i_{d_\text{par}}+1}).
  /// \f]
  ///
  /// The indices are returned as `utils::TensorArray<parDim_>` in the
  /// same order as provided in `xi`
  ///
  /// @param[in] xi Point(s) where to evaluate the B-spline object
  ///
  /// @result Indices of the knot spans containing `xi`
  ///
  /// @{
  inline auto find_knot_indices(const torch::Tensor &xi) const noexcept {
    if constexpr (parDim_ == 0)
      return torch::zeros_like(coeffs_[0]).to(torch::kInt64);
    else
      return find_knot_indices(utils::TensorArray1({xi}));
  }

  inline utils::TensorArray<parDim_>
  find_knot_indices(const utils::TensorArray<parDim_> &xi) const noexcept {
    if constexpr (parDim_ == 0)
      return utils::TensorArray<parDim_>{};
    else {
      utils::TensorArray<parDim_> result;

      for (short_t i = 0; i < parDim_; ++i)
        result[i] =
            torch::min(
                torch::full_like(xi[i], ncoeffs_[i] - 1, options_),
                torch::floor(xi[i] * (ncoeffs_[i] - degrees_[i]) + degrees_[i]))
                .to(torch::kInt64);

      return result;
    }
  }
  /// @}

  /// @brief Returns the indices of the coefficients corresponding to the knot
  /// indices `indices`
  ///
  /// @param[in] indices Indices of the knot spans
  ///
  /// @result Indices of the coefficients corresponding to the knot indices
  ///
  /// @{
  template <bool memory_optimized = false>
  inline auto find_coeff_indices(const torch::Tensor &indices) const {
    if constexpr (parDim_ == 0)
      return torch::zeros_like(coeffs_[0]).to(torch::kInt64);
    else
      return find_coeff_indices<memory_optimized>(
          utils::TensorArray1({indices}));
  }

  template <bool memory_optimized = false>
  inline auto
  find_coeff_indices(const utils::TensorArray<parDim_> &indices) const {
    using utils::operator-;

    if constexpr (parDim_ == 0)
      return torch::zeros_like(coeffs_[0]).to(torch::kInt64);
    else if constexpr (parDim_ == 1)
      return utils::VSlice<memory_optimized>(indices[0].flatten(), -degrees_[0],
                                             1);
    else {
      return utils::VSlice<memory_optimized>(
          TENSORARRAY_FORALL(indices, flatten),
          utils::make_array<int64_t>(-degrees_),
          utils::make_array<int64_t, parDim_>(1),
          utils::remove_from_back(ncoeffs_));
    }
  }
  /// @}

  /// @brief Returns the vector of multivariate B-spline basis
  /// functions (or their derivatives) evaluated in the point `xi`
  ///
  /// @param[in] xi Point(s) where to evaluate the B-spline object
  ///
  /// @result Multivariate B-spline basis functions (or their derivatives)
  /// evaluated in the point `xi`
  ///
  /// @{
  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval_basfunc(const torch::Tensor &xi) const {
    if constexpr (parDim_ == 0) {
      if constexpr (deriv == deriv::func)
        return torch::ones_like(coeffs_[0]);
      else
        return torch::zeros_like(coeffs_[0]);
    } else
      return eval_basfunc<deriv, memory_optimized>(utils::TensorArray1({xi}));
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval_basfunc(const utils::TensorArray<parDim_> &xi) const {
    if constexpr (parDim_ == 0) {
      if constexpr (deriv == deriv::func)
        return torch::ones_like(coeffs_[0]);
      else
        return torch::zeros_like(coeffs_[0]);
    } else
      return eval_basfunc<deriv, memory_optimized>(xi, find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns the vector of multivariate B-spline basis
  /// functions (or their derivatives) evaluated in the point `xi`
  ///
  /// @param[in] xi Point(s) where to evaluate the B-spline object
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the univariate
  /// B-spline object
  ///
  /// @result Multivariate B-spline basis functions (or their derivatives)
  /// evaluated in the point `xi`
  ///
  /// @{
  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval_basfunc(const torch::Tensor &xi,
                           const torch::Tensor &knot_indices) const {
    if constexpr (parDim_ == 0) {
      if constexpr (deriv == deriv::func)
        return torch::ones_like(coeffs_[0]);
      else
        return torch::zeros_like(coeffs_[0]);
    } else
      return eval_basfunc<deriv, memory_optimized>(
          utils::TensorArray1({xi}), utils::TensorArray1({knot_indices}));
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto
  eval_basfunc(const utils::TensorArray<parDim_> &xi,
               const utils::TensorArray<parDim_> &knot_indices) const {

    if constexpr (parDim_ == 0) {
      if constexpr (deriv == deriv::func)
        return torch::ones_like(coeffs_[0]);
      else
        return torch::zeros_like(coeffs_[0]);
    }

    else {
      // Check compatibility of arguments
      for (short_t i = 0; i < parDim_; ++i)
        assert(xi[i].sizes() == knot_indices[i].sizes());
      for (short_t i = 1; i < parDim_; ++i)
        assert(xi[0].sizes() == xi[i].sizes());

      if constexpr (memory_optimized) {

        // Lambda expression to evaluate the vector of basis functions
        auto basfunc_ = [&,
                         this]<std::size_t... Is>(std::index_sequence<Is...>) {
          return utils::TensorArray<parDim_>{
              (eval_prefactor<degrees_[Is],
                              (short_t)deriv /
                                  utils::integer_pow<10, Is>::value % 10>() *
               eval_basfunc_univariate<
                   degrees_[Is], Is,
                   (short_t)deriv / utils::integer_pow<10, Is>::value % 10>(
                   xi[Is].flatten(), knot_indices[Is].flatten())
                   .transpose(0, 1))...};
        };

        return basfunc_(std::make_index_sequence<parDim_>{});

      }

      else /* not memory optimize */ {

        if constexpr (parDim_ == 1) {
          return eval_prefactor<degrees_[0], (short_t)deriv % 10>() *
                 eval_basfunc_univariate<degrees_[0], 0, (short_t)deriv % 10>(
                     xi[0].flatten(), knot_indices[0].flatten());

        } else {

          // Lambda expression to evaluate the cumulated basis function
          auto basfunc_ = [&, this]<std::size_t... Is>(
                              std::index_sequence<Is...>) {
            return (1 * ... *
                    (eval_prefactor<degrees_[Is],
                                    (short_t)deriv /
                                        utils::integer_pow<10, Is>::value %
                                        10>())) *
                   utils::kronproduct(
                       eval_basfunc_univariate<
                           degrees_[Is], Is,
                           (short_t)deriv / utils::integer_pow<10, Is>::value %
                               10>(xi[Is].flatten(),
                                   knot_indices[Is].flatten())...);
          };

          // Note that the kronecker product must be called in reverse order
          return basfunc_(utils::make_reverse_index_sequence<parDim_>{});
        }
      }
    }
  }
  /// @}

  /// @brief Transforms the coefficients based on the given mapping
  inline UniformBSplineCore &
  transform(const std::function<
            std::array<real_t, geoDim_>(const std::array<real_t, parDim_> &)>
                transformation) {
    static_assert(parDim_ <= 4, "Unsupported parametric dimension");

    // 0D
    if constexpr (parDim_ == 0) {
      auto c = transformation(std::array<real_t, parDim_>{});
      for (short_t d = 0; d < geoDim_; ++d)
        coeffs_[d].detach()[0] = c[d];
    }

    // 1D
    else if constexpr (parDim_ == 1) {
#pragma omp parallel for
      for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
        auto c = transformation(
            std::array<real_t, parDim_>{i / real_t(ncoeffs_[0] - 1)});
        for (short_t d = 0; d < geoDim_; ++d)
          coeffs_[d].detach()[i] = c[d];
      }
    }

    // 2D
    else if constexpr (parDim_ == 2) {
#pragma omp parallel for collapse(2)
      for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          auto c = transformation(std::array<real_t, parDim_>{
              i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1)});
          for (short_t d = 0; d < geoDim_; ++d)
            coeffs_[d].detach()[j * ncoeffs_[0] + i] = c[d];
        }
      }
    }

    // 3D
    else if constexpr (parDim_ == 3) {
#pragma omp parallel for collapse(3)
      for (int64_t k = 0; k < ncoeffs_[2]; ++k) {
        for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
          for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
            auto c = transformation(std::array<real_t, parDim_>{
                i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1),
                k / real_t(ncoeffs_[2] - 1)});
            for (short_t d = 0; d < geoDim_; ++d)
              coeffs_[d].detach()[k * ncoeffs_[0] * ncoeffs_[1] +
                                  j * ncoeffs_[0] + i] = c[d];
          }
        }
      }
    }

    // 4D
    else if constexpr (parDim_ == 4) {
#pragma omp parallel for collapse(4)
      for (int64_t l = 0; l < ncoeffs_[3]; ++l) {
        for (int64_t k = 0; k < ncoeffs_[2]; ++k) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
            for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
              auto c = transformation(std::array<real_t, parDim_>{
                  i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1),
                  k / real_t(ncoeffs_[2] - 1), l / real_t(ncoeffs_[3] - 1)});
              for (short_t d = 0; d < geoDim_; ++d)
                coeffs_[d]
                    .detach()[l * ncoeffs_[0] * ncoeffs_[1] * ncoeffs_[2] +
                              k * ncoeffs_[0] * ncoeffs_[1] + j * ncoeffs_[0] +
                              i] = c[d];
            }
          }
        }
      }
    } else
      throw std::runtime_error("Unsupported parametric dimension");

    return *this;
  }

  /// @brief Transforms the coefficients based on the given mapping
  template <std::size_t N>
  inline UniformBSplineCore &
  transform(const std::function<
                std::array<real_t, N>(const std::array<real_t, parDim_> &)>
                transformation,
            std::array<short_t, N> dims) {
    static_assert(parDim_ <= 4, "Unsupported parametric dimension");

    // 0D
    if constexpr (parDim_ == 0) {
      auto c = transformation(std::array<real_t, parDim_>{});
      for (std::size_t d = 0; d < N; ++d)
        coeffs_[dims[d]].detach()[0] = c[d];
    }

    // 1D
    else if constexpr (parDim_ == 1) {
#pragma omp parallel for
      for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
        auto c = transformation(
            std::array<real_t, parDim_>{i / real_t(ncoeffs_[0] - 1)});
        for (std::size_t d = 0; d < N; ++d)
          coeffs_[dims[d]].detach()[i] = c[d];
      }
    }

    // 2D
    else if constexpr (parDim_ == 2) {
#pragma omp parallel for collapse(2)
      for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          auto c = transformation(std::array<real_t, parDim_>{
              i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1)});
          for (std::size_t d = 0; d < N; ++d)
            coeffs_[dims[d]].detach()[j * ncoeffs_[0] + i] = c[d];
        }
      }
    }

    // 3D
    else if constexpr (parDim_ == 3) {
#pragma omp parallel for collapse(3)
      for (int64_t k = 0; k < ncoeffs_[2]; ++k) {
        for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
          for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
            auto c = transformation(std::array<real_t, parDim_>{
                i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1),
                k / real_t(ncoeffs_[2] - 1)});
            for (std::size_t d = 0; d < N; ++d)
              coeffs_[dims[d]].detach()[k * ncoeffs_[0] * ncoeffs_[1] +
                                        j * ncoeffs_[0] + i] = c[d];
          }
        }
      }
    }

    // 4D
    else if constexpr (parDim_ == 4) {
#pragma omp parallel for collapse(4)
      for (int64_t l = 0; l < ncoeffs_[3]; ++l) {
        for (int64_t k = 0; k < ncoeffs_[2]; ++k) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
            for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
              auto c = transformation(std::array<real_t, parDim_>{
                  i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1),
                  k / real_t(ncoeffs_[2] - 1), l / real_t(ncoeffs_[3] - 1)});
              for (std::size_t d = 0; d < N; ++d)
                coeffs_[dims[d]]
                    .detach()[l * ncoeffs_[0] * ncoeffs_[1] * ncoeffs_[2] +
                              k * ncoeffs_[0] * ncoeffs_[1] + j * ncoeffs_[0] +
                              i] = c[d];
            }
          }
        }
      }
    } else
      throw std::runtime_error("Unsupported parametric dimension");

    return *this;
  }

  /// @brief Returns the B-spline object as JSON object
  inline nlohmann::json to_json() const override {
    nlohmann::json json;
    json["degrees"] = degrees_;
    json["geoDim"] = geoDim_;
    json["parDim"] = parDim_;
    json["ncoeffs"] = ncoeffs_;
    json["nknots"] = nknots_;
    json["knots"] = knots_to_json();
    json["coeffs"] = coeffs_to_json();

    return json;
  }

  /// @brief Returns the B-spline object's knots as JSON object
  inline nlohmann::json knots_to_json() const {
    return ::iganet::utils::to_json<real_t, 1>(knots_);
  }

  /// @brief Returns the B-spline object's coefficients as JSON object
  inline nlohmann::json coeffs_to_json() const {
    auto coeffs_json = nlohmann::json::array();
    for (short_t g = 0; g < geoDim_; ++g) {
      auto [coeffs_cpu, coeffs_accessor] =
          utils::to_tensorAccessor<real_t, 1>(coeffs_[g], torch::kCPU);

      auto json = nlohmann::json::array();

      if constexpr (parDim_ == 0) {
        json.push_back(coeffs_accessor[0]);
      }

      else {
        for (int64_t i = 0; i < ncumcoeffs(); ++i)
          json.push_back(coeffs_accessor[i]);
      }

      coeffs_json.push_back(json);
    }
    return coeffs_json;
  }

  /// @brief Updates the B-spline object from JSON object
  inline UniformBSplineCore &from_json(const nlohmann::json &json) {

    if (json["geoDim"].get<short_t>() != geoDim_)
      throw std::runtime_error(
          "JSON object provides incompatible geometric dimensions");

    if (json["parDim"].get<short_t>() != parDim_)
      throw std::runtime_error(
          "JSON object provides incompatible parametric dimensions");

    if (json["degrees"].get<std::array<short_t, parDim_>>() != degrees_)
      throw std::runtime_error("JSON object provides incompatible degrees");

    nknots_ = json["nknots"].get<std::array<int64_t, parDim_>>();
    ncoeffs_ = json["ncoeffs"].get<std::array<int64_t, parDim_>>();

    // Reverse ncoeffs
    ncoeffs_reverse_ = ncoeffs_;
    std::reverse(ncoeffs_reverse_.begin(), ncoeffs_reverse_.end());

    auto kv = json["knots"].get<std::array<std::vector<real_t>, parDim_>>();

    for (short_t i = 0; i < parDim_; ++i)
      knots_[i] = utils::to_tensor(kv[i], options_);

    auto c = json["coeffs"].get<std::array<std::vector<real_t>, geoDim_>>();

    for (short_t i = 0; i < geoDim_; ++i)
      coeffs_[i] = utils::to_tensor(c[i], options_);

    return *this;
  }

  /// @brief Returns the B-spline object as XML object
  inline pugi::xml_document to_xml(int id = 0, std::string label = "",
                                   int index = -1) const {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("xml");
    to_xml(root, id, label, index);

    return doc;
  }

  /// @brief Returns the B-spline object as XML node
  inline pugi::xml_node &to_xml(pugi::xml_node &root, int id = 0,
                                std::string label = "", int index = -1) const {
    // add Geometry node
    pugi::xml_node geo = root.append_child("Geometry");

    // 0D parametric dimension
    if constexpr (parDim_ == 0) {
      geo.append_attribute("type") = "Point";

      if (id >= 0)
        geo.append_attribute("id") = id;

      if (index >= 0)
        geo.append_attribute("index") = index;

      if (!label.empty())
        geo.append_attribute("label") = label.c_str();
    }

    // 1D parametric dimension
    else if constexpr (parDim_ == 1) {
      geo.append_attribute("type") = "BSpline";

      if (id >= 0)
        geo.append_attribute("id") = id;

      if (index >= 0)
        geo.append_attribute("index") = index;

      if (!label.empty())
        geo.append_attribute("label") = label.c_str();

      // add Basis node
      pugi::xml_node basis = geo.append_child("Basis");
      basis.append_attribute("type") = "BSplineBasis";

      // add KnotVector node
      pugi::xml_node knots = basis.append_child("KnotVector");
      knots.append_attribute("degree") = degrees_[0];

      std::stringstream ss;
      auto [knots_cpu, knots_accessor] =
          utils::to_tensorAccessor<real_t, 1>(knots_[0], torch::kCPU);
      for (int64_t i = 0; i < nknots_[0]; ++i)
        ss << std::to_string(knots_accessor[i])
           << (i < nknots_[0] - 1 ? " " : "");
      knots.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
    }

    // >1D parametric dimension
    else {
      geo.append_attribute("type") =
          std::string("TensorBSpline").append(std::to_string(parDim_)).c_str();

      if (id >= 0)
        geo.append_attribute("id") = id;

      if (index >= 0)
        geo.append_attribute("index") = index;

      if (!label.empty())
        geo.append_attribute("label") = label.c_str();

      // add Basis node
      pugi::xml_node bases = geo.append_child("Basis");
      bases.append_attribute("type") = std::string("TensorBSplineBasis")
                                           .append(std::to_string(parDim_))
                                           .c_str();

      for (short_t index = 0; index < parDim_; ++index) {
        pugi::xml_node basis = bases.append_child("Basis");
        basis.append_attribute("type") = "BSplineBasis";
        basis.append_attribute("index") = index;

        // add KnotVector node
        pugi::xml_node knots = basis.append_child("KnotVector");
        knots.append_attribute("degree") = degrees_[index];

        std::stringstream ss;
        auto [knots_cpu, knots_accessor] =
            utils::to_tensorAccessor<real_t, 1>(knots_[index], torch::kCPU);
        for (int64_t i = 0; i < nknots_[index]; ++i)
          ss << std::to_string(knots_accessor[i])
             << (i < nknots_[index] - 1 ? " " : "");
        knots.append_child(pugi::node_pcdata).set_value(ss.str().c_str());
      }

    } // parametric dimension

    // add Coefs node
    pugi::xml_node coefs = geo.append_child("coefs");
    coefs.append_attribute("geoDim") = geoDim_;

    auto [coeffs_cpu, coeffs_accessors] =
        utils::to_tensorAccessor<real_t, 1>(coeffs_, torch::kCPU);
    std::stringstream ss;

    if constexpr (parDim_ == 0) {
      for (short_t g = 0; g < geoDim_; ++g)
        ss << std::to_string(coeffs_accessors[g][0]) << " ";

    } else {
      for (int64_t i = 0; i < utils::prod(ncoeffs_); ++i)
        for (short_t g = 0; g < geoDim_; ++g)
          ss << std::to_string(coeffs_accessors[g][i]) << " ";
    }

    coefs.append_child(pugi::node_pcdata).set_value(ss.str().c_str());

    return root;
  }

  /// @brief Updates the B-spline object from XML object
  inline UniformBSplineCore &from_xml(const pugi::xml_document &doc, int id = 0,
                                      std::string label = "", int index = -1) {
    return from_xml(doc.child("xml"), id, label, index);
  }

  /// @brief Updates the B-spline object from XML node
  inline UniformBSplineCore &from_xml(const pugi::xml_node &root, int id = 0,
                                      std::string label = "", int index = -1) {

    std::array<bool, std::max(parDim_, short_t{1})> nknots_found{false},
        ncoeffs_found{false};

    // Loop through all geometry nodes
    for (pugi::xml_node geo : root.children("Geometry")) {

      // 0D parametric dimension
      if constexpr (parDim_ == 0) {

        // Check for "Point" with given id, index, label
        if (geo.attribute("type").value() == std::string("Point") &&
            (id >= 0 ? geo.attribute("id").as_int() == id : true) &&
            (index >= 0 ? geo.attribute("index").as_int() == index : true) &&
            (!label.empty() ? geo.attribute("label").value() == label : true)) {

          nknots_found[0] = true;
          ncoeffs_found[0] = true;
        } // "Point"
        else
          continue; // try next "Geometry"
      }

      // 1D parametric dimension
      else if constexpr (parDim_ == 1) {

        // Check for "BSpline" with given id, index, label
        if (geo.attribute("type").value() == std::string("BSpline") &&
            (id >= 0 ? geo.attribute("id").as_int() == id : true) &&
            (index >= 0 ? geo.attribute("index").as_int() == index : true) &&
            (!label.empty() ? geo.attribute("label").value() == label : true)) {

          // Check for "BSplineBasis"
          if (pugi::xml_node basis = geo.child("Basis");
              basis.attribute("type").value() == std::string("BSplineBasis")) {

            // Check for "KnotVector"
            if (pugi::xml_node knots = basis.child("KnotVector");
                knots.attribute("degree").as_int() == degrees_[0]) {

              std::vector<real_t> kv;
              std::string values = std::regex_replace(
                  knots.text().get(), std::regex("[\t\r\n\a]+| +"), " ");
              for (auto value = strtok(&values[0], " "); value != NULL;
                   value = strtok(NULL, " "))
                kv.push_back(static_cast<real_t>(std::stod(value)));

              knots_[0] = utils::to_tensor(kv, options_);
              nknots_[0] = kv.size();
              ncoeffs_[0] = nknots_[0] - degrees_[0] - 1;

              nknots_found[0] = true;
              ncoeffs_found[0] = true;

            } // "KnotVector"

          } // "BSplineBasis"

        } // "Bspline"
        else
          continue; // try next "Geometry"
      }

      // >1D parametric dimension
      else {

        // Check for "TensorBSpline<parDim>" with given id, index, label
        if (geo.attribute("type").value() ==
                std::string("TensorBSpline").append(std::to_string(parDim_)) &&
            (id >= 0 ? geo.attribute("id").as_int() == id : true) &&
            (index >= 0 ? geo.attribute("index").as_int() == index : true) &&
            (!label.empty() ? geo.attribute("label").value() == label : true)) {

          // Check for "TensorBSplineBasis<parDim>"
          if (pugi::xml_node bases = geo.child("Basis");
              bases.attribute("type").value() ==
              std::string("TensorBSplineBasis")
                  .append(std::to_string(parDim_))) {

            // Loop through all basis nodes
            for (pugi::xml_node basis : bases.children("Basis")) {

              // Check for "BSplineBasis"
              if (basis.attribute("type").value() ==
                  std::string("BSplineBasis")) {

                short_t index = basis.attribute("index").as_int();

                // Check for "KnotVector"
                if (pugi::xml_node knots = basis.child("KnotVector");
                    knots.attribute("degree").as_int() == degrees_[index]) {

                  std::vector<real_t> kv;
                  std::string values = std::regex_replace(
                      knots.text().get(), std::regex("[\t\r\n\a]+| +"), " ");

                  for (auto value = strtok(&values[0], " "); value != NULL;
                       value = strtok(NULL, " "))
                    kv.push_back(static_cast<real_t>(std::stod(value)));

                  knots_[index] = utils::to_tensor(kv, options_);
                  nknots_[index] = kv.size();
                  ncoeffs_[index] = nknots_[index] - degrees_[index] - 1;

                  nknots_found[index] = true;
                  ncoeffs_found[index] = true;

                } // "KnotVector"

              } // "BSplineBasis"

            } // "Basis"

          } // "TensorBSplineBasis<parDim>"

        } // "TensorBSpline<parDim>"
        else
          continue; // try next "Geometry"

      } // parametric dimension

      if (std::any_of(std::begin(nknots_found), std::end(nknots_found),
                      [](bool i) { return !i; }))
        throw std::runtime_error(
            "XML object is not compatible with B-spline object");

      // Reverse ncoeffs
      ncoeffs_reverse_ = ncoeffs_;
      std::reverse(ncoeffs_reverse_.begin(), ncoeffs_reverse_.end());

      // Fill coefficients with zeros
      int64_t size = ncumcoeffs();
      for (short_t i = 0; i < geoDim_; ++i)
        coeffs_[i] = torch::zeros(size, options_.device(torch::kCPU));

      // Check for "coefs"
      if (pugi::xml_node coefs = geo.child("coefs")) {

        std::string values = std::regex_replace(
            coefs.text().get(), std::regex("[\t\r\n\a]+| +"), " ");
        auto coeffs_accessors = utils::to_tensorAccessor<real_t, 1>(coeffs_);

        if constexpr (parDim_ == 0) {
          auto value = strtok(&values[0], " ");

          for (short_t g = 0; g < geoDim_; ++g) {
            if (value == NULL)
              throw std::runtime_error(
                  "XML object does not provide enough coefficients");

            coeffs_accessors[g][0] = static_cast<real_t>(std::stod(value));
            value = strtok(NULL, " ");
          }

          if (value != NULL)
            throw std::runtime_error(
                "XML object provides too many coefficients");

        } else {
          auto value = strtok(&values[0], " ");

          for (int64_t i = 0; i < utils::prod(ncoeffs_); ++i)
            for (short_t g = 0; g < geoDim_; ++g) {
              if (value == NULL)
                throw std::runtime_error(
                    "XML object does not provide enough coefficients");

              coeffs_accessors[g][i] = static_cast<real_t>(std::stod(value));
              value = strtok(NULL, " ");
            }

          if (value != NULL)
            throw std::runtime_error(
                "XML object provides too many coefficients");
        }

        // Copy coefficients to device (if needed)
        for (short_t i = 0; i < geoDim_; ++i)
          coeffs_[i] = coeffs_[i].to(options_.device());

        if constexpr (parDim_ == 0) {
          if (nknots_found[0] && ncoeffs_found[0])
            return *this;
        } else if (std::all_of(std::begin(nknots_found), std::end(nknots_found),
                               [](bool i) { return i; }) &&
                   std::all_of(std::begin(ncoeffs_found),
                               std::end(ncoeffs_found),
                               [](bool i) { return i; }))
          return *this;

        else
          throw std::runtime_error(
              "XML object is not compatible with B-spline object");

      } // Coefs
      else
        throw std::runtime_error("XML object does not provide coefficients");

    } // "Geometry"

    throw std::runtime_error("XML object does not provide geometry with given "
                             "id, index, and/or label");
    return *this;
  }

  /// @brief Loads the B-spline from file
  inline void load(const std::string &filename,
                   const std::string &key = "bspline") {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    read(archive, key);
  }

  /// @brief Reads the B-spline from a torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "bspline") {
    torch::Tensor tensor;

    archive.read(key + ".parDim", tensor);
    if (tensor.item<int64_t>() != parDim_)
      throw std::runtime_error("parDim mismatch");

    archive.read(key + ".geoDim", tensor);
    if (tensor.item<int64_t>() != geoDim_)
      throw std::runtime_error("geoDim mismatch");

    for (short_t i = 0; i < parDim_; ++i) {
      archive.read(key + ".degree[" + std::to_string(i) + "]", tensor);
      if (tensor.item<int64_t>() != degrees_[i])
        throw std::runtime_error("degrees mismatch");
    }

    for (short_t i = 0; i < parDim_; ++i) {
      archive.read(key + ".nknots[" + std::to_string(i) + "]", tensor);
      nknots_[i] = tensor.item<int64_t>();
    }

    for (short_t i = 0; i < parDim_; ++i)
      archive.read(key + ".knots[" + std::to_string(i) + "]", knots_[i]);

    for (short_t i = 0; i < parDim_; ++i) {
      archive.read(key + ".ncoeffs[" + std::to_string(i) + "]", tensor);
      ncoeffs_[i] = tensor.item<int64_t>();
    }

    for (short_t i = 0; i < geoDim_; ++i)
      archive.read(key + ".coeffs[" + std::to_string(i) + "]", coeffs_[i]);

    return archive;
  }

  /// @brief Saves the B-spline to file
  inline void save(const std::string &filename,
                   const std::string &key = "bspline") const {
    torch::serialize::OutputArchive archive;
    write(archive, key).save_to(filename);
  }

  /// @brief Writes the B-spline into a torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "bspline") const {
    archive.write(key + ".parDim", torch::full({1}, parDim_));
    archive.write(key + ".geoDim", torch::full({1}, geoDim_));

    for (short_t i = 0; i < parDim_; ++i)
      archive.write(key + ".degree[" + std::to_string(i) + "]",
                    torch::full({1}, degrees_[i]));

    for (short_t i = 0; i < parDim_; ++i)
      archive.write(key + ".nknots[" + std::to_string(i) + "]",
                    torch::full({1}, nknots_[i]));

    for (short_t i = 0; i < parDim_; ++i)
      archive.write(key + ".knots[" + std::to_string(i) + "]", knots_[i]);

    for (short_t i = 0; i < parDim_; ++i)
      archive.write(key + ".ncoeffs[" + std::to_string(i) + "]",
                    torch::full({1}, ncoeffs_[i]));

    for (short_t i = 0; i < geoDim_; ++i)
      archive.write(key + ".coeffs[" + std::to_string(i) + "]", coeffs_[i]);

    return archive;
  }

  /// @brief Returns true if both B-spline objects are close up to the given
  /// tolerances
  template <typename other_t, short_t GeoDim_, short_t... Degrees_>
  bool isclose(const UniformBSplineCore<other_t, GeoDim_, Degrees_...> &other,
               real_t rtol = real_t{1e-5}, real_t atol = real_t{1e-8}) const {
    if constexpr (!std::is_same<real_t, other_t>::value)
      return false;
    bool result(true);

    result *= (parDim_ == other.parDim());
    result *= (geoDim_ == other.geoDim());

    for (short_t i = 0; i < parDim_; ++i)
      result *= (degree(i) == other.degree(i));

    for (short_t i = 0; i < parDim_; ++i)
      result *= (nknots(i) == other.nknots(i));

    for (short_t i = 0; i < parDim_; ++i)
      result *= (ncoeffs(i) == other.ncoeffs(i));

    if (!result)
      return result;

    for (short_t i = 0; i < parDim_; ++i)
      result *= torch::allclose(knots(i), other.knots(i), rtol, atol);

    for (short_t i = 0; i < geoDim_; ++i)
      result *= torch::allclose(coeffs(i), other.coeffs(i), rtol, atol);

    return result;
  }

  /// @brief Returns true if both B-spline objects are the same
  template <typename other_t, short_t GeoDim_, short_t... Degrees_>
  bool operator==(
      const UniformBSplineCore<other_t, GeoDim_, Degrees_...> &other) const {
    if constexpr (!std::is_same<real_t, other_t>::value)
      return false;
    bool result(true);

    result *= (parDim_ == other.parDim());
    result *= (geoDim_ == other.geoDim());

    if (!result)
      return result;

    for (short_t i = 0; i < parDim_; ++i)
      result *= (degree(i) == other.degree(i));

    for (short_t i = 0; i < parDim_; ++i)
      result *= (nknots(i) == other.nknots(i));

    for (short_t i = 0; i < parDim_; ++i)
      result *= (ncoeffs(i) == other.ncoeffs(i));

    for (short_t i = 0; i < parDim_; ++i)
      result *= torch::equal(knots(i), other.knots(i));

    for (short_t i = 0; i < geoDim_; ++i)
      result *= torch::equal(coeffs(i), other.coeffs(i));

    return result;
  }

  /// @brief Returns true if both B-spline objects are different
  template <typename other_t, short_t GeoDim_, short_t... Degrees_>
  bool operator!=(
      const UniformBSplineCore<other_t, GeoDim_, Degrees_...> &other) const {
    return !(
        *this ==
        other); // Do not change this to (*this != other) is it does not work
  }

  /// @brief Returns the B-spline object with uniformly refined knot
  /// and coefficient vectors
  ///
  /// If `dim = -1`, new knot values are inserted uniformly in each
  /// knot span in all spatial dimensions. Otherwise, i.e., `dim !=
  /// -1` new knots are only inserted in the specified dimension.
  inline UniformBSplineCore &uniform_refine(int numRefine = 1, int dim = -1) {
    assert(numRefine > 0);
    assert(dim == -1 || (dim >= 0 && dim < parDim_));

    // Update number of knots and coefficients
    std::array<int64_t, parDim_> nknots(nknots_);
    std::array<int64_t, parDim_> ncoeffs(ncoeffs_);

    for (short_t refine = 0; refine < numRefine; ++refine) {
      if (dim == -1)
        for (short_t i = 0; i < parDim_; ++i) {
          ncoeffs[i] += nknots[i] - 2 * degrees_[i] - 1; // must be done first
          nknots[i] += nknots[i] - 2 * degrees_[i] - 1;
        }
      else {
        ncoeffs[dim] +=
            nknots[dim] - 2 * degrees_[dim] - 1; // must be done first
        nknots[dim] += nknots[dim] - 2 * degrees_[dim] - 1;
      }
    }

    // Update knot vectors
    utils::TensorArray<parDim_> knots, knots_indices;

    for (short_t i = 0; i < parDim_; ++i) {
      std::vector<real_t> kv;
      kv.reserve(nknots[i]);

      for (int64_t j = 0; j < degrees_[i]; ++j)
        kv.push_back(static_cast<real_t>(0));

      for (int64_t j = 0; j < ncoeffs[i] - degrees_[i] + 1; ++j)
        kv.push_back(static_cast<real_t>(j) /
                     static_cast<real_t>(ncoeffs[i] - degrees_[i]));

      for (int64_t j = 0; j < degrees_[i]; ++j)
        kv.push_back(static_cast<real_t>(1));

      knots[i] = utils::to_tensor(kv, options_);
    }

    // The updated knot vectors have lengths \f$m_d+p_d+1\f$, where
    // \f$m_d\f$ is the number of coefficients after the update. To
    // update the coefficients using the Oslo algorithm (Algorithm
    // 4.11 from \cite Lyche:2011) we need to neglect the last
    // \f$p_d+1\f$ knots in what follows
    for (short_t i = 0; i < parDim_; ++i)
      knots_indices[i] = knots[i].index(
          {torch::indexing::Slice(0, knots[i].numel() - degrees_[i] - 1)});

    // Get indices of the first \f$m_d\f$ new knots relative to old
    // knot vectors
    auto new_knot_indices = find_knot_indices(knots_indices);

    // Update coefficient vector
    update_coeffs(knots, new_knot_indices);

    // Swap old and new data
    knots.swap(knots_);
    nknots.swap(nknots_);
    ncoeffs.swap(ncoeffs_);

    ncoeffs_reverse_ = ncoeffs_;
    std::reverse(ncoeffs_reverse_.begin(), ncoeffs_reverse_.end());

    return *this;
  }

private:
  /// @brief Computes the prefactor \f$p_d!/(p_d-r_d)! = p_d \cdots
  /// (p_d-r_d+1)\f$
  template <int64_t degree, int64_t deriv, int64_t terminal = degree - deriv>
  inline int64_t constexpr eval_prefactor() const {
    if constexpr (degree > terminal)
      return degree * eval_prefactor<degree - 1, deriv, terminal>();
    else
      return 1;
  }

public:
  /// @brief Initializes the B-spline knots
  inline void init_knots() {

    for (short_t i = 0; i < parDim_; ++i) {

      // Check that open knot vector can be created
      if ((ncoeffs_[i] < degrees_[i] + 1) || (ncoeffs_[i] < 2))
        throw std::runtime_error(
            "Not enough coefficients to create open knot vector");

      nknots_[i] = ncoeffs_[i] + degrees_[i] + 1;
      int64_t num_inner = ncoeffs_[i] - degrees_[i];

      auto start = torch::zeros({degrees_[i]}, options_);
      auto end = torch::ones({degrees_[i]}, options_);
      auto inner = torch::empty({0}, options_);

      if (num_inner > 0) {
        inner = torch::arange(0, num_inner + 1, options_);
        inner = inner / static_cast<real_t>(num_inner);
      }
      
      knots_[i] = torch::cat({start, inner, end});           
    }

  }

  /// @brief Initializes the B-spline coefficients
  inline void init_coeffs(enum init init) {
    switch (init) {

    case (init::none): {
      break;
    }

    case (init::zeros): {

      // Fill coefficients with zeros
      int64_t size = ncumcoeffs();
      for (short_t i = 0; i < geoDim_; ++i)
        coeffs_[i] = torch::zeros(size, options_);

      break;
    }

    case (init::ones): {

      // Fill coefficients with ones
      int64_t size = ncumcoeffs();
      for (short_t i = 0; i < geoDim_; ++i)
        coeffs_[i] = torch::ones(size, options_);

      break;
    }

    case (init::random): {

      // Fill coefficients with random values
      int64_t size = ncumcoeffs();
      for (short_t i = 0; i < geoDim_; ++i)
        coeffs_[i] = torch::rand(size, options_);

      break;
    }

    case (init::linear): {

      // Fill coefficients with the tensor-product of linearly
      // increasing values between 0 and 1 per univariate dimension
      for (short_t i = 0; i < geoDim_; ++i) {
        coeffs_[i] = torch::ones(1, options_);

        for (short_t j = 0; j < parDim_; ++j) {
          if (i == j)
            coeffs_[i] = torch::kron(torch::linspace(static_cast<real_t>(0),
                                                     static_cast<real_t>(1),
                                                     ncoeffs_[j], options_),
                                     coeffs_[i]);
          else
            coeffs_[i] =
                torch::kron(torch::ones(ncoeffs_[j], options_), coeffs_[i]);
        }

        // Enable gradient calculation for non-leaf tensor
        if (options_.requires_grad())
          coeffs_[i].retain_grad();
      }
      break;
    }

    case (init::greville): {

      // Fill coefficients with the tensor-product of Greville
      // abscissae values per univariate dimension
      for (short_t i = 0; i < geoDim_; ++i) {
        coeffs_[i] = torch::ones(1, options_);

        for (short_t j = 0; j < parDim_; ++j) {
          if (i == j) {

	    int64_t count = ncoeffs_[j];

	    // idx_base: (count, 1)
	    auto idx_base = torch::arange(count, options_.requires_grad(false).template dtype<int64_t>()).unsqueeze(1);

	    // offsets: (1, degree)
	    auto offsets = torch::arange(1, degrees_[j] + 1, options_.requires_grad(false).template dtype<int64_t>()).unsqueeze(0);

	    // indices: (count, degree)
	    auto indices = idx_base + offsets;

	    // Gather relevant knot values: shape (count, degree)
	    auto gathered = knots_[j].index_select(0, indices.flatten()).view({count, degrees_[j]});

	    // Compute mean along degree dimension (dim=1)
	    auto greville_ = gathered.mean(1);

            coeffs_[i] = torch::kron(greville_, coeffs_[i]);
          } else
            coeffs_[i] =
                torch::kron(torch::ones(ncoeffs_[j], options_), coeffs_[i]);
        }

        // Enable gradient calculation for non-leaf tensor
        if (options_.requires_grad())
          coeffs_[i].retain_grad();
      }
      break;
    }

    case (init::linspace): {

      // Fill coefficients with increasing values
      int64_t size = ncumcoeffs();
      for (short_t i = 0; i < geoDim_; ++i)
        coeffs_[i] = torch::linspace(
            std::pow(10, i) * 0, std::pow(10, i) * (size - 1), size, options_);

      break;
    }

    default:
      throw std::runtime_error("Unsupported init option");
    }
  }

protected:
  /// @brief Updates the B-spline coefficients after knot insertion
  inline void update_coeffs(const utils::TensorArray<parDim_> &knots,
                            const utils::TensorArray<parDim_> &knot_indices) {

    // Check compatibility of arguments
    for (short_t i = 0; i < parDim_; ++i)
      assert(knots[i].numel() == knot_indices[i].numel() + degrees_[i] + 1);

    if constexpr (parDim_ == 1) {

      auto basfunc = update_coeffs_univariate<degrees_[0], 0>(
          knots[0].flatten(), knot_indices[0].flatten());

      auto coeff_indices = find_coeff_indices(knot_indices);

      for (short_t i = 0; i < geoDim_; ++i)
        coeffs(i) =
            utils::dotproduct(basfunc, coeffs(i)
                                           .index_select(0, coeff_indices)
                                           .view({-1, knot_indices[0].numel()}))
                .view(knot_indices[0].sizes());

    } else {

      // Lambda expressions to evaluate the basis functions
      auto basfunc_ = [&, this]<std::size_t... Is>(std::index_sequence<Is...>) {
        if constexpr (sizeof...(Is) == 1)
          return (update_coeffs_univariate<degrees_[Is], Is>(
                      knots[Is].flatten(), knot_indices[Is].flatten()),
                  ...);
        else
          return utils::kron(update_coeffs_univariate<degrees_[Is], Is>(
              knots[Is].flatten(), knot_indices[Is].flatten())...);
      };

      auto basfunc = basfunc_(utils::make_reverse_index_sequence<parDim_>{});

      // Lambda expression to calculate the partial product of array
      // entry from start_index to stop_index (including the latter)
      auto prod_ = [](utils::TensorArray<parDim_> array, short_t start_index,
                      short_t stop_index) {
        int64_t result{1};
        for (short_t i = start_index; i <= stop_index; ++i)
          result *= array[i].numel();
        return result;
      };

      utils::TensorArray<parDim_> knot_indices_;

      for (short_t i = 0; i < parDim_; ++i)
        knot_indices_[i] =
            knot_indices[i]
                .repeat_interleave(prod_(knot_indices, 0, i - 1), 0)
                .repeat(prod_(knot_indices, i + 1, parDim_ - 1));

      auto coeff_indices = find_coeff_indices(knot_indices_);

      for (short_t i = 0; i < geoDim_; ++i)
        coeffs(i) = utils::dotproduct(basfunc,
                                      coeffs(i)
                                          .index_select(0, coeff_indices)
                                          .view({-1, knot_indices_[0].numel()}))
                        .view(knot_indices_[0].sizes());
    }
  }

  //  clang-format off
  /// @brief Returns the vector of univariate B-spline basis
  /// functions (or their derivatives) evaluated in the point `xi`
  ///
  /// This function implements step 2 of algorithm \ref
  /// BSplineEvaluation, that is, it evaluates the vector of
  /// univariate B-spline basis functions (or their derivatives)
  /// that are non-zero at \f$\xi_d \in [t_{i_d}, t_{i_d+1})\f$
  ///
  /// \f[
  ///   D^{r_d}\mathbf{B}_d(\xi_d)
  ///   = \left( D^{r_d} B_{i_d-p_d,p_d}(\xi_d), \dots, D^{r_d}
  ///   B_{i_d,p_d}(\xi_d) \right)^\top,
  /// \f]
  ///
  /// where \f$ p_d \f$ is the degree of the \f$d\f$-th univariate
  /// B-spline and \f$ r_d \f$ denotes the requested derivative in
  /// the \f$d\f$-direction.
  ///
  /// According to the procedure described in Chapters 2 and 3 of
  /// \cite Lyche:2011 this can be accomplished by the following
  /// expression
  ///
  /// \f[
  ///   D^{r_d}\mathbf{B}_d(\xi_d)
  ///   = \frac{p_d!}{(p_d-r_d)!}\mathbf{R}_1(\xi_d)\cdot \cdots \cdot
  ///   \mathbf{R}_{p_d-r_d}(\xi_d)
  ///     D\mathbf{R}_{p_d-r_d+1}\cdot \cdots \cdot D\mathbf{R}_{p_d}(\xi_d),
  /// \f]
  ///
  /// where (cf. Equation (2.20) in \cite Lyche:2011)
  ///
  /// \f[
  ///   \mathbf{R}_k(\xi_d) =
  ///   \begin{pmatrix}
  ///     \frac{t_{i_p+1} - \xi_d}{t_{i_p+1} - t_{i_p+1-k}} & \frac{\xi_d -
  ///     t_{i_p+1-k}}{t_{i_p+1} - t_{i_p+1-k}} & 0 & \cdots & 0 \\
  ///     0 & \frac{t_{i_p+2} - \xi_d}{t_{i_p+2} - t_{i_p+2-k}} & \frac{\xi_d -
  ///     t_{i_p+2-k}}{t_{i_p+2} - t_{i_p+1-k}} & \cdots & 0 \\
  ///     \vdots & \vdots & \ddots & \ddots & \vdots \\
  ///     0 & 0 & \cdots & \frac{t_{i_p+k} - \xi_d}{t_{i_p+k} - t_{i_p}} &
  ///     \frac{\xi_d - t_{i_p}}{t_{i_p+k} - t_{i_p}}
  ///   \end{pmatrix}
  /// \f]
  ///
  /// and (cf. Equation (3.30) in \cite Lyche:2011)
  ///
  /// \f[
  ///   D\mathbf{R}_k(\xi_d) =
  ///   \begin{pmatrix}
  ///     \frac{-1}{t_{i_p+1} - t_{i_p+1-k}} & \frac{1}{t_{i_p+1} - t_{i_p+1-k}}
  ///     & 0 & \cdots & 0 \\
  ///     0 & \frac{-1}{t_{i_p+2} - t_{i_p+2-k}} & \frac{1}{t_{i_p+2} -
  ///     t_{i_p+1-k}} & \cdots & 0 \\
  ///     \vdots & \vdots & \ddots & \ddots & \vdots \\
  ///     0 & 0 & \cdots & \frac{-1}{t_{i_p+k} - t_{i_p}} & \frac{1}{t_{i_p+k} -
  ///     t_{i_p}}
  ///   \end{pmatrix}.
  /// \f]
  ///
  /// To improve computational efficiency, the prefactor
  ///
  /// \f[
  ///    \frac{p_d!}{(p_d-r_d)!}=p_d \cdots (p_d-r_d+1)
  /// \f]
  ///
  /// is computed as
  /// compile-time expression by the eval_prefactor() function.
  ///
  /// Moreover, the above expression for
  /// \f$D^{r_d}\mathbf{B}_d(\xi_d)\f$ is evaluated as described in
  /// Algorithm 2.22 (R-vector version) in \cite Lyche:2011) and its
  /// generalization to derivatives, respectively.
  ///
  /// The algorithm goes as follows:
  ///
  /// 1. \f$\mathbf{b} = 1\f$
  ///
  /// 2. For \f$k = 1, \dots, p_d-r_d\f$
  ///
  ///    1. \f$\mathbf{t}_1 = \left(t_{i_d-k+1},\dots,t_{i_d}\right)\f$
  ///
  ///    2. \f$\mathbf{t}_2 = \left(t_{i_d+1},\dots,t_{i_d+k}\right)\f$
  ///
  ///    3. \f$\mathbf{w}   =
  ///    \left(\xi_d-\mathbf{t}_1\right)\div\left(\mathbf{t}_2-\mathbf{t}_1\right)\f$
  ///
  ///    4. \f$\mathbf{b}   = \left[\left(1-\mathbf{w}\right)\odot\mathbf{b},
  ///    0\right]
  ///                       + \left[0, \mathbf{w}\odot\mathbf{b}\right]\f$
  ///
  /// 3. For \f$k = p_d-r_d+1, \dots, p_d\f$
  ///
  ///    1. \f$\mathbf{t}_1 = \left(t_{i_d-k+1},\dots,t_{i_d}\right)\f$
  ///
  ///    2. \f$\mathbf{t}_2 = \left(t_{i_d+1},\dots,t_{i_d+k}\right)\f$
  ///
  ///    3. \f$\mathbf{w}   = 1\div\left(\mathbf{t}_2-\mathbf{t}_1\right)\f$
  ///
  ///    4. \f$\mathbf{b}   = \left[-\mathbf{w}\odot\mathbf{b}, 0\right]
  ///                       + \left[0, \mathbf{w}\odot\mathbf{b}\right]\f$
  ///
  /// where \f$\div\f$ and \f$\odot\f$ denote the element-wise
  /// division and multiplication of vectors, respectively.
  //  clang-format on
  template <short_t degree, short_t dim, short_t deriv>
  inline auto eval_basfunc_univariate(const torch::Tensor &xi,
                                      const torch::Tensor &knot_indices) const {
    assert(xi.sizes() == knot_indices.sizes());

    if constexpr (deriv > degree) {
      return torch::zeros({degree + 1, xi.numel()}, options_);
    } else {
      // Algorithm 2.22 from \cite Lyche:2011
      torch::Tensor b = torch::ones({xi.numel()}, options_);

      // Calculate R_k, k = 1, ..., p_d-r_d
      for (short_t k = 1; k <= degree - deriv; ++k) {

        // Instead of calculating t1 and t2 we calculate t1 and t21=(t2-t1)
        auto t1 =
            knots_[dim].index_select(0, utils::VSlice(knot_indices, -k + 1, 1));
        auto t21 =
            knots_[dim].index_select(0, utils::VSlice(knot_indices, 1, k + 1)) -
            t1;

        // We handle the special case 0/0:=0 by first creating a
        // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
        // we do not have to take the absolute value as t2 >= t1.
        auto mask = (t21 < std::numeric_limits<real_t>::epsilon())
                        .to(::iganet::dtype<real_t>());

        // Instead of computing (xi-t1)/(t2-t1) which is prone to
        // yielding 0/0 we compute (xi-t1-mask)/(t2-t1-mask) which
        // equals the original expression if the mask is 0, i.e.,
        // t2-t1 >= eps and 1 otherwise since t1 <= xi < t2.
        auto w = torch::div(xi.repeat(k) - t1 - mask, t21 - mask);

        // Calculate the vector of B-splines evaluated at xi
        b = torch::cat({torch::mul(torch::ones_like(w, options_) - w, b),
                        torch::zeros_like(xi, options_)},
                       0) +
            torch::cat({torch::zeros_like(xi, options_), torch::mul(w, b)}, 0);
      }

      // Calculate DR_k, k = p_d-r_d+1, ..., p_d
      for (short_t k = degree - deriv + 1; k <= degree; ++k) {

        // Instead of calculating t1 and t2 we calculate t1 and t21=(t2-t1)
        auto t21 =
            knots_[dim].index_select(0, utils::VSlice(knot_indices, 1, k + 1)) -
            knots_[dim].index_select(0, utils::VSlice(knot_indices, -k + 1, 1));

        // We handle the special case 0/0:=0 by first creating a
        // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
        // we do not have to take the absolute value as t2 >= t1.
        auto mask = (t21 < std::numeric_limits<real_t>::epsilon())
                        .to(::iganet::dtype<real_t>());

        // Instead of computing 1/(t2-t1) which is prone to yielding
        // 0/0 we compute (1-mask)/(t2-t1-mask) which equals the
        // original expression if the mask is 0, i.e., t2-t1 >= eps
        // and 1 otherwise since t1 <= xi < t2.
        auto w = torch::div(torch::ones_like(t21, options_) - mask, t21 - mask);

        // Calculate the vector of B-splines evaluated at xi
        b = torch::cat({torch::mul(-w, b), torch::zeros_like(xi, options_)},
                       0) +
            torch::cat({torch::zeros_like(xi, options_), torch::mul(w, b)}, 0);
      }

      return b.view({degree + 1, xi.numel()});
    }
  }

  /// @brief Returns the knot insertion matrix
  ///
  /// This functions implements the Oslo algorithm (Algorithm 4.11
  /// in \cite Lyche:2011) to compute the univariate knot insertion
  /// matrix from the given knot vector to the new knot vector
  /// passed as argument `knots`.
  template <short_t degree, short_t dim>
  inline auto
  update_coeffs_univariate(const torch::Tensor &knots,
                           const torch::Tensor &knot_indices) const {
    // Algorithm 2.22 from \cite Lyche:2011 modified to implement
    // the Oslo algorithm (Algorithm 4.11 from \cite Lyche:2011)
    torch::Tensor b = torch::ones({knot_indices.numel()}, options_);

    // Calculate R_k, k = 1, ..., p_d
    for (short_t k = 1; k <= degree; ++k) {

      // Instead of calculating t1 and t2 we calculate t1 and t21=(t2-t1)
      auto t1 =
          knots_[dim].index_select(0, utils::VSlice(knot_indices, -k + 1, 1));
      auto t21 =
          knots_[dim].index_select(0, utils::VSlice(knot_indices, 1, k + 1)) -
          t1;

      // We handle the special case 0/0:=0 by first creating a
      // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
      // we do not have to take the absolute value as t2 >= t1.
      auto mask = (t21 < std::numeric_limits<real_t>::epsilon())
                      .to(::iganet::dtype<real_t>());

      // Instead of computing (xi-t1)/(t2-t1) which is prone to
      // yielding 0/0 we compute (xi-t1-mask)/(t2-t1-mask) which
      // equals the original expression if the mask is 0, i.e.,
      // t2-t1 >= eps and 1 otherwise since t1 <= xi < t2.
      auto w = torch::div(
          knots.index({torch::indexing::Slice(k, knot_indices.numel() + k)})
                  .repeat(k) -
              t1 - mask,
          t21 - mask);

      // Calculate the vector of B-splines evaluated at xi
      b = torch::cat({torch::mul(torch::ones_like(w, options_) - w, b),
                      torch::zeros_like(knot_indices, options_)},
                     0) +
          torch::cat(
              {torch::zeros_like(knot_indices, options_), torch::mul(w, b)}, 0);
    }

    return b.view({degree + 1, knot_indices.numel()});
  }

public:
  /// @brief Converts the B-spline object into a gsBSpline object of
  /// the parametric dimension is one and a gsTensorBSpline object
  /// otherwise
  auto to_gismo() const {

#ifdef IGANET_WITH_GISMO

    gismo::gsMatrix<real_t> coefs(ncumcoeffs(), geoDim_);

    for (short_t g = 0; g < geoDim_; ++g) {
      auto [coeffs_cpu, coeffs_accessor] =
          utils::to_tensorAccessor<real_t, 1>(coeffs_[g], torch::kCPU);
      auto coeffs_cpu_ptr = coeffs_cpu.template data_ptr<real_t>();
      coefs.col(g) =
        gismo::gsAsConstVector<real_t>(coeffs_cpu_ptr, coeffs_cpu.size(0));
    }

    std::array<gismo::gsKnotVector<real_t>, parDim_> kv;

    for (short_t i = 0; i < parDim_; ++i) {
      auto [knots_cpu, knots_accessor] =
          utils::to_tensorAccessor<real_t, 1>(knots_[i], torch::kCPU);
      auto knots_cpu_ptr = knots_cpu.template data_ptr<real_t>();
      kv[i] = gismo::gsKnotVector<real_t>(degrees_[i], knots_cpu_ptr,
                                          knots_cpu_ptr + knots_cpu.size(0));
    }

    if constexpr (parDim_ == 1) {

      return gismo::gsBSpline<real_t>(gismo::give(kv[0]), gismo::give(coefs));

    } else if constexpr (parDim_ == 2) {

      return gismo::gsTensorBSpline<parDim_, real_t>(
          gismo::give(kv[0]), gismo::give(kv[1]), gismo::give(coefs));

    } else if constexpr (parDim_ == 3) {

      return gismo::gsTensorBSpline<parDim_, real_t>(
          gismo::give(kv[0]), gismo::give(kv[1]), gismo::give(kv[2]),
          gismo::give(coefs));

    } else if constexpr (parDim_ == 4) {

      return gismo::gsTensorBSpline<parDim_, real_t>(
          gismo::give(kv[0]), gismo::give(kv[1]), gismo::give(kv[2]),
          gismo::give(kv[3]), gismo::give(coefs));

    } else
      throw std::runtime_error("Invalid parametric dimension");

#else
    throw std::runtime_error(
        "This functions must be compiled with -DIGANET_WITH_GISMO turned on");
#endif
  }

#ifdef IGANET_WITH_GISMO

  // @brief Updates a given gsBSpline object from the B-spline object
  gismo::gsBSpline<real_t> &to_gismo(gismo::gsBSpline<real_t> &bspline,
                                     bool updateKnotVector = true,
                                     bool updateCoeffs = true) const {

    if (updateKnotVector) {

      if constexpr (parDim_ == 1) {

        if (bspline.degree(0) != degrees_[0])
          throw std::runtime_error("Degrees mismatch");

        auto [knots_cpu, knots_accessor] =
            utils::to_tensorAccessor<real_t, 1>(knots_[0], torch::kCPU);
        auto knots_cpu_ptr = knots_cpu.template data_ptr<real_t>();

        gismo::gsKnotVector<real_t> kv(degrees_[0], knots_cpu_ptr,
                                       knots_cpu_ptr + knots_cpu.size(0));

        bspline.knots(0).swap(kv);

      } else
        throw std::runtime_error("Invalid parametric dimension");
    }

    if (updateCoeffs) {

      for (short_t g = 0; g < geoDim_; ++g) {
        auto [coeffs_cpu, coeffs_accessor] =
            utils::to_tensorAccessor<real_t, 1>(coeffs_[g], torch::kCPU);
        auto coeffs_cpu_ptr = coeffs_cpu.template data_ptr<real_t>();
        bspline.coefs().col(g) =
          gismo::gsAsConstVector<real_t>(coeffs_cpu_ptr, coeffs_cpu.size(0));
      }
    }

    return bspline;
  }

  // @brief Updates a given gsTensorBSpline object from the B-spline object
  gismo::gsTensorBSpline<parDim_, real_t> &
  to_gismo(gismo::gsTensorBSpline<parDim_, real_t> &bspline,
           bool updateKnotVector = true, bool updateCoeffs = true) const {

    if (updateKnotVector) {

      // Check compatibility of arguments
      for (short_t i = 0; i < parDim_; ++i)
        assert(bspline.degree(i) == degrees_[i]);

      for (short_t i = 0; i < parDim_; ++i) {
        auto [knots_cpu, knots_accessor] =
            utils::to_tensorAccessor<real_t, 1>(knots_[i], torch::kCPU);
        auto knots_cpu_ptr = knots_cpu.template data_ptr<real_t>();

        gismo::gsKnotVector<real_t> kv(degrees_[i], knots_cpu_ptr,
                                       knots_cpu_ptr + knots_cpu.size(0));
        bspline.knots(i).swap(kv);
      }
    }

    if (updateCoeffs) {

      for (short_t g = 0; g < geoDim_; ++g) {
        auto [coeffs_cpu, coeffs_accessor] =
            utils::to_tensorAccessor<real_t, 1>(coeffs_[g], torch::kCPU);
        auto coeffs_cpu_ptr = coeffs_cpu.template data_ptr<real_t>();
        bspline.coefs().col(g) =
          gismo::gsAsConstVector<real_t>(coeffs_cpu_ptr, coeffs_cpu.size(0));
      }
    }

    return bspline;
  }

#else // IGANET_WITH_GISMO

  template <typename BSpline>
  BSpline &to_gismo(BSpline &bspline, bool, bool) const {
    throw std::runtime_error(
        "This functions must be compiled with -DIGANET_WITH_GISMO turned on");
    return bspline;
  }

#endif // IGANET_WITH_GISMO

#ifdef IGANET_WITH_GISMO

  // @brief Updates the B-spline object from a given gsBSpline object
  auto &from_gismo(const gismo::gsBSpline<real_t> &bspline,
                   bool updateCoeffs = true, bool updateKnotVector = false) {

    if (updateKnotVector) {

      throw std::runtime_error(
          "Knot vectors can only be updated for Non-uniform B-splines");
    }

    if (updateCoeffs) {

      if (bspline.coefs().cols() != geoDim_)
        throw std::runtime_error("Geometric dimensions mismatch");

      if (bspline.coefs().rows() != ncumcoeffs())
        throw std::runtime_error("Coefficient vector dimensions mismatch");

      for (short_t g = 0; g < geoDim_; ++g) {

        auto [coeffs_cpu, coeffs_accessor] =
            utils::to_tensorAccessor<real_t, 1>(coeffs_[g], torch::kCPU);

        const real_t *coeffs_ptr = bspline.coefs().col(g).data();

        for (int64_t i = 0; i < ncoeffs_[g]; ++i)
          coeffs_accessor[i] = coeffs_ptr[i];

        coeffs_[g] = coeffs_[g].to(options_.device());
      }
    }

    return *this;
  }

  // @brief Updates the B-spline object from a given gsTensorBSpline object
  auto &from_gismo(const gismo::gsTensorBSpline<parDim_, real_t> &bspline,
                   bool updateCoeffs = true, bool updateKnotVector = false) {

    if (updateKnotVector) {

      throw std::runtime_error(
          "Knot vectors can only be updated for Non-uniform B-splines");
    }

    if (updateCoeffs) {

      if (bspline.coefs().cols() != geoDim_)
        throw std::runtime_error("Geometric dimensions mismatch");

      if (bspline.coefs().rows() != ncumcoeffs())
        throw std::runtime_error("Coefficient vector dimensions mismatch");

      for (short_t g = 0; g < geoDim_; ++g) {

        auto [coeffs_cpu, coeffs_accessor] =
            utils::to_tensorAccessor<real_t, 1>(coeffs_[g], torch::kCPU);

        const real_t *coeffs_ptr = bspline.coefs().col(g).data();

        for (int64_t i = 0; i < ncoeffs_[g]; ++i)
          coeffs_accessor[i] = coeffs_ptr[i];

        coeffs_[g] = coeffs_[g].to(options_.device());
      }
    }

    return *this;
  }

#else // IGANET_WITH_GISMO

  template <typename BSpline> auto &from_gismo(BSpline &bspline, bool, bool) {
    throw std::runtime_error(
        "This functions must be compiled with -DIGANET_WITH_GISMO turned on");
    return *this;
  }

#endif // IGANET_WITH_GISMO
};

/// @brief Serializes a B-spline object
template <typename real_t, short_t GeoDim, short_t... Degrees>
inline torch::serialize::OutputArchive &
operator<<(torch::serialize::OutputArchive &archive,
           const UniformBSplineCore<real_t, GeoDim, Degrees...> &obj) {
  return obj.write(archive);
}

/// @brief De-serializes a B-spline object
template <typename real_t, short_t GeoDim, short_t... Degrees>
inline torch::serialize::InputArchive &
operator>>(torch::serialize::InputArchive &archive,
           UniformBSplineCore<real_t, GeoDim, Degrees...> &obj) {
  return obj.read(archive);
}
  
/// @brief Tensor-product non-uniform B-spline (core functionality)
///
/// This class extends the base class UniformBSplineCore to
/// non-uniform B-splines. Like its base class it only implements
/// the core functionality of non-uniform B-splines
template <typename real_t, short_t GeoDim, short_t... Degrees>
class NonUniformBSplineCore
    : public NonUniformSplineCore_,
      public UniformBSplineCore<real_t, GeoDim, Degrees...> {
private:
  /// @brief Base type
  using Base = UniformBSplineCore<real_t, GeoDim, Degrees...>;

public:
  /// @brief Value type
  using value_type = real_t;

  /// @brief Deduces the type of the template template parameter `BSpline`
  /// when exposed to the class template parameters `real_t` and
  /// `GeoDim`, and the `Degrees` parameter pack. The optional
  /// template parameter `degree_elevate` can be used to
  /// (de-)elevate the degrees by an additive constant
  template <template <typename, short_t, short_t...> class BSpline,
            std::make_signed<short_t>::type degree_elevate = 0>
  using derived_type = BSpline<real_t, GeoDim, (Degrees + degree_elevate)...>;

  /// @brief Deduces the self-type possibly degrees (de-)elevated by
  /// the additive constant `degree_elevate`
  template <std::make_signed<short_t>::type degree_elevate = 0>
  using self_type = typename Base::template derived_type<NonUniformBSplineCore,
                                                         degree_elevate>;

  /// @brief Deduces the derived self-type when exposed to different
  /// class template parameters `real_t` and `GeoDim`, and the
  /// `Degrees` parameter pack
  template <typename other_t, short_t GeoDim_, short_t... Degrees_>
  using derived_self_type =
      NonUniformBSplineCore<other_t, GeoDim_, Degrees_...>;

  /// @brief Deduces the derived self-type when exposed to a
  /// different class template parameter `real_t`
  template <typename other_t>
  using real_derived_self_type =
      NonUniformBSplineCore<other_t, GeoDim, Degrees...>;

  /// @brief Returns true if the B-spline is uniform
  static constexpr bool is_uniform() { return false; }

  /// @brief Returns true if the B-spline is non-uniform
  static constexpr bool is_nonuniform() { return true; }

  /// @brief Constructor for equidistant knot vectors
  using UniformBSplineCore<real_t, GeoDim, Degrees...>::UniformBSplineCore;

  /// @brief Constructor for non-equidistant knot vectors
  ///
  /// @param[in] kv Knot vectors
  ///
  /// @param[in] init Type of initialization
  ///
  /// @param[in] options Options configuration
  NonUniformBSplineCore(const std::array<std::vector<typename Base::value_type>,
                                         Base::parDim_> &kv,
                        enum init init = init::greville,
                        Options<real_t> options = Options<real_t>{})
      : Base(options) {
    init_knots(kv);
    Base::init_coeffs(init);
  }

  /// @brief Constructor for non-equidistant knot vectors
  ///
  /// @param[in] kv Knot vectors
  ///
  /// @param[in] coeffs Vectors of coefficients per parametric dimension
  ///
  /// @param[in] clone  If true, coefficients will be cloned. Otherwise,
  /// coefficients will be aliased
  ///
  /// @param[in] options Options configuration
  ///
  /// @note It is not checked whether vectors of coefficients are
  /// compatible with the given Options object if clone is false.
  NonUniformBSplineCore(const std::array<std::vector<typename Base::value_type>,
                                         Base::parDim_> &kv,
                        const utils::TensorArray<Base::geoDim_> &coeffs,
                        bool clone = false,
                        Options<real_t> options = Options<real_t>{})
      : Base(options) {
    init_knots(kv);

    // Copy/clone coefficients
    if (clone)
      for (short_t i = 0; i < Base::geoDim_; ++i)
        Base::coeffs_[i] = coeffs[i]
                               .clone()
                               .to(options.requires_grad(false))
                               .requires_grad_(Base::options.requires_grad());
    else
      for (short_t i = 0; i < Base::geoDim_; ++i)
        Base::coeffs_[i] = coeffs[i];
  }

private:
  /// @brief Initializes the B-spline knots
  inline void init_knots(
      const std::array<std::vector<typename Base::value_type>, Base::parDim_>
          &kv) {
    for (short_t i = 0; i < Base::parDim_; ++i) {

      // Check that knot vector has enough (n+p+1) entries
      if (2 * Base::degrees_[i] > kv[i].size() - 2)
        throw std::runtime_error("Knot vector is too short for an open knot "
                                 "vector (n+p+1 > 2*(p+1))");

      Base::knots_[i] = utils::to_tensor(kv[i], Base::options_);
      Base::nknots_[i] = Base::knots_[i].size(0);
      Base::ncoeffs_[i] = Base::nknots_[i] - Base::degrees_[i] - 1;
      Base::ncoeffs_reverse_[i] = Base::ncoeffs_[i];
    }
    // Reverse ncoeffs
    std::reverse(Base::ncoeffs_reverse_.begin(), Base::ncoeffs_reverse_.end());
  }

public:
  /// @brief Returns the value of the multivariate B-spline object in the point
  /// `xi`
  /// @{
  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval(const torch::Tensor &xi) const {
    return eval<deriv, memory_optimized>(utils::TensorArray1({xi}));
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval(const utils::TensorArray<Base::parDim_> &xi) const {
    if constexpr (Base::parDim_ == 0) {
      utils::BlockTensor<torch::Tensor, 1, Base::geoDim_> result;
      for (short_t i = 0; i < Base::geoDim_; ++i)
        if constexpr (deriv == deriv::func)
          result.set(i, Base::coeffs_[i]);
        else
          result.set(i, torch::zeros_like(Base::coeffs_[i]));
      return result;
    } else
      return Base::template eval<deriv, memory_optimized>(
          xi, find_knot_indices(xi));
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto
  eval(const utils::TensorArray<Base::parDim_> &xi,
       const utils::TensorArray<Base::parDim_> &knot_indices) const {
    if constexpr (Base::parDim_ == 0) {
      utils::BlockTensor<torch::Tensor, 1, Base::geoDim_> result;
      for (short_t i = 0; i < Base::geoDim_; ++i)
        if constexpr (deriv == deriv::func)
          result.set(i, Base::coeffs_[i]);
        else
          result.set(i, torch::zeros_like(Base::coeffs_[i]));
      return result;
    } else
      return Base::template eval<deriv, memory_optimized>(xi, knot_indices);
  }

  template <deriv deriv = deriv::func, bool memory_optimized = false>
  inline auto eval(const utils::TensorArray<Base::parDim_> &xi,
                   const utils::TensorArray<Base::parDim_> &knot_indices,
                   const torch::Tensor &coeff_indices) const {
    if constexpr (Base::parDim_ == 0) {
      utils::BlockTensor<torch::Tensor, 1, Base::geoDim_> result;
      for (short_t i = 0; i < Base::geoDim_; ++i)
        if constexpr (deriv == deriv::func)
          result.set(i, Base::coeffs_[i]);
        else
          result.set(i, torch::zeros_like(Base::coeffs_[i]));
      return result;
    } else
      return Base::template eval<deriv, memory_optimized>(xi, knot_indices,
                                                          coeff_indices);
  }
  /// @}

  /// @brief Returns the indices of knot spans containing `xi`
  ///
  /// This function returns the indices
  /// \f$(i_d)_{d=1}^{d_\text{par}}\f$ of the knot spans such that
  ///
  /// \f[
  ///   \boldsymbol{\xi} \in [t_{i_1}, t_{i_1+1}) \times [t_{i_2}, t_{i_2+1})
  ///   \times \dots \times [t_{i_{d_\text{par}}}, t_{i_{d_\text{par}}+1}).
  /// \f]
  ///
  /// The indices are returned as `utils::TensorArray<parDim_>` in the
  /// same order as provided in `xi`
  /// @{
  inline auto find_knot_indices(const torch::Tensor &xi) const {
    if constexpr (Base::parDim_ == 0)
      return torch::zeros_like(Base::coeffs_[0]).to(torch::kInt64);
    else
      return find_knot_indices(utils::TensorArray1({xi}));
  }

  inline auto
  find_knot_indices(const utils::TensorArray<Base::parDim_> &xi) const {

    utils::TensorArray<Base::parDim_> indices;
    for (short_t i = 0; i < Base::parDim_; ++i) {
      auto nnz = Base::knots_[i].repeat({xi[i].numel(), 1}) >
                 xi[i].flatten().view({-1, 1});
      indices[i] =
          torch::remainder(std::get<1>(((nnz.cumsum(1) == 1) & nnz).max(1)) - 1,
                           Base::nknots_[i] - Base::degrees_[i] - 1)
              .view(xi[i].sizes());
    }
    return indices;
  }
  /// @}

  /// @brief Returns the B-spline object with uniformly refined knot
  /// and coefficient vectors
  ///
  /// If `dim = -1`, new knot values are inserted uniformly in each
  /// knot span in all spatial dimensions. Otherwise, i.e., `dim !=
  /// -1` new knots are only inserted in the specified dimension.
  inline NonUniformBSplineCore &uniform_refine(int numRefine = 1,
                                               int dim = -1) {
    assert(numRefine > 0);
    assert(dim == -1 || (dim >= 0 && dim < Base::parDim_));

    // Update knot vectors, number of knots and coefficients
    std::array<int64_t, Base::parDim_> nknots, ncoeffs;
    utils::TensorArray<Base::parDim_> knots, knots_indices;

    for (short_t i = 0; i < Base::parDim_; ++i) {
      auto [kv_cpu, kv_accessor] =
          utils::to_tensorAccessor<typename Base::value_type, 1>(
              Base::knots_[i], torch::kCPU);

      std::vector<typename Base::value_type> kv;
      kv.reserve(Base::nknots_[i]);
      kv.push_back(kv_accessor[0]);

      for (int64_t j = 1; j < kv_accessor.size(0); ++j) {

        if ((dim == -1 || dim == i) && (kv_accessor[j - 1] < kv_accessor[j]))
          for (short_t refine = 1; refine < (2 << (numRefine - 1)); ++refine)
            kv.push_back(kv_accessor[j - 1] +
                         static_cast<typename Base::value_type>(refine) /
                             static_cast<typename Base::value_type>(
                                 2 << (numRefine - 1)) *
                             (kv_accessor[j] - kv_accessor[j - 1]));

        kv.push_back(kv_accessor[j]);
      }

      knots[i] = utils::to_tensor(kv, Base::options_);
      nknots[i] = kv.size();
      ncoeffs[i] = nknots[i] - Base::degrees_[i] - 1;
    }

    // The updated knot vectors have lengths \f$m_d+p_d+1\f$, where
    // \f$m_d\f$ is the number of coefficients after the update. To
    // update the coefficients using the Oslo algorithm (Algorithm
    // 4.11 from \cite Lyche:2011) we need to neglect the last
    // \f$p_d+1\f$ knots in what follows
    for (short_t i = 0; i < Base::parDim_; ++i)
      knots_indices[i] = knots[i].index({torch::indexing::Slice(
          0, knots[i].numel() - Base::degrees_[i] - 1)});

    // Get indices of the first \f$m_d\f$ new knots relative to old
    // knot vectors
    auto new_knot_indices = find_knot_indices(knots_indices);

    // Update coefficient vector
    Base::update_coeffs(knots, new_knot_indices);

    // Swap old and new data
    knots.swap(Base::knots_);
    nknots.swap(Base::nknots_);
    ncoeffs.swap(Base::ncoeffs_);

    Base::ncoeffs_reverse_ = Base::ncoeffs_;
    std::reverse(Base::ncoeffs_reverse_.begin(), Base::ncoeffs_reverse_.end());

    return *this;
  }

  /// @brief Returns the B-spline object with refined knot and
  /// coefficient vectors
  inline NonUniformBSplineCore &
  insert_knots(const utils::TensorArray<Base::parDim_> &knots) {
    std::array<int64_t, Base::parDim_> nknots(Base::nknots_);
    std::array<int64_t, Base::parDim_> ncoeffs(Base::ncoeffs_);
    utils::TensorArray<Base::parDim_> knots_, knots_indices;

    // Update number of knots and coefficients and generate new knot
    // vectors
    for (short_t i = 0; i < Base::parDim_; ++i) {
      nknots[i] += knots[i].numel();
      ncoeffs[i] += knots[i].numel();
      knots_[i] =
          std::get<0>(torch::sort(torch::cat({Base::knots_[i], knots[i]})));
    }

    // The updated knot vectors have lengths \f$m_d+p_d+1\f$, where
    // \f$m_d\f$ is the number of coefficients after the update. To
    // update the coefficients using the Oslo algorithm (Algorithm
    // 4.11 from \cite Lyche:2011) we need to neglect the last
    // \f$p_d+1\f$ knots in what follows
    for (short_t i = 0; i < Base::parDim_; ++i)
      knots_indices[i] = knots_[i].index({torch::indexing::Slice(
          0, knots_[i].numel() - Base::degrees_[i] - 1)});

    // Get indices of the first \f$m_d\f$ new knots relative to old
    // knot vectors
    auto new_knot_indices = find_knot_indices(knots_indices);

    // Update coefficient vector
    Base::update_coeffs(knots_, new_knot_indices);

    // Swap old and new data
    knots_.swap(Base::knots_);
    nknots.swap(Base::nknots_);
    ncoeffs.swap(Base::ncoeffs_);

    Base::ncoeffs_reverse_ = Base::ncoeffs_;
    std::reverse(Base::ncoeffs_reverse_.begin(), Base::ncoeffs_reverse_.end());

    return *this;
  }

  /// @brief Returns the B-spline object with updated knot and
  /// coefficient vectors with reduced continuity
  inline NonUniformBSplineCore &reduce_continuity(int numReduce = 1,
                                                  int dim = -1) {
    assert(numReduce > 0);
    assert(dim == -1 || (dim >= 0 && dim < Base::parDim_));

    // Update knot vectors, number of knots and coefficients
    std::array<int64_t, Base::parDim_> nknots, ncoeffs;
    utils::TensorArray<Base::parDim_> knots, knots_indices;

    for (short_t i = 0; i < Base::parDim_; ++i) {
      auto [kv_cpu, kv_accessor] =
          utils::to_tensorAccessor<typename Base::value_type, 1>(
              Base::knots_[i], torch::kCPU);

      std::vector<typename Base::value_type> kv;
      kv.reserve(Base::nknots_[i]);
      kv.push_back(kv_accessor[0]);

      for (int64_t j = 1; j < kv_accessor.size(0); ++j) {

        if ((dim == -1 || dim == i) && (kv_accessor[j - 1] < kv_accessor[j]) &&
            (kv_accessor[j] < kv_accessor[kv_accessor.size(0) - 1]))
          for (short_t reduce = 0; reduce < numReduce; ++reduce)
            kv.push_back(kv_accessor[j]);

        kv.push_back(kv_accessor[j]);
      }

      knots[i] = utils::to_tensor(kv, Base::options_);
      nknots[i] = kv.size();
      ncoeffs[i] = nknots[i] - Base::degrees_[i] - 1;
    }

    // The updated knot vectors have lengths \f$m_d+p_d+1\f$, where
    // \f$m_d\f$ is the number of coefficients after the update. To
    // update the coefficients using the Oslo algorithm (Algorithm
    // 4.11 from \cite Lyche:2011) we need to neglect the last
    // \f$p_d+1\f$ knots in what follows
    for (short_t i = 0; i < Base::parDim_; ++i)
      knots_indices[i] = knots[i].index({torch::indexing::Slice(
          0, knots[i].numel() - Base::degrees_[i] - 1)});

    // Get indices of the first \f$m_d\f$ new knots relative to old
    // knot vectors
    auto new_knot_indices = find_knot_indices(knots_indices);

    // Update coefficient vector
    Base::update_coeffs(knots, new_knot_indices);

    // Swap old and new data
    knots.swap(Base::knots_);
    nknots.swap(Base::nknots_);
    ncoeffs.swap(Base::ncoeffs_);

    Base::ncoeffs_reverse_ = Base::ncoeffs_;
    std::reverse(Base::ncoeffs_reverse_.begin(), Base::ncoeffs_reverse_.end());

    return *this;
  }

#ifdef IGANET_WITH_GISMO

  // @brief Updates the B-spline object from a given gsBSpline object
  auto &from_gismo(const gismo::gsBSpline<typename Base::value_type> &bspline,
                   bool updateCoeffs = true, bool updateKnotVector = false) {

    if (updateKnotVector) {

      if constexpr (Base::parDim_ == 1) {

        if (bspline.degree(0) != Base::degrees_[0])
          throw std::runtime_error("Degrees mismatch");

        if (bspline.knots(0).size() != Base::nknots_[0])
          throw std::runtime_error("Knot vector dimensions mismatch");

        auto [knots0_cpu, knots0_accessor] =
            utils::to_tensorAccessor<typename Base::value_type, 1>(
                Base::knots_[0], torch::kCPU);

        const typename Base::value_type *knots0_ptr =
            bspline.knots(0).asMatrix().data();

        for (int64_t i = 0; i < Base::nknots_[0]; ++i)
          knots0_accessor[i] = knots0_ptr[i];

        Base::knots_[0] = Base::knots_[0].to(Base::options_.device());

      } else
        throw std::runtime_error("Invalid parametric dimension");
    }

    if (updateCoeffs) {

      if (bspline.coefs().rows() != Base::geoDim_)
        throw std::runtime_error("Geometric dimensions mismatch");

      if (bspline.coefs().cols() != Base::ncumcoeffs())
        throw std::runtime_error("Coefficient vector dimensions mismatch");

      for (short_t g = 0; g < Base::geoDim_; ++g) {

        auto [coeffs_cpu, coeffs_accessor] =
            utils::to_tensorAccessor<typename Base::value_type, 1>(
                Base::coeffs_[g], torch::kCPU);

        const typename Base::value_type *coeffs_ptr =
            bspline.coefs().row(g).data();

        for (int64_t i = 0; i < Base::ncoeffs_[g]; ++i)
          coeffs_accessor[i] = coeffs_ptr[i];

        Base::coeffs_[g] = Base::coeffs_[g].to(Base::options_.device());
      }
    }

    return *this;
  }

  // @brief Updates the B-spline object from a given gsTensorBSpline object
  auto &from_gismo(
      const gismo::gsTensorBSpline<Base::parDim_, typename Base::value_type>
          &bspline,
      bool updateCoeffs = true, bool updateKnotVector = false) {

    if (updateKnotVector) {

      for (short_t i = 0; i < Base::parDim_; ++i) {
        if (bspline.degree(i) != Base::degrees_[i])
          throw std::runtime_error("Degrees mismatch");

        if (bspline.knots(i).size() != Base::nknots_[i])
          throw std::runtime_error("Knot vector dimensions mismatch");

        auto [knots_cpu, knots_accessor] =
            utils::to_tensorAccessor<typename Base::value_type, 1>(
                Base::knots_[i], torch::kCPU);

        const typename Base::value_type *knots_ptr =
            bspline.knots(i).asMatrix().data();

        for (int64_t i = 0; i < Base::nknots_[i]; ++i)
          knots_accessor[i] = knots_ptr[i];

        Base::knots_[i] = Base::knots_[i].to(Base::options_.device());
      }
    }

    if (updateCoeffs) {

      if (bspline.coefs().rows() != Base::geoDim_)
        throw std::runtime_error("Geometric dimensions mismatch");

      if (bspline.coefs().cols() != Base::ncumcoeffs())
        throw std::runtime_error("Coefficient vector dimensions mismatch");

      for (short_t g = 0; g < Base::geoDim_; ++g) {

        auto [coeffs_cpu, coeffs_accessor] =
            utils::to_tensorAccessor<typename Base::value_type, 1>(
                Base::coeffs_[g], torch::kCPU);

        const typename Base::value_type *coeffs_ptr =
            bspline.coefs().row(g).data();

        for (int64_t i = 0; i < Base::ncoeffs_[g]; ++i)
          coeffs_accessor[i] = coeffs_ptr[i];

        Base::coeffs_[g] = Base::coeffs_[g].to(Base::options_.device());
      }
    }

    return *this;
  }

#else // IGANET_WITH_GISMO

  template <typename BSpline> auto &from_gismo(BSpline &bspline, bool, bool) {
    throw std::runtime_error(
        "This functions must be compiled with -DIGANET_WITH_GISMO turned on");
    return *this;
  }

#endif // IGANET_WITH_GISMO
};

  /// @brief Spline base class
  class Spline_ {};

  /// @brief Concept to identify template parameters that are derived from iganet::Spline_
  template<typename T>
  concept SplineType = std::is_base_of_v<Spline_, T>;

  /// @brief Concept to identify template parameters that are derived from iganet::Spline_ and iganet::UniformSplineCore_
  template<typename T>
  concept UniformSplineType = std::is_base_of_v<Spline_, T> &&
    std::is_base_of_v<UniformSplineCore_, T> && !std::is_base_of_v<NonUniformSplineCore_, T>;

    /// @brief Concept to identify template parameters that are derived from iganet::Spline_ and iganet::NonUniformSplineCore_
  template<typename T>
  concept NonUniformSplineType = std::is_base_of_v<Spline_, T> &&
    std::is_base_of_v<NonUniformSplineCore_, T>;
  
/// @brief B-spline (common high-level functionality)
///
/// This class implements some high-level common functionality of
/// all B-spline classes, e.g., plotting which rely on low-level
/// functionality that is implemented differently for uniform and
/// non-uniform B-spline. C++ suggests to use virtual methods for
/// this purpose and implement the common functionality in a base
/// class. However, this is not performant for low-level
/// functionality, e.g., point-wise function evaluation which is
/// called repeatedly. Moreover, virtual methods do not work with
/// templated functions, which is why we implement high-level common
/// functionality here and 'inject' the core functionality by
/// deriving from a particular base class.
  template <typename BSplineCore>
  requires SplineCoreType<BSplineCore>
  class BSplineCommon : public Spline_,
                      public BSplineCore,
                      protected utils::FullQualifiedName {
public:
  /// @brief Constructors from the base class
  using BSplineCore::BSplineCore;

  /// @brief Deduces the type of the template template parameter `T`
  /// when exposed to the class template parameters `real_t` and
  /// `GeoDim`, and the `Degrees` parameter pack. The optional
  /// template parameter `degree_elevate` can be used to
  /// (de-)elevate the degrees by an additive constant
  template <template <typename, short_t, short_t...> class T,
            std::make_signed<short_t>::type degree_elevate = 0>
  using derived_type = BSplineCommon<
      typename BSplineCore::template derived_type<T, degree_elevate>>;

  /// @brief Deduces the self-type possibly degrees (de-)elevated by
  /// the additive constant `degree_elevate`
  template <std::make_signed<short_t>::type degree_elevate = 0>
  using self_type =
      BSplineCommon<typename BSplineCore::template self_type<degree_elevate>>;

  /// @brief Deduces the derived self-type when exposed to different
  /// class template parameters `real_t` and `GeoDim`, and the
  /// `Degrees` parameter pack
  template <typename real_t, short_t GeoDim, short_t... Degrees>
  using derived_self_type =
      BSplineCommon<typename BSplineCore::template derived_self_type<
          real_t, GeoDim, Degrees...>>;

  /// @brief Deduces the derived self-type when exposed to a
  /// different class template parameter `real_t`
  template <typename other_t>
  using real_derived_self_type = BSplineCommon<
      typename BSplineCore::template real_derived_self_type<other_t>>;

  /// @brief Shared pointer for BSplineCommon
  using Ptr = std::shared_ptr<BSplineCommon>;

  /// @brief Unique pointer for BSplineCommon
  using uPtr = std::unique_ptr<BSplineCommon>;

  /// @brief Copy constructor
  BSplineCommon(const BSplineCommon &) = default;

  /// @brief Copy/clone constructor
  BSplineCommon(const BSplineCommon &other, bool clone) : BSplineCommon(other) {
    if (clone)
      for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
        BSplineCore::coeffs_[i] = other.coeffs(i).clone();
  }

  /// @brief Copy constructor with external coefficients
  BSplineCommon(const BSplineCommon &other,
                const utils::TensorArray<BSplineCore::geoDim_> &coeffs,
                bool clone = false)
      : BSplineCommon(other) {
    if (clone)
      for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
        BSplineCore::coeffs_[i] = coeffs[i].clone();
    else
      for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
        BSplineCore::coeffs_[i] = coeffs[i];
  }

  /// @brief Move constructor
  BSplineCommon(BSplineCommon &&) = default;

  /// @brief Move constructor with external coefficients
  BSplineCommon(BSplineCommon &&other,
                utils::TensorArray<BSplineCore::geoDim_> &&coeffs)
      : BSplineCommon(std::move(other)) {
    for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
      BSplineCore::coeffs_[i] = std::move(coeffs[i]);
  }

  /// @brief Creates a new B-spline object as unique pointer
  /// @{
  inline static Ptr
  make_unique(Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return uPtr(new BSplineCommon(options));
  }

  inline static Ptr
  make_unique(const std::array<int64_t, BSplineCore::parDim_> &ncoeffs,
              enum init init = init::greville,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return uPtr(new BSplineCommon(ncoeffs, init, options));
  }

  inline static Ptr
  make_unique(const std::array<int64_t, BSplineCore::parDim_> &ncoeffs,
              const utils::TensorArray<BSplineCore::geoDim_> &coeffs,
              bool clone = false,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return uPtr(new BSplineCommon(ncoeffs, coeffs, clone, options));
  }

  inline static Ptr
  make_unique(const std::array<int64_t, BSplineCore::parDim_> &ncoeffs,
              utils::TensorArray<BSplineCore::geoDim_> &&coeffs,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return uPtr(new BSplineCommon(ncoeffs, coeffs, options));
  }

  inline static Ptr
  make_unique(const std::array<std::vector<typename BSplineCore::value_type>,
                               BSplineCore::parDim_> &kv,
              enum init init = init::greville,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return uPtr(new BSplineCommon(kv, init, options));
  }

  inline static Ptr
  make_unique(const std::array<std::vector<typename BSplineCore::value_type>,
                               BSplineCore::parDim_> &kv,
              const utils::TensorArray<BSplineCore::geoDim_> &coeffs,
              bool clone = false,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return uPtr(new BSplineCommon(kv, coeffs, clone, options));
  }
  /// @}

  /// @brief Creates a new B-spline object as shared pointer
  /// @{
  inline static Ptr
  make_shared(Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return Ptr(new BSplineCommon(options));
  }

  inline static Ptr
  make_shared(const std::array<int64_t, BSplineCore::parDim_> &ncoeffs,
              enum init init = init::greville,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return Ptr(new BSplineCommon(ncoeffs, init, options));
  }

  inline static Ptr
  make_shared(const std::array<int64_t, BSplineCore::parDim_> &ncoeffs,
              const utils::TensorArray<BSplineCore::geoDim_> &coeffs,
              bool clone = false,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return Ptr(new BSplineCommon(ncoeffs, coeffs, clone, options));
  }

  inline static Ptr
  make_shared(const std::array<int64_t, BSplineCore::parDim_> &ncoeffs,
              utils::TensorArray<BSplineCore::geoDim_> &&coeffs,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return Ptr(new BSplineCommon(ncoeffs, coeffs, options));
  }

  inline static Ptr
  make_shared(const std::array<std::vector<typename BSplineCore::value_type>,
                               BSplineCore::parDim_> &kv,
              enum init init = init::greville,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return Ptr(new BSplineCommon(kv, init, options));
  }

  inline static Ptr
  make_shared(const std::array<std::vector<typename BSplineCore::value_type>,
                               BSplineCore::parDim_> &kv,
              const utils::TensorArray<BSplineCore::geoDim_> &coeffs,
              bool clone = false,
              Options<typename BSplineCore::value_type> options =
                  Options<typename BSplineCore::value_type>{}) {
    return Ptr(new BSplineCommon(kv, coeffs, clone, options));
  }
  /// @}

  /// @brief Returns the B-spline object with uniformly refined knot
  /// and coefficient vectors
  ///
  /// If `dim = -1`, new knot values are inserted uniformly in each
  /// knot span in all spatial dimensions. Otherwise, i.e., `dim !=
  /// -1` new knots are only inserted in the specified dimension.
  inline BSplineCommon &uniform_refine(int numRefine = 1, int dim = -1) {
    BSplineCore::uniform_refine(numRefine, dim);
    return *this;
  }

  /// @brief Returns a clone of the B-spline object
  inline auto clone() const {
    BSplineCommon result;

    result.nknots_ = BSplineCore::nknots_;
    result.ncoeffs_ = BSplineCore::ncoeffs_;
    result.ncoeffs_reverse_ = BSplineCore::ncoeffs_reverse_;

    for (short_t i = 0; i < BSplineCore::parDim_; ++i)
      result.knots_[i] = BSplineCore::knots_[i].clone();

    for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
      result.coeffs_[i] = BSplineCore::coeffs_[i].clone();

    return result;
  }

  /// @brief Returns a copy of the B-spline object with settings from options
  template <typename real_t> inline auto to(Options<real_t> options) const {
    BSplineCommon<typename BSplineCore::template real_derived_self_type<real_t>>
        result(options);

    result.nknots_ = BSplineCore::nknots_;
    result.ncoeffs_ = BSplineCore::ncoeffs_;
    result.ncoeffs_reverse_ = BSplineCore::ncoeffs_reverse_;

    for (short_t i = 0; i < BSplineCore::parDim_; ++i)
      result.knots_[i] = BSplineCore::knots_[i].to(options);

    for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
      result.coeffs_[i] = BSplineCore::coeffs_[i].to(options);

    return result;
  }

  /// @brief Returns a copy of the B-spline object with settings from device
  inline auto to(torch::Device device) const {
    BSplineCommon result(BSplineCore::options_.device(device));

    result.nknots_ = BSplineCore::nknots_;
    result.ncoeffs_ = BSplineCore::ncoeffs_;
    result.ncoeffs_reverse_ = BSplineCore::ncoeffs_reverse_;

    for (short_t i = 0; i < BSplineCore::parDim_; ++i)
      result.knots_[i] = BSplineCore::knots_[i].to(device);

    for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
      result.coeffs_[i] = BSplineCore::coeffs_[i].to(device);

    return result;
  }

  /// @brief Returns a copy of the B-spline object with real_t type
  template <typename real_t> inline auto to() const {
    return to(BSplineCore::options_.template dtype<real_t>());
  }

  /// @brief Computes the difference between two compatible B-spline
  /// objects
  ///
  /// If `dim = -1` the full coefficient vector of \a other is
  /// subtracted from that of the current B-spline object. Otherwise,
  /// only the specified direction is subtracted
  inline auto diff(const BSplineCommon &other, int dim = -1) {

    bool compatible(true);

    for (short_t i = 0; i < BSplineCore::parDim_; ++i)
      compatible *= (BSplineCore::nknots(i) == other.nknots(i));

    for (short_t i = 0; i < BSplineCore::parDim_; ++i)
      compatible *= (BSplineCore::ncoeffs(i) == other.ncoeffs(i));

    if (!compatible)
      throw std::runtime_error("B-splines are not compatible");

    if (dim == -1) {
      for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
        BSplineCore::coeffs(i) -= other.coeffs(i);
    } else
      BSplineCore::coeffs(dim) -= other.coeffs(dim);

    return *this;
  }

  /// @brief Computes the absolute difference between two compatible
  /// B-spline objects
  ///
  /// If `dim = -1` the full coefficient vector of \a other is
  /// subtracted from that of the current B-spline object. Otherwise,
  /// only the specified direction is subtracted
  inline auto abs_diff(const BSplineCommon &other, int dim = -1) {

    bool compatible(true);

    for (short_t i = 0; i < BSplineCore::parDim_; ++i)
      compatible *= (BSplineCore::nknots(i) == other.nknots(i));

    for (short_t i = 0; i < BSplineCore::parDim_; ++i)
      compatible *= (BSplineCore::ncoeffs(i) == other.ncoeffs(i));

    if (!compatible)
      throw std::runtime_error("B-splines are not compatible");

    if (dim == -1) {
      for (short_t i = 0; i < BSplineCore::geoDim_; ++i)
        BSplineCore::coeffs(i) =
            torch::abs(BSplineCore::coeffs(i) - other.coeffs(i));
    } else
      BSplineCore::coeffs(dim) =
          torch::abs(BSplineCore::coeffs(dim) - other.coeffs(dim));

    return *this;
  }

  /// @brief Scales the B-spline object by a scalar
  inline auto scale(typename BSplineCore::value_type s, int dim = -1) {
    if (dim == -1)
      for (int i = 0; i < BSplineCore::geoDim(); ++i)
        BSplineCore::coeffs(i) *= s;
    else
      BSplineCore::coeffs(dim) *= s;
    return *this;
  }

  /// @brief Scales the B-spline object by a vector
  inline auto
  scale(std::array<typename BSplineCore::value_type, BSplineCore::geoDim()> v) {
    for (int i = 0; i < BSplineCore::geoDim(); ++i)
      BSplineCore::coeffs(i) *= v[i];
    return *this;
  }

  /// @brief Translates the B-spline object by a vector
  inline auto translate(
      std::array<typename BSplineCore::value_type, BSplineCore::geoDim()> v) {
    for (int i = 0; i < BSplineCore::geoDim(); ++i)
      BSplineCore::coeffs(i) += v[i];
    return *this;
  }

  /// @brief Rotates the B-spline object by an angle in 2d
  inline auto rotate(typename BSplineCore::value_type angle) {

    static_assert(BSplineCore::geoDim() == 2,
                  "Rotation about one angle is only available in 2D");

    utils::TensorArray<2> coeffs;
    coeffs[0] = std::cos(angle) * BSplineCore::coeffs(0) -
                std::sin(angle) * BSplineCore::coeffs(1);
    coeffs[1] = std::sin(angle) * BSplineCore::coeffs(0) +
                std::cos(angle) * BSplineCore::coeffs(1);

    BSplineCore::coeffs().swap(coeffs);
    return *this;
  }

  /// @brief Rotates the B-spline object by three angles in 3d
  inline auto rotate(std::array<typename BSplineCore::value_type, 3> angle) {

    static_assert(BSplineCore::geoDim() == 3,
                  "Rotation about two angles is only available in 3D");

    utils::TensorArray<3> coeffs;
    coeffs[0] =
        std::cos(angle[0]) * std::cos(angle[1]) * BSplineCore::coeffs(0) +
        (std::sin(angle[0]) * std::sin(angle[1]) * std::cos(angle[2]) -
         std::cos(angle[0]) * std::sin(angle[2])) *
            BSplineCore::coeffs(1) +
        (std::cos(angle[0]) * std::sin(angle[1]) * std::cos(angle[2]) +
         std::sin(angle[0]) * std::sin(angle[2])) *
            BSplineCore::coeffs(2);

    coeffs[1] =
        std::cos(angle[1]) * std::sin(angle[2]) * BSplineCore::coeffs(0) +
        (std::sin(angle[0]) * std::sin(angle[1]) * std::sin(angle[2]) +
         std::cos(angle[0]) * std::cos(angle[2])) *
            BSplineCore::coeffs(1) +
        (std::cos(angle[0]) * std::sin(angle[1]) * std::sin(angle[2]) -
         std::sin(angle[0]) * std::cos(angle[2])) *
            BSplineCore::coeffs(2);

    coeffs[2] =
        -std::sin(angle[1]) * BSplineCore::coeffs(0) +
        std::sin(angle[0]) * std::cos(angle[1]) * BSplineCore::coeffs(1) +
        std::cos(angle[0]) * std::cos(angle[1]) * BSplineCore::coeffs(2);

    BSplineCore::coeffs().swap(coeffs);
    return *this;
  }

  /// @brief Computes the bounding box of the B-spline object
  inline auto boundingBox() const {

    // Lambda expression to compute the minimum value of all dimensions
    auto min_ = [&, this]<std::size_t... Is>(std::index_sequence<Is...>) {
      return torch::stack({BSplineCore::coeffs(Is).min()...});
    };

    // Lambda expression to compute the maximum value of all dimensions
    auto max_ = [&, this]<std::size_t... Is>(std::index_sequence<Is...>) {
      return torch::stack({BSplineCore::coeffs(Is).max()...});
    };

    std::pair<torch::Tensor, torch::Tensor> bbox;
    bbox.first = min_(std::make_index_sequence<BSplineCore::geoDim_>{});
    bbox.second = max_(std::make_index_sequence<BSplineCore::geoDim_>{});
    return bbox;
  }

  //  clang-format off
  /// @brief Returns a block-tensor with the curl of the
  /// B-spline object with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the curl
  ///
  /// @result Block-tensor with the curl with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}} \times \mathbf{u}
  ///        =
  ///     \begin{bmatrix}
  ///        \mathbf{i}_0 & \cdots & \mathbf{i}_{d_\text{par}} \\
  ///        \frac{\partial}{\partial\xi_0} & \cdots &
  ///        \frac{\partial}{\partial\xi_{d_\text{par}}} \\
  ///        u_0 & \cdots & u_{d_\text{par}}
  ///     \end{bmatrix}
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// equal parametric and geometric dimensionality.
  //  clang-format off
  /// @{
  template <bool memory_optimized = false>
  auto curl(const torch::Tensor &xi) const {
    return curl<memory_optimized>(utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false>
  inline auto curl(const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    return curl<memory_optimized>(xi, BSplineCore::find_knot_indices(xi));
  }
  /// @}

  //  clang-format off
  /// @brief Returns a block-tensor with the curl of the
  /// B-spline object with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the curl
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the curl
  ///
  /// @result Block-tensor with the curl with respect to
  /// the parametric variables
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}} \times \mathbf{u}
  ///        =
  ///     \begin{bmatrix}
  ///        \mathbf{i}_0 & \cdots & \mathbf{i}_{d_\text{par}} \\
  ///        \frac{\partial}{\partial\xi_0} & \cdots &
  ///        \frac{\partial}{\partial\xi_{d_\text{par}}} \\
  ///        u_0 & \cdots & u_{d_\text{par}}
  ///     \end{bmatrix}
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// equal parametric and geometric dimensionality.
  //  clang-format on
  template <bool memory_optimized = false>
  inline auto
  curl(const utils::TensorArray<BSplineCore::parDim_> &xi,
       const utils::TensorArray<BSplineCore::parDim_> &knot_indices) const {
    return curl<memory_optimized>(
        xi, knot_indices,
        BSplineCore::template find_coeff_indices<memory_optimized>(
            knot_indices));
  }

  /// @brief Returns a block-tensor with the curl of the B-spline
  /// object with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the curl
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the curl
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate
  /// the curl
  ///
  /// @result Block-tensor with the curl of the B-spline with respect
  /// to the parametric variables
  ///
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}} \cdot \mathbf{u}
  ///        =
  ///     \text{trace} ( J_{\boldsymbol{\xi}}(u) )
  ///        =
  ///     \frac{\partial u_0}{\partial \xi_0} +
  ///     \frac{\partial u_1}{\partial \xi_1} +
  ///        \dots
  ///     \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// equal parametric and geometric dimensionality.
  template <bool memory_optimized = false>
  inline auto curl(const utils::TensorArray<BSplineCore::parDim_> &xi,
                   const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
                   const torch::Tensor &coeff_indices) const {

    static_assert(BSplineCore::parDim_ == BSplineCore::geoDim_,
                  "curl(.) requires that parametric and geometric dimension "
                  "are the same");

    // Check compatibility of arguments
    for (short_t i = 0; i < BSplineCore::parDim_; ++i)
      assert(xi[i].sizes() == knot_indices[i].sizes());
    for (short_t i = 1; i < BSplineCore::parDim_; ++i)
      assert(xi[0].sizes() == xi[i].sizes());

    if constexpr (BSplineCore::parDim_ == 2)

      /// curl = 0,
      ///        0,
      ///        du_y / dx - du_x / dy
      ///
      /// Only the third component is returned
      return utils::BlockTensor<torch::Tensor, 1, 1>(
          *BSplineCore::template eval<deriv::dx, memory_optimized>(
              xi, knot_indices, coeff_indices)[1] -
          *BSplineCore::template eval<deriv::dy, memory_optimized>(
              xi, knot_indices, coeff_indices)[0]);

    else if constexpr (BSplineCore::parDim_ == 3)

      /// curl = du_z / dy - du_y / dz,
      ///        du_x / dz - du_z / dx,
      ///        du_y / dx - du_x / dy
      return utils::BlockTensor<torch::Tensor, 1, 3>(
          *BSplineCore::template eval<deriv::dy, memory_optimized>(
              xi, knot_indices, coeff_indices)[2] -
              *BSplineCore::template eval<deriv::dz, memory_optimized>(
                  xi, knot_indices, coeff_indices)[1],
          *BSplineCore::template eval<deriv::dz, memory_optimized>(
              xi, knot_indices, coeff_indices)[0] +
              *BSplineCore::template eval<deriv::dx, memory_optimized>(
                  xi, knot_indices, coeff_indices)[2],
          *BSplineCore::template eval<deriv::dx, memory_optimized>(
              xi, knot_indices, coeff_indices)[1] +
              *BSplineCore::template eval<deriv::dy, memory_optimized>(
                  xi, knot_indices, coeff_indices)[0]);

    else {
      throw std::runtime_error("Unsupported parametric/geometric dimension");
      return utils::BlockTensor<torch::Tensor, 1, 1>{};
    }
  }

  /// @brief Returns a block-tensor with the curl of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the curl
  ///
  /// @result Block-tensor with the curl with respect to the
  /// parametric variables
  /// \f[
  ///     \nabla \times {\mathbf{x}} u
  ///        =
  ///     \nabla_{\boldsymbol{\xi}} \times u \,
  ///     \operatorname{det}(\operatorname{det}(J_{\boldsymbol{\xi}}(G))^{-1} \,
  ///     J_{\boldsymbol{\xi}}(G) , \quad \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  ///
  /// @{
  template <bool memory_optimized = false, typename Geometry>
  auto icurl(const Geometry &G, const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return icurl<memory_optimized, Geometry>(G, utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false, typename Geometry>
  inline auto icurl(const Geometry &G,
                    const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return icurl<memory_optimized, Geometry>(
          G, xi, BSplineCore::find_knot_indices(xi), G.find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns a block-tensor with the curl of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the gradient
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate Jacobian of `G`
  ///
  /// @result Block-tensor with the curl with respect to the
  /// physical variables
  /// \f[
  ///     \nabla \times {\mathbf{x}} u
  ///        =
  ///     \nabla_{\boldsymbol{\xi}} \times u \,
  ///     \operatorname{det}(\operatorname{det}(J_{\boldsymbol{\xi}}(G))^{-1} \,
  ///     J_{\boldsymbol{\xi}}(G) , \quad \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  icurl(const Geometry G, const utils::TensorArray<BSplineCore::parDim_> &xi,
        const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
        const utils::TensorArray<Geometry::parDim()> &knot_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return icurl<memory_optimized, Geometry>(
          G, xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices),
          knot_indices_G,
          G.template find_coeff_indices<memory_optimized>(knot_indices_G));
  }

  /// @brief Returns a block-tensor with the curl of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the gradient
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate the Jacobian of
  /// `G`
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// gradient
  ///
  /// @param[in] coeff_indices_G Coefficient indices where to evaluate the
  /// Jacobian of `G`
  ///
  /// @result Block-tensor with the curl with respect to the
  /// physical variables
  /// \f[
  ///     \nabla \times {\mathbf{x}} u
  ///        =
  ///     \nabla_{\boldsymbol{\xi}} \times u \,
  ///     \operatorname{det}(\operatorname{det}(J_{\boldsymbol{\xi}}(G))^{-1} \,
  ///     J_{\boldsymbol{\xi}}(G) , \quad \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  icurl(const Geometry &G, const utils::TensorArray<BSplineCore::parDim_> &xi,
        const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
        const torch::Tensor &coeff_indices,
        const utils::TensorArray<Geometry::parDim()> &knot_indices_G,
        const torch::Tensor &coeff_indices_G) const {

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else {
      utils::BlockTensor<torch::Tensor, 1, 1> det;
      det[0] = std::make_shared<torch::Tensor>(torch::reciprocal(
          G.template jac<memory_optimized>(xi, knot_indices_G, coeff_indices_G)
              .det()));

      return det * (curl<memory_optimized>(xi, knot_indices, coeff_indices) *
                    G.template jac<memory_optimized>(xi, knot_indices_G,
                                                     coeff_indices_G));
    }
  }

  /// @brief Returns a block-tensor with the divergence of the
  /// B-spline object with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the divergence
  ///
  /// @result Block-tensor with the divergence with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}} \cdot \mathbf{u}
  ///        =
  ///     \text{trace} ( J_{\boldsymbol{\xi}}(u) )
  ///        =
  ///     \frac{\partial u_0}{\partial \xi_0} +
  ///     \frac{\partial u_1}{\partial \xi_1} +
  ///        \dots
  ///     \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// equal parametric and geometric dimensionality.
  /// @{
  template <bool memory_optimized = false>
  auto div(const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    return div<memory_optimized>(utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false>
  inline auto div(const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    return div<memory_optimized>(xi, BSplineCore::find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns a block-tensor with the divergence of the
  /// B-spline object with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the divergence
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the divergence
  ///
  /// @result Block-tensor with the divergence with respect to
  /// the parametric variables
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}} \cdot \mathbf{u}
  ///        =
  ///     \text{trace} ( J_{\boldsymbol{\xi}}(u) )
  ///        =
  ///     \frac{\partial u_0}{\partial \xi_0} +
  ///     \frac{\partial u_1}{\partial \xi_1} +
  ///        \dots
  ///     \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// equal parametric and geometric dimensionality.
  template <bool memory_optimized = false>
  inline auto
  div(const utils::TensorArray<BSplineCore::parDim_> &xi,
      const utils::TensorArray<BSplineCore::parDim_> &knot_indices) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    return div<memory_optimized>(
        xi, knot_indices,
        BSplineCore::template find_coeff_indices<memory_optimized>(
            knot_indices));
  }

  /// @brief Returns a block-tensor with the divergence of the
  /// B-spline object with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the divergence
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the divergence
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// divergence
  ///
  /// @result Block-tensor with the divergence of the B-spline with
  /// respect to the parametric variables
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}} \cdot \mathbf{u}
  ///        =
  ///     \text{trace} ( J_{\boldsymbol{\xi}}(u) )
  ///        =
  ///     \frac{\partial u_0}{\partial \xi_0} +
  ///     \frac{\partial u_1}{\partial \xi_1} +
  ///        \dots
  ///     \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// equal parametric and geometric dimensionality.
  template <bool memory_optimized = false>
  inline auto div(const utils::TensorArray<BSplineCore::parDim_> &xi,
                  const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
                  const torch::Tensor &coeff_indices) const {

    static_assert(BSplineCore::parDim_ == BSplineCore::geoDim_,
                  "div(.) requires parDim == geoDim");

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    // return torch::zeros_like(BSplineCore::coeffs_[0]);

    else {
      // Check compatibility of arguments
      for (short_t i = 0; i < BSplineCore::parDim_; ++i)
        assert(xi[i].sizes() == knot_indices[i].sizes());
      for (short_t i = 1; i < BSplineCore::parDim_; ++i)
        assert(xi[0].sizes() == xi[i].sizes());

      // Lambda expression to evaluate the divergence
      auto div_ = [&, this]<std::size_t... Is>(std::index_sequence<Is...>) {
        return utils::BlockTensor<torch::Tensor, 1, 1>{
            (*BSplineCore::template eval<
                 (deriv)utils::integer_pow<10, Is>::value, memory_optimized>(
                 xi, knot_indices, coeff_indices)[Is] +
             ...)};
      };

      return div_(std::make_index_sequence<BSplineCore::parDim_>{});
    }
  }

  /// @brief Returns a block-tensor with the divergence of the
  /// B-spline object with respect to the physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the divergence
  ///
  /// @result Block-tensor with the divergence with respect to the
  /// parametric variables
  /// \f[
  ///     \nabla_{\mathbf{x}} \cdot \mathbf{u}
  ///        =
  ///     \text{trace} ( J_{\mathbf{x}}(u) )
  ///        =
  ///     \frac{\partial u_0}{\partial x_0} +
  ///     \frac{\partial u_1}{\partial x_1} +
  ///     \frac{\partial u_{d_\text{geo}}}{\partial x_{d_\text{par}}}
  /// \f]
  ///
  /// @{
  template <bool memory_optimized = false, typename Geometry>
  auto idiv(const Geometry &G, const torch::Tensor &xi) {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return idiv<memory_optimized, Geometry>(G, utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false, typename Geometry>
  inline auto idiv(const Geometry &G,
                   const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return idiv<memory_optimized, Geometry>(
          G, xi, BSplineCore::find_knot_indices(xi), G.find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns a block-tensor with the divergence of the
  /// B-spline object with respect to the physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the divergence
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the divergence
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate the Jacobian of
  /// `G`
  ///
  /// @result Block-tensor with the divergence with respect to the
  /// physical variables
  /// \f[
  ///     \nabla_{\mathbf{x}} \cdot \mathbf{u}
  ///        =
  ///     \text{trace} ( J_{\mathbf{x}}(u) )
  ///        =
  ///     \frac{\partial u_0}{\partial x_0} +
  ///     \frac{\partial u_1}{\partial x_1} +
  ///     \frac{\partial u_{d_\text{geo}}}{\partial x_{d_\text{par}}}
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  idiv(const Geometry G, const utils::TensorArray<BSplineCore::parDim_> &xi,
       const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
       const utils::TensorArray<Geometry::parDim()> &knot_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return idiv<memory_optimized, Geometry>(
          G, xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices),
          knot_indices_G,
          G.template find_coeff_indices<memory_optimized>(knot_indices_G));
  }

  /// @brief Returns a block-tensor with the divergence of the
  /// B-spline object with respect to the physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the divergence
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the divergence
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate the Jacobian of
  /// `G`
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// divergence
  ///
  /// @param[in] coeff_indices_G Coefficient indices where to evaluate the
  /// Jacobian of `G`
  ///
  /// @result Block-tensor with the divergence with respect to the
  /// physical variables
  /// \f[
  ///     \nabla_{\mathbf{x}} \cdot \mathbf{u}
  ///        =
  ///     \text{trace} ( J_{\mathbf{x}}(u) )
  ///        =
  ///     \frac{\partial u_0}{\partial x_0} +
  ///     \frac{\partial u_1}{\partial x_1} +
  ///     \frac{\partial u_{d_\text{geo}}}{\partial x_{d_\text{par}}}
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto idiv(const Geometry &G,
                   const utils::TensorArray<BSplineCore::parDim_> &xi,
                   const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
                   const torch::Tensor &coeff_indices,
                   const utils::TensorArray<Geometry::parDim()> &knot_indices_G,
                   const torch::Tensor &coeff_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ijac<memory_optimized, Geometry>(G, xi, knot_indices,
                                              coeff_indices, knot_indices_G,
                                              coeff_indices_G)
          .trace();
  }

  /// @brief Returns a block-tensor with the gradient of the B-spline
  /// object in the points `xi` with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @result Block-tensor with the gradient with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}}u
  ///        =
  ///     \left(\frac{\partial u}{\partial \xi_0},
  ///           \frac{\partial u}{\partial \xi_1},
  ///           \dots
  ///           \frac{\partial u}{\partial \xi_{d_\text{par}}}\right)
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// geometric dimensionality 1, i.e. scalar fields.
  ///
  /// @{
  template <bool memory_optimized = false>
  inline auto grad(const torch::Tensor &xi) const {

    static_assert(BSplineCore::geoDim_ == 1,
                  "grad(.) requires 1D variable, use jac(.) instead");

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return grad<memory_optimized>(utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false>
  inline auto grad(const utils::TensorArray<BSplineCore::parDim_> &xi) const {

    static_assert(BSplineCore::geoDim_ == 1,
                  "grad(.) requires 1D variable, use jac(.) instead");

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return grad<memory_optimized>(xi, BSplineCore::find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns a block-tensor with the gradient of the
  /// B-spline object in the points `xi` with respect to the
  /// parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the gradient
  ///
  /// @result Block-tensor with the gradient with respect to
  /// the parametric variables
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}}u
  ///        =
  ///     \left(\frac{\partial u}{\partial \xi_0},
  ///           \frac{\partial u}{\partial \xi_1},
  ///           \dots
  ///           \frac{\partial u}{\partial \xi_{d_\text{par}}}\right)
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// geometric dimensionality 1, i.e. scalar fields.
  template <bool memory_optimized = false>
  inline auto
  grad(const utils::TensorArray<BSplineCore::parDim_> &xi,
       const utils::TensorArray<BSplineCore::parDim_> &knot_indices) const {

    static_assert(BSplineCore::geoDim_ == 1,
                  "grad(.) requires 1D variable, use jac(.) instead");

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return grad<memory_optimized>(
          xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices));
  }

  /// @brief Returns a block-tensor with the gradient of the
  /// B-spline object in the points `xi` with respect to the
  /// parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the gradient
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// gradient
  ///
  /// @result Block-tensor with the gradient with respect to the
  /// parametric variables
  /// \f[
  ///     \nabla_{\boldsymbol{\xi}}u
  ///        =
  ///     \left(\frac{\partial u}{\partial \xi_0},
  ///           \frac{\partial u}{\partial \xi_1},
  ///           \dots
  ///           \frac{\partial u}{\partial \xi_{d_\text{par}}}\right)
  /// \f]
  ///
  /// @note This function can only be applied to B-spline objects with
  /// geometric dimensionality 1, i.e. scalar fields.
  template <bool memory_optimized = false>
  inline auto grad(const utils::TensorArray<BSplineCore::parDim_> &xi,
                   const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
                   const torch::Tensor &coeff_indices) const {

    static_assert(BSplineCore::geoDim_ == 1,
                  "grad(.) requires 1D variable, use jac(.) instead");

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};

    else {
      // Check compatibility of arguments
      for (short_t i = 0; i < BSplineCore::parDim_; ++i)
        assert(xi[i].sizes() == knot_indices[i].sizes());
      for (short_t i = 1; i < BSplineCore::parDim_; ++i)
        assert(xi[0].sizes() == xi[i].sizes());

      // Lambda expression to evaluate the gradient
      auto grad_ = [&, this]<std::size_t... Is>(std::index_sequence<Is...>) {
        return utils::BlockTensor<torch::Tensor, 1, BSplineCore::parDim_>{
            BSplineCore::template eval<(deriv)utils::integer_pow<10, Is>::value,
                                       memory_optimized>(xi, knot_indices,
                                                         coeff_indices)...};
      };

      return grad_(std::make_index_sequence<BSplineCore::parDim_>{});
    }
  }

  /// @brief Returns a block-tensor with the gradient of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @result Block-tensor with the gradient with respect to the
  /// parametric variables
  /// \f[
  ///     \nabla_{\mathbf{x}} u
  ///        =
  ///     \nabla_{\boldsymbol{\xi}} u \, J_{\boldsymbol{\xi}}(G)^{-T} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  ///
  /// @{
  template <bool memory_optimized = false, typename Geometry>
  auto igrad(const Geometry &G, const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return igrad<memory_optimized, Geometry>(G, utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false, typename Geometry>
  inline auto igrad(const Geometry &G,
                    const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return igrad<memory_optimized, Geometry>(
          G, xi, BSplineCore::find_knot_indices(xi), G.find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns a block-tensor with the gradient of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the gradient
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate Jacobian of `G`
  ///
  /// @result Block-tensor with the gradient with respect to the
  /// physical variables
  /// \f[
  ///     \nabla_{\mathbf{x}} u
  ///        =
  ///     \nabla_{\boldsymbol{\xi}} u \, J_{\boldsymbol{\xi}}(G)^{-T} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  igrad(const Geometry G, const utils::TensorArray<BSplineCore::parDim_> &xi,
        const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
        const utils::TensorArray<Geometry::parDim()> &knot_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return igrad<memory_optimized, Geometry>(
          G, xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices),
          knot_indices_G,
          G.template find_coeff_indices<memory_optimized>(knot_indices_G));
  }

  /// @brief Returns a block-tensor with the gradient of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the gradient
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate the Jacobian of
  /// `G`
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// gradient
  ///
  /// @param[in] coeff_indices_G Coefficient indices where to evaluate the
  /// Jacobian of `G`
  ///
  /// @result Block-tensor with the gradient with respect to the
  /// physical variables
  /// \f[
  ///     \nabla_{\mathbf{x}} u
  ///        =
  ///     \nabla_{\boldsymbol{\xi}} u \, J_{\boldsymbol{\xi}}(G)^{-T} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  igrad(const Geometry &G, const utils::TensorArray<BSplineCore::parDim_> &xi,
        const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
        const torch::Tensor &coeff_indices,
        const utils::TensorArray<Geometry::parDim()> &knot_indices_G,
        const torch::Tensor &coeff_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return grad<memory_optimized>(xi, knot_indices, coeff_indices) *
             G.template jac<memory_optimized>(xi, knot_indices_G,
                                              coeff_indices_G)
                 .ginv();
  }

  //  clang-format off
  /// @brief Returns a block-tensor with the Hessian of the B-spline
  /// object in the points `xi` with respect to the parametric
  /// variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Hessian
  ///
  /// @result Block-tensor with the Hessian with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     H_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \begin{bmatrix}
  ///           \frac{\partial^2 u}{\partial^2 \xi_0}&
  ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_{d_\text{par}}}
  ///           \\ \frac{\partial^2 u}{\partial \xi_1\partial \xi_0}&
  ///           \frac{\partial^2 u}{\partial^2 \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_{d_\text{par}}}
  ///           \\
  ///           \vdots& \vdots & \ddots & \vdots \\
  ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_0}&
  ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial^2 \xi_{d_\text{par}}}
  ///     \end{bmatrix}
  /// \f]
  ///
  /// @note If the B-spline object has geometric dimension larger
  /// then one then all Hessian matrices are returned as slices of a
  /// rank-3 tensor.
  //  clang-format on
  ///
  /// @{
  template <bool memory_optimized = false>
  inline auto hess(const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return hess<memory_optimized>(utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false>
  inline auto hess(const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return hess<memory_optimized>(xi, BSplineCore::find_knot_indices(xi));
  }
  /// @}

  //  clang-format off
  /// @brief Returns a block-tensor with the Hessian of the B-spline
  /// object in the points `xi` with respect to the parametric
  /// variables
  ///
  /// @param[in] xi Point(s) where to evaluate the gradient
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Hessian
  ///
  /// @result Block-tensor with the Hessian with respect to
  /// the parametric variables
  /// \f[
  ///     H_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \begin{bmatrix}
  ///           \frac{\partial^2 u}{\partial^2 \xi_0}&
  ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_{d_\text{par}}}
  ///           \\ \frac{\partial^2 u}{\partial \xi_1\partial \xi_0}&
  ///           \frac{\partial^2 u}{\partial^2 \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_{d_\text{par}}}
  ///           \\
  ///           \vdots& \vdots & \ddots & \vdots \\
  ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_0}&
  ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial^2 \xi_{d_\text{par}}}
  ///     \end{bmatrix}
  /// \f]
  ///
  /// @note If the B-spline object has geometric dimension larger
  /// then one then all Hessian matrices are returned as slices of a
  /// rank-3 tensor.
  //  clang-format on
  template <bool memory_optimized = false>
  inline auto
  hess(const utils::TensorArray<BSplineCore::parDim_> &xi,
       const utils::TensorArray<BSplineCore::parDim_> &knot_indices) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return hess<memory_optimized>(
          xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices));
  }

  //  clang-format off
  /// @brief Returns a block-tensor with the Hessian of the B-spline
  /// object in the points `xi` with respect to the parametric
  /// variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Hessian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Hessian
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the Hessian
  ///
  /// @result Block-tensor with the Hessian with respect to the
  /// parametric variables
  /// \f[
  ///     H_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \begin{bmatrix}
  ///           \frac{\partial^2 u}{\partial^2 \xi_0}&
  ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_{d_\text{par}}}
  ///           \\ \frac{\partial^2 u}{\partial \xi_1\partial \xi_0}&
  ///           \frac{\partial^2 u}{\partial^2 \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_{d_\text{par}}}
  ///           \\
  ///           \vdots& \vdots & \ddots & \vdots \\
  ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_0}&
  ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial^2 u}{\partial^2 \xi_{d_\text{par}}}
  ///     \end{bmatrix}
  /// \f]
  ///
  /// @note If the B-spline object has geometric dimension larger
  /// then one then all Hessian matrices are returned as slices of a
  /// rank-3 tensor.
  //  clang-format on
  template <bool memory_optimized = false>
  inline auto hess(const utils::TensorArray<BSplineCore::parDim_> &xi,
                   const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
                   const torch::Tensor &coeff_indices) const {

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};

    else {
      // Check compatibility of arguments
      for (short_t i = 0; i < BSplineCore::parDim_; ++i)
        assert(xi[i].sizes() == knot_indices[i].sizes());
      for (short_t i = 1; i < BSplineCore::parDim_; ++i)
        assert(xi[0].sizes() == xi[i].sizes());

      // Lambda expression to evaluate the hessian
      auto hess_ = [&, this]<std::size_t... Is>(std::index_sequence<Is...>) {
        return utils::BlockTensor<torch::Tensor, BSplineCore::parDim_,
                                  BSplineCore::geoDim_, BSplineCore::parDim_>{
            BSplineCore::template eval<
                (deriv)utils::integer_pow<10,
                                          Is / BSplineCore::parDim_>::value +
                    (deriv)utils::integer_pow<10,
                                              Is % BSplineCore::parDim_>::value,
                memory_optimized>(xi, knot_indices, coeff_indices)...}
            .reorder_ikj();
      };

      return hess_(std::make_index_sequence<BSplineCore::parDim_ *
                                            BSplineCore::parDim_>{});
    }
  }

  /// @brief Returns a block-tensor with the Hessian of the B-spline
  /// object in the points `xi` with respect to the physical
  /// variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Hessian
  ///
  /// @result Block-tensor with the Hessian with respect to the
  /// parametric variables
  /// \f[
  ///     H_{\mathbf{x}}(u)
  ///        =
  ///     J_{\boldsymbol{\xi}}(G)^{-T}
  ///     \left(
  ///       H_\boldsymbol{\xi}(u)
  ///       -
  ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
  ///     \right)
  ///     J_{\boldsymbol{\xi}}(G)^{-1} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  ///
  /// @{
  template <bool memory_optimized = false, typename Geometry>
  auto ihess(const Geometry &G, const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ihess<memory_optimized, Geometry>(G, utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false, typename Geometry>
  inline auto ihess(const Geometry &G,
                    const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ihess<memory_optimized, Geometry>(
          G, xi, BSplineCore::find_knot_indices(xi), G.find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns a block-tensor with the Hessian of the B-spline
  /// object in the points `xi` with respect to the physical
  /// variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Hessian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Hessian
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate Jacobian of `G`
  ///
  /// @result Block-tensor with the Hessian with respect to the
  /// physical variables
  /// \f[
  ///     H_{\mathbf{x}}(u)
  ///        =
  ///     J_{\boldsymbol{\xi}}(G)^{-T}
  ///     \left(
  ///       H_\boldsymbol{\xi}(u)
  ///       -
  ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
  ///     \right)
  ///     J_{\boldsymbol{\xi}}(G)^{-1} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  ihess(const Geometry G, const utils::TensorArray<BSplineCore::parDim_> &xi,
        const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
        const utils::TensorArray<Geometry::parDim()> &knot_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ihess<memory_optimized, Geometry>(
          G, xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices),
          knot_indices_G,
          G.template find_coeff_indices<memory_optimized>(knot_indices_G));
  }

  /// @brief Returns a block-tensor with the Hessian of the B-spline
  /// object in the points `xi` with respect to the physical
  /// variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Hessian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Hessian
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate the Jacobian of
  /// `G`
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the Hessian
  ///
  /// @param[in] coeff_indices_G Coefficient indices where to evaluate the
  /// Jacobian of `G`
  ///
  /// @result Block-tensor with the Hessian with respect to the
  /// physical variables
  /// \f[
  ///     H_{\mathbf{x}}(u)
  ///        =
  ///     J_{\boldsymbol{\xi}}(G)^{-T}
  ///     \left(
  ///       H_\boldsymbol{\xi}(u)
  ///       -
  ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
  ///     \right)
  ///     J_{\boldsymbol{\xi}}(G)^{-1} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  ihess(const Geometry &G, const utils::TensorArray<BSplineCore::parDim_> &xi,
        const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
        const torch::Tensor &coeff_indices,
        const utils::TensorArray<Geometry::parDim()> &knot_indices_G,
        const torch::Tensor &coeff_indices_G) const {

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else {
      utils::BlockTensor<torch::Tensor, BSplineCore::parDim_, BSplineCore::parDim_, BSplineCore::geoDim_> hessu;

      auto hessG = G.template hess<memory_optimized>(xi, knot_indices_G,
                                                     coeff_indices_G);
      auto ijacG = ijac<memory_optimized>(G, xi, knot_indices, coeff_indices,
                                          knot_indices_G, coeff_indices_G);

      for (short_t component = 0; component < BSplineCore::geoDim_; ++component) {
          auto hess_component = hess<memory_optimized>(xi, knot_indices, coeff_indices).slice(component);

          for (short_t k = 0; k < hessG.slices(); ++k) {
              hess_component -= ijacG(component, k) * hessG.slice(k);
          }

          auto jacInv = G.template jac<memory_optimized>(xi, knot_indices_G, coeff_indices_G).ginv();
          auto hessu_component = jacInv.tr() * hess_component * jacInv;
          
          for (short_t i = 0; i < BSplineCore::parDim_; ++i)
            for (short_t j = 0; j < BSplineCore::parDim_; ++j)
              hessu.set(i, j, component, hessu_component(i, j));
      }

      return hessu;

    }
  }

  //  clang-format off
  /// @brief Returns a block-tensor with the Jacobian of the
  /// B-spline object in the points `xi` with respect to the
  /// parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Jacobian
  ///
  /// @result Block-tensor with the Jacobian with respect to the
  /// parametric variables
  /// \f[
  ///     J_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \begin{bmatrix}
  ///           \frac{\partial u_0}{\partial \xi_0}&
  ///           \frac{\partial u_0}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_0}{\partial \xi_{d_\text{par}}} \\
  ///           \frac{\partial u_1}{\partial \xi_0}&
  ///           \frac{\partial u_1}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_1}{\partial \xi_{d_\text{par}}} \\
  ///           \vdots& \vdots & \ddots & \vdots \\
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_0}&
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  ///     \end{bmatrix}
  /// \f]
  //  clang-format on
  ///
  /// @{
  template <bool memory_optimized = false>
  inline auto jac(const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return jac<memory_optimized>(utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false>
  inline auto jac(const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return jac<memory_optimized>(xi, BSplineCore::find_knot_indices(xi));
  }
  /// @}

  //  clang-format off
  /// @brief Returns a block-tensor with the Jacobian of the
  /// B-spline object in the points `xi` with respect to the
  /// parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Jacobian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Jacobian
  ///
  /// @result Block-tensor with the Jacobian with respect to the
  /// parametric variables
  /// \f[
  ///     J_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \begin{bmatrix}
  ///           \frac{\partial u_0}{\partial \xi_0}&
  ///           \frac{\partial u_0}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_0}{\partial \xi_{d_\text{par}}} \\
  ///           \frac{\partial u_1}{\partial \xi_0}&
  ///           \frac{\partial u_1}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_1}{\partial \xi_{d_\text{par}}} \\
  ///           \vdots& \vdots & \ddots & \vdots \\
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_0}&
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  ///     \end{bmatrix}
  /// \f]
  //  clang-format on
  template <bool memory_optimized = false>
  inline auto
  jac(const utils::TensorArray<BSplineCore::parDim_> &xi,
      const utils::TensorArray<BSplineCore::parDim_> &knot_indices) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return jac<memory_optimized>(
          xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices));
  }

  //  clang-format off
  /// @brief Returns a block-tensor with the Jacobian of the
  /// B-spline object in the points `xi` with respect to the
  /// parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Jacobian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Jacobian
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// Jacobian
  ///
  /// @result Block-tensor with the Jacobian with respect to the
  /// parametric variables
  /// \f[
  ///     J_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \begin{bmatrix}
  ///           \frac{\partial u_0}{\partial \xi_0}&
  ///           \frac{\partial u_0}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_0}{\partial \xi_{d_\text{par}}} \\
  ///           \frac{\partial u_1}{\partial \xi_0}&
  ///           \frac{\partial u_1}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_1}{\partial \xi_{d_\text{par}}} \\
  ///           \vdots& \vdots & \ddots & \vdots \\
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_0}&
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_1}&
  ///           \dots&
  ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
  ///     \end{bmatrix}
  /// \f]
  ///
  /// @note Since the B-spline evaluation function computes the
  /// specified derivatives for all variables simultaneously we
  /// compute the transpose of the Jacobian and return its
  /// tranposed, hence, the Jacobian.
  //  clang-format on
  template <bool memory_optimized = false>
  inline auto jac(const utils::TensorArray<BSplineCore::parDim_> &xi,
                  const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
                  const torch::Tensor &coeff_indices) const {

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};

    else {
      // Check compatibility of arguments
      for (short_t i = 0; i < BSplineCore::parDim_; ++i)
        assert(xi[i].sizes() == knot_indices[i].sizes());
      for (short_t i = 1; i < BSplineCore::parDim_; ++i)
        assert(xi[0].sizes() == xi[i].sizes());

      // Lambda expression to evaluate the jacobian
      auto jac_ = [&, this]<std::size_t... Is>(std::index_sequence<Is...>) {
        return utils::BlockTensor<torch::Tensor, BSplineCore::parDim_,
                                  BSplineCore::geoDim_>{
            BSplineCore::template eval<(deriv)utils::integer_pow<10, Is>::value,
                                       memory_optimized>(xi, knot_indices,
                                                         coeff_indices)...}
            .tr();
      };

      return jac_(std::make_index_sequence<BSplineCore::parDim_>{});
    }
  }

  /// @brief Returns a block-tensor with the Jacobian of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Jacobian
  ///
  /// @result Block-tensor with the Jacobian with respect to the
  /// parametric variables
  /// \f[
  ///     J_{\mathbf{x}}(u)
  ///        =
  ///     J_{\boldsymbol{\xi}}(u) \, J_{\boldsymbol{\xi}}(G)^{-T} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  ///
  /// @{
  template <bool memory_optimized = false, typename Geometry>
  auto ijac(const Geometry &G, const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ijac<memory_optimized, Geometry>(G, utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false, typename Geometry>
  inline auto ijac(const Geometry &G,
                   const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ijac<memory_optimized, Geometry>(
          G, xi, BSplineCore::find_knot_indices(xi), G.find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns a block-tensor with the Jacobian of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Jacobian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Jacobian
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate the Jacobian of
  /// `G`
  ///
  /// @result Block-tensor with the Jacobian with respect to the
  /// physical variables
  /// \f[
  ///     J_{\mathbf{x}}(u)
  ///        =
  ///     J_{\boldsymbol{\xi}}(u) \, J_{\boldsymbol{\xi}}(G)^{-T} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  ijac(const Geometry G, const utils::TensorArray<BSplineCore::parDim_> &xi,
       const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
       const utils::TensorArray<Geometry::parDim()> &knot_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ijac<memory_optimized, Geometry>(
          G, xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices),
          knot_indices_G,
          G.template find_coeff_indices<memory_optimized>(knot_indices_G));
  }

  /// @brief Returns a block-tensor with the Jacobian of the
  /// B-spline object in the points `xi` with respect to the
  /// physical variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Jacobian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Jacobain
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate the Jacobian of
  /// `G`
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// Jacobian
  ///
  /// @param[in] coeff_indices_G Coefficient indices where to evaluate the
  /// Jacobian of `G`
  ///
  /// @result Block-tensor with the Jacobian with respect to the
  /// physical variables
  /// \f[
  ///     J_{\mathbf{x}}(u)
  ///        =
  ///     J_{\boldsymbol{\xi}}(u) \, J_{\boldsymbol{\xi}}(G)^{-T} ,
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto ijac(const Geometry &G,
                   const utils::TensorArray<BSplineCore::parDim_> &xi,
                   const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
                   const torch::Tensor &coeff_indices,
                   const utils::TensorArray<Geometry::parDim()> &knot_indices_G,
                   const torch::Tensor &coeff_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return jac<memory_optimized>(xi, knot_indices, coeff_indices) *
             G.template jac<memory_optimized>(xi, knot_indices_G,
                                              coeff_indices_G)
                 .ginv();
  }

  //  clang-format off
  /// @brief Returns a block-tensor with the Laplacian of the B-spline
  /// object in the points `xi` with respect to the parametric variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Laplacian
  ///
  /// @result Block-tensor with the Laplacian with respect to the
  /// parametric variables `xi`
  /// \f[
  ///     L_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \sum_{i,j=0\atop|i+j|=2}^2
  ///     \frac{\partial^2 u}{\partial \xi_i\partial \xi_{j}}
  /// \f]
  ///
  /// @note If the B-spline object has geometric dimension larger
  /// then one then all Laplacians are returned as a vector.
  //  clang-format on
  ///
  /// @{
  template <bool memory_optimized = false>
  inline auto lapl(const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return lapl<memory_optimized>(utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false>
  inline auto lapl(const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return lapl<memory_optimized>(xi, BSplineCore::find_knot_indices(xi));
  }
  /// @}

  //  clang-format off
  /// @brief Returns a block-tensor with the Laplacian of the B-spline
  /// object in the points `xi` with respect to the parametric
  /// variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Laplacian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Laplacian
  ///
  /// @result Block-tensor with the Laplacian with respect to
  /// the parametric variables
  /// \f[
  ///     L_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \sum_{i,j=0\atop|i+j|=2}^2
  ///     \frac{\partial^2 u}{\partial \xi_i\partial \xi_{j}}
  /// \f]
  ///
  /// @note If the B-spline object has geometric dimension larger
  /// then one then all Laplacian matrices are returned as a vector.
  //  clang-format on
  template <bool memory_optimized = false>
  inline auto
  lapl(const utils::TensorArray<BSplineCore::parDim_> &xi,
       const utils::TensorArray<BSplineCore::parDim_> &knot_indices) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return lapl<memory_optimized>(
          xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices));
  }

  //  clang-format off
  /// @brief Returns a block-tensor with the Laplacian of the B-spline
  /// object in the points `xi` with respect to the parametric
  /// variables
  ///
  /// @param[in] xi Point(s) where to evaluate the Laplacian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Laplacian
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// Laplacian
  ///
  /// @result Block-tensor with the Laplacian with respect to the
  /// parametric variables
  /// \f[
  ///     L_{\boldsymbol{\xi}}(u)
  ///        =
  ///     \sum_{i,j=0\atop|i+j|=2}^2
  ///     \frac{\partial^2 u}{\partial \xi_i\partial \xi_{j}}
  /// \f]
  ///
  /// @note If the B-spline object has geometric dimension larger
  /// then one then all Laplacians are returned as a vector.
  //  clang-format on
  template <bool memory_optimized = false>
  inline auto lapl(const utils::TensorArray<BSplineCore::parDim_> &xi,
                   const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
                   const torch::Tensor &coeff_indices) const {

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};

    else {
      // Check compatibility of arguments
      for (short_t i = 0; i < BSplineCore::parDim_; ++i)
        assert(xi[i].sizes() == knot_indices[i].sizes());
      for (short_t i = 1; i < BSplineCore::parDim_; ++i)
        assert(xi[0].sizes() == xi[i].sizes());

      // Lambda expression to evaluate the laplacian
      auto lapl_ = [&, this]<std::size_t... Is>(std::index_sequence<Is...>) {
        return utils::BlockTensor<torch::Tensor, 1, 1, BSplineCore::geoDim_>{
            (BSplineCore::template eval<
                 (deriv)utils::integer_pow<10, Is>::value ^ 2,
                 memory_optimized>(xi, knot_indices, coeff_indices) +
             ...)}
            .reorder_ikj();
      };

      return lapl_(std::make_index_sequence<BSplineCore::parDim_>{});
    }
  }

  /// @brief Returns a block-tensor with the Laplacian of the B-spline
  /// object in the points `xi` with respect to the physical
  /// variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Laplacian
  ///
  /// @result Block-tensor with the Laplacian with respect to the
  /// parametric variables
  /// \f[
  ///     L_{\mathbf{x}}(u)
  ///        =
  ///     \text{trace} \left(
  ///     J_{\boldsymbol{\xi}}(G)^{-T}
  ///     \left(
  ///       H_\boldsymbol{\xi}(u)
  ///       -
  ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
  ///     \right)
  ///     J_{\boldsymbol{\xi}}(G)^{-1} \right),
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  ///
  /// @{
  template <bool memory_optimized = false, typename Geometry>
  auto ilapl(const Geometry &G, const torch::Tensor &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ilapl<memory_optimized, Geometry>(G, utils::TensorArray1({xi}));
  }

  template <bool memory_optimized = false, typename Geometry>
  inline auto ilapl(const Geometry &G,
                    const utils::TensorArray<BSplineCore::parDim_> &xi) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ilapl<memory_optimized, Geometry>(
          G, xi, BSplineCore::find_knot_indices(xi), G.find_knot_indices(xi));
  }
  /// @}

  /// @brief Returns a block-tensor with the Laplacian of the B-spline
  /// object in the points `xi` with respect to the physical
  /// variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Laplacian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Laplacian
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate Jacobian of `G`
  ///
  /// @result Block-tensor with the Laplacian with respect to the
  /// physical variables
  /// \f[
  ///     L_{\mathbf{x}}(u)
  ///        =
  ///     \text{trace} \left(
  ///     J_{\boldsymbol{\xi}}(G)^{-T}
  ///     \left(
  ///       H_\boldsymbol{\xi}(u)
  ///       -
  ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
  ///     \right)
  ///     J_{\boldsymbol{\xi}}(G)^{-1} \right),
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  ilapl(const Geometry G, const utils::TensorArray<BSplineCore::parDim_> &xi,
        const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
        const utils::TensorArray<Geometry::parDim()> &knot_indices_G) const {
    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};
    else
      return ilapl<memory_optimized, Geometry>(
          G, xi, knot_indices,
          BSplineCore::template find_coeff_indices<memory_optimized>(
              knot_indices),
          knot_indices_G,
          G.template find_coeff_indices<memory_optimized>(knot_indices_G));
  }

  /// @brief Returns a block-tensor with the Laplacian of the B-spline
  /// object in the points `xi` with respect to the physical
  /// variables
  ///
  /// @tparam Geometry Type of the geometry B-spline object
  ///
  /// @param[in] G B-spline geometry object
  ///
  /// @param[in] xi Point(s) where to evaluate the Laplacian
  ///
  /// @param[in] knot_indices Knot indices where to evaluate the Laplacian
  ///
  /// @param[in] knot_indices_G Knot indices where to evaluate the Jacobian of
  /// `G`
  ///
  /// @param[in] coeff_indices Coefficient indices where to evaluate the
  /// Laplacian
  ///
  /// @param[in] coeff_indices_G Coefficient indices where to evaluate the
  /// Jacobian of `G`
  ///
  /// @result Block-tensor with the Laplacian with respect to the
  /// physical variables
  /// \f[
  ///     L_{\mathbf{x}}(u)
  ///        =
  ///     \text{trace} \left(
  ///     J_{\boldsymbol{\xi}}(G)^{-T}
  ///     \left(
  ///       H_\boldsymbol{\xi}(u)
  ///       -
  ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
  ///     \right)
  ///     J_{\boldsymbol{\xi}}(G)^{-1} \right),
  ///     \quad
  ///     \mathbf{x} = G(\boldsymbol{\xi})
  /// \f]
  template <bool memory_optimized = false, typename Geometry>
  inline auto
  ilapl(const Geometry &G, const utils::TensorArray<BSplineCore::parDim_> &xi,
        const utils::TensorArray<BSplineCore::parDim_> &knot_indices,
        const torch::Tensor &coeff_indices,
        const utils::TensorArray<Geometry::parDim()> &knot_indices_G,
        const torch::Tensor &coeff_indices_G) const {

    if constexpr (BSplineCore::parDim_ == 0)
      return utils::BlockTensor<torch::Tensor, 1, 1>{
          torch::zeros_like(BSplineCore::coeffs_[0])};

    else {
      auto hessu =
          hess<memory_optimized>(xi, knot_indices, coeff_indices).slice(0);

      {
        auto igradG =
            igrad<memory_optimized>(G, xi, knot_indices, coeff_indices,
                                    knot_indices_G, coeff_indices_G);
        auto hessG = G.template hess<memory_optimized>(xi, knot_indices_G,
                                                       coeff_indices_G);
        assert(igradG.cols() == hessG.slices());
        for (short_t k = 0; k < hessG.slices(); ++k)
          hessu -= igradG(0, k) * hessG.slice(k);
      }

      auto jacInv =
          G.template jac<memory_optimized>(xi, knot_indices_G, coeff_indices_G)
              .ginv();

      return (jacInv.tr() * hessu * jacInv).trace();
    }
  }

  /// Plots the B-spline object
  ///
  /// @param[in] json JSON configuration
  ///
  /// @result Plot of the B-spline object
#ifdef IGANET_WITH_MATPLOT
  template <typename Backend = matplot::backend::gnuplot>
#else
  template <typename Backend = void>
#endif
  inline auto plot(const nlohmann::json &json = {}) const {
    return plot<Backend>(*this, json);
  }

  /// Plots the B-spline object together with a set of sampling points
  ///
  /// @param[in] xi Sampling points
  ///
  /// @param[in] json JSON configuration
  ///
  /// @result Plot of the B-spline object
#ifdef IGANET_WITH_MATPLOT
  template <typename Backend = matplot::backend::gnuplot>
#else
  template <typename Backend = void>
#endif
  inline auto plot(const utils::TensorArray<BSplineCore::parDim_> &xi,
                   const nlohmann::json &json = {}) const {

    return plot<Backend>(*this, xi, json);
  }

  /// Plots the B-spline object together with a set of sampling points
  ///
  /// @param[in] xi Vector of sampling points
  ///
  /// @param[in] json JSON configuration
  ///
  /// @result Plot of the B-spline object
#ifdef IGANET_WITH_MATPLOT
  template <typename Backend = matplot::backend::gnuplot>
#else
  template <typename Backend = void>
#endif
  inline auto plot(
      const std::initializer_list<utils::TensorArray<BSplineCore::parDim_>> &xi,
      const nlohmann::json &json = {}) const {

    return plot<Backend>(*this, xi, json);
  }

  /// Plots the B-spline object colored by another B-spline object
  ///
  /// @param[in] color B-spline object representing the color
  ///
  /// @param[in] json JSON configuration
  ///
  /// @result Plot of the B-spline object
#ifdef IGANET_WITH_MATPLOT
  template <typename Backend = matplot::backend::gnuplot,
            typename BSplineCoreColor>
#else
  template <typename Backend = void, typename BSplineCoreColor>
#endif
  inline auto plot(const BSplineCommon<BSplineCoreColor> &color,
                   const nlohmann::json &json = {}) const {
#ifdef IGANET_WITH_MATPLOT
    static_assert(BSplineCore::parDim() == BSplineCoreColor::parDim(),
                  "Parametric dimensions must match");

    if ((void *)this != (void *)&color && BSplineCoreColor::geoDim() > 1)
      throw std::runtime_error("BSpline for coloring must have geoDim=1");

    if constexpr (BSplineCore::parDim() == 1 && BSplineCore::geoDim() == 1) {

      //
      // mapping: [0,1] -> R^1
      //

      int64_t res0 = BSplineCore::ncoeffs(0);
      if (json.contains("res0"))
        res0 = json["res0"].get<int64_t>();

      // Create figure with specified backend
      auto f = matplot::figure<Backend>(true);
      auto ax = f->current_axes();

      // Create line
      auto Coords =
          BSplineCore::eval(torch::linspace(0, 1, res0, BSplineCore::options_));
#ifdef __clang__
      auto Coords_cpu =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords(0), torch::kCPU);
      auto XAccessor = std::get<1>(Coords_cpu);
#else
      auto [Coords_cpu, XAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords(0), torch::kCPU);
#endif

      matplot::vector_1d Xfine(res0, 0.0);
      matplot::vector_1d Yfine(res0, 0.0);

#pragma omp parallel for simd
      for (int64_t i = 0; i < res0; ++i)
        Xfine[i] = XAccessor[i];

      // Plot (colored) line
      if ((void *)this != (void *)&color) {
        if constexpr (BSplineCoreColor::geoDim_ == 1) {

          // Create colors
          auto Color =
              color.eval(torch::linspace(0, 1, res0, BSplineCore::options_));
#ifdef __clang__
          auto Color_cpu =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       1>(Color(0), torch::kCPU);
          auto CAccessor = std::get<1>(Color_cpu);
#else
          auto [Color_cpu, CAccessor] =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       1>(Color(0), torch::kCPU);
#endif

          matplot::vector_1d Cfine(res0, 0.0);

#pragma omp parallel for simd
          for (int64_t i = 0; i < res0; ++i)
            Cfine[i] = CAccessor[i];

          auto Cmin = *std::min_element(Cfine.begin(), Cfine.end());
          auto Cmax = *std::max_element(Cfine.begin(), Cfine.end());

          auto Cmap = matplot::colormap();

          auto a = Cmap.size() / (Cmax - Cmin);
          auto b = -a * Cmin;

          // Plot colored line
          ax->hold(matplot::on);
          for (std::size_t i = 0; i < Xfine.size() - 1; ++i)
            ax->plot({Xfine[i], Xfine[i + 1]}, {Yfine[i], Yfine[i + 1]})
                ->line_width(2)
                .color({Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][0],
                        Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][1],
                        Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][2]});
          ax->hold(matplot::off);
          matplot::colorbar(ax);
        } else
          throw std::runtime_error("BSpline for coloring must have geoDim=1");
      } else {
        // Plot unicolor line
        ax->plot(Xfine, Yfine, "b-")->line_width(2);
      }

      bool cnet = false;
      if (json.contains("cnet"))
        cnet = json["cnet"].get<bool>();

      if (cnet) {
        // Create control net
#ifdef __clang__
        auto coeffs_cpu =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(0), torch::kCPU);
        auto xAccessor = std::get<1>(coeffs_cpu);
#else
        auto [coeffs_cpu, xAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(0), torch::kCPU);
#endif
        matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
        matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);

#pragma omp parallel for simd
        for (int64_t i = 0; i < BSplineCore::ncoeffs(0); ++i) {
          X[i] = xAccessor[i];
        }

        // Plot control net
        ax->hold(matplot::on);
        ax->plot(X, Y, ".k-")->line_width(1);
        ax->hold(matplot::off);
      }

      // Title
      if (json.contains("title"))
        ax->title(json["title"].get<std::string>());
      else
        ax->title("BSpline: [0,1] -> R");

      // X-axis label
      if (json.contains("xlabel"))
        ax->xlabel(json["xlabel"].get<std::string>());
      else
        ax->xlabel("x");

      // Y-axis label
      if (json.contains("ylabel"))
        ax->ylabel(json["ylabel"].get<std::string>());
      else
        ax->ylabel("y");

      return f;
    }

    else if constexpr (BSplineCore::parDim_ == 1 && BSplineCore::geoDim_ == 2) {

      //
      // mapping: [0,1] -> R^2
      //

      int64_t res0 = BSplineCore::ncoeffs(0);
      if (json.contains("res0"))
        res0 = json["res0"].get<int64_t>();

      // Create figure with specified backend
      auto f = matplot::figure<Backend>(true);
      auto ax = f->current_axes();

      // Create curve
      auto Coords =
          BSplineCore::eval(torch::linspace(0, 1, res0, BSplineCore::options_));
#ifdef __clang__
      auto Coords_cpu =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords, torch::kCPU);
      auto XAccessor = std::get<1>(Coords_cpu)[0];
      auto YAccessor = std::get<1>(Coords_cpu)[1];
#else
      auto [Coords0_cpu, XAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords(0), torch::kCPU);
      auto [Coords1_cpu, YAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords(1), torch::kCPU);
#endif

      matplot::vector_1d Xfine(res0, 0.0);
      matplot::vector_1d Yfine(res0, 0.0);

#pragma omp parallel for simd
      for (int64_t i = 0; i < res0; ++i) {
        Xfine[i] = XAccessor[i];
        Yfine[i] = YAccessor[i];
      }

      // Plot (colored) curve
      if ((void *)this != (void *)&color) {
        if constexpr (BSplineCoreColor::geoDim() == 1) {

          // Create colors
          auto Color =
              color.eval(torch::linspace(0, 1, res0, BSplineCore::options_));
#ifdef __clang__
          auto Color_cpu =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       1>(Color(0), torch::kCPU);
          auto CAccessor = std::get<1>(Color_cpu);
#else
          auto [Color_cpu, CAccessor] =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       1>(Color(0), torch::kCPU);
#endif

          matplot::vector_1d Cfine(res0, 0.0);

#pragma omp parallel for simd
          for (int64_t i = 0; i < res0; ++i) {
            Cfine[i] = CAccessor[i];
          }

          auto Cmin = *std::min_element(Cfine.begin(), Cfine.end());
          auto Cmax = *std::max_element(Cfine.begin(), Cfine.end());

          auto Cmap = matplot::colormap();

          auto a = Cmap.size() / (Cmax - Cmin);
          auto b = -a * Cmin;

          // Plot colored curve
          ax->hold(matplot::on);
          for (std::size_t i = 0; i < Xfine.size() - 1; ++i)
            ax->plot({Xfine[i], Xfine[i + 1]}, {Yfine[i], Yfine[i + 1]})
                ->line_width(2)
                .color({Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][0],
                        Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][1],
                        Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][2]});
          ax->hold(matplot::off);
          matplot::colorbar(ax);
        } else
          throw std::runtime_error("BSpline for coloring must have geoDim=1");
      } else {
        // Plot unicolor curve
        ax->plot(Xfine, Yfine, "b-")->line_width(2);
      }

      bool cnet = false;
      if (json.contains("cnet"))
        cnet = json["cnet"].get<bool>();

      if (cnet) {
        // Create control net
#ifdef __clang__
        auto coeffs_cpu =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(), torch::kCPU);
        auto xAccessor = std::get<1>(coeffs_cpu)[0];
        auto yAccessor = std::get<1>(coeffs_cpu)[1];
#else
        auto [coeffs0_cpu, xAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(0), torch::kCPU);
        auto [coeffs1_cpu, yAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(1), torch::kCPU);
#endif

        matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
        matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);

#pragma omp parallel for simd
        for (int64_t i = 0; i < BSplineCore::ncoeffs(0); ++i) {
          X[i] = xAccessor[i];
          Y[i] = yAccessor[i];
        }

        // Plot control net
        ax->hold(matplot::on);
        ax->plot(X, Y, ".k-")->line_width(1);
        ax->hold(matplot::off);
      }

      // Title
      if (json.contains("title"))
        ax->title(json["title"].get<std::string>());
      else
        ax->title("BSpline: [0,1] -> R^2");

      // X-axis label
      if (json.contains("xlabel"))
        ax->xlabel(json["xlabel"].get<std::string>());
      else
        ax->xlabel("x");

      // Y-axis label
      if (json.contains("ylabel"))
        ax->ylabel(json["ylabel"].get<std::string>());
      else
        ax->ylabel("y");

      return f;
    }

    else if constexpr (BSplineCore::parDim() == 1 &&
                       BSplineCore::geoDim() == 3) {

      //
      // mapping: [0,1] -> R^3
      //

      int64_t res0 = BSplineCore::ncoeffs(0);
      if (json.contains("res0"))
        res0 = json["res0"].get<int64_t>();

      // Create figure with specified backend
      auto f = matplot::figure<Backend>(true);
      auto ax = f->current_axes();

      auto Coords =
          BSplineCore::eval(torch::linspace(0, 1, res0, BSplineCore::options_));
#ifdef __clang__
      auto Coords_cpu =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords, torch::kCPU);
      auto XAccessor = std::get<1>(Coords_cpu)[0];
      auto YAccessor = std::get<1>(Coords_cpu)[1];
      auto ZAccessor = std::get<1>(Coords_cpu)[2];
#else
      auto [Coords0_cpu, XAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords(0), torch::kCPU);
      auto [Coords1_cpu, YAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords(1), torch::kCPU);
      auto [Coords2_cpu, ZAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              Coords(2), torch::kCPU);
#endif

      // Create curve
      matplot::vector_1d Xfine(res0, 0.0);
      matplot::vector_1d Yfine(res0, 0.0);
      matplot::vector_1d Zfine(res0, 0.0);

#pragma omp parallel for simd
      for (int64_t i = 0; i < res0; ++i) {
        Xfine[i] = XAccessor[i];
        Yfine[i] = YAccessor[i];
        Zfine[i] = ZAccessor[i];
      }

      // Plot (colored) curve
      if ((void *)this != (void *)&color) {
        if constexpr (BSplineCoreColor::geoDim() == 1) {

          auto Color =
              color.eval(torch::linspace(0, 1, res0, BSplineCore::options_));
#ifdef __clang__
          auto Color_cpu =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       1>(Color(0), torch::kCPU);
          auto CAccessor = std::get<1>(Color_cpu);
#else
          auto [Color_cpu, CAccessor] =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       1>(Color(0), torch::kCPU);
#endif

          // Create colors
          matplot::vector_1d Cfine(matplot::vector_1d(res0, 0.0));

#pragma omp parallel for simd
          for (int64_t i = 0; i < res0; ++i) {
            Cfine[i] = CAccessor[i];
          }

          auto Cmin = *std::min_element(Cfine.begin(), Cfine.end());
          auto Cmax = *std::max_element(Cfine.begin(), Cfine.end());

          auto Cmap = matplot::colormap();

          auto a = Cmap.size() / (Cmax - Cmin);
          auto b = -a * Cmin;

          // Plot colored line
          ax->hold(matplot::on);
          for (std::size_t i = 0; i < Xfine.size() - 1; ++i)
            ax->plot3({Xfine[i], Xfine[i + 1]}, {Yfine[i], Yfine[i + 1]},
                      {Zfine[i], Zfine[i + 1]})
                ->line_width(2)
                .color({Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][0],
                        Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][1],
                        Cmap[a * (Cfine[i] + Cfine[i + 1]) / 2.0 - b][2]});
          ax->hold(matplot::off);
          matplot::colorbar(ax);
        } else
          throw std::runtime_error("BSpline for coloring must have geoDim=1");
      } else {
        // Plot curve
        ax->plot3(Xfine, Yfine, Zfine, "b-")->line_width(2);
      }

      bool cnet = false;
      if (json.contains("cnet"))
        cnet = json["cnet"].get<bool>();

      if (cnet) {
        // Create control net
#ifdef __clang__
        auto coeffs_cpu =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(), torch::kCPU);
        auto xAccessor = std::get<1>(coeffs_cpu)[0];
        auto yAccessor = std::get<1>(coeffs_cpu)[1];
        auto zAccessor = std::get<1>(coeffs_cpu)[2];
#else
        auto [coeffs0_cpu, xAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(0), torch::kCPU);
        auto [coeffs1_cpu, yAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(1), torch::kCPU);
        auto [coeffs2_cpu, zAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(2), torch::kCPU);
#endif

        matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
        matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);
        matplot::vector_1d Z(BSplineCore::ncoeffs(0), 0.0);

#pragma omp parallel for simd
        for (int64_t i = 0; i < BSplineCore::ncoeffs(0); ++i) {
          X[i] = xAccessor[i];
          Y[i] = yAccessor[i];
          Z[i] = zAccessor[i];
        }

        // Plot control net
        ax->hold(matplot::on);
        ax->plot3(X, Y, Z, ".k-")->line_width(1);
        ax->hold(matplot::off);
      }

      // Title
      if (json.contains("title"))
        ax->title(json["title"].get<std::string>());
      else
        ax->title("BSpline: [0,1] -> R^3");

      // X-axis label
      if (json.contains("xlabel"))
        ax->xlabel(json["xlabel"].get<std::string>());
      else
        ax->xlabel("x");

      // Y-axis label
      if (json.contains("ylabel"))
        ax->ylabel(json["ylabel"].get<std::string>());
      else
        ax->ylabel("y");

      // Z-axis label
      if (json.contains("zlabel"))
        ax->zlabel(json["zlabel"].get<std::string>());
      else
        ax->zlabel("z");

      return f;
    }

    else if constexpr (BSplineCore::parDim() == 2 &&
                       BSplineCore::geoDim() == 2) {

      //
      // mapping: [0,1]^2 -> R^2
      //

      int64_t res0 = BSplineCore::ncoeffs(0);
      int64_t res1 = BSplineCore::ncoeffs(1);
      if (json.contains("res0"))
        res0 = json["res0"].get<int64_t>();
      if (json.contains("res1"))
        res1 = json["res1"].get<int64_t>();

      // Create figure with specified backend
      auto f = matplot::figure<Backend>(true);
      auto ax = f->current_axes();

      // Create mesh
      utils::TensorArray<2> meshgrid = utils::to_array<2>(
          torch::meshgrid({torch::linspace(0, 1, res0, BSplineCore::options_),
                           torch::linspace(0, 1, res1, BSplineCore::options_)},
                          "xy"));
      auto Coords = BSplineCore::eval(meshgrid);
#ifdef __clang__
      auto Coords_cpu =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 2>(
              Coords, torch::kCPU);
      auto XAccessor = std::get<1>(Coords_cpu)[0];
      auto YAccessor = std::get<1>(Coords_cpu)[1];
#else
      auto [Coords0_cpu, XAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 2>(
              Coords(0), torch::kCPU);
      auto [Coords1_cpu, YAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 2>(
              Coords(1), torch::kCPU);
#endif

      matplot::vector_2d Xfine(res1, matplot::vector_1d(res0, 0.0));
      matplot::vector_2d Yfine(res1, matplot::vector_1d(res0, 0.0));
      matplot::vector_2d Zfine(res1, matplot::vector_1d(res0, 0.0));

#pragma omp parallel for simd collapse(2)
      for (int64_t i = 0; i < res0; ++i)
        for (int64_t j = 0; j < res1; ++j) {
          Xfine[j][i] = XAccessor[j][i];
          Yfine[j][i] = YAccessor[j][i];
        }

      // Plot (colored) mesh
      if ((void *)this != (void *)&color) {
        if constexpr (BSplineCoreColor::geoDim() == 1) {

          // Create colors
          auto Color = color.eval(meshgrid);
#ifdef __clang__
          auto Color_cpu =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       2>(Color, torch::kCPU);
          auto CAccessor = std::get<1>(Color_cpu)[0];
#else
          auto [Color0_cpu, CAccessor] =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       2>(Color(0), torch::kCPU);
#endif

          matplot::vector_2d Cfine(res1, matplot::vector_1d(res0, 0.0));

#pragma omp parallel for simd collapse(2)
          for (int64_t i = 0; i < res0; ++i)
            for (int64_t j = 0; j < res1; ++j)
              Cfine[j][i] = CAccessor[j][i];

          // Plot colored mesh
          matplot::view(2);
          ax->mesh(Xfine, Yfine, Cfine)->hidden_3d(false);
          matplot::colorbar(ax);
        } else
          throw std::runtime_error("BSpline for coloring must have geoDim=1");
      } else {
        // Plot unicolor mesh
        matplot::view(2);
        matplot::colormap(std::vector<std::vector<double>>{{ 0.0, 0.0, 1.0 }});
        ax->mesh(Xfine, Yfine, Zfine)->hidden_3d(false).line_width(2);
      }

      bool cnet = false;
      if (json.contains("cnet"))
        cnet = json["cnet"].get<bool>();

      if (cnet) {
        // Create control net
#ifdef __clang__
        auto coeffs_cpu =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(), torch::kCPU);
        auto xAccessor = std::get<1>(coeffs_cpu)[0];
        auto yAccessor = std::get<1>(coeffs_cpu)[1];
#else
        auto [coeffs0_cpu, xAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(0), torch::kCPU);
        auto [coeffs1_cpu, yAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(1), torch::kCPU);
#endif

        matplot::vector_2d X(BSplineCore::ncoeffs(1),
                             matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
        matplot::vector_2d Y(BSplineCore::ncoeffs(1),
                             matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
        matplot::vector_2d Z(BSplineCore::ncoeffs(1),
                             matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));

#pragma omp parallel for simd collapse(2)
        for (int64_t i = 0; i < BSplineCore::ncoeffs(0); ++i)
          for (int64_t j = 0; j < BSplineCore::ncoeffs(1); ++j) {
            X[j][i] = xAccessor[j * BSplineCore::ncoeffs(0) + i];
            Y[j][i] = yAccessor[j * BSplineCore::ncoeffs(0) + i];
          }

        // Plot control net
        ax->hold(matplot::on);
        ax->surf(X, Y, Z)
            ->palette_map_at_surface(true)
            .face_alpha(0)
            .line_width(1);
        for (std::size_t i = 0; i < X.size(); ++i)
          ax->scatter3(X[i], Y[i], Z[i], "k.");
        ax->hold(matplot::off);
      }

      // Title
      if (json.contains("title"))
        ax->title(json["title"].get<std::string>());
      else
        ax->title("BSpline: [0,1]^2 -> R^2");

      // X-axis label
      if (json.contains("xlabel"))
        ax->xlabel(json["xlabel"].get<std::string>());
      else
        ax->xlabel("x");

      // Y-axis label
      if (json.contains("ylabel"))
        ax->ylabel(json["ylabel"].get<std::string>());
      else
        ax->ylabel("y");

      // Z-axis label
      if (json.contains("zlabel"))
        ax->zlabel(json["zlabel"].get<std::string>());
      else
        ax->zlabel("z");

      return f;
    }

    else if constexpr (BSplineCore::parDim() == 2 &&
                       BSplineCore::geoDim() == 3) {

      ///
      // mapping: [0,1]^2 -> R^3
      ///

      int64_t res0 = BSplineCore::ncoeffs(0);
      int64_t res1 = BSplineCore::ncoeffs(1);
      if (json.contains("res0"))
        res0 = json["res0"].get<int64_t>();
      if (json.contains("res1"))
        res1 = json["res1"].get<int64_t>();

      // Create figure with specified backend
      auto f = matplot::figure<Backend>(true);
      auto ax = f->current_axes();

      // Create surface
      utils::TensorArray<2> meshgrid = utils::to_array<2>(
          torch::meshgrid({torch::linspace(0, 1, res0, BSplineCore::options_),
                           torch::linspace(0, 1, res1, BSplineCore::options_)},
                          "xy"));
      auto Coords = BSplineCore::eval(meshgrid);
#ifdef __clang__
      auto Coords_cpu =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 2>(
              Coords, torch::kCPU);
      auto XAccessor = std::get<1>(Coords_cpu)[0];
      auto YAccessor = std::get<1>(Coords_cpu)[1];
      auto ZAccessor = std::get<1>(Coords_cpu)[2];
#else
      auto [Coords0_cpu, XAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 2>(
              Coords(0), torch::kCPU);
      auto [Coords1_cpu, YAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 2>(
              Coords(1), torch::kCPU);
      auto [Coords2_cpu, ZAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 2>(
              Coords(2), torch::kCPU);
#endif

      matplot::vector_2d Xfine(res1, matplot::vector_1d(res0, 0.0));
      matplot::vector_2d Yfine(res1, matplot::vector_1d(res0, 0.0));
      matplot::vector_2d Zfine(res1, matplot::vector_1d(res0, 0.0));

#pragma omp parallel for simd collapse(2)
      for (int64_t i = 0; i < res0; ++i)
        for (int64_t j = 0; j < res1; ++j) {
          Xfine[j][i] = XAccessor[j][i];
          Yfine[j][i] = YAccessor[j][i];
          Zfine[j][i] = ZAccessor[j][i];
        }

      // Plot (colored) surface
      if ((void *)this != (void *)&color) {
        if constexpr (BSplineCoreColor::geoDim() == 1) {

          // Create colors
          auto Color = color.eval(meshgrid);
#ifdef __clang__
          auto Color_cpu =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       2>(Color, torch::kCPU);
          auto CAccessor = std::get<1>(Color_cpu)[0];
#else
          auto [Color_cpu, CAccessor] =
              utils::to_tensorAccessor<typename BSplineCoreColor::value_type,
                                       2>(Color(0), torch::kCPU);
#endif

          matplot::vector_2d Cfine(res1, matplot::vector_1d(res0, 0.0));

#pragma omp parallel for simd collapse(2)
          for (int64_t i = 0; i < res0; ++i)
            for (int64_t j = 0; j < res1; ++j) {
              Cfine[j][i] = CAccessor[j][i];
            }

          // Plot colored surface
          ax->mesh(Xfine, Yfine, Zfine, Cfine)->hidden_3d(false);
          matplot::colorbar(ax);
        } else
          throw std::runtime_error("BSpline for coloring must have geoDim=1");
      } else {
        // Plot unicolor surface
        matplot::colormap(std::vector<std::vector<double>>{{ 0.0, 0.0, 1.0 }});
        ax->mesh(Xfine, Yfine, Zfine)->hidden_3d(false).line_width(2);
      }

      bool cnet = false;
      if (json.contains("cnet"))
        cnet = json["cnet"].get<bool>();

      if (cnet) {
        // Create control net
#ifdef __clang__
        auto coeffs_cpu =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(), torch::kCPU);
        auto xAccessor = std::get<1>(coeffs_cpu)[0];
        auto yAccessor = std::get<1>(coeffs_cpu)[1];
        auto zAccessor = std::get<1>(coeffs_cpu)[2];
#else
        auto [coeffs0_cpu, xAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(0), torch::kCPU);
        auto [coeffs1_cpu, yAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(1), torch::kCPU);
        auto [coeffs2_cpu, zAccessor] =
            utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
                BSplineCore::coeffs(2), torch::kCPU);
#endif

        matplot::vector_2d X(BSplineCore::ncoeffs(1),
                             matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
        matplot::vector_2d Y(BSplineCore::ncoeffs(1),
                             matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
        matplot::vector_2d Z(BSplineCore::ncoeffs(1),
                             matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));

#pragma omp parallel for simd collapse(2)
        for (int64_t i = 0; i < BSplineCore::ncoeffs(0); ++i)
          for (int64_t j = 0; j < BSplineCore::ncoeffs(1); ++j) {
            X[j][i] = xAccessor[j * BSplineCore::ncoeffs(0) + i];
            Y[j][i] = yAccessor[j * BSplineCore::ncoeffs(0) + i];
            Z[j][i] = zAccessor[j * BSplineCore::ncoeffs(0) + i];
          }

        // Plot control net
        ax->hold(matplot::on);
        ax->surf(X, Y, Z)
            ->palette_map_at_surface(true)
            .face_alpha(0)
            .line_width(1);
        for (std::size_t i = 0; i < X.size(); ++i)
          ax->scatter3(X[i], Y[i], Z[i], "k.");
        ax->hold(matplot::off);
      }

      // Title
      if (json.contains("title"))
        ax->title(json["title"].get<std::string>());
      else
        ax->title("BSpline: [0,1]^2 -> R^3");

      // X-axis label
      if (json.contains("xlabel"))
        ax->xlabel(json["xlabel"].get<std::string>());
      else
        ax->xlabel("x");

      // Y-axis label
      if (json.contains("ylabel"))
        ax->ylabel(json["ylabel"].get<std::string>());
      else
        ax->ylabel("y");

      // Z-axis label
      if (json.contains("zlabel"))
        ax->zlabel(json["zlabel"].get<std::string>());
      else
        ax->zlabel("z");

      return f;
    }

    else
      throw std::runtime_error(
          "Unsupported combination of parametric/geometric dimensions");
#else
    throw std::runtime_error(
        "This functions must be compiled with -DIGANET_WITH_MATPLOT turned on");
#endif
  }

  /// Plots the B-spline object colored by another B-spline object
  /// together with a set of sampling points
  ///
  /// @param[in] color B-spline object representing the color
  ///
  /// @param[in] xi Sampling points
  ///
  /// @param[in] json JSON configuration
  ///
  /// @result Plot of the B-spline object
#ifdef IGANET_WITH_MATPLOT
  template <typename Backend = matplot::backend::gnuplot,
            typename BSplineCoreColor>
#else
  template <typename Backend = void, typename BSplineCoreColor>
#endif
  inline auto plot(const BSplineCommon<BSplineCoreColor> &color,
                   const utils::TensorArray<BSplineCore::parDim_> &xi,
                   const nlohmann::json &json = {}) const {

#ifdef IGANET_WITH_MATPLOT
    auto f = plot<Backend>(color, json);
    auto ax = f->current_axes();

#ifdef __clang__
    auto xi_cpu =
        utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
            xi, torch::kCPU);
    auto xiAccessor = std::get<1>(xi_cpu);
#else
    auto [xi_cpu, xiAccessor] =
        utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
            xi, torch::kCPU);
#endif

    if constexpr (BSplineCore::parDim_ == 1) {
      matplot::vector_1d X(xi[0].size(0), 0.0);
      matplot::vector_1d Y(xi[0].size(0), 0.0);

#pragma omp parallel for simd
      for (int64_t i = 0; i < xi[0].size(0); ++i) {
        X[i] = xiAccessor[0][i];
      }

      ax->hold(matplot::on);
      ax->scatter(X, Y, ".");
      ax->hold(matplot::off);
    } else if constexpr (BSplineCore::parDim_ == 2) {
      matplot::vector_1d X(xi[0].size(0), 0.0);
      matplot::vector_1d Y(xi[0].size(0), 0.0);
      matplot::vector_1d Z(xi[0].size(0), 0.0);

#pragma omp parallel for simd
      for (int64_t i = 0; i < xi[0].size(0); ++i) {
        X[i] = xiAccessor[0][i];
        Y[i] = xiAccessor[1][i];
      }

      ax->hold(matplot::on);
      ax->scatter3(X, Y, Z, ".");
      ax->hold(matplot::off);
    } else if constexpr (BSplineCore::parDim_ == 3) {
      matplot::vector_1d X(xi[0].size(0), 0.0);
      matplot::vector_1d Y(xi[0].size(0), 0.0);
      matplot::vector_1d Z(xi[0].size(0), 0.0);

#pragma omp parallel for simd
      for (int64_t i = 0; i < xi[0].size(0); ++i) {
        X[i] = xiAccessor[0][i];
        Y[i] = xiAccessor[1][i];
        Z[i] = xiAccessor[2][i];
      }

      ax->hold(matplot::on);
      ax->scatter3(X, Y, Z, ".");
      ax->hold(matplot::off);
    } else
      throw std::runtime_error("Invalid parametric dimension");

    return f;
#else
    throw std::runtime_error(
        "This functions must be compiled with -DIGANET_WITH_MATPLOT turned on");
#endif
  }

  /// Plots the B-spline object colored by another B-spline object
  /// together with a set of sampling points
  ///
  /// @param[in] color B-spline object representing the color
  ///
  /// @param[in] xi Vector of sampling points
  ///
  /// @param[in] json JSON configuration
  ///
  /// @result Plot of the B-spline object
#ifdef IGANET_WITH_MATPLOT
  template <typename Backend = matplot::backend::gnuplot,
            typename BSplineCoreColor>
#else
  template <typename Backend = void, typename BSplineCoreColor>
#endif
  inline auto plot(
      const BSplineCommon<BSplineCoreColor> &color,
      const std::initializer_list<utils::TensorArray<BSplineCore::parDim_>> &xi,
      const nlohmann::json &json = {}) const {

#ifdef IGANET_WITH_MATPLOT
    auto f = plot<Backend>(color, json);
    auto ax = f->current_axes();

    for (const auto &xi : xi) {
#ifdef __clang__
      auto xi_cpu =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              xi, torch::kCPU);
      auto xiAccessor = std::get<1>(xi_cpu);
#else
      auto [xi_cpu, xiAccessor] =
          utils::to_tensorAccessor<typename BSplineCoreColor::value_type, 1>(
              xi, torch::kCPU);
#endif

      if constexpr (BSplineCore::parDim_ == 1) {
        matplot::vector_1d X(xi[0].size(0), 0.0);
        matplot::vector_1d Y(xi[0].size(0), 0.0);

#pragma omp parallel for simd
        for (int64_t i = 0; i < xi[0].size(0); ++i) {
          X[i] = xiAccessor[0][i];
        }

        ax->hold(matplot::on);
        ax->scatter(X, Y, ".");
        ax->hold(matplot::off);
      } else if constexpr (BSplineCore::parDim_ == 2) {
        matplot::vector_1d X(xi[0].size(0), 0.0);
        matplot::vector_1d Y(xi[0].size(0), 0.0);
        matplot::vector_1d Z(xi[0].size(0), 0.0);

#pragma omp parallel for simd
        for (int64_t i = 0; i < xi[0].size(0); ++i) {
          X[i] = xiAccessor[0][i];
          Y[i] = xiAccessor[1][i];
        }

        ax->hold(matplot::on);
        ax->scatter3(X, Y, Z, ".");
        ax->hold(matplot::off);
      } else if constexpr (BSplineCore::parDim_ == 3) {
        matplot::vector_1d X(xi[0].size(0), 0.0);
        matplot::vector_1d Y(xi[0].size(0), 0.0);
        matplot::vector_1d Z(xi[0].size(0), 0.0);

#pragma omp parallel for simd
        for (int64_t i = 0; i < xi[0].size(0); ++i) {
          X[i] = xiAccessor[0][i];
          Y[i] = xiAccessor[1][i];
          Z[i] = xiAccessor[2][i];
        }

        ax->hold(matplot::on);
        ax->scatter3(X, Y, Z, ".");
        ax->hold(matplot::off);

      } else
        throw std::runtime_error("Invalid parametric dimension");
    }
    return f;
#else
    throw std::runtime_error(
        "This functions must be compiled with -DIGANET_WITH_MATPLOT turned on");
#endif
  }

  /// @brief Returns a string representation of the BSplineCommon object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\nparDim = " << BSplineCore::parDim()
       << ", geoDim = " << BSplineCore::geoDim() << ", degrees = ";

    for (short_t i = 0; i < BSplineCore::parDim() - 1; ++i)
      os << BSplineCore::degree(i) << "x";
    if (BSplineCore::parDim() > 0)
      os << BSplineCore::degree(BSplineCore::parDim() - 1);
    else
      os << 0;

    os << ", knots = ";
    for (short_t i = 0; i < BSplineCore::parDim() - 1; ++i)
      os << BSplineCore::nknots(i) << "x";
    if (BSplineCore::parDim() > 0)
      os << BSplineCore::nknots(BSplineCore::parDim() - 1);
    else
      os << 0;

    os << ", coeffs = ";
    for (short_t i = 0; i < BSplineCore::parDim() - 1; ++i)
      os << BSplineCore::ncoeffs(i) << "x";
    if (BSplineCore::parDim() > 0)
      os << BSplineCore::ncoeffs(BSplineCore::parDim() - 1);
    else
      os << 1;

    os << ", options = "
       << static_cast<torch::TensorOptions>(BSplineCore::options_);

    if (is_verbose(os)) {
      os << "\nknots [ ";
      for (const torch::Tensor &knots : BSplineCore::knots()) {
        os << (knots.is_view() ? "view/" : "owns/");
        os << (knots.is_contiguous() ? "cont " : "non-cont ");
      }
      if (BSplineCore::parDim() > 0)
        os << "] = " << BSplineCore::knots();
      else
        os << "] = {}";

      os << "\ncoeffs [ ";
      for (const torch::Tensor &coeffs : BSplineCore::coeffs()) {
        os << (coeffs.is_view() ? "view/" : "owns/");
        os << (coeffs.is_contiguous() ? "cont " : "non-cont ");
      }
      if (BSplineCore::ncumcoeffs() > 0)
        os << "] = " << BSplineCore::coeffs_view();
      else
        os << "] = {}";
    }

    os << "\n)";
  }

  /// @brief Returns a new B-spline object whose coefficients are the
  /// sum of that of two compatible B-spline objects
  ///
  /// @note This method does not check if the knot vectors of the two
  /// B-spline objects are compatible. It simply adds the two
  /// coefficients arrays and throws an error if their sizes do not
  /// match. Any compatibility checks must be performed outside.
  BSplineCommon operator+(const BSplineCommon &other) const {

    BSplineCommon result{*this};

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      result.coeffs(i) += other.coeffs(i);

    return result;
  }

  /// @brief Returns a new B-spline object whose coefficients are the
  /// difference of that of two compatible B-spline objects
  ///
  /// @note This method does not check if the knot vectors of the two
  /// B-spline objects are compatible. It simply subtracts the
  /// coefficients arrays of two B-spline objects from each other and
  /// throws an error if their sizes do not match. Any compatibility
  /// checks must be performed outside.
  BSplineCommon operator-(const BSplineCommon &other) const {

    BSplineCommon result{*this};

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      result.coeffs(i) -= other.coeffs(i);

    return result;
  }

  /// @brief Returns a new B-spline object whose coefficients are
  /// scaled by a scalar
  BSplineCommon operator*(typename BSplineCore::value_type s) const {

    BSplineCommon result{*this};

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      result.coeffs(i) *= s;

    return result;
  }

  /// @brief Returns a new B-spline object whose coefficients are
  /// scaled by a vector
  BSplineCommon operator*(
      std::array<typename BSplineCore::value_type, BSplineCore::geoDim()> v)
      const {

    BSplineCommon result{*this};

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      result.coeffs(i) *= v[i];

    return result;
  }

  /// @brief Returns a new B-spline object whose coefficients are
  /// scaled by a scalar
  BSplineCommon operator/(typename BSplineCore::value_type s) const {

    BSplineCommon result{*this};

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      result.coeffs(i) /= s;

    return result;
  }

  /// @brief Returns a new B-spline object whose coefficients are
  /// scaled by a vector
  BSplineCommon operator/(
      std::array<typename BSplineCore::value_type, BSplineCore::geoDim()> v)
      const {

    BSplineCommon result{*this};

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      result.coeffs(i) /= v[i];

    return result;
  }

  /// @brief Adds the coefficients of another B-spline object
  ///
  /// @note This method does not check if the knot vectors of the two
  /// B-spline objects are compatible. It simply adds the two
  /// coefficients arrays and throws an error if their sizes do not
  /// match. Any compatibility checks must be performed outside.
  BSplineCommon &operator+=(const BSplineCommon &other) {

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      BSplineCore::coeffs(i) += other.coeffs(i);

    return *this;
  }

  /// @brief Substracts the coefficients of another B-spline object
  ///
  /// @note This method does not check if the knot vectors of the two
  /// B-spline objects are compatible. It simply substracts the two
  /// coefficients arrays and throws an error if their sizes do not
  /// match. Any compatibility checks must be performed outside.
  BSplineCommon &operator-=(const BSplineCommon &other) {

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      BSplineCore::coeffs(i) -= other.coeffs(i);

    return *this;
  }

  /// @brief Scales the coefficients by a scalar
  BSplineCommon &operator*=(typename BSplineCore::value_type s) {

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      BSplineCore::coeffs(i) *= s;

    return *this;
  }

  /// @brief Scales the coefficients by a vector
  BSplineCommon &operator*=(
      std::array<typename BSplineCore::value_type, BSplineCore::geoDim()> v) {

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      BSplineCore::coeffs(i) *= v[i];

    return *this;
  }

  /// @brief Scales the coefficients by a scalar
  BSplineCommon &operator/=(typename BSplineCore::value_type s) {

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      BSplineCore::coeffs(i) /= s;

    return *this;
  }

  /// @brief Scales the coefficients by a vector
  BSplineCommon &operator/=(
      std::array<typename BSplineCore::value_type, BSplineCore::geoDim()> v) {

    for (short_t i = 0; i < BSplineCore::geoDim(); ++i)
      BSplineCore::coeffs(i) /= v[i];

    return *this;
  }
};

/// @brief Tensor-product uniform B-spline
template <typename real_t, short_t GeoDim, short_t... Degrees>
using UniformBSpline =
    BSplineCommon<UniformBSplineCore<real_t, GeoDim, Degrees...>>;

/// @brief Print (as string) a UniformBSpline object
template <typename real_t, short_t GeoDim, short_t... Degrees>
inline std::ostream &
operator<<(std::ostream &os,
           const UniformBSpline<real_t, GeoDim, Degrees...> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Tensor-product non-uniform B-spline
template <typename real_t, short_t GeoDim, short_t... Degrees>
using NonUniformBSpline =
    BSplineCommon<NonUniformBSplineCore<real_t, GeoDim, Degrees...>>;

/// @brief Print (as string) a UniformBSpline object
template <typename real_t, short_t GeoDim, short_t... Degrees>
inline std::ostream &
operator<<(std::ostream &os,
           const NonUniformBSpline<real_t, GeoDim, Degrees...> &obj) {
  obj.pretty_print(os);
  return os;
}
} // namespace iganet
