/**
   @file include/bspline.hpp

   @brief Multivariate B-splines

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <exception>
#include <filesystem>
#include <functional>

#include <core.hpp>
#include <utils.hpp>

#pragma once

namespace iganet {
  
  /// @brief Enumerator for specifying the initialization of B-spline coefficients
  enum class BSplineInit : short_t
    {
      zeros    = 0, /*!< set coefficient values to zero */
      ones     = 1, /*!< set coefficient values to one */
      linear   = 2, /*!< set coefficient values to \f$0,1,\dots \#\text{coeffs}-1\f$ */
      random   = 3, /*!< set coefficient values to random numbers */
      greville = 4  /*!< set coefficient values to the Greville abscissae */
    };

  /// @brief Enumerator for specifying the derivative of B-spline evaluation
  ///
  /// **Examples**
  ///
  /// * 3d Laplace operator `dx2+dy2+dz2`
  /// * 2d convection operator with time derivative dt+dx+dy`
  enum class BSplineDeriv : short_t
    {
      func   =    0, /*!< function value */
      dx     =    1, /*!< first derivative in x-direction */
      dx1    =    1, /*!< first derivative in x-direction */
      dx2    =    2, /*!< second derivative in x-direction */
      dx3    =    3, /*!< third derivative in x-direction */
      dx4    =    4, /*!< fourth derivative in x-direction */
      dy     =   10, /*!< first derivative in y-direction */
      dy1    =   10, /*!< first derivative in y-direction */
      dy2    =   20, /*!< second derivative in y-direction */
      dy3    =   30, /*!< third derivative in y-direction */
      dy4    =   40, /*!< fourth derivative in y-direction */
      dz     =  100, /*!< first derivative in z-direction */
      dz1    =  100, /*!< first derivative in z-direction */
      dz2    =  200, /*!< second derivative in z-direction */
      dz3    =  300, /*!< third derivative in z-direction */
      dz4    =  400, /*!< fourth derivative in z-direction */
      dt     = 1000, /*!< first derivative in t-direction */
      dt1    = 1000, /*!< first derivative in t-direction */
      dt2    = 2000, /*!< second derivative in t-direction */
      dt3    = 3000, /*!< third derivative in t-direction */
      dt4    = 4000  /*!< fourth derivative in t-direction */
    };

  /// @brief Tensor-product uniform B-spline (core functionality)
  ///
  /// This class implements the core functionality of all B-spline
  /// classes and serves as base class for non-uniform B-splines.
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
  /// \mathbf{f}(\boldsymbol{\xi}) = \sum_{I=1}^N B_I(\boldsymbol{\xi}) \mathbf{c}_I
  /// \f]
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
  
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  class UniformBSplineCore : public core<real_t> {
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

    /// @brief Array storing the knot vectors
    /// \f$\left(\left(t_{i_d}\right)_{i_d=1}^{n_d+p_d+1}\right)_{d=1}^{d_\text{par}}\f$
    std::array<torch::Tensor, parDim_> knots_;

    /// @brief Array storing the sizes of the knot vectors
    /// \f$\left(n_d+p_d+1\right)_{d=1}^{d_\text{par}}\f$
    std::array<int64_t, parDim_> nknots_;

    /// @brief Array storing the coefficients of the control net
    /// \f$\left(\mathbf{c}_{i_d}\right)_{i_d=1}^{n_d}\f$,
    /// \f$\mathbf{c}_{i_d}\in\mathbb{R}^{d_\text{geo}}\f$
    std::array<torch::Tensor, geoDim_> coeffs_;

    /// @brief Array storing the sizes of the coefficients of the
    /// control net \f$\left(n_d\right)_{d=1}^{d_\text{par}}\f$
    std::array<int64_t, parDim_> ncoeffs_;

    /// @brief LibTorch constants
    /// @{
    const torch::Tensor one_, zero_;
    /// @}

  public:
    /// @brief Default constructor
    UniformBSplineCore()
        : core<real_t>(),
          one_(torch::ones(1, core<real_t>::options_)),
          zero_(torch::zeros(1, core<real_t>::options_)) {}

    /// @brief Constructor for equidistant knot vectors
    UniformBSplineCore(const std::array<int64_t, parDim_>& ncoeffs,
                       BSplineInit init = BSplineInit::zeros)
        : core<real_t>(),
          ncoeffs_(ncoeffs),
          one_(torch::ones(1, core<real_t>::options_)),
          zero_(torch::zeros(1, core<real_t>::options_))
    {
      for (short_t i = 0; i < parDim_; ++i) {

        // Check that open knot vector can be created
        if (degrees_[i] > ncoeffs_[i] + 1)
          throw std::runtime_error("Not enough coefficients to create open knot vector");

        // Create open uniform knot vector
        std::vector<real_t> kv;

        for (int64_t j = 0; j < degrees_[i]; ++j)
          kv.push_back(static_cast<real_t>(0));

        for (int64_t j = 0; j < ncoeffs[i] - degrees_[i] + 1; ++j)
          kv.push_back(static_cast<real_t>(j / real_t(ncoeffs[i] - degrees_[i])));

        for (int64_t j = 0; j < degrees_[i]; ++j)
          kv.push_back(static_cast<real_t>(1));

        if (core<real_t>::options_.device() == torch::kCPU)
          knots_[i] = torch::from_blob(static_cast<real_t *>(kv.data()),
                                       kv.size(), core<real_t>::options_).clone();
        else
          knots_[i] = torch::from_blob(static_cast<real_t *>(kv.data()),
                                       kv.size(), core<real_t>::options_.device(torch::kCPU))
            .to(core<real_t>::options_.device());
        
        // Store the size of the knot vector
        nknots_[i] = knots_[i].size(0);
      }

      // Initialize coefficients
      init_coeffs(init);
    }

    /// @brief Returns the parametric dimension
    inline static constexpr short_t parDim()
    {
      return parDim_;
    }

    /// @brief Returns the geometric dimension
    inline static constexpr short_t geoDim()
    {
      return geoDim_;
    }

    /// @brief Returns a constant reference to the array of degrees
    inline static constexpr const std::array<short_t, parDim_>& degrees()
    {
      return degrees_;
    }

    /// @brief Returns a constant reference to the degree in the
    /// \f$i\f$-th dimension
    inline static constexpr const short_t& degree(short_t i)
    {
      assert(i >= 0 && i < parDim_);
      return degrees_[i];
    }

    /// @brief Returns a constant reference to the array of knot
    /// vectors
    inline const std::array<torch::Tensor, parDim_>& knots() const
    {
      return knots_;
    }

    /// @brief Returns a constant reference to the knot vector in the
    /// \f$i\f$-th dimension
    inline const torch::Tensor& knots(short_t i) const {
      assert(i >= 0 && i < parDim_);
      return knots_[i];
    }

    /// @brief Returns a non-constant reference to the array of knot
    /// vectors
    inline std::array<torch::Tensor, parDim_>& knots()
    {
      return knots_;
    }

    /// @brief Returns a non-constant reference to the knot vector in
    /// the \f$i\f$-th dimension
    inline torch::Tensor& knots(short_t i)
    {
      assert(i >= 0 && i < parDim_);
      return knots_[i];
    }

    /// @brief Returns a constant reference to the array of knot
    /// vector dimensions
    inline const std::array<int64_t, parDim_>& nknots() const
    {
      return nknots_;
    }

    /// @brief Returns the dimension of the knot vector in the
    /// \f$i\f$-th dimension
    inline int64_t nknots(short_t i) const
    {
      assert(i >= 0 && i < parDim_);
      return nknots_[i];
    }

    /// @brief Returns a constant reference to the array of
    /// coefficients.
    ///
    /// @note If `flatten=false`, this function returns an
    /// `std::array` of `torch::Tensor` objects with coefficients
    /// reshaped according to the dimensions of the knot vectors
    /// (return by value in this case)
    template<bool flatten = true>
    inline auto coeffs() const
      -> typename std::conditional<flatten,
                                   const std::array<torch::Tensor, geoDim_>& ,
                                   std::array<torch::Tensor, geoDim_>>::type
    {
      if constexpr (flatten || parDim_ == 0)
        return coeffs_;
      else {
        std::array<torch::Tensor, geoDim_> result;
        for (short_t i = 0; i < geoDim_; ++i)
          result[i] = coeffs_[i].view(ncoeffs_);
        return result;
      }
    }

    /// @brief Returns a constant reference to the coefficients in the
    /// \f$i\f$-th dimension.
    ///
    /// @note If `flatten=false`, this function returns a
    /// `torch::Tensor` with coefficients in the \f$i\f$-th dimension
    /// reshaped according to the dimensions of the knot vectors
    /// (return by value in this case)
    template<bool flatten = true>
    inline auto coeffs(short_t i) const
      -> typename std::conditional<flatten,
                                   const torch::Tensor& ,
                                   torch::Tensor>::type
    {
      assert(i >= 0 && i < geoDim_);
      if constexpr (flatten || parDim_ == 0)
        return coeffs_[i];
      else
        return coeffs_[i].view(ncoeffs_);
    }

    /// @brief Returns a non-constant reference to the array of
    /// coefficients
    inline std::array<torch::Tensor, geoDim_>& coeffs()
    {
      return coeffs_;
    }

    /// @brief Returns a non-constant reference to the coefficients in
    /// the \f$i\f$-th dimension
    inline torch::Tensor& coeffs(short_t i)
    {
      assert(i >= 0 && i < geoDim_);
      return coeffs_[i];
    }

    /// @brief Returns the total number of coefficients
    inline int64_t ncoeffs() const
    {
      int64_t s = 1;
      for (short_t i = 0; i < parDim_; ++i)
        s *= ncoeffs(i);
      return s;
    }

    /// @brief Returns the total number of coefficients in the
    /// \f$i\f$-th direction
    inline int64_t ncoeffs(short_t i) const
    {
      assert(i >= 0 && i < parDim_);
      return ncoeffs_[i];
    }

    /// @brief Returns the Greville abscissae
    inline std::array<torch::Tensor, geoDim_> greville() const
    {
      std::array<torch::Tensor, geoDim_> coeffs;

      // Fill coefficients with the tensor-product of Greville
      // abscissae values per univariate dimension
      for (short_t i = 0; i < geoDim_; ++i) {
        coeffs[i] = torch::ones(1, core<real_t>::options_);

        for (short_t j = 0; j < parDim_; ++j) {
          if (i == j) {
            auto greville_ = torch::zeros(ncoeffs_[j], core<real_t>::options_);
            auto greville = greville_.template accessor<real_t, 1>();
            auto knots = knots_[j].template accessor<real_t, 1>();
            for (int64_t k = 0; k < ncoeffs_[j]; ++k) {
              for (short_t l = 1; l <= degrees_[j]; ++l)
                greville[k] += knots[k + l];
              greville[k] /= degrees_[j];
            }
            coeffs[i] = coeffs[i].kron(greville_);
          } else
            coeffs[i] = coeffs[i].kron(torch::ones(ncoeffs_[j],
                                                   core<real_t>::options_));
        }
      }

      return coeffs;
    }

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
    ///      \boldsymbol{\xi} = \left(\xi_1, \dots, \xi_{d_\text{par}}\right)^\top \in
    ///      \bigotimes_{d=1}^{d_\text{par}}
    ///      [t_{i_d}, t_{i_d+1}).
    ///    \f]
    ///
    /// 2. Evaluate the vectors of univariate B-spline basis functions (or
    ///    their derivatives) that are non-zero at \f$\boldsymbol{\xi}\f$
    ///
    ///    \f[
    ///      D^{r_d}\mathbf{B}_d =
    ///      \left( D^{r_d} B_{i_d-p_d,p_d}, \dots, D^{r_d} B_{i_d,p_d} \right)^\top,
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
    ///    \left( \bigotimes_{d=1}^{d_\text{par}} D^{r_d}\mathbf{B}_d \right) \cdot \mathbf{c}_\mathcal{J},
    ///    \f]
    ///
    ///    where \f$\mathcal{J}\f$ is the subset of global indices
    ///    that belong to the coefficients
    ///
    ///    \f[
    ///    \mathbf{c}_{i_1-p_1:i_1,\dots,i_\text{par}-p_\text{par}:i_\text{par}}
    ///    \f]
    ///
    /// @sa UniformBSplineCore::eval, UniformBSplineCore::eval_
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the multivariate B-spline object
    ///
    /// @result Value(s) of the multivariate B-spline evaluated at the point(s) `xi`
    
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi) const
    {
      static_assert(parDim_ <= 4, "Unsupported parametric dimension");

      // 0D (point value)
      if constexpr (parDim_ == 0) {
        return coeffs_[0];
      }

        // 1D
      else if constexpr (parDim_ == 1) {
        int64_t i = std::min(ncoeffs_[0]-1,
                             int64_t(xi[0].item<real_t>() * (ncoeffs_[0] - degrees_[0]) + degrees_[0]));
        return eval<deriv>(xi, i);
      }

        // 2D
      else if constexpr (parDim_ == 2) {
        int64_t i = std::min(ncoeffs_[0]-1,
                             int64_t(xi[0].item<real_t>() * (ncoeffs_[0] - degrees_[0]) + degrees_[0]));
        int64_t j = std::min(ncoeffs_[1]-1,
                             int64_t(xi[1].item<real_t>() * (ncoeffs_[1] - degrees_[1]) + degrees_[1]));
        return eval<deriv>(xi, i, j);
      }

        // 3D
      else if constexpr (parDim_ == 3) {
        int64_t i = std::min(ncoeffs_[0]-1,
                             int64_t(xi[0].item<real_t>() * (ncoeffs_[0] - degrees_[0]) + degrees_[0]));
        int64_t j = std::min(ncoeffs_[1]-1,
                             int64_t(xi[1].item<real_t>() * (ncoeffs_[1] - degrees_[1]) + degrees_[1]));
        int64_t k = std::min(ncoeffs_[2]-1,
                             int64_t(xi[2].item<real_t>() * (ncoeffs_[2] - degrees_[2]) + degrees_[2]));
        return eval<deriv>(xi, i, j, k);
      }

        // 4D
      else if constexpr (parDim_ == 4) {
        int64_t i = std::min(ncoeffs_[0]-1,
                             int64_t(xi[0].item<real_t>() * (ncoeffs_[0] - degrees_[0]) + degrees_[0]));
        int64_t j = std::min(ncoeffs_[1]-1,
                             int64_t(xi[1].item<real_t>() * (ncoeffs_[1] - degrees_[1]) + degrees_[1]));
        int64_t k = std::min(ncoeffs_[2]-1,
                             int64_t(xi[2].item<real_t>() * (ncoeffs_[2] - degrees_[2]) + degrees_[2]));
        int64_t l = std::min(ncoeffs_[3]-1,
                             int64_t(xi[3].item<real_t>() * (ncoeffs_[3] - degrees_[3]) + degrees_[3]));
        return eval<deriv>(xi, i, j, k, l);
      } else {
        throw std::runtime_error("Unsupported parametric dimension");
      }
    }

    /// @brief Returns the value of the univariate B-spline object in
    /// the point `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for univariate B-splines
    /// (i.e. \f$d_\text{par}=1\f$)
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi, int64_t i) const
    {
      if constexpr (geoDim_ > 1) {
        auto basfunc =
          eval_prefactor<degrees_[0], (short_t) deriv % 10>() *
          eval_univariate<degrees_[0], 0, (short_t) deriv % 10>(xi[0], i);
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t g = 0; g < geoDim_; ++g)
          result[g] = dotproduct(basfunc,
                                 coeffs<false>(g).index(
                                                        {
                                                          torch::indexing::Slice(i - degrees_[0], i + 1, 1)
                                                        }
                                                        ).flatten(), 0);
        return result;
      } else        
        return
          eval_prefactor<degrees_[0], (short_t) deriv % 10>() *
          dotproduct(eval_univariate<degrees_[0], 0, (short_t) deriv % 10>(xi[0], i),
                     coeffs<false>(0).index(
                                            {
                                              torch::indexing::Slice(i - degrees_[0], i + 1, 1)
                                            }
                                            ).flatten(), 0);
    }

    /// @brief Returns the value of the bivariate B-spline object in
    /// the point `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for bivariate B-splines
    /// (i.e. \f$d_\text{par}=2\f$)
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi, int64_t i, int64_t j) const
    {

      if constexpr (geoDim_ > 1) {
        auto basfunc =
          eval_prefactor<degrees_[0],  (short_t)deriv    %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/10)%10>() *
          kronproduct(eval_univariate<degrees_[0], 0,  (short_t)deriv    %10>(xi[0], i),
                      eval_univariate<degrees_[1], 1, ((short_t)deriv/10)%10>(xi[1], j),
                      0);
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t g=0; g<geoDim_; ++g)
          result[g] = dotproduct(basfunc,
                                 coeffs<false>(g).index(
                                                        {
                                                          torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                          torch::indexing::Slice(j-degrees_[1], j+1, 1)
                                                        }
                                                        ).flatten(), 0);
        return result;
      } else
        return
          eval_prefactor<degrees_[0],  (short_t)deriv    %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/10)%10>() *
          dotproduct(kronproduct(eval_univariate<degrees_[0], 0,  (short_t)deriv    %10>(xi[0], i),
                                 eval_univariate<degrees_[1], 1, ((short_t)deriv/10)%10>(xi[1], j),
                                 0),
                     coeffs<false>(0).index(
                                            {
                                              torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                              torch::indexing::Slice(j-degrees_[1], j+1, 1)
                                            }
                                            ).flatten(), 0);
    }

    /// @brief Returns the value of the trivariate B-spline object in
    /// the point `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for trivariate B-splines
    /// (i.e. \f$d_\text{par}=3\f$)
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi, int64_t i, int64_t j, int64_t k) const
    {
      if constexpr (geoDim_ > 1) {
        auto basfunc =
          eval_prefactor<degrees_[0],  (short_t)deriv     %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/ 10)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/100)%10>() *
          kronproduct(kronproduct(eval_univariate<degrees_[0], 0,  (short_t)deriv    %10>(xi[0], i),
                                  eval_univariate<degrees_[1], 1, ((short_t)deriv/10)%10>(xi[1], j),
                                  0),
                      eval_univariate<degrees_[2], 2, ((short_t)deriv/100)%10>(xi[2], k),
                      0);
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t g=0; g<geoDim_; ++g)
          result[g] = dotproduct(basfunc,
                                 coeffs<false>(g).index(
                                                        {
                                                          torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                          torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                                          torch::indexing::Slice(k-degrees_[2], k+1, 1)
                                                        }
                                                        ).flatten(), 0);
        return result;
      } else
        return
          eval_prefactor<degrees_[0],  (short_t)deriv     %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/ 10)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/100)%10>() *
          dotproduct(kronproduct(kronproduct(eval_univariate<degrees_[0], 0,  (short_t)deriv    %10>(xi[0], i),
                                             eval_univariate<degrees_[1], 1, ((short_t)deriv/10)%10>(xi[1], j),
                                             0),
                                 eval_univariate<degrees_[2], 2, ((short_t)deriv/100)%10>(xi[2], k),
                                 0),
                     coeffs<false>(0).index(
                                            {
                                              torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                              torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                              torch::indexing::Slice(k-degrees_[2], k+1, 1)
                                            }
                                            ).flatten(), 0);
    }

    /// @brief Returns the value of the quartvariate B-spline object in
    /// the point `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for quartvariate B-splines
    /// (i.e. \f$d_\text{par}=4\f$)
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi, int64_t i, int64_t j, int64_t k, int64_t l) const
    {
      if constexpr (geoDim_ > 1) {
        auto basfunc =
          eval_prefactor<degrees_[0],  (short_t)deriv      %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/  10)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/ 100)%10>() *
          eval_prefactor<degrees_[3], ((short_t)deriv/1000)%10>() *
          kronproduct(kronproduct(eval_univariate<degrees_[0], 0,  (short_t)deriv      %10>(xi[0], i),
                                  eval_univariate<degrees_[1], 1, ((short_t)deriv/  10)%10>(xi[1], j),
                                  0),
                      kronproduct(eval_univariate<degrees_[2], 2, ((short_t)deriv/ 100)%10>(xi[2], k),
                                  eval_univariate<degrees_[3], 3, ((short_t)deriv/1000)%10>(xi[3], l),
                                  0),
                      0);
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t g=0; g<geoDim_; ++g)
          result[g] = dotproduct(basfunc,
                                 coeffs<false>(g).index(
                                                        {
                                                          torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                          torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                                          torch::indexing::Slice(k-degrees_[2], k+1, 1),
                                                          torch::indexing::Slice(l-degrees_[3], l+1, 1)
                                                        }
                                                        ).flatten(), 0);
        return result;
      } else
        return
          eval_prefactor<degrees_[0],  (short_t)deriv      %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/  10)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/ 100)%10>() *
          eval_prefactor<degrees_[3], ((short_t)deriv/1000)%10>() *
          dotproduct(kronproduct(kronproduct(eval_univariate<degrees_[0],0, (short_t)deriv      %10>(xi[0], i),
                                             eval_univariate<degrees_[1],1,((short_t)deriv/  10)%10>(xi[1], j),
                                             0),
                                 kronproduct(eval_univariate<degrees_[2],2,((short_t)deriv/ 100)%10>(xi[2], k),
                                             eval_univariate<degrees_[3],3,((short_t)deriv/1000)%10>(xi[3], l),
                                             0),
                                 0),
                     coeffs<false>(0).index(
                                            {
                                              torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                              torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                              torch::indexing::Slice(k-degrees_[2], k+1, 1),
                                              torch::indexing::Slice(l-degrees_[3], l+1, 1)
                                            }
                                            ).flatten(), 0);
    }

    /// @brief Returns the values of the B-spline object in the points `xi`
    ///
    /// @copydetails UniformBSplineCore::eval
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_(const torch::Tensor& xi, short_t dim = 0) const
    {
      return eval_<deriv>(TensorArray1({xi}), dim);
    }
          
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_(const std::array<torch::Tensor, parDim_>& xi, short_t dim = 0) const
    {
      if constexpr (parDim_ == 0)
        return coeffs_[0];
      else
        return eval_<deriv>(xi, eval_indices(xi), dim);
    }

    /// @brief Returns the value of the univariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for univariate B-splines
    /// (i.e. \f$d_\text{par}=1\f$)
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_(const TensorArray1& xi, const TensorArray1& idx, short_t dim = 0) const
    {
      assert(xi[0].sizes() == idx[0].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc =
          eval_prefactor<degrees_[0], (short_t) deriv % 10>() *
          eval_univariate_<degrees_[0], 0, (short_t) deriv % 10>( xi[0].flatten(),
                                                                 idx[0].flatten());
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result[i] = dotproduct(basfunc,
                                 coeffs(i).index_select(0,
                                                        VSlice(idx[0].flatten(),
                                                               -degrees_[0],
                                                               1)
                                                        ).view({-1, xi[0].numel()}),
                                 0).view(xi[0].sizes());
        return result;
      } else
        return
          eval_prefactor<degrees_[0], (short_t) deriv % 10>() *
          dotproduct(eval_univariate_<degrees_[0], 0, (short_t) deriv % 10>( xi[0].flatten(),
                                                                            idx[0].flatten()),
                     coeffs(0).index_select(0,
                                            VSlice(idx[0].flatten(),
                                                   -degrees_[0],
                                                   1)
                                            ).view({-1, xi[0].numel()}),
                     0).view(xi[0].sizes());
    }

    /// @brief Returns the value of the bivariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for bivariate B-splines
    /// (i.e. \f$d_\text{par}=2\f$)
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_(const TensorArray2& xi, const TensorArray2& idx, short_t dim = 0) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc =
          eval_prefactor<degrees_[0],  (short_t)deriv    %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/10)%10>() *
          kronproduct(eval_univariate_<degrees_[0], 0,  (short_t)deriv    %10>( xi[0].flatten(),
                                                                               idx[0].flatten()),
                      eval_univariate_<degrees_[1], 1, ((short_t)deriv/10)%10>( xi[1].flatten(),
                                                                               idx[1].flatten()),
                      0);
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result[i] = dotproduct(basfunc,
                                 coeffs(i).index_select(0,
                                                        VSlice(TensorArray2({idx[0].flatten(),
                                                                             idx[1].flatten()}),
                                                          std::array<int64_t,2>{-degrees_[0],
                                                                                -degrees_[1]},
                                                          std::array<int64_t,2>{1,1},
                                                          ncoeffs(1))
                                                        ).view({-1, xi[0].numel()}),
                                 0).view(xi[0].sizes());
        return result;
      } else
        return
          eval_prefactor<degrees_[0],  (short_t)deriv    %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/10)%10>() *
          dotproduct(kronproduct(eval_univariate_<degrees_[0], 0,  (short_t)deriv    %10>( xi[0].flatten(),
                                                                                          idx[0].flatten()),
                                 eval_univariate_<degrees_[1], 1, ((short_t)deriv/10)%10>( xi[1].flatten(),
                                                                                          idx[1].flatten()),
                                 0),
                     coeffs(0).index_select(0,
                                            VSlice(TensorArray2({idx[0].flatten(),
                                                                 idx[1].flatten()}),
                                              std::array<int64_t,2>{-degrees_[0],
                                                                    -degrees_[1]},
                                              std::array<int64_t,2>{1,1},
                                              ncoeffs(1))
                                            ).view({-1, xi[0].numel()}),
                     0).view(xi[0].sizes());
    }
    
    /// @brief Returns the value of the trivariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for trivariate B-splines
    /// (i.e. \f$d_\text{par}=3\f$)
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_(const TensorArray3& xi, const TensorArray3& idx, short_t dim = 0) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc =
          eval_prefactor<degrees_[0],  (short_t)deriv     %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/ 10)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/100)%10>() *
          kronproduct(kronproduct(eval_univariate_<degrees_[0], 0,  (short_t)deriv    %10>( xi[0].flatten(),
                                                                                           idx[0].flatten()),
                                  eval_univariate_<degrees_[1], 1, ((short_t)deriv/10)%10>( xi[1].flatten(),
                                                                                           idx[1].flatten()),
                                  0),
                      eval_univariate_<degrees_[2], 2, ((short_t)deriv/100)%10>( xi[2].flatten(),
                                                                                idx[2].flatten()),
                      0);
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result[i] = dotproduct(basfunc,
                                 coeffs(i).index_select(0,
                                                        VSlice(TensorArray3({idx[0].flatten(),
                                                                             idx[1].flatten(),
                                                                             idx[2].flatten}),
                                                          std::array<int64_t,3>{-degrees_[0],
                                                                                -degrees_[1],
                                                                                -degrees_[2]},
                                                          std::array<int64_t,3>{1,1,1},
                                                          ncoeffs(2)) // NEEDS TO BE FIXED!!!
                                                        ).view({-1, xi[0].numel()}),
                                 0).view(xi[0].sizes());
        return result;
      } else        
        return
          eval_prefactor<degrees_[0],  (short_t)deriv     %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/ 10)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/100)%10>() *
          dotproduct(kronproduct(kronproduct(eval_univariate_<degrees_[0], 0,  (short_t)deriv    %10>( xi[0].flatten(),
                                                                                                      idx[0].flatten()),
                                             eval_univariate_<degrees_[1], 1, ((short_t)deriv/10)%10>( xi[1].flatten(),
                                                                                                      idx[1].flatten()),
                                             0),
                                 eval_univariate_<degrees_[2], 2, ((short_t)deriv/100)%10>( xi[2].flatten(),
                                                                                           idx[2].flatten()),
                                 0),
                     coeffs(0).index_select(0,
                                            VSlice(TensorArray3({idx[0].flatten(),
                                                                 idx[1].flatten(),
                                                                 idx[2].flatten()}),
                                              std::array<int64_t,3>{-degrees_[0],
                                                                    -degrees_[1],
                                                                    -degrees_[2]},
                                              std::array<int64_t,3>{1,1,1},
                                              ncoeffs(2)) // NEEDS TO BE FIXED!!!
                                            ).view({-1, xi[0].numel()}),
                     0).view(xi[0].sizes());
    }

    
    /// @brief Returns the value of the quartvariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for quartvariate B-splines
    /// (i.e. \f$d_\text{par}=4\f$)
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_(const TensorArray4& xi, const TensorArray4& idx, short_t dim = 0) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[3].sizes() == idx[3].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc =
          eval_prefactor<degrees_[0],  (short_t)deriv      %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/  10)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/ 100)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/1000)%10>() *
          kronproduct(kronproduct(eval_univariate_<degrees_[0], 0,  (short_t)deriv      %10>( xi[0].flatten(),
                                                                                             idx[0].flatten()),
                                  eval_univariate_<degrees_[1], 1, ((short_t)deriv/  10)%10>( xi[1].flatten(),
                                                                                             idx[1].flatten()),
                                  0),
                      kronproduct(eval_univariate_<degrees_[2], 2, ((short_t)deriv/ 100)%10>( xi[2].flatten(),
                                                                                             idx[2].flatten()),
                                  eval_univariate_<degrees_[3], 3, ((short_t)deriv/1000)%10>( xi[3].flatten(),
                                                                                             idx[3].flatten()),
                                  0),
                      0);
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result[i] = dotproduct(basfunc,
                                 coeffs(i).index_select(0,
                                                        VSlice(TensorArray4({idx[0].flatten(),
                                                                             idx[1].flatten(),
                                                                             idx[2].flatten(),
                                                                             idx[3].flatten()}),
                                                          std::array<int64_t,4>{-degrees_[0],
                                                                                -degrees_[1],
                                                                                -degrees_[2],
                                                                                -degrees_[3]},
                                                          std::array<int64_t,4>{1,1,1,1},
                                                          ncoeffs(2)) // NEEDS TO BE FIXED!!!
                                                        ).view({-1, xi[0].numel()}),
                                 0).view(xi[0].sizes());
        return result;
      } else        
        return
          eval_prefactor<degrees_[0],  (short_t)deriv      %10>() *
          eval_prefactor<degrees_[1], ((short_t)deriv/  10)%10>() *
          eval_prefactor<degrees_[2], ((short_t)deriv/ 100)%10>() *
          eval_prefactor<degrees_[3], ((short_t)deriv/1000)%10>() *
          dotproduct(kronproduct(kronproduct(eval_univariate_<degrees_[0], 0,  (short_t)deriv      %10>( xi[0].flatten(),
                                                                                                        idx[0].flatten()),
                                             eval_univariate_<degrees_[1], 1, ((short_t)deriv/  10)%10>( xi[1].flatten(),
                                                                                                        idx[1].flatten()),
                                             0),
                                 kronproduct(eval_univariate_<degrees_[2], 2, ((short_t)deriv/ 100)%10>( xi[2].flatten(),
                                                                                                        idx[2].flatten()),
                                             eval_univariate_<degrees_[3], 3, ((short_t)deriv/1000)%10>( xi[3].flatten(),
                                                                                                        idx[3].flatten()),
                                             0),
                                 0),
                     coeffs(0).index_select(0,
                                            VSlice(TensorArray4({idx[0].flatten(),
                                                                 idx[1].flatten(),
                                                                 idx[2].flatten(),
                                                                 idx[3].flatten()}),
                                              std::array<int64_t,4>{-degrees_[0],
                                                                    -degrees_[1],
                                                                    -degrees_[2],
                                                                    -degrees_[3]},
                                              std::array<int64_t,4>{1,1,1,1},
                                              ncoeffs(2)) // NEEDS TO BE FIXED!!!@
                                            ).view({-1, xi[0].numel()}),
                     0).view(xi[0].sizes());
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
    /// The indices are returned as `std::array<torch::Tensor,
    /// parDim_>` in the same order as provided in `xi`    
    inline auto eval_indices(const TensorArray1& xi) const
    {
      assert(parDim_ == 1);
      return TensorArray1({
          torch::min(torch::full_like(xi[0], ncoeffs_[0]-1, core<real_t>::options_),
                     torch::floor(xi[0] * (ncoeffs_[0] - degrees_[0]) + degrees_[0])).to(torch::kInt64)
        });
    }
      
    inline auto eval_indices(const TensorArray2& xi) const
    {
      assert(parDim_ == 2);
      return TensorArray2({
          torch::min(torch::full_like(xi[0], ncoeffs_[0]-1, core<real_t>::options_),
                     torch::floor(xi[0] * (ncoeffs_[0] - degrees_[0]) + degrees_[0])).to(torch::kInt64),
          torch::min(torch::full_like(xi[1], ncoeffs_[1]-1, core<real_t>::options_),
                     torch::floor(xi[1] * (ncoeffs_[1] - degrees_[1]) + degrees_[1])).to(torch::kInt64)
        });
    }

    inline auto eval_indices(const TensorArray3& xi) const
    {
      assert(parDim_ == 3);
      return TensorArray3({
          torch::min(torch::full_like(xi, ncoeffs_[0]-1, core<real_t>::options_),
                     torch::floor(xi[0] * (ncoeffs_[0] - degrees_[0]) + degrees_[0])).to(torch::kInt64),
          torch::min(torch::full_like(xi, ncoeffs_[1]-1, core<real_t>::options_),
                     torch::floor(xi[1] * (ncoeffs_[1] - degrees_[1]) + degrees_[1])).to(torch::kInt64),
          torch::min(torch::full_like(xi, ncoeffs_[2]-1, core<real_t>::options_),
                     torch::floor(xi[2] * (ncoeffs_[2] - degrees_[2]) + degrees_[2])).to(torch::kInt64)
        });
    }

    inline auto eval_indices(const TensorArray4& xi) const
    {
      assert(parDim_ == 4);
      return TensorArray4({
          torch::min(torch::full_like(xi, ncoeffs_[0]-1, core<real_t>::options_),
                     torch::floor(xi[0] * (ncoeffs_[0] - degrees_[0]) + degrees_[0])).to(torch::kInt64),
          torch::min(torch::full_like(xi, ncoeffs_[1]-1, core<real_t>::options_),
                     torch::floor(xi[1] * (ncoeffs_[1] - degrees_[1]) + degrees_[1])).to(torch::kInt64),
          torch::min(torch::full_like(xi, ncoeffs_[2]-1, core<real_t>::options_),
                     torch::floor(xi[2] * (ncoeffs_[2] - degrees_[2]) + degrees_[2])).to(torch::kInt64),
          torch::min(torch::full_like(xi, ncoeffs_[3]-1, core<real_t>::options_),
                     torch::floor(xi[3] * (ncoeffs_[3] - degrees_[3]) + degrees_[3])).to(torch::kInt64)
        });
    }     

    /// @brief Transforms the coefficients based on the given mapping
    inline UniformBSplineCore& 
    transform(const std::function<std::array<real_t, geoDim_>(const std::array<real_t, parDim_>& )> transformation)
    {
      static_assert(parDim_ <= 4, "Unsupported parametric dimension");

      // 1D
      if constexpr (parDim_ == 1) {
#pragma omp parallel for simd
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          auto c = transformation(std::array<real_t, 1>{i / real_t(ncoeffs_[0] - 1)});
          for (short_t d = 0; d < geoDim_; ++d)
            coeffs_[d].detach()[i] = c[d];
        }
      }

        // 2D
      else if constexpr (parDim_ == 2) {
#pragma omp parallel for simd collapse(2)
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
            auto c = transformation(std::array<real_t, 2>{i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1)});
            for (short_t d = 0; d < geoDim_; ++d)
              coeffs_[d].detach()[i * ncoeffs_[1] + j] = c[d];
          }
        }
      }

        // 3D
      else if constexpr (parDim_ == 3) {
#pragma omp parallel for simd collapse(3)
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
            for (int64_t k = 0; k < ncoeffs_[2]; ++k) {
              auto c = transformation(std::array<real_t, 3>{i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1),
                                                            k / real_t(ncoeffs_[2] - 1)});
              for (short_t d = 0; d < geoDim_; ++d)
                coeffs_[d].detach()[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[2] + k] = c[d];
            }
          }
        }
      }

        // 4D
      else if constexpr (parDim_ == 4) {
#pragma omp parallel for simd collapse(4)
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
            for (int64_t k = 0; k < ncoeffs_[2]; ++k) {
              for (int64_t l = 0; l < ncoeffs_[3]; ++l) {
                auto c = transformation(std::array<real_t, 4>{i / real_t(ncoeffs_[0] - 1), j / real_t(ncoeffs_[1] - 1),
                                                              k / real_t(ncoeffs_[2] - 1),
                                                              l / real_t(ncoeffs_[3] - 1)});
                for (short_t d = 0; d < geoDim_; ++d)
                  coeffs_[d].detach()[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[2] * ncoeffs_[3] +
                                      k * ncoeffs_[3] + l] = c[d];
              }
            }
          }
        }
      } else {
        throw std::runtime_error("Unsupported parametric dimension");
      }

      return *this;
    }

    /// @brief Returns the B-spline object as XML string
    std::string to_xml() const
    {
      std::stringstream ss;

      // Write preamble and knot vectors
      ss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
         << "<xml>\n";

      // 1D parametric dimension
      if constexpr (parDim_ == 1) {
        ss << " <Geometry type=\"BSpline\">\n"
           << "  <Basis type=\"BSplineBasis\">\n"
           << "   <KnotVector degree=\"" << degrees_[0] << "\">";
        auto knots = knots_[0].template accessor<real_t, 1>();
        for (int64_t i = 0; i < nknots_[0]; ++i)
          ss << knots[i] << " ";
        ss << "</KnotVector>\n"
           << "  </Basis>\n";
      }

        // >1D parametric dimension
      else {
        ss << " <Geometry type=\"TensorBSpline" << parDim_ << "\" id=\"0\">\n"
           << "  <Basis type=\"TensorBSplineBasis" << parDim_ << "\">\n";
        for (short_t i = 0; i < parDim_; ++i) {
          ss << "   <Basis type=\"BSplineBasis\" index=\"" << i << "\">\n"
             << "    <KnotVector degree=\"" << degrees_[i] << "\">";
          auto knots = knots_[i].template accessor<real_t, 1>();
          for (int64_t j = 0; j < nknots_[i]; ++j)
            ss << knots[j] << " ";
          ss << "</KnotVector>\n"
             << "   </Basis>\n";
        }
        ss << "  </Basis>\n";
      }

      // Write coefficients
      ss << "  <coefs geoDim=\"" << geoDim_ << "\">\n";

      // 1D geometric dimension
      if constexpr (geoDim_ == 1) {
        auto coeffs0 = coeffs_[0].template accessor<real_t, 1>();
        for (int64_t i = 0; i < ncoeffs(); ++i)
          ss << "   " << coeffs() << "\n";
      }

        // 2D geometric dimension
      else if constexpr (geoDim_ == 2) {
        auto coeffs0 = coeffs_[0].template accessor<real_t, 1>();
        auto coeffs1 = coeffs_[1].template accessor<real_t, 1>();
        if constexpr (parDim_ == 1) {
          for (int64_t i = 0; i < ncoeffs(); ++i)
            ss << "   " << coeffs() << "\n";
        } else if constexpr (parDim_ == 2) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j)
            for (int64_t i = 0; i < ncoeffs_[0]; ++i)
              ss << "   " << coeffs0[i * ncoeffs_[1] + j]
                 << " " << coeffs1[i * ncoeffs_[1] + j] << "\n";
        } else if constexpr (parDim_ == 3) {
          for (int64_t k = 0; k < ncoeffs_[2]; ++k)
            for (int64_t j = 0; j < ncoeffs_[1]; ++j)
              for (int64_t i = 0; i < ncoeffs_[0]; ++i)
                ss << "   " << coeffs0[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs1[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k] << "\n";
        }
      }

        // 3D geometric dimension
      else if constexpr (geoDim_ == 3) {
        auto coeffs0 = coeffs_[0].template accessor<real_t, 1>();
        auto coeffs1 = coeffs_[1].template accessor<real_t, 1>();
        auto coeffs2 = coeffs_[2].template accessor<real_t, 1>();
        if constexpr (parDim_ == 1) {
          for (int64_t i = 0; i < ncoeffs(); ++i)
            ss << "   " << coeffs() << "\n";
        } else if constexpr (parDim_ == 2) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j)
            for (int64_t i = 0; i < ncoeffs_[0]; ++i)
              ss << "   " << coeffs0[i * ncoeffs_[1] + j]
                 << " " << coeffs1[i * ncoeffs_[1] + j]
                 << " " << coeffs2[i * ncoeffs_[1] + j] << "\n";
        } else if constexpr (parDim_ == 3) {
          for (int64_t k = 0; k < ncoeffs_[2]; ++k)
            for (int64_t j = 0; j < ncoeffs_[1]; ++j)
              for (int64_t i = 0; i < ncoeffs_[0]; ++i)
                ss << "   " << coeffs0[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs1[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs2[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k] << "\n";
        }
      }

        // 4D geometric dimension
      else if constexpr (geoDim_ == 4) {
        auto coeffs0 = coeffs_[0].template accessor<real_t, 1>();
        auto coeffs1 = coeffs_[1].template accessor<real_t, 1>();
        auto coeffs2 = coeffs_[2].template accessor<real_t, 1>();
        auto coeffs3 = coeffs_[3].template accessor<real_t, 1>();
        if constexpr (parDim_ == 1) {
          for (int64_t i = 0; i < ncoeffs(); ++i)
            ss << "   " << coeffs() << "\n";
        } else if constexpr (parDim_ == 2) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j)
            for (int64_t i = 0; i < ncoeffs_[0]; ++i)
              ss << "   " << coeffs0[i * ncoeffs_[1] + j]
                 << " " << coeffs1[i * ncoeffs_[1] + j]
                 << " " << coeffs2[i * ncoeffs_[1] + j]
                 << " " << coeffs3[i * ncoeffs_[1] + j] << "\n";
        } else if constexpr (parDim_ == 3) {
          for (int64_t k = 0; k < ncoeffs_[2]; ++k)
            for (int64_t j = 0; j < ncoeffs_[1]; ++j)
              for (int64_t i = 0; i < ncoeffs_[0]; ++i)
                ss << "   " << coeffs0[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs1[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs2[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs3[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k] << "\n";
        } else if constexpr (parDim_ == 4) {
          for (int64_t l = 0; l < ncoeffs_[3]; ++l)
            for (int64_t k = 0; k < ncoeffs_[2]; ++k)
              for (int64_t j = 0; j < ncoeffs_[1]; ++j)
                for (int64_t i = 0; i < ncoeffs_[0]; ++i)
                  ss << "   " << coeffs0[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[1] * ncoeffs_[2] +
                                         k * ncoeffs_[1] + l]
                     << " " << coeffs1[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[1] * ncoeffs_[2] +
                                       k * ncoeffs_[1] + l]
                     << " " << coeffs2[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[1] * ncoeffs_[2] +
                                       k * ncoeffs_[1] + l]
                     << " " << coeffs3[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[1] * ncoeffs_[2] +
                                       k * ncoeffs_[1] + l] << "\n";
        }
      } else {
        throw std::runtime_error("Unsupported parametric dimension");
      }

      ss << "  </coefs>\n"
         << " </Geometry>\n"
         << "</xml>\n";

      return ss.str();
    }

    /// @brief Saves the B-spline to file
    inline void save(const std::string& filename,
                     const std::string& key = "bspline") const
    {
      torch::serialize::OutputArchive archive;
      write(archive, key).save_to(filename);
    }

    /// @brief Loads the B-spline from file
    inline void load(const std::string& filename,
                     const std::string& key = "bspline")
    {
      torch::serialize::InputArchive archive;
      archive.load_from(filename);
      read(archive, key);
    }

    /// @brief Writes the B-spline into a torch::serialize::OutputArchive object
    inline torch::serialize::OutputArchive& write(torch::serialize::OutputArchive& archive,
                                                  const std::string& key = "bspline") const
    {
      archive.write(key + ".parDim", torch::full({1}, parDim_));
      archive.write(key + ".geoDim", torch::full({1}, geoDim_));

      for (short_t i = 0; i < parDim_; ++i)
        archive.write(key + ".degree[" + std::to_string(i) + "]", torch::full({1}, degrees_[i]));

      for (short_t i = 0; i < parDim_; ++i)
        archive.write(key + ".nknots[" + std::to_string(i) + "]", torch::full({1}, nknots_[i]));

      for (short_t i = 0; i < parDim_; ++i)
        archive.write(key + ".knots[" + std::to_string(i) + "]", knots_[i]);

      for (short_t i = 0; i < parDim_; ++i)
        archive.write(key + ".ncoeffs[" + std::to_string(i) + "]", torch::full({1}, ncoeffs_[i]));

      for (short_t i = 0; i < geoDim_; ++i)
        archive.write(key + ".coeffs[" + std::to_string(i) + "]", coeffs_[i]);

      return archive;
    }

    /// @brief Reads the B-spline from a torch::serialize::InputArchive object
    inline torch::serialize::InputArchive& read(torch::serialize::InputArchive& archive,
                                                const std::string& key = "bspline")
    {
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

    /// @brief Returns true if both B-spline objects are the same
    bool operator==(const UniformBSplineCore& other) const
    {
      bool result(true);

      result *= (parDim_ == other.parDim());
      result *= (geoDim_ == other.geoDim());

      for (short_t i = 0; i < parDim_; ++i)
        result *= (degree(i) == other.degree(i));

      for (short_t i = 0; i < parDim_; ++i)
        result *= (nknots(i) == other.nknots(i));

      for (short_t i = 0; i < parDim_; ++i)
        result *= (ncoeffs(i) == other.ncoeffs(i));

      for (short_t i = 0; i < parDim_; ++i)
        result *= torch::all(knots(i) == other.knots(i)).template item<bool>();

      for (short_t i = 0; i < parDim_; ++i)
        result *= torch::all(coeffs(i) == other.coeffs(i)).template item<bool>();

      return result;
    }

    /// @brief Returns true if both B-spline objects are different
    bool operator!=(const UniformBSplineCore& other) const {
      return !(*this==other); // Do not change this to (*this != other) is it does not work
    }

  private:
    /// @brief Computes the prefactor \f$p_d!/(p_d-r_d)! = p_d \cdots (p_d-r_d+1)\f$
    template<int64_t degree, int64_t deriv, int64_t terminal=degree-deriv>
    int64_t constexpr eval_prefactor() const
    {
      if constexpr (degree > terminal)
        return degree * eval_prefactor<degree-1, deriv, terminal>();
      else
        return 1;
    }
    
  protected:
    /// @brief Initializes the B-spline coefficients
    inline void init_coeffs(BSplineInit init)
    {
      switch (init) {

        case (BSplineInit::zeros): {

          // Fill coefficients with zeros
          for (short_t i = 0; i < geoDim_; ++i) {

            int64_t size = 1;
            for (short_t j = 0; j < parDim_; ++j)
              size *= ncoeffs_[j];

            coeffs_[i] = torch::zeros(size, core<real_t>::options_);
          }
          break;
        }

        case (BSplineInit::ones): {

          // Fill coefficients with ones
          for (short_t i = 0; i < geoDim_; ++i) {

            int64_t size = 1;
            for (short_t j = 0; j < parDim_; ++j)
              size *= ncoeffs_[j];

            coeffs_[i] = torch::ones(size, core<real_t>::options_);
          }
          break;
        }

        case (BSplineInit::linear): {

          // Fill coefficients with the tensor-product of linearly
          // increasing values between 0 and 1 per univariate dimension
          for (short_t i = 0; i < geoDim_; ++i) {
            coeffs_[i] = torch::ones(1, core<real_t>::options_);

            for (short_t j = 0; j < parDim_; ++j) {
              if (i == j)
                coeffs_[i] = coeffs_[i].kron(torch::linspace(static_cast<real_t>(0),
                                                             static_cast<real_t>(1),
                                                             ncoeffs_[j],
                                                             core<real_t>::options_));
              else
                coeffs_[i] = coeffs_[i].kron(torch::ones(ncoeffs_[j],
                                                         core<real_t>::options_));
            }
          }
          break;
        }

        case (BSplineInit::random): {

          // Fill coefficients with random values
          for (short_t i = 0; i < geoDim_; ++i) {

            int64_t size = 1;
            for (short_t j = 0; j < parDim_; ++j)
              size *= ncoeffs_[j];

            coeffs_[i] = torch::rand(size, core<real_t>::options_);
          }
          break;
        }

        case (BSplineInit::greville): {

          // Fill coefficients with the tensor-product of Greville
          // abscissae values per univariate dimension
          for (short_t i = 0; i < geoDim_; ++i) {
            coeffs_[i] = torch::ones(1, core<real_t>::options_);

            for (short_t j = 0; j < parDim_; ++j) {
              if (i == j) {
                auto greville_ = torch::zeros(ncoeffs_[j], core<real_t>::options_);
                auto greville = greville_.template accessor<real_t, 1>();
                auto knots = knots_[j].template accessor<real_t, 1>();
                for (int64_t k = 0; k < ncoeffs_[j]; ++k) {
                  for (short_t l = 1; l <= degrees_[j]; ++l)
                    greville[k] += knots[k + l];
                  greville[k] /= degrees_[j];
                }
                coeffs_[i] = coeffs_[i].kron(greville_);
              } else
                coeffs_[i] = coeffs_[i].kron(torch::ones(ncoeffs_[j],
                                                         core<real_t>::options_));
            }
          }
          break;
        }

        default:
          throw std::runtime_error("Unsupported BSplineInit option");
      }
    }

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
    ///   = \left( D^{r_d} B_{i_d-p_d,p_d}(\xi_d), \dots, D^{r_d} B_{i_d,p_d}(\xi_d) \right)^\top,
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
    ///   = \frac{p_d!}{(p_d-r_d)!}\mathbf{R}_1(\xi_d)\cdot \cdots \cdot \mathbf{R}_{p_d-r_d}(\xi_d)
    ///     D\mathbf{R}_{p_d-r_d+1}\cdot \cdots \cdot D\mathbf{R}_{p_d}(\xi_d),
    /// \f]
    ///
    /// where (cf. Equation (2.20) in \cite Lyche:2011)
    ///
    /// \f[
    ///   \mathbf{R}_k(\xi_d) =
    ///   \begin{pmatrix}
    ///     \frac{t_{i_p+1} - \xi_d}{t_{i_p+1} - t_{i_p+1-k}} & \frac{\xi_d - t_{i_p+1-k}}{t_{i_p+1} - t_{i_p+1-k}} & 0 & \cdots & 0 \\
    ///     0 & \frac{t_{i_p+2} - \xi_d}{t_{i_p+2} - t_{i_p+2-k}} & \frac{\xi_d - t_{i_p+2-k}}{t_{i_p+2} - t_{i_p+1-k}} & \cdots & 0 \\
    ///     \vdots & \vdots & \ddots & \ddots & \vdots \\
    ///     0 & 0 & \cdots & \frac{t_{i_p+k} - \xi_d}{t_{i_p+k} - t_{i_p}} & \frac{\xi_d - t_{i_p}}{t_{i_p+k} - t_{i_p}}
    ///   \end{pmatrix}
    /// \f]
    ///
    /// and (cf. Equation (3.30) in \cite Lyche:2011)
    ///
    /// \f[
    ///   D\mathbf{R}_k(\xi_d) =
    ///   \begin{pmatrix}
    ///     \frac{-1}{t_{i_p+1} - t_{i_p+1-k}} & \frac{1}{t_{i_p+1} - t_{i_p+1-k}} & 0 & \cdots & 0 \\
    ///     0 & \frac{-1}{t_{i_p+2} - t_{i_p+2-k}} & \frac{1}{t_{i_p+2} - t_{i_p+1-k}} & \cdots & 0 \\
    ///     \vdots & \vdots & \ddots & \ddots & \vdots \\
    ///     0 & 0 & \cdots & \frac{-1}{t_{i_p+k} - t_{i_p}} & \frac{1}{t_{i_p+k} - t_{i_p}}
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
    ///    3. \f$\mathbf{w}   = \left(\xi_d-\mathbf{t}_1\right)\div\left(\mathbf{t}_2-\mathbf{t}_1\right)\f$
    ///
    ///    4. \f$\mathbf{b}   = \left[\left(1-\mathbf{w}\right)\odot\mathbf{b}, 0\right]
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
    template<short_t degree, short_t dim, short_t deriv>
    inline auto eval_univariate(const torch::Tensor& xi, int64_t i) const
    {
      if constexpr (deriv > degree+1) {
        // It might be enough to return zero as a scalar
        return torch::zeros({degree+1}, core<real_t>::options_);
      } else {
        // Algorithm 2.22 from \cite Lyche:2011
        torch::Tensor b = torch::ones({1}, core<real_t>::options_);        

        // Calculate R_k, k = 1, ..., p_d-r_d
        for (short_t k=1; k<= degree-deriv; ++k) {

          // Instead of calculating t1 and t2 we calculate t1 and t21=(t2-t1)
          auto t1 = knots_[dim].index({torch::indexing::Slice(i-k+1, i+1, 1)});
          auto t21 = knots_[dim].index({torch::indexing::Slice(i+1, i+k+1, 1)}) - t1;

          // We handle the special case 0/0:=0 by first creating a
          // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
          // we do not have to take the absolute value as t2 >= t1.
          auto mask = (t21 < std::numeric_limits<real_t>::epsilon()).to(dtype<real_t>());

          // Instead of computing (xi-t1)/(t2-t1) which is prone to
          // yielding 0/0 we compute (xi-t1-mask)/(t2-t1-mask) which
          // equals the original expression if the mask is 0, i.e.,
          // t2-t1 >= eps and 1 otherwise since t1 <= xi < t2.          
          auto w  = torch::div(xi-t1-mask, t21-mask);

          // Calculate the vector of B-splines evaluated at xi
          b = torch::cat({ torch::mul(torch::ones_like(w, core<real_t>::options_)-w, b), zero_ }, 0)
            + torch::cat({ zero_, torch::mul(w, b) }, 0);
        }

        // Calculate DR_k, k = p_d-r_d+1, ..., p_d
        for (short_t k=degree-deriv+1; k<=degree; ++k) {

          // Instead of calculating t1 and t2 we calculate t21=(t2-t1)
          auto t21 = knots_[dim].index({torch::indexing::Slice(i+1, i+k+1, 1)})
            -        knots_[dim].index({torch::indexing::Slice(i-k+1, i+1, 1)});
          
          // We handle the special case 0/0:=0 by first creating a
          // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
          // we do not have to take the absolute value as t2 >= t1.
          auto mask = (t21 < std::numeric_limits<real_t>::epsilon()).to(dtype<real_t>());

          // Instead of computing 1/(t2-t1) which is prone to yielding
          // 0/0 we compute (1-mask)/(t2-t1-mask) which equals the
          // original expression if the mask is 0, i.e., t2-t1 >= eps
          // and 1 otherwise since t1 <= xi < t2.
          auto w  = torch::div(torch::ones_like(t21, core<real_t>::options_)-mask, t21-mask);

          // Calculate the vector of B-splines evaluated at xi
          b = torch::cat({ torch::mul(-w, b), zero_ }, 0)
            + torch::cat({ zero_, torch::mul(w, b) }, 0);
        }
        return b;
      }
    }
    
    /// @brief Returns the vector of univariate B-spline basis
    /// functions (or their derivatives) evaluated in the points `xi`
    ///
    /// @note This is the vectorized version of
    /// `UniformBSplineCore::eval_univariate(const torch::Tensor& xi, int64_t i)`
    template<short_t degree, short_t dim, short_t deriv>
    inline auto eval_univariate_(const torch::Tensor& xi, const torch::Tensor& i) const
    {
      assert(xi.sizes() == i.sizes());
      
      if constexpr (deriv > degree+1) {
        // It might be enough to return zero as a scalar
        return torch::zeros({degree+1, xi.numel()}, core<real_t>::options_);
      } else {
        // Algorithm 2.22 from \cite Lyche:2011
        torch::Tensor b = torch::ones({xi.numel()}, core<real_t>::options_);

        // Calculate R_k, k = 1, ..., p_d-r_d
        for (short_t k=1; k<= degree-deriv; ++k) {

          // Instead of calculating t1 and t2 we calculate t1 and t21=(t2-t1)
          auto t1 = knots_[dim].index_select(0, VSlice(i, -k+1, 1) );
          auto t21 = knots_[dim].index_select(0, VSlice(i, 1, k+1) ) - t1;

          // We handle the special case 0/0:=0 by first creating a
          // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
          // we do not have to take the absolute value as t2 >= t1.
          auto mask = (t21 < std::numeric_limits<real_t>::epsilon()).to(dtype<real_t>());

          // Instead of computing (xi-t1)/(t2-t1) which is prone to
          // yielding 0/0 we compute (xi-t1-mask)/(t2-t1-mask) which
          // equals the original expression if the mask is 0, i.e.,
          // t2-t1 >= eps and 1 otherwise since t1 <= xi < t2.          
          auto w  = torch::div(xi.repeat(k)-t1-mask, t21-mask);

          // Calculate the vector of B-splines evaluated at xi
          b = torch::cat({ torch::mul(torch::ones_like(w, core<real_t>::options_)-w, b),
              torch::zeros_like(xi, core<real_t>::options_) }, 0)
            + torch::cat({ torch::zeros_like(xi, core<real_t>::options_),
                torch::mul(w, b) }, 0);
        }

        // Calculate DR_k, k = p_d-r_d+1, ..., p_d
        for (short_t k=degree-deriv+1; k<=degree; ++k) {
          
          // Instead of calculating t1 and t2 we calculate t1 and t21=(t2-t1)
          auto t21 = knots_[dim].index_select(0, VSlice(i, 1, k+1) )
            -        knots_[dim].index_select(0, VSlice(i, -k+1, 1) );

          // We handle the special case 0/0:=0 by first creating a
          // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
          // we do not have to take the absolute value as t2 >= t1.
          auto mask = (t21 < std::numeric_limits<real_t>::epsilon()).to(dtype<real_t>());

          // Instead of computing 1/(t2-t1) which is prone to yielding
          // 0/0 we compute (1-mask)/(t2-t1-mask) which equals the
          // original expression if the mask is 0, i.e., t2-t1 >= eps
          // and 1 otherwise since t1 <= xi < t2.
          auto w  = torch::div(torch::ones_like(t21, core<real_t>::options_)-mask, t21-mask);

          // Calculate the vector of B-splines evaluated at xi
          b = torch::cat({ torch::mul(-w, b),
              torch::zeros_like(xi, core<real_t>::options_) }, 0)
            + torch::cat({ torch::zeros_like(xi, core<real_t>::options_),
                torch::mul(w, b) }, 0);
        }
        
        return b.view({degree+1, xi.numel()});
      }
    }      
  };

  /// @brief Serializes a B-spline object
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  inline torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive,
                                                     const UniformBSplineCore<real_t, GeoDim, Degrees...>& obj)
  {
    return obj.write(archive);
  }

  /// @brief De-serializes a B-spline object
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  inline torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive,
                                                    UniformBSplineCore<real_t, GeoDim, Degrees...>& obj)
  {
    return obj.read(archive);
  }

  /// @brief Tensor-product non-uniform B-spline (core functionality)
  ///
  /// This class extends the base class UniformBSplineCore to
  /// non-uniform B-splines. Like its base class it only implements
  /// the core functionality of non-uniform B-splines
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  class NonUniformBSplineCore : public UniformBSplineCore<real_t, GeoDim, Degrees...>
  {
  private:
    using Base = UniformBSplineCore<real_t, GeoDim, Degrees...>;    
    
  public:
    /// @brief Constructor for equidistant knot vectors
    using UniformBSplineCore<real_t, GeoDim, Degrees...>::UniformBSplineCore;

    /// @brief Constructor for non-equidistant knot vectors
    NonUniformBSplineCore(std::array<std::vector<real_t>, Base::parDim_> kv,
                          BSplineInit init = BSplineInit::zeros)
      : Base(std::array<int64_t, Base::parDim_>{Degrees...}, init)
    {
      for (short_t i=0; i<Base::parDim_; ++i) {

        // Check that knot vector has enough (n+p+1) entries
        if (2*Base::degrees_[i]>kv[i].size()-2)
          throw std::runtime_error("Knot vector is too short for an open knot vector (n+p+1 > 2*(p+1))");

        if (core<real_t>::options_.device() == torch::kCPU)
          Base::knots_[i] = torch::from_blob(static_cast<real_t*>(kv[i].data()),
                                             kv[i].size(), core<real_t>::options_).clone();
        else
          Base::knots_[i] = torch::from_blob(static_cast<real_t *>(kv[i].data()),
                                             kv[i].size(), core<real_t>::options_.device(torch::kCPU))
            .to(core<real_t>::options_.device());

        // Store the size of the knot vector
        Base::nknots_[i] = Base::knots_[i].size(0);

        // Store the size of the coefficient vector
        Base::ncoeffs_[i] = Base::nknots_[i]-Base::degrees_[i]-1;
      }

      // Initialize coefficients
      Base::init_coeffs(init);
    }

    /// @brief Evaluates the B-spline in the point `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi) const
    {
      static_assert(Base::parDim_ <= 4, "Unsupported parametric dimension");

      // 0D (point value)
      if constexpr (Base::parDim_ == 0) {
        return Base::coeffs_[0];
      }
      
      // 1D
      else if constexpr (Base::parDim_ == 1) {
        int64_t i;
        auto knots = Base::knots_[0].template accessor<real_t,1>();
        for (i=Base::degrees_[0]; i<Base::nknots_[0]-Base::degrees_[0]-1; ++i)
          if (knots[i+1] > xi[0].item<real_t>())
            break;

        return Base::eval(xi, i);
      }

      // 2D
      else if constexpr (Base::parDim_ == 2) {
        int64_t i;
        auto knots = Base::knots_[0].template accessor<real_t,1>();
        for (i=Base::degrees_[0]; i<Base::nknots_[0]-Base::degrees_[0]-1; ++i) {
          if (knots[i+1] > xi[0].item<real_t>())
            break;
        }

        int64_t j;
        knots = Base::knots_[1].template accessor<real_t,1>();
        for (j=Base::degrees_[1]; j<Base::nknots_[1]-Base::degrees_[1]-1; ++j) {
          if (knots[j+1] > xi[1].item<real_t>())
            break;
        }
        return Base::eval(xi, i, j);
      }

      // 3D
      else if constexpr (Base::parDim_ == 3) {
        int64_t i;
        auto knots = Base::knots_[0].template accessor<real_t,1>();
        for (i=Base::degrees_[0]; i<Base::nknots_[0]-Base::degrees_[0]-1; ++i)
          if (knots[i+1] > xi[0].item<real_t>())
            break;

        int64_t j;
        knots = Base::knots_[1].template accessor<real_t,1>();
        for (j=Base::degrees_[1]; j<Base::nknots_[1]-Base::degrees_[1]-1; ++j)
          if (knots[j+1] > xi[1].item<real_t>())
            break;

        int64_t k;
        knots = Base::knots_[2].template accessor<real_t,1>();
        for (k=Base::degrees_[2]; k<Base::nknots_[2]-Base::degrees_[2]-1; ++k)
          if (knots[k+1] > xi[2].item<real_t>())
            break;

        return Base::eval(xi, i, j, k);
      }

      // 4D
      else if constexpr (Base::parDim_ == 4) {
        int64_t i;
        auto knots = Base::knots_[0].template accessor<real_t,1>();
        for (i=Base::degrees_[0]; i<Base::nknots_[0]-Base::degrees_[0]-1; ++i)
          if (knots[i+1] > xi[0].item<real_t>())
            break;

        int64_t j;
        knots = Base::knots_[1].template accessor<real_t,1>();
        for (j=Base::degrees_[1]; j<Base::nknots_[1]-Base::degrees_[1]-1; ++j)
          if (knots[j+1] > xi[1].item<real_t>())
            break;

        int64_t k;
        knots = Base::knots_[2].template accessor<real_t,1>();
        for (k=Base::degrees_[2]; k<Base::nknots_[2]-Base::degrees_[2]-1; ++k)
          if (knots[k+1] > xi[2].item<real_t>())
            break;

        int64_t l;
        knots = Base::knots_[3].template accessor<real_t,1>();
        for (l=Base::degrees_[3]; l<Base::nknots_[3]-Base::degrees_[3]-1; ++l)
          if (knots[l+1] > xi[3].item<real_t>())
            break;

        return Base::eval(xi, i, j, k, l);
      }

      else {
        throw std::runtime_error("Unsupported parametric dimension");
      }
    }
  };

  /// B-spline (common high-level functionality)
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
  template<typename real_t, typename BSplineCore>
  class BSplineCommon : public BSplineCore
  {
  public:
    using BSplineCore::BSplineCore;

    /// Plots the B-spline object using matplotlibcpp
    inline auto plot(int64_t res0=10, int64_t res1=10, int64_t res2=10) const
    {
      return plot(*this, res0, res1, res2);
    }

    /// Plots the B-spline object using matplotlibcpp
    template<typename BSplineCore_t>
    inline auto plot(const BSplineCommon<real_t, BSplineCore_t>& color,
                     int64_t res0=10, int64_t res1=10, int64_t res2=10) const
    {
      static_assert(BSplineCore::parDim() == BSplineCore_t::parDim(),
                    "Parametric dimensions must match");

      if ((void*)this != (void*)&color && BSplineCore_t::geoDim() > 1)
        throw std::runtime_error("BSpline for coloring must have geoDim=1");

      if constexpr(BSplineCore::parDim()==1 && BSplineCore::geoDim()==1) {

        //
        // mapping: [0,1] -> R^1
        //

        matplot::vector_1d Xfine(res0, 0.0);
        matplot::vector_1d Yfine(res0, 0.0);

        auto Coords = BSplineCore::eval_(torch::linspace(0, 1, res0));
        auto XAccessor = Coords.template accessor<real_t,1>();
        
#pragma omp parallel for simd
        for (int64_t i=0; i<res0; ++i)
          Xfine[i] = XAccessor[i];

        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
            torch::Tensor Color = color.eval_(torch::linspace(0, 1, res0));
            auto CAccessor = Color.accessor<real_t,1>();
            
#pragma omp parallel for simd
            for (int64_t i=0; i<res0; ++i)
              Yfine[i] = CAccessor[i];
          }
        }

        // Plotting ...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
            matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          }
        } else {
          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);

          auto xaccessor = BSplineCore::template coeffs<true>(0).template accessor<real_t,1>();

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = xaccessor[i];
          }

          matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot(X, Y, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R");
        matplot::xlabel("x");
        matplot::ylabel("y");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim_==1 && BSplineCore::geoDim_==2) {

        //
        // mapping: [0,1] -> R^2
        //

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim()==1) {
            matplot::vector_2d Xfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Yfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Zfine(1, matplot::vector_1d(res0, 0.0));

            auto Coords = BSplineCore::eval_(torch::linspace(0, 1, res0));
            auto XAccessor = Coords[0].template accessor<real_t,1>();
            auto YAccessor = Coords[1].template accessor<real_t,1>();

            auto Color = color.eval_(torch::linspace(0, 1, res0));
            auto CAccessor = Color.template accessor<real_t,1>();
            
#pragma omp parallel for simd
            for (int64_t i=0; i<res0; ++i) {              
              Xfine[0][i] = XAccessor[i];
              Yfine[0][i] = YAccessor[i];
              Zfine[0][i] = CAccessor[i];
            }
            matplot::view(2);
            matplot::mesh(Xfine, Yfine, Zfine);
          }
        } else {
          matplot::vector_1d Xfine(res0, 0.0);
          matplot::vector_1d Yfine(res0, 0.0);

          auto Coords = BSplineCore::eval_(torch::linspace(0, 1, res0));
          auto XAccessor = Coords[0].template accessor<real_t,1>();
          auto YAccessor = Coords[1].template accessor<real_t,1>();
          
#pragma omp parallel for simd
          for (int64_t i=0; i<res0; ++i) {
            Xfine[i] = XAccessor[i];
            Yfine[i] = YAccessor[i];
          }

          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);

          auto xaccessor = BSplineCore::template coeffs<true>(0).template accessor<real_t,1>();
          auto yaccessor = BSplineCore::template coeffs<true>(1).template accessor<real_t,1>();

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = xaccessor[i];
            Y[i] = yaccessor[i];
          }

          matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot(X, Y, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R^2");
        matplot::xlabel("x");
        matplot::ylabel("y");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim()==1 && BSplineCore::geoDim()==3) {

        //
        // mapping: [0,1] -> R^3
        //

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim()==1) {
            matplot::vector_2d Xfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Yfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Zfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Cfine(1, matplot::vector_1d(res0, 0.0));

            auto Coords = BSplineCore::eval_(torch::linspace(0, 1, res0));
            auto XAccessor = Coords[0].template accessor<real_t,1>();
            auto YAccessor = Coords[1].template accessor<real_t,1>();
            auto ZAccessor = Coords[2].template accessor<real_t,1>();

            auto Color = color.eval_(torch::linspace(0, 1, res0));
            auto CAccessor = Color.template accessor<real_t,1>();
            
#pragma omp parallel for simd
            for (int64_t i=0; i<res0; ++i) {
              Xfine[0][i] = XAccessor[i];
              Yfine[0][i] = YAccessor[i];
              Zfine[0][i] = ZAccessor[i];
              Cfine[0][i] = CAccessor[i];
            }

            matplot::mesh(Xfine, Yfine, Zfine, Cfine);
          }
        } else {
          matplot::vector_1d Xfine(res0, 0.0);
          matplot::vector_1d Yfine(res0, 0.0);
          matplot::vector_1d Zfine(res0, 0.0);

          auto Coords = BSplineCore::eval_(torch::linspace(0, 1, res0));
          auto XAccessor = Coords[0].template accessor<real_t,1>();
          auto YAccessor = Coords[1].template accessor<real_t,1>();
          auto ZAccessor = Coords[2].template accessor<real_t,1>();
          
#pragma omp parallel for simd
          for (int64_t i=0; i<res0; ++i) {
            Xfine[i] = XAccessor[i];
            Yfine[i] = YAccessor[i];
            Zfine[i] = ZAccessor[i];
          }

          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Z(BSplineCore::ncoeffs(0), 0.0);

          auto xaccessor = BSplineCore::template coeffs<true>(0).template accessor<real_t,1>();
          auto yaccessor = BSplineCore::template coeffs<true>(1).template accessor<real_t,1>();
          auto zaccessor = BSplineCore::template coeffs<true>(2).template accessor<real_t,1>();

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = xaccessor[i];
            Y[i] = yaccessor[i];
            Z[i] = zaccessor[i];
          }

          matplot::plot3(Xfine, Yfine, Zfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot3(X, Y, Z, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R^3");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim()==2 && BSplineCore::geoDim()==2) {

        //
        // mapping: [0,1]^2 -> R^2
        //

        matplot::vector_2d Xfine(res1, matplot::vector_1d(res0, 0.0));
        matplot::vector_2d Yfine(res1, matplot::vector_1d(res0, 0.0));
        matplot::vector_2d Zfine(res1, matplot::vector_1d(res0, 0.0));

        std::array<torch::Tensor,2> meshgrid = convert<2>(torch::meshgrid({torch::linspace(0, 1, res0),
                                                                           torch::linspace(0, 1, res1)}, "xy"));
        auto Coords = BSplineCore::eval_(meshgrid);
        auto XAccessor = Coords[0].template accessor<real_t,2>();
        auto YAccessor = Coords[1].template accessor<real_t,2>();

        auto xAccessor = meshgrid[0].template accessor<float,2>();
        auto yAccessor = meshgrid[1].template accessor<float,2>();

#pragma omp parallel for simd collapse(2)
        for (int64_t i=0; i<res0; ++i)
          for (int64_t j=0; j<res1; ++j) {
            Xfine[j][i] = XAccessor[j][i];
            Yfine[j][i] = YAccessor[j][i];
          }
        
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim()==1) {            
            auto Color = color.eval_(meshgrid);
            auto CAccessor = Color.template accessor<real_t,2>();
            
#pragma omp parallel for simd collapse(2)
            for (int64_t i=0; i<res0; ++i)
              for (int64_t j=0; j<res1; ++j)
                Zfine[j][i] = CAccessor[j][i];
          }
        }

        // Plotting...
        if ((void*)this != (void*)&color && BSplineCore_t::geoDim()==1) {
          matplot::view(2);
          matplot::colormap(matplot::palette::hsv());
          matplot::mesh(Xfine, Yfine, Zfine)->palette_map_at_surface(true).face_alpha(0.7);
        } else {
          matplot::view(2);
          matplot::vector_2d X(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Y(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Z(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));

          auto x = BSplineCore::template coeffs<false>(0);
          auto y = BSplineCore::template coeffs<false>(1);

#pragma omp parallel for simd collapse(2)
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i)
            for (int64_t j=0; j<BSplineCore::ncoeffs(1); ++j) {
              X[j][i] = x[i][j].template item<real_t>();
              Y[j][i] = y[i][j].template item<real_t>();
            }

          matplot::colormap(matplot::palette::winter());
          matplot::mesh(Xfine, Yfine, Zfine);
          matplot::hold(matplot::on);
          matplot::surf(X, Y, Z)->palette_map_at_surface(true).face_alpha(0);
          matplot::hold(matplot::off);
        }
        matplot::title("BSpline: [0,1]^2 -> R^2");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim()==2 && BSplineCore::geoDim()==3) {

        ///
        // mapping: [0,1]^2 -> R^3
        ///

        matplot::vector_2d Xfine(res1, matplot::vector_1d(res0, 0.0));
        matplot::vector_2d Yfine(res1, matplot::vector_1d(res0, 0.0));
        matplot::vector_2d Zfine(res1, matplot::vector_1d(res0, 0.0));

#pragma omp parallel for simd collapse(2)
        for (int64_t i=0; i<res0; ++i)
          for (int64_t j=0; j<res1; ++j) {
            auto coords = BSplineCore::eval(torch::stack(
                                                         {
                                                           torch::full({1}, i/real_t(res0-1)),
                                                           torch::full({1}, j/real_t(res1-1))
                                                         }
                                                         ).view({2})
                                            );
            Xfine[j][i] = coords[0].template item<real_t>();
            Yfine[j][i] = coords[1].template item<real_t>();
            Zfine[j][i] = coords[2].template item<real_t>();
          }

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim()==1) {
            matplot::vector_2d Cfine(res1, matplot::vector_1d(res0, 0.0));

#pragma omp parallel for simd collapse(2)
            for (int64_t i=0; i<res0; ++i)
              for (int64_t j=0; j<res1; ++j) {
                Cfine[j][i] = color.eval(torch::stack(
                                                      {
                                                        torch::full({1}, i/real_t(res0-1)),
                                                        torch::full({1}, j/real_t(res1-1))
                                                      }
                                                      ).view({2})
                                         ).template item<real_t>();
              }
            matplot::colormap(matplot::palette::hsv());
            matplot::mesh(Xfine, Yfine, Zfine, Cfine);
          }
        }
        else {
          matplot::vector_2d X(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Y(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Z(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));

          auto x = BSplineCore::template coeffs<false>(0);
          auto y = BSplineCore::template coeffs<false>(1);
          auto z = BSplineCore::template coeffs<false>(2);

#pragma omp parallel for simd collapse(2)
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i)
            for (int64_t j=0; j<BSplineCore::ncoeffs(1); ++j) {
              X[j][i] = x[i][j].template item<real_t>();
              Y[j][i] = y[i][j].template item<real_t>();
              Z[j][i] = z[i][j].template item<real_t>();
            }

          matplot::colormap(matplot::palette::winter());
          matplot::mesh(Xfine, Yfine, Zfine);

          matplot::hold(matplot::on);
          matplot::surf(X, Y, Z)->palette_map_at_surface(true).face_alpha(0);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1]^2 -> R^3");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else
        throw std::runtime_error("Unsupported combination of parametric/geometric dimensions");
    }

    /// Returns a string representation of the BSplineCommon object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << BSplineCore::name()
         << "(\n  parDim=" << BSplineCore::parDim_
         << ", geoDim=" << BSplineCore::geoDim_

         << ", degrees=";
      for (short_t i=0; i<BSplineCore::parDim_-1; ++i)
        os << BSplineCore::degree(i) << "x";
      if (BSplineCore::parDim_ > 0)
        os << BSplineCore::degree(BSplineCore::parDim_-1);
      else
        os << 0;

      os << ", knots=";
      for (short_t i=0; i<BSplineCore::parDim_-1; ++i)
        os << BSplineCore::nknots(i) << "x";
      if (BSplineCore::parDim_ > 0)
        os << BSplineCore::nknots(BSplineCore::parDim_-1);
      else
        os << 0;
          
      os << ", coeffs=";
      for (short_t i=0; i<BSplineCore::parDim_-1; ++i)
        os << BSplineCore::ncoeffs(i) << "x";
      if (BSplineCore::parDim_ > 0)
        os << BSplineCore::ncoeffs(BSplineCore::parDim_-1);
      else
        os << 1;

      if (is_verbose(os)) {
        os << "\nknots = ";
        if (BSplineCore::parDim_ > 0)
          os << BSplineCore::knots();
        else
          os << "{}";
        os << "\ncoeffs = "
           << BSplineCore::template coeffs<false>();
      }
      
      os << "\n)";
    }
  };

  /// Tensor-product uniform B-spline
  template<typename real_t, short_t geoDim, short_t... Degrees>
  using UniformBSpline = BSplineCommon<real_t, UniformBSplineCore<real_t, geoDim, Degrees...>>;

  /// Print (as string) a UniformBSpline object
  template<typename real_t, short_t geoDim, short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const UniformBSpline<real_t, geoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

  /// Tensor-product non-uniform B-spline
  template<typename real_t, short_t geoDim, short_t... Degrees>
  using NonUniformBSpline = BSplineCommon<real_t, NonUniformBSplineCore<real_t, geoDim, Degrees...>>;

  /// Print (as string) a UniformBSpline object
  template<typename real_t, short_t geoDim, short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const NonUniformBSpline<real_t, geoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

} // namespace iganet
