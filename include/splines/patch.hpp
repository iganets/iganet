/**
   @file splines/bspline.hpp

   @brief Abstract patch function base class.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core/core.hpp>

#include <utils/blocktensor.hpp>
#include <utils/tensorarray.hpp>

#include <nlohmann/json.hpp>
#include <pugixml.hpp>

namespace iganet {

namespace detail {

// @brief Concept to identify template parameters that have an
// as_tensor function
template <typename T>
concept HasAsTensor = requires(T a) {
  { a.as_tensor() };
};

// @brief Concept to identify template parameters that have an
// as_tensor_size function
template <typename T>
concept HasAsTensorSize = requires(T a) {
  { a.as_tensor_size() };
};

// @brief Concept to identify template parameters that have a
// from_tensor function
template <typename T>
concept HasFromTensor = requires(T a) {
  { a.from_tensor() };
};

} // namespace detail

/// @brief Abstract patch function base class.
template <typename real_t, short_t GeoDim, short_t ParDim> class BSplinePatch {
public:
  /// @brief Scalar type.
  using value_type = real_t;

  /// @brief Dimension of the physical space.
  /// @return Result of the operation.
  inline static constexpr short_t geoDim() noexcept { return GeoDim; }

  /// @brief Dimension of the parametric space.
  /// @return Result of the operation.
  inline static constexpr short_t parDim() noexcept { return ParDim; }

  /// @brief Destructor.
  virtual ~BSplinePatch() = default;

  /// @brief Returns the `device` property.
  /// @return Result of the operation.
  virtual torch::Device device() const noexcept = 0;

  /// @brief Returns the `device_index` property.
  /// @return Result of the operation.
  virtual int32_t device_index() const noexcept = 0;

  /// @brief Returns the `dtype` property.
  /// @return Result of the operation.
  virtual torch::Dtype dtype() const noexcept = 0;

  /// @brief Returns the `layout` property.
  /// @return Result of the operation.
  virtual torch::Layout layout() const noexcept = 0;

  /// @brief Returns the `requires_grad` property.
  /// @return Result of the operation.
  virtual bool requires_grad() const noexcept = 0;

  /// @brief Returns the `pinned_memory` property.
  /// @return Result of the operation.
  virtual bool pinned_memory() const noexcept = 0;

  /// @brief Returns if the layout is sparse.
  /// @return Result of the operation.
  virtual bool is_sparse() const noexcept = 0;

  /// @brief Sets the B-spline object's `requires_grad` property.
  /// @param requires_grad Value of `requires_grad`.
  /// @return Result of the operation.
  virtual BSplinePatch &set_requires_grad(bool requires_grad) noexcept = 0;

  /// @brief Provides the `as_tensor` operation.
  /// @return Result of the operation.
  // @brief Returns all coefficients as a single tensor
  virtual torch::Tensor as_tensor() const noexcept = 0;

  /// @brief Sets all coefficients from a single tensor.
  /// @param tensor Tensor to process.
  /// @return Result of the operation.
  virtual BSplinePatch &from_tensor(const torch::Tensor &tensor) noexcept = 0;

  /// @brief Returns the size of the single tensor representation of
  /// all coefficients.
  /// @return Result of the operation.
  virtual int64_t as_tensor_size() const noexcept = 0;

  /// @brief Returns the value of the spline function from precomputed
  /// basis function
  /// @{
  /// @param basfunc Value of `basfunc`.
  /// @param coeff_indices Value of `coeff_indices`.
  /// @param numeval Value of `numeval`.
  /// @param sizes Value of `sizes`.
  /// @return Result of the operation.
  virtual utils::BlockTensor<torch::Tensor, 1, GeoDim>
  eval_from_precomputed(const torch::Tensor &basfunc,
                        const torch::Tensor &coeff_indices, int64_t numeval,
                        torch::IntArrayRef sizes) const = 0;

  /// @brief Provides the `eval_from_precomputed` operation.
  /// @param basfunc Value of `basfunc`.
  /// @param coeff_indices Value of `coeff_indices`.
  /// @param numeval Value of `numeval`.
  /// @param sizes Value of `sizes`.
  /// @return Result of the operation.
  virtual utils::BlockTensor<torch::Tensor, 1, GeoDim>
  eval_from_precomputed(const utils::TensorArray<ParDim> &basfunc,
                        const torch::Tensor &coeff_indices, int64_t numeval,
                        torch::IntArrayRef sizes) const = 0;
  /// @}

  /// @brief Returns the B-spline patch as a JSON object.
  /// @return Result of the operation.
  [[nodiscard]] virtual nlohmann::json to_json() const = 0;

  /// @brief Updates the B-spline patch from a JSON object.
  /// @param json JSON value to process.
  /// @return Result of the operation.
  virtual BSplinePatch &from_json(const nlohmann::json &json) = 0;

  /// @brief Returns the B-spline patch as an XML document.
  /// @param id Object identifier.
  /// @param label Object label.
  /// @param index Object index.
  /// @return Result of the operation.
  [[nodiscard]] virtual pugi::xml_document
  to_xml(int id = 0, const std::string &label = "", int index = -1) const = 0;

  /// @brief Appends the B-spline patch to an XML node.
  /// @param root Root XML node.
  /// @param id Object identifier.
  /// @param label Object label.
  /// @param index Object index.
  /// @return Result of the operation.
  virtual pugi::xml_node &to_xml(pugi::xml_node &root, int id = 0,
                                 const std::string &label = "",
                                 int index = -1) const = 0;

  /// @brief Updates the B-spline patch from an XML document.
  /// @param doc Value of `doc`.
  /// @param id Object identifier.
  /// @param label Object label.
  /// @param index Object index.
  /// @return Result of the operation.
  virtual BSplinePatch &from_xml(const pugi::xml_document &doc, int id = 0,
                                 const std::string &label = "",
                                 int index = -1) = 0;

  /// @brief Updates the B-spline patch from an XML node.
  /// @param root Root XML node.
  /// @param id Object identifier.
  /// @param label Object label.
  /// @param index Object index.
  /// @return Result of the operation.
  virtual BSplinePatch &from_xml(const pugi::xml_node &root, int id = 0,
                                 const std::string &label = "",
                                 int index = -1) = 0;

  /// @brief Returns a string representation.
  /// @param os Output stream.
  virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept = 0;
};

/// @brief Print (as string) a BSplinePatch object.
template <typename real_t, short_t GeoDim, short_t ParDim>
inline std::ostream &
operator<<(std::ostream &os, const BSplinePatch<real_t, GeoDim, ParDim> &obj) {
  obj.pretty_print(os);
  return os;
}

} // namespace iganet
