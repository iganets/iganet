/**
   @file splines/multipatch.hpp

   @brief Multi-patch geometry and topology support

   @author Guenther Obermair

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <pugixml.hpp>

#include <splines/bspline.hpp>
#include <splines/boundary.hpp>

namespace iganet {

namespace detail {

/// @brief Disjoint-set data structure for global DOF identification
class DisjointSet {
public:
  explicit DisjointSet(std::size_t size = 0) { reset(size); }

  void reset(std::size_t size) {
    parent_.resize(size);
    rank_.assign(size, 0);
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  std::size_t find(std::size_t index) {
    assert(index < parent_.size());
    if (parent_[index] != index)
      parent_[index] = find(parent_[index]);
    return parent_[index];
  }

  void unite(std::size_t lhs, std::size_t rhs) {
    lhs = find(lhs);
    rhs = find(rhs);

    if (lhs == rhs)
      return;

    if (rank_[lhs] < rank_[rhs])
      std::swap(lhs, rhs);

    parent_[rhs] = lhs;

    if (rank_[lhs] == rank_[rhs])
      ++rank_[lhs];
  }

private:
  std::vector<std::size_t> parent_;
  std::vector<std::size_t> rank_;
};

template <typename Patch, short_t ValueDim, typename Sequence>
struct StaticIsoparametricPatchType;

template <typename Patch, short_t ValueDim, std::size_t... Is>
struct StaticIsoparametricPatchType<Patch, ValueDim,
                                    std::index_sequence<Is...>> {
  using type = typename Patch::template derived_self_type<
      typename Patch::value_type, ValueDim,
      Patch::degree(static_cast<short_t>(Is))...>;
};

template <typename Patch, short_t ValueDim, typename = void>
struct IsoparametricPatchType {
  using type =
      typename StaticIsoparametricPatchType<
          Patch, ValueDim,
          std::make_index_sequence<static_cast<std::size_t>(Patch::parDim())>>::
          type;
};

template <typename Patch, short_t ValueDim>
struct IsoparametricPatchType<
    Patch, ValueDim,
    std::void_t<typename Patch::template isoparametric_patch_type<ValueDim>>> {
  using type = typename Patch::template isoparametric_patch_type<ValueDim>;
};

template <typename Patch, short_t ValueDim>
using isoparametric_patch_t =
    typename IsoparametricPatchType<Patch, ValueDim>::type;

template <typename Patch, short_t ValueDim>
concept HasMakeIsoparametricPatch =
    requires(const Patch &patch, Options<typename Patch::value_type> options) {
      { patch.template make_isoparametric<ValueDim>(options) };
    };

template <typename value_type>
inline std::vector<value_type> tensor_to_vector(const torch::Tensor &tensor) {
  auto cpu = tensor.to(torch::kCPU).contiguous();
  auto accessor = cpu.template accessor<value_type, 1>();

  std::vector<value_type> values(cpu.numel());
  for (int64_t i = 0; i < cpu.numel(); ++i)
    values[i] = accessor[i];

  return values;
}

template <typename Patch, short_t ValueDim>
inline auto make_isoparametric_patch(
    const Patch &patch,
    Options<typename Patch::value_type> options =
        Options<typename Patch::value_type>{}) {

  if constexpr (HasMakeIsoparametricPatch<Patch, ValueDim>) {
    return patch.template make_isoparametric<ValueDim>(options);
  } else if constexpr (Patch::is_nonuniform()) {
    using SolutionPatch = isoparametric_patch_t<Patch, ValueDim>;
    std::array<std::vector<typename Patch::value_type>, Patch::parDim()> kv;
    for (short_t i = 0; i < Patch::parDim(); ++i)
      kv[i] = tensor_to_vector<typename Patch::value_type>(patch.knots(i));

    return SolutionPatch(kv, init::zeros, options);
  } else {
    using SolutionPatch = isoparametric_patch_t<Patch, ValueDim>;
    return SolutionPatch(patch.ncoeffs(), init::zeros, options);
  }
}

} // namespace detail

/// @brief Runtime-degree tensor-product B-spline patch
///
/// This patch type stores tensor-product B-spline metadata with degrees known
/// at runtime. It is intended for heterogeneous multi-patch XML geometries
/// where different patches use different degree tuples.
template <typename real_t, short_t GeoDim, short_t ParDim>
class DynamicBSplinePatch {
public:
  using value_type = real_t;
  struct PreparedEvaluation {
    utils::TensorArray<ParDim> xi;
    utils::TensorArray<ParDim> knot_indices;
    torch::Tensor coeff_indices;
    int64_t numeval{0};
    std::vector<int64_t> sizes;
    std::map<short_t, torch::Tensor> basfuncs;

    [[nodiscard]] bool has_basfunc(deriv d) const noexcept {
      return basfuncs.contains(static_cast<short_t>(d));
    }

    [[nodiscard]] const torch::Tensor &basfunc(deriv d) const {
      const auto it = basfuncs.find(static_cast<short_t>(d));
      if (it == basfuncs.end())
        throw std::runtime_error(
            "PreparedEvaluation does not contain requested basis derivative");
      return it->second;
    }
  };

  template <typename other_t, short_t GeoDim_, short_t...>
  using derived_self_type = DynamicBSplinePatch<other_t, GeoDim_, ParDim>;

  template <short_t ValueDim>
  using isoparametric_patch_type =
      DynamicBSplinePatch<value_type, ValueDim, ParDim>;

  static_assert(ParDim >= 1 && ParDim <= 4,
                "DynamicBSplinePatch supports parDim 1..4");
  static_assert(GeoDim >= 1,
                "DynamicBSplinePatch requires at least one value component");

public:
  /// @brief Default constructor
  explicit DynamicBSplinePatch(Options<value_type> options =
                                   Options<value_type>{})
      : options_(options) {
    degrees_.fill(0);
    nknots_.fill(0);
    ncoeffs_.fill(0);
  }

  /// @brief Constructor from degrees and knot vectors
  DynamicBSplinePatch(
      const std::array<short_t, ParDim> &degrees,
      const std::array<std::vector<value_type>, ParDim> &knots,
      enum init init = init::zeros,
      Options<value_type> options = Options<value_type>{})
      : degrees_(degrees), options_(options) {
    for (short_t i = 0; i < ParDim; ++i) {
      knots_[i] = utils::to_tensor(knots[i], options_);
      nknots_[i] = static_cast<int64_t>(knots[i].size());
      ncoeffs_[i] = nknots_[i] - degrees_[i] - 1;

      if (ncoeffs_[i] <= 0)
        throw std::runtime_error(
            "Dynamic B-spline knot vector is incompatible with degree");
    }

    init_coeffs_(init);
  }

  /// @brief Returns the parametric dimension
  inline static constexpr short_t parDim() noexcept { return ParDim; }

  /// @brief Returns the value/geometric dimension
  inline static constexpr short_t geoDim() noexcept { return GeoDim; }

  /// @brief Returns true if the B-spline is uniform
  inline static constexpr bool is_uniform() noexcept { return false; }

  /// @brief Returns true if the B-spline is non-uniform
  inline static constexpr bool is_nonuniform() noexcept { return true; }

  /// @brief Returns the `device` property
  [[nodiscard]] inline torch::Device device() const noexcept {
    return options_.device();
  }

  /// @brief Returns the `dtype` property
  [[nodiscard]] inline torch::Dtype dtype() const noexcept {
    return options_.dtype();
  }

  /// @brief Returns the options object
  [[nodiscard]] inline const Options<value_type> &options() const noexcept {
    return options_;
  }

  /// @brief Returns the degrees
  [[nodiscard]] inline const std::array<short_t, ParDim> &degrees()
      const noexcept {
    return degrees_;
  }

  /// @brief Returns the degree in a parametric direction
  [[nodiscard]] inline short_t degree(short_t i) const noexcept {
    assert(i >= 0 && i < ParDim);
    return degrees_[i];
  }

  /// @brief Returns the knot vectors
  [[nodiscard]] inline const std::array<torch::Tensor, ParDim> &knots()
      const noexcept {
    return knots_;
  }

  /// @brief Returns the knot vector in a parametric direction
  [[nodiscard]] inline const torch::Tensor &knots(short_t i) const noexcept {
    assert(i >= 0 && i < ParDim);
    return knots_[i];
  }

  /// @brief Returns the number of coefficients per direction
  [[nodiscard]] inline const std::array<int64_t, ParDim> &ncoeffs()
      const noexcept {
    return ncoeffs_;
  }

  /// @brief Returns the number of coefficients in a parametric direction
  [[nodiscard]] inline int64_t ncoeffs(short_t i) const noexcept {
    assert(i >= 0 && i < ParDim);
    return ncoeffs_[i];
  }

  /// @brief Returns coefficient tensors component-wise
  [[nodiscard]] inline const std::array<torch::Tensor, GeoDim> &coeffs()
      const noexcept {
    return coeffs_;
  }

  /// @brief Returns one coefficient component tensor
  [[nodiscard]] inline const torch::Tensor &coeffs(short_t i) const noexcept {
    assert(i >= 0 && i < GeoDim);
    return coeffs_[i];
  }

  /// @brief Returns one coefficient component tensor
  [[nodiscard]] inline torch::Tensor &coeffs(short_t i) noexcept {
    assert(i >= 0 && i < GeoDim);
    return coeffs_[i];
  }

  /// @brief Returns the total number of scalar control points per component
  [[nodiscard]] inline int64_t ncumcoeffs() const noexcept {
    int64_t result = 1;
    for (short_t i = 0; i < ParDim; ++i)
      result *= ncoeffs_[i];
    return result;
  }

  /// @brief Returns all coefficients as a single tensor
  [[nodiscard]] inline torch::Tensor as_tensor() const noexcept {
    std::vector<torch::Tensor> blocks;
    blocks.reserve(GeoDim);
    for (short_t i = 0; i < GeoDim; ++i)
      blocks.push_back(coeffs_[i]);
    return torch::cat(blocks);
  }

  /// @brief Sets all coefficients from a single tensor
  inline DynamicBSplinePatch &from_tensor(const torch::Tensor &tensor) {
    if (tensor.numel() != as_tensor_size())
      throw std::runtime_error(
          "Tensor size is not compatible with DynamicBSplinePatch");

    const int64_t n = ncumcoeffs();
    for (short_t i = 0; i < GeoDim; ++i)
      coeffs_[i] = tensor.index({torch::indexing::Slice(i * n, (i + 1) * n)})
                       .to(options_.device());

    return *this;
  }

  /// @brief Returns the size of the single tensor representation
  [[nodiscard]] inline int64_t as_tensor_size() const noexcept {
    return GeoDim * ncumcoeffs();
  }

  /// @brief Returns tensor-product Greville abscissae
  [[nodiscard]] inline utils::TensorArray<ParDim>
  greville(bool interior = false) const {
    std::array<std::vector<value_type>, ParDim> points_1d;
    std::array<int64_t, ParDim> sizes{};
    int64_t num_points = 1;

    for (short_t d = 0; d < ParDim; ++d) {
      const auto kv = knot_vector_(d);
      points_1d[d].reserve(static_cast<std::size_t>(ncoeffs_[d]));

      for (int64_t i = 0; i < ncoeffs_[d]; ++i) {
        if (degrees_[d] == 0) {
          points_1d[d].push_back(
              static_cast<value_type>(0.5) * (kv[i] + kv[i + 1]));
        } else {
          value_type value = 0;
          for (short_t j = 1; j <= degrees_[d]; ++j)
            value += kv[i + j];
          points_1d[d].push_back(value / degrees_[d]);
        }
      }

      if (interior && points_1d[d].size() > 2) {
        points_1d[d].erase(points_1d[d].begin());
        points_1d[d].pop_back();
      }

      sizes[d] = static_cast<int64_t>(points_1d[d].size());
      num_points *= sizes[d];
    }

    utils::TensorArray<ParDim> result;
    std::vector<std::vector<value_type>> values(ParDim);
    for (short_t d = 0; d < ParDim; ++d)
      values[d].resize(static_cast<std::size_t>(num_points));

    for (int64_t p = 0; p < num_points; ++p) {
      int64_t rest = p;
      for (short_t d = 0; d < ParDim; ++d) {
        const int64_t i = rest % sizes[d];
        rest /= sizes[d];
        values[d][static_cast<std::size_t>(p)] =
            points_1d[d][static_cast<std::size_t>(i)];
      }
    }

    for (short_t d = 0; d < ParDim; ++d)
      result[d] = utils::to_tensor(values[d], options_);

    return result;
  }

  /// @brief Finds knot span indices for univariate evaluation
  [[nodiscard]] inline auto find_knot_indices(const torch::Tensor &xi) const {
    if constexpr (ParDim == 1) {
      return find_knot_indices(utils::TensorArray<ParDim>({xi}));
    } else {
      throw std::runtime_error("Invalid parametric dimension");
    }
  }

  /// @brief Finds knot span indices for tensor-product evaluation
  [[nodiscard]] inline auto
  find_knot_indices(const utils::TensorArray<ParDim> &xi) const {
    utils::TensorArray<ParDim> result;

    for (short_t d = 0; d < ParDim; ++d) {
      const auto kv = knot_vector_(d);
      auto flat = xi[d].flatten().to(torch::kCPU).contiguous();
      torch::Tensor spans =
          torch::empty({flat.numel()}, torch::TensorOptions().dtype(torch::kInt64));
      auto out = spans.accessor<int64_t, 1>();

      for (int64_t i = 0; i < flat.numel(); ++i) {
        const double value = flat.index({i}).template item<double>();
        out[i] = find_span_value_(d, static_cast<value_type>(value), kv);
      }

      result[d] = spans.view(xi[d].sizes()).to(xi[d].device());
    }

    return result;
  }

  /// @brief Finds coefficient indices for tensor-product evaluation
  template <bool memory_optimized = false>
  [[nodiscard]] inline auto
  find_coeff_indices(const torch::Tensor &knot_indices) const {
    if constexpr (ParDim == 1) {
      return find_coeff_indices<memory_optimized>(
          utils::TensorArray<ParDim>({knot_indices}));
    } else {
      throw std::runtime_error("Invalid parametric dimension");
    }
  }

  /// @brief Finds coefficient indices for tensor-product evaluation
  template <bool memory_optimized = false>
  [[nodiscard]] inline auto
  find_coeff_indices(const utils::TensorArray<ParDim> &knot_indices) const {
    if constexpr (memory_optimized) {
      throw std::runtime_error(
          "DynamicBSplinePatch memory-optimized evaluation is not implemented");
    } else {
      const int64_t numeval = knot_indices[0].numel();
      const int64_t support = support_size_();
      torch::Tensor result = torch::empty(
          {support, numeval}, torch::TensorOptions().dtype(torch::kInt64));
      auto out = result.accessor<int64_t, 2>();

      std::array<torch::Tensor, ParDim> spans_cpu;
      for (short_t d = 0; d < ParDim; ++d)
        spans_cpu[d] = knot_indices[d].flatten().to(torch::kCPU).contiguous();

      for (int64_t e = 0; e < numeval; ++e) {
        std::array<int64_t, ParDim> span{};
        for (short_t d = 0; d < ParDim; ++d)
          span[d] = spans_cpu[d].index({e}).template item<int64_t>();

        for (int64_t s = 0; s < support; ++s) {
          const auto local = support_multi_index_(s);
          std::array<int64_t, ParDim> coeff{};
          for (short_t d = 0; d < ParDim; ++d)
            coeff[d] = span[d] - degrees_[d] + local[d];
          out[s][e] = linear_index_(coeff, ncoeffs_);
        }
      }

      return result.to(knot_indices[0].device());
    }
  }

  /// @brief Evaluates the patch
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto eval(const torch::Tensor &xi) const {
    if constexpr (ParDim == 1) {
      return eval<Deriv, memory_optimized>(utils::TensorArray<ParDim>({xi}));
    } else {
      throw std::runtime_error("Invalid parametric dimension");
    }
  }

  /// @brief Evaluates the patch
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto
  eval(const utils::TensorArray<ParDim> &xi) const {
    return eval<Deriv, memory_optimized>(xi, find_knot_indices(xi));
  }

  /// @brief Evaluates the patch
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto
  eval(const utils::TensorArray<ParDim> &xi,
       const utils::TensorArray<ParDim> &knot_indices) const {
    return eval<Deriv, memory_optimized>(
        xi, knot_indices, find_coeff_indices<memory_optimized>(knot_indices));
  }

  /// @brief Evaluates the patch
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto
  eval(const utils::TensorArray<ParDim> &xi,
       const utils::TensorArray<ParDim> &knot_indices,
       const torch::Tensor &coeff_indices) const {
    if constexpr (memory_optimized) {
      throw std::runtime_error(
          "DynamicBSplinePatch memory-optimized evaluation is not implemented");
    } else {
      auto basfunc = eval_basfunc<Deriv, memory_optimized>(xi, knot_indices);
      return eval_from_precomputed(basfunc, coeff_indices, xi[0].numel(),
                                   xi[0].sizes());
    }
  }

  /// @brief Evaluates the patch with transposed basis ordering
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto eval_tr(const torch::Tensor &xi) const {
    return eval<Deriv, memory_optimized>(xi);
  }

  /// @brief Evaluates the patch with transposed basis ordering
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto
  eval_tr(const utils::TensorArray<ParDim> &xi) const {
    return eval<Deriv, memory_optimized>(xi);
  }

  /// @brief Evaluates basis functions
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto eval_basfunc(const torch::Tensor &xi) const {
    if constexpr (ParDim == 1) {
      return eval_basfunc<Deriv, memory_optimized>(
          utils::TensorArray<ParDim>({xi}));
    } else {
      throw std::runtime_error("Invalid parametric dimension");
    }
  }

  /// @brief Evaluates basis functions
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto
  eval_basfunc(const utils::TensorArray<ParDim> &xi) const {
    return eval_basfunc<Deriv, memory_optimized>(xi, find_knot_indices(xi));
  }

  /// @brief Evaluates basis functions
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto
  eval_basfunc(const torch::Tensor &xi,
               const torch::Tensor &knot_indices) const {
    if constexpr (ParDim == 1) {
      return eval_basfunc<Deriv, memory_optimized>(
          utils::TensorArray<ParDim>({xi}),
          utils::TensorArray<ParDim>({knot_indices}));
    } else {
      throw std::runtime_error("Invalid parametric dimension");
    }
  }

  /// @brief Evaluates basis functions
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto
  eval_basfunc(const utils::TensorArray<ParDim> &xi,
               const utils::TensorArray<ParDim> &knot_indices) const {
    if constexpr (memory_optimized) {
      throw std::runtime_error(
          "DynamicBSplinePatch memory-optimized evaluation is not implemented");
    } else {
      for (short_t d = 1; d < ParDim; ++d)
        if (xi[0].sizes() != xi[d].sizes())
          throw std::runtime_error(
              "DynamicBSplinePatch evaluation point tensors have incompatible sizes");

      const int64_t numeval = xi[0].numel();
      const int64_t support = support_size_();
      std::array<torch::Tensor, ParDim> univariate;

      for (short_t d = 0; d < ParDim; ++d)
        univariate[d] = eval_basis_univariate_<Deriv>(d, xi[d], knot_indices[d]);

      torch::Tensor result =
          torch::empty({support, numeval}, xi[0].options());

      for (int64_t s = 0; s < support; ++s) {
        const auto local = support_multi_index_(s);
        torch::Tensor value = torch::ones({numeval}, xi[0].options());
        for (short_t d = 0; d < ParDim; ++d)
          value = value * univariate[d].index({local[d]});
        result.index_put_({s}, value);
      }

      return result;
    }
  }

  /// @brief Evaluates basis functions with transposed basis ordering
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto eval_basfunc_tr(const torch::Tensor &xi) const {
    return eval_basfunc<Deriv, memory_optimized>(xi);
  }

  /// @brief Evaluates basis functions with transposed basis ordering
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] inline auto
  eval_basfunc_tr(const utils::TensorArray<ParDim> &xi) const {
    return eval_basfunc<Deriv, memory_optimized>(xi);
  }

  /// @brief Evaluates from precomputed basis and coefficient indices
  [[nodiscard]] inline utils::BlockTensor<torch::Tensor, 1, GeoDim>
  eval_from_precomputed(const torch::Tensor &basfunc,
                        const torch::Tensor &coeff_indices, int64_t numeval,
                        torch::IntArrayRef sizes) const {
    utils::BlockTensor<torch::Tensor, 1, GeoDim> result;
    const auto flat_indices =
        coeff_indices.flatten().to(coeffs_[0].device()).to(torch::kInt64);

    for (short_t g = 0; g < GeoDim; ++g) {
      auto local_coeffs =
          coeffs(g).index_select(0, flat_indices).view({-1, numeval});
      result.set(g, (basfunc * local_coeffs).sum(0).view(sizes));
    }

    return result;
  }

  /// @brief Prepares knot indices, coefficient indices, and basis values for fixed points
  template <deriv... Derivs, bool memory_optimized = false>
  [[nodiscard]] inline PreparedEvaluation
  prepare_evaluation(const utils::TensorArray<ParDim> &xi) const {
    auto knot_indices = find_knot_indices(xi);
    return prepare_evaluation<Derivs...>(xi, knot_indices);
  }

  /// @brief Prepares coefficient indices and basis values from known knot indices
  template <deriv... Derivs, bool memory_optimized = false>
  [[nodiscard]] inline PreparedEvaluation
  prepare_evaluation(const utils::TensorArray<ParDim> &xi,
                     const utils::TensorArray<ParDim> &knot_indices) const {
    PreparedEvaluation result;
    result.xi = xi;
    result.knot_indices = knot_indices;
    result.coeff_indices = find_coeff_indices<memory_optimized>(knot_indices);
    result.numeval = xi[0].numel();
    result.sizes.assign(xi[0].sizes().begin(), xi[0].sizes().end());

    ([&] {
      result.basfuncs.emplace(static_cast<short_t>(Derivs),
                              eval_basfunc<Derivs, memory_optimized>(
                                  xi, knot_indices));
    }(),
     ...);

    return result;
  }

  /// @brief Evaluates one derivative from a prepared cache
  template <deriv Deriv>
  [[nodiscard]] inline utils::BlockTensor<torch::Tensor, 1, GeoDim>
  eval_from_prepared(const PreparedEvaluation &prepared) const {
    return eval_from_precomputed(prepared.basfunc(Deriv), prepared.coeff_indices,
                                 prepared.numeval, prepared.sizes);
  }

  /// @brief Updates the B-spline patch from an XML document
  inline DynamicBSplinePatch &from_xml(const pugi::xml_document &doc,
                                       int id = 0,
                                       const std::string &label = "",
                                       int index = -1) {
    return from_xml(doc.child("xml"), id, label, index);
  }

  /// @brief Updates the B-spline patch from an XML root node
  inline DynamicBSplinePatch &from_xml(const pugi::xml_node &root, int id = 0,
                                       const std::string &label = "",
                                       int index = -1) {
    for (pugi::xml_node geo : root.children("Geometry")) {
      const std::string expected_type =
          ParDim == 1
              ? std::string("BSpline")
              : std::string("TensorBSpline").append(std::to_string(ParDim));

      if (geo.attribute("type").value() != expected_type ||
          (id >= 0 && geo.attribute("id").as_int() != id) ||
          (index >= 0 && geo.attribute("index").as_int() != index) ||
          (!label.empty() && geo.attribute("label").value() != label))
        continue;

      read_basis_from_xml_(geo);
      read_coeffs_from_xml_(geo);
      return *this;
    }

    throw std::runtime_error("XML object does not provide geometry with given "
                             "id, index, and/or label");
  }

  /// @brief Creates an isoparametric patch with different value dimension
  template <short_t ValueDim>
  [[nodiscard]] auto
  make_isoparametric(Options<value_type> options) const {
    DynamicBSplinePatch<value_type, ValueDim, ParDim> result(options);
    result.degrees_ = degrees_;
    result.nknots_ = nknots_;
    result.ncoeffs_ = ncoeffs_;
    for (short_t i = 0; i < ParDim; ++i)
      result.knots_[i] = knots_[i].clone().to(options.device());
    result.init_coeffs_(init::zeros);
    return result;
  }

private:
  template <typename, short_t, short_t> friend class DynamicBSplinePatch;

  static int64_t linear_index_(const std::array<int64_t, ParDim> &indices,
                               const std::array<int64_t, ParDim> &sizes) {
    int64_t index = 0;
    int64_t stride = 1;
    for (short_t d = 0; d < ParDim; ++d) {
      index += indices[d] * stride;
      stride *= sizes[d];
    }
    return index;
  }

  [[nodiscard]] int64_t support_size_() const noexcept {
    int64_t result = 1;
    for (short_t d = 0; d < ParDim; ++d)
      result *= degrees_[d] + 1;
    return result;
  }

  [[nodiscard]] std::array<int64_t, ParDim>
  support_multi_index_(int64_t linear) const noexcept {
    std::array<int64_t, ParDim> result{};
    for (short_t d = 0; d < ParDim; ++d) {
      const int64_t size = degrees_[d] + 1;
      result[d] = linear % size;
      linear /= size;
    }
    return result;
  }

  [[nodiscard]] std::vector<value_type> knot_vector_(short_t d) const {
    auto cpu = knots_[d].to(torch::kCPU).contiguous();
    std::vector<value_type> result(static_cast<std::size_t>(cpu.numel()));
    for (int64_t i = 0; i < cpu.numel(); ++i)
      result[static_cast<std::size_t>(i)] =
          static_cast<value_type>(cpu.index({i}).template item<double>());
    return result;
  }

  [[nodiscard]] int64_t
  find_span_value_(short_t d, value_type xi,
                   const std::vector<value_type> &knots) const noexcept {
    const int64_t degree = degrees_[d];
    const int64_t ncoeff = ncoeffs_[d];

    if (xi <= knots[static_cast<std::size_t>(degree)])
      return degree;
    if (xi >= knots[static_cast<std::size_t>(ncoeff)])
      return ncoeff - 1;

    for (int64_t i = degree; i < ncoeff; ++i)
      if (xi >= knots[static_cast<std::size_t>(i)] &&
          xi < knots[static_cast<std::size_t>(i + 1)])
        return i;

    return ncoeff - 1;
  }

  template <deriv Deriv>
  static constexpr short_t derivative_order_(short_t d) noexcept {
    short_t value = static_cast<short_t>(Deriv);
    for (short_t i = 0; i < d; ++i)
      value /= 10;
    return value % 10;
  }

  [[nodiscard]] static std::vector<std::vector<value_type>>
  ders_basis_funs_(int64_t span, value_type xi, short_t degree,
                   short_t order,
                   const std::vector<value_type> &knots) {
    const short_t nder = std::min(order, degree);

    std::vector<std::vector<value_type>> ndu(
        static_cast<std::size_t>(degree + 1),
        std::vector<value_type>(static_cast<std::size_t>(degree + 1), 0));
    std::vector<value_type> left(static_cast<std::size_t>(degree + 1), 0);
    std::vector<value_type> right(static_cast<std::size_t>(degree + 1), 0);
    std::vector<std::vector<value_type>> ders(
        static_cast<std::size_t>(nder + 1),
        std::vector<value_type>(static_cast<std::size_t>(degree + 1), 0));

    ndu[0][0] = static_cast<value_type>(1);

    for (short_t j = 1; j <= degree; ++j) {
      left[static_cast<std::size_t>(j)] =
          xi - knots[static_cast<std::size_t>(span + 1 - j)];
      right[static_cast<std::size_t>(j)] =
          knots[static_cast<std::size_t>(span + j)] - xi;

      value_type saved = 0;
      for (short_t r = 0; r < j; ++r) {
        ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)] =
            right[static_cast<std::size_t>(r + 1)] +
            left[static_cast<std::size_t>(j - r)];

        const auto den =
            ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(r)];
        const value_type temp =
            std::abs(den) > std::numeric_limits<value_type>::epsilon()
                ? ndu[static_cast<std::size_t>(r)][static_cast<std::size_t>(j - 1)] /
                      den
                : static_cast<value_type>(0);

        ndu[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] =
            saved + right[static_cast<std::size_t>(r + 1)] * temp;
        saved = left[static_cast<std::size_t>(j - r)] * temp;
      }
      ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] = saved;
    }

    for (short_t j = 0; j <= degree; ++j)
      ders[0][static_cast<std::size_t>(j)] =
          ndu[static_cast<std::size_t>(j)][static_cast<std::size_t>(degree)];

    std::array<std::vector<value_type>, 2> a{
        std::vector<value_type>(static_cast<std::size_t>(degree + 1), 0),
        std::vector<value_type>(static_cast<std::size_t>(degree + 1), 0)};

    for (short_t r = 0; r <= degree; ++r) {
      short_t s1 = 0;
      short_t s2 = 1;
      a[0][0] = static_cast<value_type>(1);

      for (short_t k = 1; k <= nder; ++k) {
        value_type dval = 0;
        const int64_t rk = static_cast<int64_t>(r) - k;
        const int64_t pk = static_cast<int64_t>(degree) - k;

        if (r >= k) {
          const auto den =
              ndu[static_cast<std::size_t>(pk + 1)][static_cast<std::size_t>(rk)];
          a[static_cast<std::size_t>(s2)][0] =
              std::abs(den) > std::numeric_limits<value_type>::epsilon()
                  ? a[static_cast<std::size_t>(s1)][0] / den
                  : static_cast<value_type>(0);
          dval = a[static_cast<std::size_t>(s2)][0] *
                 ndu[static_cast<std::size_t>(rk)][static_cast<std::size_t>(pk)];
        }

        const int64_t j1 = rk >= -1 ? 1 : -rk;
        const int64_t j2 = static_cast<int64_t>(r) - 1 <= pk
                               ? k - 1
                               : static_cast<int64_t>(degree) - r;

        for (int64_t j = j1; j <= j2; ++j) {
          const auto den = ndu[static_cast<std::size_t>(pk + 1)]
                              [static_cast<std::size_t>(rk + j)];
          a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(j)] =
              std::abs(den) > std::numeric_limits<value_type>::epsilon()
                  ? (a[static_cast<std::size_t>(s1)][static_cast<std::size_t>(j)] -
                     a[static_cast<std::size_t>(s1)][static_cast<std::size_t>(j - 1)]) /
                        den
                  : static_cast<value_type>(0);
          dval += a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(j)] *
                  ndu[static_cast<std::size_t>(rk + j)][static_cast<std::size_t>(pk)];
        }

        if (r <= pk) {
          const auto den =
              ndu[static_cast<std::size_t>(pk + 1)][static_cast<std::size_t>(r)];
          a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(k)] =
              std::abs(den) > std::numeric_limits<value_type>::epsilon()
                  ? -a[static_cast<std::size_t>(s1)][static_cast<std::size_t>(k - 1)] /
                        den
                  : static_cast<value_type>(0);
          dval += a[static_cast<std::size_t>(s2)][static_cast<std::size_t>(k)] *
                  ndu[static_cast<std::size_t>(r)][static_cast<std::size_t>(pk)];
        }

        ders[static_cast<std::size_t>(k)][static_cast<std::size_t>(r)] = dval;
        std::swap(s1, s2);
      }
    }

    value_type factor = static_cast<value_type>(degree);
    for (short_t k = 1; k <= nder; ++k) {
      for (short_t j = 0; j <= degree; ++j)
        ders[static_cast<std::size_t>(k)][static_cast<std::size_t>(j)] *= factor;
      factor *= static_cast<value_type>(degree - k);
    }

    return ders;
  }

  template <deriv Deriv>
  [[nodiscard]] torch::Tensor
  eval_basis_univariate_(short_t d, const torch::Tensor &xi,
                         const torch::Tensor &knot_indices) const {
    const short_t degree = degrees_[d];
    const short_t order = derivative_order_<Deriv>(d);
    const int64_t numeval = xi.numel();
    const auto knots = knot_vector_(d);

    torch::Tensor result = torch::empty(
        {degree + 1, numeval}, xi.options().device(torch::kCPU));
    auto out = result.template accessor<value_type, 2>();

    auto xi_cpu = xi.flatten().to(torch::kCPU).contiguous();
    auto spans_cpu = knot_indices.flatten().to(torch::kCPU).contiguous();

    for (int64_t e = 0; e < numeval; ++e) {
      const value_type value =
          static_cast<value_type>(xi_cpu.index({e}).template item<double>());
      const int64_t span = spans_cpu.index({e}).template item<int64_t>();
      const auto ders =
          ders_basis_funs_(span, value, degree, order, knots);
      for (short_t local = 0; local <= degree; ++local) {
        out[local][e] =
            ders[static_cast<std::size_t>(order)][static_cast<std::size_t>(local)];
      }
    }

    return result.to(xi.device());
  }

  void init_coeffs_(enum init init_) {
    const int64_t size = ncumcoeffs();
    for (short_t i = 0; i < GeoDim; ++i) {
      switch (init_) {
      case init::zeros:
      case init::none:
        coeffs_[i] = torch::zeros({size}, options_);
        break;
      case init::ones:
        coeffs_[i] = torch::ones({size}, options_);
        break;
      case init::random:
        coeffs_[i] = torch::rand({size}, options_);
        break;
      default:
        coeffs_[i] = torch::zeros({size}, options_);
        break;
      }
    }
  }

  static std::vector<value_type> parse_values_(const char *text) {
    std::string values =
        std::regex_replace(text, std::regex("[\t\r\n\a]+| +"), " ");
    std::stringstream ss(values);
    std::vector<value_type> result;
    value_type value{};

    while (ss >> value)
      result.push_back(value);

    return result;
  }

  void read_basis_from_xml_(const pugi::xml_node &geo) {
    if constexpr (ParDim == 1) {
      pugi::xml_node basis = geo.child("Basis");
      if (!basis ||
          basis.attribute("type").value() != std::string("BSplineBasis"))
        throw std::runtime_error("XML geometry does not provide BSplineBasis");

      pugi::xml_node knots = basis.child("KnotVector");
      if (!knots)
        throw std::runtime_error("XML basis does not provide KnotVector");

      degrees_[0] = static_cast<short_t>(knots.attribute("degree").as_int());
      const auto kv = parse_values_(knots.text().get());
      knots_[0] = utils::to_tensor(kv, options_);
      nknots_[0] = static_cast<int64_t>(kv.size());
      ncoeffs_[0] = nknots_[0] - degrees_[0] - 1;
    } else {
      pugi::xml_node bases = geo.child("Basis");
      const std::string expected_type =
          std::string("TensorBSplineBasis").append(std::to_string(ParDim));
      if (!bases || bases.attribute("type").value() != expected_type)
        throw std::runtime_error(
            "XML geometry does not provide TensorBSplineBasis");

      std::array<bool, ParDim> found{};
      found.fill(false);

      for (pugi::xml_node basis : bases.children("Basis")) {
        if (basis.attribute("type").value() != std::string("BSplineBasis"))
          continue;

        const short_t i = static_cast<short_t>(basis.attribute("index").as_int());
        if (i < 0 || i >= ParDim)
          throw std::runtime_error("XML basis index is out of range");

        pugi::xml_node knots = basis.child("KnotVector");
        if (!knots)
          throw std::runtime_error("XML basis does not provide KnotVector");

        degrees_[i] = static_cast<short_t>(knots.attribute("degree").as_int());
        const auto kv = parse_values_(knots.text().get());
        knots_[i] = utils::to_tensor(kv, options_);
        nknots_[i] = static_cast<int64_t>(kv.size());
        ncoeffs_[i] = nknots_[i] - degrees_[i] - 1;
        found[i] = true;
      }

      if (std::any_of(found.begin(), found.end(), [](bool value) {
            return !value;
          }))
        throw std::runtime_error("XML geometry has incomplete tensor basis");
    }

    for (short_t i = 0; i < ParDim; ++i)
      if (ncoeffs_[i] <= 0)
        throw std::runtime_error(
            "XML knot vector is incompatible with B-spline degree");
  }

  void read_coeffs_from_xml_(const pugi::xml_node &geo) {
    pugi::xml_node coefs = geo.child("coefs");
    if (!coefs)
      throw std::runtime_error("XML geometry does not provide coefficients");

    if (coefs.attribute("geoDim").as_int() != GeoDim)
      throw std::runtime_error(
          "XML geometry value dimension is not compatible with patch type");

    const auto values = parse_values_(coefs.text().get());
    const int64_t n = ncumcoeffs();
    if (static_cast<int64_t>(values.size()) != n * GeoDim)
      throw std::runtime_error(
          "XML geometry provides incompatible number of coefficients");

    for (short_t g = 0; g < GeoDim; ++g)
      coeffs_[g] = torch::empty({n}, options_.device(torch::kCPU));

    for (int64_t i = 0; i < n; ++i)
      for (short_t g = 0; g < GeoDim; ++g)
        coeffs_[g].index_put_({i}, values[i * GeoDim + g]);

    for (short_t g = 0; g < GeoDim; ++g)
      coeffs_[g] = coeffs_[g].to(options_.device());
  }

private:
  std::array<short_t, ParDim> degrees_;
  std::array<int64_t, ParDim> nknots_;
  std::array<int64_t, ParDim> ncoeffs_;
  std::array<torch::Tensor, ParDim> knots_;
  std::array<torch::Tensor, GeoDim> coeffs_;
  Options<value_type> options_;
};

/// @brief Interface between two patch sides
template <short_t ParDim> struct MultiPatchInterface {
  static_assert(ParDim >= 1 && ParDim <= 4,
                "Multi-patch interfaces are supported for parDim 1..4");

  std::size_t patch1{0};
  short_t side1{side::none};
  std::size_t patch2{0};
  short_t side2{side::none};
  std::array<short_t, ParDim> direction_map{};
  std::array<bool, ParDim> direction_orientation{};
  std::string label{};

  /// @brief Returns the parametric direction fixed by a side
  static constexpr short_t side_direction(short_t side_) noexcept {
    return static_cast<short_t>((side_ - 1) / 2);
  }

  /// @brief Returns the fixed side parameter, false=0 and true=1
  static constexpr bool side_parameter(short_t side_) noexcept {
    return (side_ - 1) % 2 != 0;
  }

  /// @brief Returns the inverse interface
  [[nodiscard]] MultiPatchInterface inverse() const {
    MultiPatchInterface result;
    result.patch1 = patch2;
    result.side1 = side2;
    result.patch2 = patch1;
    result.side2 = side1;
    result.label = label;

    for (short_t i = 0; i < ParDim; ++i) {
      result.direction_map[direction_map[i]] = i;
      result.direction_orientation[direction_map[i]] =
          direction_orientation[i];
    }

    return result;
  }
};

/// @brief Outer boundary side of a multi-patch geometry
struct MultiPatchBoundary {
  std::size_t patch{0};
  short_t side{side::none};
  std::string label{};
};

/// @brief Global C0 DOF map for a multi-patch space
struct MultiPatchDofMap {
  std::vector<std::vector<int64_t>> local_to_global;
  std::vector<std::pair<std::size_t, int64_t>> representatives;
  int64_t ndofs{0};

  [[nodiscard]] bool empty() const noexcept { return ndofs == 0; }
};

/// @brief Multi-patch container class
///
/// This class stores a set of tensor-product patches, their box topology,
/// and a global C0 DOF map for matching interfaces. The XML topology format is
/// intentionally compatible with the G+Smo-style <MultiPatch> section:
/// each interface row stores patch/side pairs followed by a full direction
/// map and orientation vector.
template <typename Patch> class MultiPatch {
public:
  using patch_type = Patch;
  using value_type = typename Patch::value_type;
  using interface_type = MultiPatchInterface<Patch::parDim()>;
  using boundary_type = MultiPatchBoundary;
  using dof_map_type = MultiPatchDofMap;
  using eval_type = utils::TensorArray<Patch::parDim()>;
  using boundary_eval_type =
      std::vector<std::pair<boundary_type, utils::TensorArray<Patch::parDim()>>>;

  static_assert(Patch::parDim() >= 1 && Patch::parDim() <= 4,
                "MultiPatch supports patch parametric dimensions 1..4");

public:
  /// @brief Default constructor
  MultiPatch() = default;

  /// @brief Copy constructor
  MultiPatch(const MultiPatch &) = default;

  /// @brief Move constructor
  MultiPatch(MultiPatch &&) noexcept = default;

  /// @brief Copy assignment operator
  MultiPatch &operator=(const MultiPatch &) = default;

  /// @brief Move assignment operator
  MultiPatch &operator=(MultiPatch &&) noexcept = default;

public:
  /// @brief Returns the parametric dimension
  inline static constexpr short_t parDim() noexcept { return Patch::parDim(); }

  /// @brief Returns the geometric dimension
  inline static constexpr short_t geoDim() noexcept { return Patch::geoDim(); }

  /// @brief Sets the absolute and relative geometry matching tolerances
  MultiPatch &set_matching_tolerance(double abs_tol = 1e-6,
                                     double rel_tol = 1e-6) noexcept {
    abs_tol_ = abs_tol;
    rel_tol_ = rel_tol;
    return *this;
  }

  /// @brief Returns the absolute matching tolerance
  [[nodiscard]] double absolute_tolerance() const noexcept { return abs_tol_; }

  /// @brief Returns the relative matching tolerance
  [[nodiscard]] double relative_tolerance() const noexcept { return rel_tol_; }

public:
  /// @brief Returns an iterator to the patches
  auto begin() { return patches_.begin(); }

  /// @brief Returns a const-iterator to the patches
  /// @{
  auto begin() const { return patches_.begin(); }
  auto cbegin() const noexcept { return patches_.cbegin(); }
  /// @}

  /// @brief Returns an iterator to the end of the patches
  auto end() { return patches_.end(); }

  /// @brief Returns a const-iterator to the end of the patches
  /// @{
  auto end() const { return patches_.end(); }
  auto cend() const noexcept { return patches_.cend(); }
  /// @}

  /// @brief Returns a reverse iterator to the patches
  auto rbegin() { return patches_.rbegin(); }

  /// @brief Returns a reverse const-iterator to the patches
  /// @{
  auto rbegin() const { return patches_.rbegin(); }
  auto crbegin() const noexcept { return patches_.crbegin(); }
  /// @}

  /// @brief Returns a reverse iterator to the end of the patches
  auto rend() { return patches_.rend(); }

  /// @brief Returns a reverse const-iterator to the end of the patches
  /// @{
  auto rend() const { return patches_.rend(); }
  auto crend() const noexcept { return patches_.crend(); }
  /// @}

public:
  /// @brief Returns the number of patches
  [[nodiscard]] std::size_t npatches() const noexcept {
    return patches_.size();
  }

  /// @brief Returns the number of interfaces
  [[nodiscard]] std::size_t ninterfaces() const noexcept {
    return interfaces_.size();
  }

  /// @brief Returns the number of outer boundaries
  [[nodiscard]] std::size_t nboundaries() const noexcept {
    return boundaries_.size();
  }

  /// @brief Returns the number of global scalar control-point DOFs
  [[nodiscard]] int64_t ndofs() const noexcept { return dof_map_.ndofs; }

public:
  /// @brief Adds a single patch
  /// @{
  std::size_t addPatch(std::shared_ptr<Patch> patch, int xml_id = -1) {
    if (!patch)
      throw std::runtime_error("Cannot add null patch to MultiPatch");

    std::size_t index = patches_.size();
    patches_.push_back(std::move(patch));
    patch_xml_ids_.push_back(xml_id);
    invalidate_dof_map_();
    return index;
  }

  std::size_t addPatch(std::unique_ptr<Patch> patch, int xml_id = -1) {
    return addPatch(std::shared_ptr<Patch>(std::move(patch)), xml_id);
  }

  std::size_t addPatch(const Patch &patch, int xml_id = -1) {
    return addPatch(std::make_shared<Patch>(patch), xml_id);
  }

  std::size_t addPatch(Patch &&patch, int xml_id = -1) {
    return addPatch(std::make_shared<Patch>(std::move(patch)), xml_id);
  }
  /// @}

  /// @brief Adds a single interface
  std::size_t addInterface(const interface_type &interface) {
    validate_patch_side_(interface.patch1, interface.side1);
    validate_patch_side_(interface.patch2, interface.side2);

    if (interface.patch1 == interface.patch2 && interface.side1 == interface.side2)
      throw std::runtime_error("Cannot add an interface from a side to itself");

    interfaces_.push_back(interface);
    invalidate_dof_map_();
    return interfaces_.size() - 1;
  }

  /// @brief Adds a single boundary
  std::size_t addBoundary(const boundary_type &boundary) {
    validate_patch_side_(boundary.patch, boundary.side);

    boundaries_.push_back(boundary);
    return boundaries_.size() - 1;
  }

  /// @brief Removes all patches and topology data
  void clear() {
    patches_.clear();
    patch_xml_ids_.clear();
    xml_id_to_patch_.clear();
    interfaces_.clear();
    boundaries_.clear();
    dof_map_ = dof_map_type{};
  }

  /// @brief Returns a non-constant reference to a single patch
  Patch &patch(std::size_t index) {
    assert(index < patches_.size());
    return *patches_[index];
  }

  /// @brief Returns a constant reference to a single patch
  const Patch &patch(std::size_t index) const {
    assert(index < patches_.size());
    return *patches_[index];
  }

  /// @brief Returns a constant reference to the vector of patches
  const std::vector<std::shared_ptr<Patch>> &patches() const noexcept {
    return patches_;
  }

  /// @brief Returns a non-constant reference to the vector of patches
  std::vector<std::shared_ptr<Patch>> &patches() noexcept {
    invalidate_dof_map_();
    return patches_;
  }

  /// @brief Returns the XML id of the index-th patch, or -1 if unknown
  [[nodiscard]] int patch_xml_id(std::size_t index) const {
    assert(index < patch_xml_ids_.size());
    return patch_xml_ids_[index];
  }

  /// @brief Returns a constant reference to the interfaces
  const std::vector<interface_type> &interfaces() const noexcept {
    return interfaces_;
  }

  /// @brief Returns the interface attached to a patch side, if any
  [[nodiscard]] std::optional<interface_type>
  interface(std::size_t patch_index, short_t side_) const {
    validate_patch_side_(patch_index, side_);

    for (const auto &interface : interfaces_) {
      if (interface.patch1 == patch_index && interface.side1 == side_)
        return interface;
      if (interface.patch2 == patch_index && interface.side2 == side_)
        return interface.inverse();
    }

    return std::nullopt;
  }

  /// @brief Returns a constant reference to the boundaries
  const std::vector<boundary_type> &boundaries() const noexcept {
    return boundaries_;
  }

  /// @brief Returns all boundaries with a given label
  [[nodiscard]] std::vector<boundary_type>
  boundaries(const std::string &label) const {
    std::vector<boundary_type> result;
    for (const auto &boundary : boundaries_)
      if (boundary.label == label)
        result.push_back(boundary);
    return result;
  }

  /// @brief Returns a constant reference to the DOF map
  const dof_map_type &dof_map() const noexcept { return dof_map_; }

  /// @brief Returns patch-local scalar control-point indices on one side
  [[nodiscard]] std::vector<int64_t>
  boundary_local_dofs(std::size_t patch_index, short_t side_) const {
    validate_patch_side_(patch_index, side_);
    return side_indices_(patch_index, side_);
  }

  /// @brief Returns global scalar control-point DOFs on one patch side
  [[nodiscard]] std::vector<int64_t>
  boundary_global_dofs(std::size_t patch_index, short_t side_) const {
    if (dof_map_.empty())
      throw std::runtime_error("MultiPatch DOF map has not been built");

    const auto local = boundary_local_dofs(patch_index, side_);
    std::vector<int64_t> result;
    result.reserve(local.size());

    for (const auto index : local)
      result.push_back(dof_map_.local_to_global[patch_index]
                                           [static_cast<std::size_t>(index)]);

    return result;
  }

  /// @brief Returns global scalar control-point DOFs on one boundary side
  [[nodiscard]] std::vector<int64_t>
  boundary_global_dofs(const boundary_type &boundary) const {
    return boundary_global_dofs(boundary.patch, boundary.side);
  }

  /// @brief Returns unique global scalar control-point DOFs for a boundary label
  [[nodiscard]] std::vector<int64_t>
  boundary_global_dofs(const std::string &label) const {
    std::vector<int64_t> result;

    for (const auto &boundary : boundaries_) {
      if (boundary.label != label)
        continue;

      const auto side_dofs = boundary_global_dofs(boundary);
      result.insert(result.end(), side_dofs.begin(), side_dofs.end());
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

  /// @brief Returns the index of a given single patch
  /// @{
  std::size_t findPatchIndex(const Patch &patch) const {
    return findPatchIndex(&patch);
  }

  std::size_t findPatchIndex(const Patch *patch) const {
    auto it = std::find_if(patches_.begin(), patches_.end(),
                           [patch](const auto &ptr) { return ptr.get() == patch; });
    if (it == patches_.end())
      throw std::runtime_error("Did not find the patch index");

    return static_cast<std::size_t>(std::distance(patches_.begin(), it));
  }
  /// @}

public:
  /// @brief Updates the multi-patch object from an XML document
  MultiPatch &from_xml(const pugi::xml_document &doc, int id = 0,
                       const std::string &label = "") {
    return from_xml(doc.child("xml"), id, label);
  }

  /// @brief Updates the multi-patch object from an XML file
  MultiPatch &from_xml(const std::string &filename, int id = 0,
                       const std::string &label = "") {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());
    if (!result)
      throw std::runtime_error("Could not read MultiPatch XML file");

    return from_xml(doc, id, label);
  }

  /// @brief Updates the multi-patch object from an XML root node
  MultiPatch &from_xml(const pugi::xml_node &root, int id = 0,
                       const std::string &label = "") {
    clear();

    pugi::xml_node mp_node;
    for (pugi::xml_node candidate : root.children("MultiPatch")) {
      if ((id >= 0 ? candidate.attribute("id").as_int() == id : true) &&
          (!label.empty() ? candidate.attribute("label").value() == label : true)) {
        mp_node = candidate;
        break;
      }
    }

    if (!mp_node)
      throw std::runtime_error("XML object does not provide MultiPatch with "
                               "given id and/or label");

    const short_t xml_par_dim =
        static_cast<short_t>(mp_node.attribute("parDim").as_int());
    if (xml_par_dim != parDim())
      throw std::runtime_error("XML MultiPatch parametric dimension is not "
                               "compatible with patch type");

    read_xml_patches_(root, mp_node);
    read_xml_boundaries_(mp_node);
    read_xml_interfaces_(mp_node);
    normalize_topology_();
    build_dof_map();

    return *this;
  }

public:
  /// @brief Builds and validates the global C0 DOF map
  MultiPatch &build_dof_map() {
    validate_topology_();

    const std::size_t total_local = total_local_dofs_();
    detail::DisjointSet dsu(total_local);

    std::vector<int64_t> patch_offsets(patches_.size(), 0);
    int64_t offset = 0;
    for (std::size_t p = 0; p < patches_.size(); ++p) {
      patch_offsets[p] = offset;
      offset += patch(p).ncumcoeffs();
    }

    for (const auto &interface : interfaces_)
      unite_interface_dofs_(interface, patch_offsets, dsu);

    std::map<std::size_t, int64_t> root_to_global;
    dof_map_.local_to_global.clear();
    dof_map_.local_to_global.resize(patches_.size());
    dof_map_.representatives.clear();

    for (std::size_t p = 0; p < patches_.size(); ++p) {
      const int64_t nlocal = patch(p).ncumcoeffs();
      dof_map_.local_to_global[p].resize(nlocal);

      for (int64_t i = 0; i < nlocal; ++i) {
        const auto root =
            dsu.find(static_cast<std::size_t>(patch_offsets[p] + i));
        auto [it, inserted] =
            root_to_global.emplace(root, static_cast<int64_t>(root_to_global.size()));

        if (inserted)
          dof_map_.representatives.emplace_back(p, i);

        dof_map_.local_to_global[p][i] = it->second;
      }
    }

    dof_map_.ndofs = static_cast<int64_t>(root_to_global.size());
    return *this;
  }

  /// @brief Returns all globally unique coefficients as a single tensor
  [[nodiscard]] torch::Tensor as_tensor() const {
    if (dof_map_.empty())
      throw std::runtime_error("MultiPatch DOF map has not been built");

    if (patches_.empty())
      return torch::empty({0}, torch::TensorOptions{});

    std::vector<torch::Tensor> local_tensors;
    std::vector<int64_t> local_sizes;
    local_tensors.reserve(patches_.size());
    local_sizes.reserve(patches_.size());

    for (std::size_t p = 0; p < patches_.size(); ++p) {
      local_tensors.push_back(patch(p).as_tensor());
      local_sizes.push_back(patch(p).ncumcoeffs());
    }

    torch::Tensor result = torch::empty(
        {geoDim() * dof_map_.ndofs}, local_tensors.front().options());

    for (int64_t global = 0; global < dof_map_.ndofs; ++global) {
      const auto [patch_index, local_index] =
          dof_map_.representatives[static_cast<std::size_t>(global)];
      const torch::Tensor &local =
          local_tensors[static_cast<std::size_t>(patch_index)];
      const int64_t nlocal = local_sizes[static_cast<std::size_t>(patch_index)];

      for (short_t g = 0; g < geoDim(); ++g)
        result.index_put_({g * dof_map_.ndofs + global},
                          local.index({g * nlocal + local_index}));
    }

    return result;
  }

  /// @brief Sets all patch-local coefficients from a global tensor
  MultiPatch &from_tensor(const torch::Tensor &tensor) {
    if (dof_map_.empty())
      throw std::runtime_error("MultiPatch DOF map has not been built");

    if (tensor.numel() != geoDim() * dof_map_.ndofs)
      throw std::runtime_error("Tensor size is not compatible with MultiPatch");

    for (std::size_t p = 0; p < patches_.size(); ++p) {
      const int64_t nlocal = patch(p).ncumcoeffs();
      torch::Tensor local =
          torch::empty({geoDim() * nlocal}, tensor.options());
      const torch::Tensor global_ids =
          torch::tensor(dof_map_.local_to_global[p],
                        torch::TensorOptions().dtype(torch::kInt64))
              .to(tensor.device());

      for (short_t g = 0; g < geoDim(); ++g) {
        const torch::Tensor source_ids = global_ids + g * dof_map_.ndofs;
        local.index_put_({torch::indexing::Slice(g * nlocal, (g + 1) * nlocal)},
                         tensor.index_select(0, source_ids));
      }

      patch(p).from_tensor(local);
    }

    return *this;
  }

  /// @brief Returns the size of the global tensor representation
  [[nodiscard]] int64_t as_tensor_size() const noexcept {
    return geoDim() * dof_map_.ndofs;
  }

  /// @brief Returns the patch-local tensor induced by a global tensor
  [[nodiscard]] torch::Tensor local_tensor(std::size_t patch_index,
                                           const torch::Tensor &tensor) const {
    if (dof_map_.empty())
      throw std::runtime_error("MultiPatch DOF map has not been built");

    validate_patch_index_(patch_index);

    if (tensor.numel() != geoDim() * dof_map_.ndofs)
      throw std::runtime_error("Tensor size is not compatible with MultiPatch");

    const int64_t nlocal = patch(patch_index).ncumcoeffs();
    torch::Tensor result = torch::empty({geoDim() * nlocal}, tensor.options());
    const torch::Tensor global_ids =
        torch::tensor(dof_map_.local_to_global[patch_index],
                      torch::TensorOptions().dtype(torch::kInt64))
            .to(tensor.device());

    for (short_t g = 0; g < geoDim(); ++g) {
      const torch::Tensor source_ids = global_ids + g * dof_map_.ndofs;
      result.index_put_({torch::indexing::Slice(g * nlocal, (g + 1) * nlocal)},
                        tensor.index_select(0, source_ids));
    }

    return result;
  }

  /// @brief Maps patch-local scalar coefficient indices to global DOF indices
  [[nodiscard]] torch::Tensor
  global_coeff_indices(std::size_t patch_index,
                       const torch::Tensor &local_indices) const {
    if (dof_map_.empty())
      throw std::runtime_error("MultiPatch DOF map has not been built");

    validate_patch_index_(patch_index);

    auto flat = local_indices.flatten().to(torch::kCPU).contiguous();
    torch::Tensor result =
        torch::empty({flat.numel()}, torch::TensorOptions().dtype(torch::kInt64));
    auto out = result.accessor<int64_t, 1>();

    for (int64_t i = 0; i < flat.numel(); ++i) {
      const int64_t local = flat.index({i}).template item<int64_t>();
      if (local < 0 ||
          local >= static_cast<int64_t>(dof_map_.local_to_global[patch_index].size()))
        throw std::runtime_error("Local coefficient index is out of range");
      out[i] = dof_map_.local_to_global[patch_index][local];
    }

    return result.view(local_indices.sizes()).to(local_indices.device());
  }

  /// @brief Evaluates one patch using a global coefficient tensor
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] auto
  eval_patch(std::size_t patch_index,
             const utils::TensorArray<parDim()> &xi,
             const torch::Tensor &tensor) const {
    Patch local = patch(patch_index);
    local.from_tensor(local_tensor(patch_index, tensor));
    return local.template eval<Deriv, memory_optimized>(xi);
  }

  /// @brief Evaluates one univariate patch using a global coefficient tensor
  template <deriv Deriv = deriv::func, bool memory_optimized = false>
  [[nodiscard]] auto eval_patch(std::size_t patch_index,
                                const torch::Tensor &xi,
                                const torch::Tensor &tensor) const {
    if constexpr (parDim() == 1) {
      return eval_patch<Deriv, memory_optimized>(
          patch_index, utils::TensorArray<parDim()>({xi}), tensor);
    } else {
      throw std::runtime_error("Invalid parametric dimension");
    }
  }

  /// @brief Returns patch-wise Greville point sets
  [[nodiscard]] std::vector<utils::TensorArray<parDim()>>
  patch_greville(bool interior = false) const {
    std::vector<utils::TensorArray<parDim()>> result;
    result.reserve(patches_.size());
    for (const auto &patch : patches_)
      result.push_back(patch->greville(interior));
    return result;
  }

  /// @brief Returns boundary Greville point sets, optionally filtered by label
  [[nodiscard]] std::vector<std::pair<boundary_type, utils::TensorArray<parDim()>>>
  boundary_greville(const std::string &label = "") const {
    std::vector<std::pair<boundary_type, utils::TensorArray<parDim()>>> result;
    for (const auto &boundary : boundaries_) {
      if (!label.empty() && boundary.label != label)
        continue;
      result.emplace_back(boundary,
                          side_greville_(boundary.patch, boundary.side));
    }
    return result;
  }

  /// @brief Returns Greville points on one patch side
  [[nodiscard]] utils::TensorArray<parDim()>
  side_greville(std::size_t patch_index, short_t side_) const {
    return side_greville_(patch_index, side_);
  }

  /// @brief Returns matching Greville point sets on both sides of one interface
  [[nodiscard]]
  std::pair<utils::TensorArray<parDim()>, utils::TensorArray<parDim()>>
  interface_greville(const interface_type &interface) const {
    auto xi1 = side_greville_(interface.patch1, interface.side1);
    auto xi2 = xi1;

    const short_t fixed1 = interface_type::side_direction(interface.side1);
    const short_t fixed2 = interface_type::side_direction(interface.side2);
    auto knots2 = patch(interface.patch2).knots(fixed2).to(torch::kCPU).contiguous();
    const double fixed_value2 =
        interface_type::side_parameter(interface.side2)
            ? knots2.index({knots2.numel() - 1}).template item<double>()
            : knots2.index({0}).template item<double>();

    xi2[fixed2] = torch::full_like(xi1[fixed1], fixed_value2);
    for (short_t d1 = 0; d1 < parDim(); ++d1) {
      if (d1 == fixed1)
        continue;

      const short_t d2 = interface.direction_map[d1];
      xi2[d2] = map_parametric_direction_(
          xi1[d1], patch(interface.patch1).knots(d1),
          patch(interface.patch2).knots(d2),
          interface.direction_orientation[d1]);
    }

    return {xi1, xi2};
  }

public:
  /// @brief Creates an isoparametric solution multi-patch space
  template <short_t ValueDim>
  [[nodiscard]] auto make_isoparametric_solution_space() const {
    if (patches_.empty())
      throw std::runtime_error(
          "Cannot create a solution space from an empty MultiPatch");

    return make_isoparametric_solution_space<ValueDim>(patch(0).options());
  }

  /// @brief Creates an isoparametric solution multi-patch space
  template <short_t ValueDim>
  [[nodiscard]] auto
  make_isoparametric_solution_space(Options<value_type> options) const {

    using SolutionPatch = detail::isoparametric_patch_t<Patch, ValueDim>;
    MultiPatch<SolutionPatch> result;
    result.set_matching_tolerance(abs_tol_, rel_tol_);

    for (std::size_t p = 0; p < patches_.size(); ++p) {
      result.addPatch(
          detail::make_isoparametric_patch<Patch, ValueDim>(patch(p),
                                                            options),
          patch_xml_ids_[p]);
    }

    for (const auto &interface : interfaces_) {
      typename MultiPatch<SolutionPatch>::interface_type other;
      other.patch1 = interface.patch1;
      other.side1 = interface.side1;
      other.patch2 = interface.patch2;
      other.side2 = interface.side2;
      other.direction_map = interface.direction_map;
      other.direction_orientation = interface.direction_orientation;
      other.label = interface.label;
      result.addInterface(other);
    }

    for (const auto &boundary : boundaries_)
      result.addBoundary(boundary);

    result.build_dof_map();
    return result;
  }

private:
  void invalidate_dof_map_() { dof_map_ = dof_map_type{}; }

  static constexpr bool valid_side_(short_t side_) noexcept {
    return side_ > side::none && side_ <= 2 * parDim();
  }

  void validate_patch_index_(std::size_t patch_index) const {
    if (patch_index >= patches_.size())
      throw std::runtime_error("Patch index is out of range");
  }

  void validate_patch_side_(std::size_t patch_index, short_t side_) const {
    validate_patch_index_(patch_index);
    if (!valid_side_(side_))
      throw std::runtime_error("Side index is not valid for patch dimension");
  }

  static int64_t linear_index_(const std::array<int64_t, parDim()> &indices,
                               const std::array<int64_t, parDim()> &sizes) {
    int64_t index = 0;
    int64_t stride = 1;
    for (short_t d = 0; d < parDim(); ++d) {
      index += indices[d] * stride;
      stride *= sizes[d];
    }
    return index;
  }

  static void next_interface_index_(
      std::array<int64_t, parDim()> &index,
      const std::array<int64_t, parDim()> &sizes, short_t fixed_dir,
      bool &done) {
    for (short_t d = 0; d < parDim(); ++d) {
      if (d == fixed_dir)
        continue;

      ++index[d];
      if (index[d] < sizes[d])
        return;

      index[d] = 0;
    }
    done = true;
  }

  std::vector<int64_t> side_indices_(std::size_t patch_index,
                                     short_t side_) const {
    const auto &p = patch(patch_index);
    const auto sizes = p.ncoeffs();
    const short_t fixed_dir = interface_type::side_direction(side_);

    std::array<int64_t, parDim()> index{};
    index.fill(0);
    index[fixed_dir] =
        interface_type::side_parameter(side_) ? sizes[fixed_dir] - 1 : 0;

    int64_t count = 1;
    for (short_t d = 0; d < parDim(); ++d)
      if (d != fixed_dir)
        count *= sizes[d];

    std::vector<int64_t> result;
    result.reserve(static_cast<std::size_t>(count));

    bool done = false;
    while (!done) {
      result.push_back(linear_index_(index, sizes));
      next_interface_index_(index, sizes, fixed_dir, done);
    }

    return result;
  }

  utils::TensorArray<parDim()> side_greville_(std::size_t patch_index,
                                              short_t side_) const {
    validate_patch_side_(patch_index, side_);

    const auto &p = patch(patch_index);
    auto points = p.greville(false);
    const short_t fixed_dir = interface_type::side_direction(side_);
    const bool upper = interface_type::side_parameter(side_);

    auto knots = p.knots(fixed_dir).to(torch::kCPU).contiguous();
    const double fixed_value =
        upper ? knots.index({knots.numel() - 1}).template item<double>()
              : knots.index({0}).template item<double>();

    auto mask = torch::abs(points[fixed_dir] - fixed_value) <=
                (abs_tol_ + rel_tol_ * std::abs(fixed_value));

    utils::TensorArray<parDim()> result;
    for (short_t d = 0; d < parDim(); ++d)
      result[d] = points[d].index({mask});

    return result;
  }

  std::vector<std::pair<int64_t, int64_t>>
  interface_index_pairs_(const interface_type &interface) const {
    const auto &p1 = patch(interface.patch1);
    const auto &p2 = patch(interface.patch2);
    const auto sizes1 = p1.ncoeffs();
    const auto sizes2 = p2.ncoeffs();
    const short_t fixed1 = interface_type::side_direction(interface.side1);

    std::array<int64_t, parDim()> index1{};
    index1.fill(0);
    index1[fixed1] =
        interface_type::side_parameter(interface.side1) ? sizes1[fixed1] - 1
                                                       : 0;

    int64_t count = 1;
    for (short_t d = 0; d < parDim(); ++d)
      if (d != fixed1)
        count *= sizes1[d];

    std::vector<std::pair<int64_t, int64_t>> result;
    result.reserve(static_cast<std::size_t>(count));

    bool done = false;
    while (!done) {
      std::array<int64_t, parDim()> index2{};
      index2.fill(0);

      for (short_t d = 0; d < parDim(); ++d) {
        const short_t mapped = interface.direction_map[d];
        if (d == fixed1) {
          index2[mapped] =
              interface_type::side_parameter(interface.side2)
                  ? sizes2[mapped] - 1
                  : 0;
        } else {
          index2[mapped] =
              interface.direction_orientation[d]
                  ? index1[d]
                  : sizes2[mapped] - 1 - index1[d];
        }
      }

      result.emplace_back(linear_index_(index1, sizes1),
                          linear_index_(index2, sizes2));
      next_interface_index_(index1, sizes1, fixed1, done);
    }

    return result;
  }

  static std::pair<double, double> knot_domain_(const torch::Tensor &knots) {
    auto cpu = knots.to(torch::kCPU).contiguous();
    if (cpu.numel() < 2)
      throw std::runtime_error("Knot vector must contain at least two entries");

    return {cpu.index({0}).template item<double>(),
            cpu.index({cpu.numel() - 1}).template item<double>()};
  }

  static double map_parametric_value_(double value, double source_min,
                                      double source_max, double target_min,
                                      double target_max,
                                      bool same_orientation) {
    const double source_length = source_max - source_min;
    if (std::abs(source_length) <= std::numeric_limits<double>::epsilon())
      throw std::runtime_error("Cannot map a degenerate parametric interval");

    double t = (value - source_min) / source_length;
    if (!same_orientation)
      t = 1.0 - t;

    return target_min + t * (target_max - target_min);
  }

  static torch::Tensor
  map_parametric_direction_(const torch::Tensor &xi,
                            const torch::Tensor &source_knots,
                            const torch::Tensor &target_knots,
                            bool same_orientation) {
    const auto [source_min, source_max] = knot_domain_(source_knots);
    const auto [target_min, target_max] = knot_domain_(target_knots);
    const double source_length = source_max - source_min;

    if (std::abs(source_length) <= std::numeric_limits<double>::epsilon())
      throw std::runtime_error("Cannot map a degenerate parametric interval");

    torch::Tensor t = (xi - source_min) / source_length;
    if (!same_orientation)
      t = 1.0 - t;

    return target_min + t * (target_max - target_min);
  }

  static bool close_(double lhs, double rhs, double abs_tol, double rel_tol) {
    return std::abs(lhs - rhs) <=
           abs_tol + rel_tol * std::max(std::abs(lhs), std::abs(rhs));
  }

  bool knot_vectors_match_(const torch::Tensor &lhs, const torch::Tensor &rhs,
                           bool same_orientation) const {
    auto lhs_cpu = lhs.to(torch::kCPU).contiguous();
    auto rhs_cpu = rhs.to(torch::kCPU).contiguous();

    if (lhs_cpu.numel() != rhs_cpu.numel())
      return false;

    const auto [lhs_min, lhs_max] = knot_domain_(lhs_cpu);
    const auto [rhs_min, rhs_max] = knot_domain_(rhs_cpu);

    for (int64_t i = 0; i < lhs_cpu.numel(); ++i) {
      const double lhs_value = lhs_cpu.index({i}).template item<double>();
      const int64_t rhs_index = same_orientation ? i : rhs_cpu.numel() - 1 - i;
      const double rhs_raw = rhs_cpu.index({rhs_index}).template item<double>();
      const double rhs_value = map_parametric_value_(
          rhs_raw, rhs_min, rhs_max, lhs_min, lhs_max, same_orientation);

      if (!close_(lhs_value, rhs_value, abs_tol_, rel_tol_))
        return false;
    }

    return true;
  }

  void validate_interface_basis_(const interface_type &interface) const {
    const auto &p1 = patch(interface.patch1);
    const auto &p2 = patch(interface.patch2);
    const auto sizes1 = p1.ncoeffs();
    const auto sizes2 = p2.ncoeffs();
    const short_t fixed1 = interface_type::side_direction(interface.side1);
    const short_t fixed2 = interface_type::side_direction(interface.side2);

    std::array<bool, parDim()> mapped{};
    mapped.fill(false);

    for (short_t d = 0; d < parDim(); ++d) {
      const short_t mapped_dir = interface.direction_map[d];
      if (mapped_dir < 0 || mapped_dir >= parDim())
        throw std::runtime_error("Interface direction map is out of range");
      if (mapped[mapped_dir])
        throw std::runtime_error("Interface direction map is not a permutation");
      mapped[mapped_dir] = true;
    }

    if (interface.direction_map[fixed1] != fixed2)
      throw std::runtime_error("Interface direction map does not map normal "
                               "direction to normal direction");

    for (short_t d = 0; d < parDim(); ++d) {
      const short_t mapped_dir = interface.direction_map[d];

      if (d == fixed1)
        continue;

      if (sizes1[d] != sizes2[mapped_dir])
        throw std::runtime_error("Interface control-point counts do not match");

      if (p1.degree(d) != p2.degree(mapped_dir))
        throw std::runtime_error("Interface spline degrees do not match");

      if (!knot_vectors_match_(p1.knots(d), p2.knots(mapped_dir),
                               interface.direction_orientation[d]))
        throw std::runtime_error("Interface knot vectors do not match");
    }
  }

  void validate_interface_geometry_(const interface_type &interface) const {
    const auto pairs = interface_index_pairs_(interface);
    const auto &p1 = patch(interface.patch1);
    const auto &p2 = patch(interface.patch2);

    for (const auto &[i1, i2] : pairs)
      for (short_t g = 0; g < geoDim(); ++g) {
        const double lhs =
            p1.coeffs(g).index({i1}).template item<double>();
        const double rhs =
            p2.coeffs(g).index({i2}).template item<double>();

        if (!close_(lhs, rhs, abs_tol_, rel_tol_))
          throw std::runtime_error(
              "Interface control points do not geometrically match");
      }
  }

  void validate_topology_() const {
    std::set<std::pair<std::size_t, short_t>> interface_sides;

    for (const auto &interface : interfaces_) {
      validate_patch_side_(interface.patch1, interface.side1);
      validate_patch_side_(interface.patch2, interface.side2);

      const auto key1 = std::make_pair(interface.patch1, interface.side1);
      const auto key2 = std::make_pair(interface.patch2, interface.side2);

      if (!interface_sides.insert(key1).second ||
          !interface_sides.insert(key2).second)
        throw std::runtime_error("A patch side is used by multiple interfaces");

      validate_interface_basis_(interface);
      validate_interface_geometry_(interface);
    }

    for (const auto &boundary : boundaries_) {
      validate_patch_side_(boundary.patch, boundary.side);
      if (interface_sides.contains(std::make_pair(boundary.patch, boundary.side)))
        throw std::runtime_error("A patch side is both interface and boundary");
    }
  }

  std::size_t total_local_dofs_() const {
    std::size_t total = 0;
    for (const auto &patch : patches_)
      total += static_cast<std::size_t>(patch->ncumcoeffs());
    return total;
  }

  void unite_interface_dofs_(const interface_type &interface,
                             const std::vector<int64_t> &patch_offsets,
                             detail::DisjointSet &dsu) const {
    for (const auto &[i1, i2] : interface_index_pairs_(interface))
      dsu.unite(static_cast<std::size_t>(patch_offsets[interface.patch1] + i1),
                static_cast<std::size_t>(patch_offsets[interface.patch2] + i2));
  }

  void normalize_topology_() {
    std::sort(boundaries_.begin(), boundaries_.end(),
              [](const auto &lhs, const auto &rhs) {
                return std::tie(lhs.patch, lhs.side, lhs.label) <
                       std::tie(rhs.patch, rhs.side, rhs.label);
              });
    boundaries_.erase(
        std::unique(boundaries_.begin(), boundaries_.end(),
                    [](const auto &lhs, const auto &rhs) {
                      return lhs.patch == rhs.patch && lhs.side == rhs.side &&
                             lhs.label == rhs.label;
                    }),
        boundaries_.end());

    std::sort(interfaces_.begin(), interfaces_.end(),
              [](const auto &lhs, const auto &rhs) {
                return std::tie(lhs.patch1, lhs.side1, lhs.patch2, lhs.side2,
                                lhs.label) <
                       std::tie(rhs.patch1, rhs.side1, rhs.patch2, rhs.side2,
                                rhs.label);
              });
    interfaces_.erase(
        std::unique(interfaces_.begin(), interfaces_.end(),
                    [](const auto &lhs, const auto &rhs) {
                      return lhs.patch1 == rhs.patch1 &&
                             lhs.side1 == rhs.side1 &&
                             lhs.patch2 == rhs.patch2 &&
                             lhs.side2 == rhs.side2 &&
                             lhs.direction_map == rhs.direction_map &&
                             lhs.direction_orientation ==
                                 rhs.direction_orientation &&
                             lhs.label == rhs.label;
                    }),
        interfaces_.end());
  }

  static std::vector<int> parse_ints_(const std::string &text) {
    std::string values =
        std::regex_replace(text, std::regex("[\t\r\n\a]+| +"), " ");
    std::stringstream ss(values);
    std::vector<int> result;
    int value = 0;

    while (ss >> value)
      result.push_back(value);

    return result;
  }

  void read_xml_patches_(const pugi::xml_node &root,
                         const pugi::xml_node &mp_node) {
    pugi::xml_node patches_node = mp_node.child("patches");
    if (!patches_node)
      throw std::runtime_error("XML MultiPatch object does not provide patches");

    const std::string type = patches_node.attribute("type").value();
    const auto ids = parse_ints_(patches_node.text().get());
    std::vector<int> patch_ids;

    if (type == "id_range") {
      if (ids.size() != 2)
        throw std::runtime_error("XML MultiPatch id_range needs two ids");

      for (int id = ids[0]; id <= ids[1]; ++id)
        patch_ids.push_back(id);
    } else if (type == "id_index") {
      patch_ids = ids;
    } else {
      throw std::runtime_error("Unsupported XML MultiPatch patches type");
    }

    for (std::size_t index = 0; index < patch_ids.size(); ++index) {
      const int xml_id = patch_ids[index];
      Patch patch;
      patch.from_xml(root, xml_id);
      addPatch(std::move(patch), xml_id);
      xml_id_to_patch_[xml_id] = index;
    }
  }

  std::size_t patch_index_from_xml_id_(int xml_id) const {
    auto it = xml_id_to_patch_.find(xml_id);
    if (it == xml_id_to_patch_.end())
      throw std::runtime_error("XML topology references unknown patch id");
    return it->second;
  }

  void read_xml_boundaries_(const pugi::xml_node &mp_node) {
    for (pugi::xml_node boundary_node : mp_node.children("boundary")) {
      const std::string label = boundary_node.attribute("name").value();
      const auto values = parse_ints_(boundary_node.text().get());

      if (values.size() % 2 != 0)
        throw std::runtime_error("XML boundary data must contain patch/side pairs");

      for (std::size_t i = 0; i < values.size(); i += 2)
        addBoundary(boundary_type{patch_index_from_xml_id_(values[i]),
                                  static_cast<short_t>(values[i + 1]), label});
    }
  }

  void read_xml_interfaces_(const pugi::xml_node &mp_node) {
    for (pugi::xml_node interface_node : mp_node.children("interfaces")) {
      const std::string type = interface_node.attribute("type").value();
      if (!type.empty() && type != "conforming")
        throw std::runtime_error("Only conforming interfaces are supported");

      const std::string label = interface_node.attribute("name").value();
      const auto values = parse_ints_(interface_node.text().get());
      const std::size_t row_size = 4 + 2 * parDim();

      if (values.size() % row_size != 0)
        throw std::runtime_error("XML interface data has invalid row size");

      for (std::size_t row = 0; row < values.size(); row += row_size) {
        interface_type interface;
        interface.patch1 = patch_index_from_xml_id_(values[row]);
        interface.side1 = static_cast<short_t>(values[row + 1]);
        interface.patch2 = patch_index_from_xml_id_(values[row + 2]);
        interface.side2 = static_cast<short_t>(values[row + 3]);
        interface.label = label;

        for (short_t d = 0; d < parDim(); ++d)
          interface.direction_map[d] =
              static_cast<short_t>(values[row + 4 + d]);

        for (short_t d = 0; d < parDim(); ++d)
          interface.direction_orientation[d] =
              values[row + 4 + parDim() + d] != 0;

        const short_t fixed1 =
            interface_type::side_direction(interface.side1);
        interface.direction_map[fixed1] =
            interface_type::side_direction(interface.side2);
        interface.direction_orientation[fixed1] =
            interface_type::side_parameter(interface.side1) ==
            interface_type::side_parameter(interface.side2);

        addInterface(interface);
      }
    }
  }

private:
  /// @brief Vector of single-patch objects
  std::vector<std::shared_ptr<Patch>> patches_;

  /// @brief External XML patch ids; -1 denotes unknown id
  std::vector<int> patch_xml_ids_;

  /// @brief Map from external XML patch id to internal patch index
  std::map<int, std::size_t> xml_id_to_patch_;

  /// @brief Interfaces between patch sides
  std::vector<interface_type> interfaces_;

  /// @brief Outer boundary sides
  std::vector<boundary_type> boundaries_;

  /// @brief Global scalar control-point DOF map
  dof_map_type dof_map_;

  /// @brief Geometry matching tolerances
  double abs_tol_{1e-6};
  double rel_tol_{1e-6};
};

template <typename Patch>
inline std::ostream &operator<<(std::ostream &os,
                                const MultiPatch<Patch> &obj) {
  os << "MultiPatch(npatches=" << obj.npatches()
     << ", ninterfaces=" << obj.ninterfaces()
     << ", nboundaries=" << obj.nboundaries()
     << ", ndofs=" << obj.ndofs() << ")";
  return os;
}

} // namespace iganet
