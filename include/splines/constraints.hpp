/**
   @file splines/constraints.hpp

   @brief Strong tensor constraints for spline and multi-patch spaces

   @author Guenther Obermair

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <splines/functionspace.hpp>
#include <splines/multipatch.hpp>

namespace iganet {

namespace detail {

template <std::size_t ParDim>
inline int64_t constraint_linear_index(
    const std::array<int64_t, ParDim> &indices,
    const std::array<int64_t, ParDim> &sizes) {
  int64_t index = 0;
  int64_t stride = 1;
  for (short_t d = 0; d < ParDim; ++d) {
    index += indices[d] * stride;
    stride *= sizes[d];
  }
  return index;
}

template <typename Patch>
inline std::vector<int64_t> single_patch_boundary_dofs(const Patch &patch,
                                                       short_t side_) {
  constexpr short_t ParDim = Patch::parDim();
  if (side_ <= side::none || side_ > 2 * ParDim)
    throw std::runtime_error("Side index is not valid for patch dimension");

  const auto sizes = patch.ncoeffs();
  const short_t fixed_dir =
      MultiPatchInterface<ParDim>::side_direction(side_);

  std::array<int64_t, ParDim> index{};
  index.fill(0);
  index[fixed_dir] =
      MultiPatchInterface<ParDim>::side_parameter(side_) ? sizes[fixed_dir] - 1
                                                        : 0;

  int64_t count = 1;
  for (short_t d = 0; d < ParDim; ++d)
    if (d != fixed_dir)
      count *= sizes[d];

  std::vector<int64_t> result;
  result.reserve(static_cast<std::size_t>(count));

  bool done = false;
  while (!done) {
    result.push_back(constraint_linear_index(index, sizes));

    bool advanced = false;
    for (short_t d = 0; d < ParDim; ++d) {
      if (d == fixed_dir)
        continue;

      ++index[d];
      if (index[d] < sizes[d]) {
        advanced = true;
        break;
      }

      index[d] = 0;
    }

    if (!advanced)
      done = true;
  }

  return result;
}

template <typename Space>
inline int64_t scalar_dofs(const Space &space) {
  if constexpr (requires { space.ndofs(); }) {
    return space.ndofs();
  } else if constexpr (iganet::FunctionSpaceType<Space>) {
    static_assert(Space::nspaces() == 1,
                  "Strong constraints currently support single-space "
                  "FunctionSpace objects");
    return space.template space<0>().ncumcoeffs();
  } else {
    return space.ncumcoeffs();
  }
}

template <typename Space> inline short_t value_dim(const Space &) {
  if constexpr (iganet::FunctionSpaceType<Space>) {
    static_assert(Space::nspaces() == 1,
                  "Strong constraints currently support single-space "
                  "FunctionSpace objects");
    return Space::template geoDim<0>();
  } else {
    return Space::geoDim();
  }
}

template <typename Space>
inline std::vector<int64_t> boundary_dofs(const Space &space, short_t side_) {
  if constexpr (iganet::FunctionSpaceType<Space>) {
    static_assert(Space::nspaces() == 1,
                  "Strong constraints currently support single-space "
                  "FunctionSpace objects");
    return single_patch_boundary_dofs(space.template space<0>(), side_);
  } else {
    return single_patch_boundary_dofs(space, side_);
  }
}

inline torch::Tensor int64_tensor(const std::vector<int64_t> &values,
                                  torch::Device device) {
  auto tensor = torch::empty(
      {static_cast<int64_t>(values.size())},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto accessor = tensor.accessor<int64_t, 1>();
  for (int64_t i = 0; i < tensor.numel(); ++i)
    accessor[i] = values[static_cast<std::size_t>(i)];
  return tensor.to(device);
}

template <typename value_type>
inline torch::Tensor value_tensor(const std::vector<value_type> &values,
                                  torch::TensorOptions options) {
  auto tensor = torch::empty(
      {static_cast<int64_t>(values.size())},
      options.device(torch::kCPU));
  auto accessor = tensor.template accessor<value_type, 1>();
  for (int64_t i = 0; i < tensor.numel(); ++i)
    accessor[i] = values[static_cast<std::size_t>(i)];
  return tensor.to(options.device());
}

} // namespace detail

/// @brief Strong Dirichlet constraints for component-block tensor layouts
///
/// The constrained tensor layout is assumed to be
/// `[component0 scalar dofs, component1 scalar dofs, ...]`, matching spline,
/// function-space, and multi-patch `as_tensor()` representations.
template <typename real_t = double> class StrongDirichletConstraints {
public:
  using value_type = real_t;

public:
  StrongDirichletConstraints() = default;

  StrongDirichletConstraints(int64_t scalar_dofs, short_t value_dim,
                             double conflict_tol = 1e-10)
      : scalar_dofs_(scalar_dofs), value_dim_(value_dim),
        tensor_size_(scalar_dofs * value_dim), conflict_tol_(conflict_tol) {
    if (scalar_dofs_ < 0)
      throw std::runtime_error("Number of scalar DOFs must be non-negative");
    if (value_dim_ <= 0)
      throw std::runtime_error("Value dimension must be positive");
  }

  template <typename Space>
  explicit StrongDirichletConstraints(const Space &space,
                                      double conflict_tol = 1e-10)
      : StrongDirichletConstraints(detail::scalar_dofs(space),
                                   detail::value_dim(space), conflict_tol) {
    if (space.as_tensor_size() != tensor_size_)
      throw std::runtime_error(
          "Space tensor layout is not compatible with strong constraints");
  }

  [[nodiscard]] int64_t scalar_dofs() const noexcept { return scalar_dofs_; }
  [[nodiscard]] short_t value_dim() const noexcept { return value_dim_; }
  [[nodiscard]] int64_t tensor_size() const noexcept { return tensor_size_; }
  [[nodiscard]] std::size_t nfixed() const noexcept { return fixed_.size(); }
  [[nodiscard]] int64_t nfree() const noexcept {
    return tensor_size_ - static_cast<int64_t>(fixed_.size());
  }
  [[nodiscard]] bool empty() const noexcept { return fixed_.empty(); }

  /// @brief Adds one fixed tensor index
  StrongDirichletConstraints &fix_tensor_index(int64_t tensor_index,
                                               value_type value) {
    if (tensor_index < 0 || tensor_index >= tensor_size_)
      throw std::runtime_error("Constraint tensor index is out of range");

    auto [it, inserted] = fixed_.emplace(tensor_index, value);
    if (!inserted &&
        std::abs(static_cast<double>(it->second - value)) > conflict_tol_)
      throw std::runtime_error(
          "Conflicting strong Dirichlet values for the same DOF");

    return *this;
  }

  /// @brief Adds fixed scalar DOFs for one component
  StrongDirichletConstraints &
  fix_component_dofs(short_t component, const std::vector<int64_t> &dofs,
                     value_type value) {
    if (component < 0 || component >= value_dim_)
      throw std::runtime_error("Constraint component is out of range");

    for (const auto dof : dofs) {
      if (dof < 0 || dof >= scalar_dofs_)
        throw std::runtime_error("Constraint scalar DOF is out of range");
      fix_tensor_index(component * scalar_dofs_ + dof, value);
    }

    return *this;
  }

  /// @brief Adds constant boundary constraints for a single patch or FunctionSpace
  template <typename Space>
  StrongDirichletConstraints &fix_boundary(const Space &space, short_t side_,
                                           short_t component,
                                           value_type value) {
    if constexpr (requires { space.ndofs(); }) {
      static_assert(!requires { space.ndofs(); },
                    "Use fix_boundary(multipatch, patch, side, component, "
                    "value) or fix_boundary_label for MultiPatch objects");
    } else {
      return fix_component_dofs(component,
                                detail::boundary_dofs(space, side_), value);
    }
  }

  /// @brief Adds constant boundary constraints for one MultiPatch side
  template <typename Patch>
  StrongDirichletConstraints &
  fix_boundary(const MultiPatch<Patch> &space, std::size_t patch_index,
               short_t side_, short_t component, value_type value) {
    return fix_component_dofs(component,
                              space.boundary_global_dofs(patch_index, side_),
                              value);
  }

  /// @brief Adds constant boundary constraints for one MultiPatch boundary
  template <typename Patch>
  StrongDirichletConstraints &
  fix_boundary(const MultiPatch<Patch> &space,
               const typename MultiPatch<Patch>::boundary_type &boundary,
               short_t component, value_type value) {
    return fix_component_dofs(component, space.boundary_global_dofs(boundary),
                              value);
  }

  /// @brief Adds constant boundary constraints for all MultiPatch boundaries
  /// with a given label
  template <typename Patch>
  StrongDirichletConstraints &
  fix_boundary_label(const MultiPatch<Patch> &space, const std::string &label,
                     short_t component, value_type value) {
    return fix_component_dofs(component, space.boundary_global_dofs(label),
                              value);
  }

  [[nodiscard]] std::vector<int64_t> fixed_indices() const {
    std::vector<int64_t> result;
    result.reserve(fixed_.size());
    for (const auto &[index, value] : fixed_)
      result.push_back(index);
    return result;
  }

  [[nodiscard]] std::vector<value_type> fixed_values() const {
    std::vector<value_type> result;
    result.reserve(fixed_.size());
    for (const auto &[index, value] : fixed_)
      result.push_back(value);
    return result;
  }

  [[nodiscard]] std::vector<int64_t> free_indices() const {
    std::vector<int64_t> result;
    result.reserve(static_cast<std::size_t>(nfree()));

    auto fixed_it = fixed_.begin();
    for (int64_t i = 0; i < tensor_size_; ++i) {
      if (fixed_it != fixed_.end() && fixed_it->first == i) {
        ++fixed_it;
      } else {
        result.push_back(i);
      }
    }

    return result;
  }

  /// @brief Returns a tensor containing only unconstrained values
  [[nodiscard]] torch::Tensor extract_free_tensor(const torch::Tensor &full) const {
    if (full.numel() != tensor_size_)
      throw std::runtime_error("Full tensor size is incompatible with constraints");

    const auto indices = detail::int64_tensor(free_indices(), full.device());
    return full.flatten().index_select(0, indices);
  }

  /// @brief Expands unconstrained values to a full tensor with fixed values
  [[nodiscard]] torch::Tensor expand_free_tensor(const torch::Tensor &free) const {
    if (free.numel() != nfree())
      throw std::runtime_error("Free tensor size is incompatible with constraints");

    auto full = torch::zeros({tensor_size_}, free.options());
    const auto free_idx = detail::int64_tensor(free_indices(), free.device());
    if (free_idx.numel() > 0)
      full.index_put_({free_idx}, free.flatten());

    const auto fixed_idx = detail::int64_tensor(fixed_indices(), free.device());
    if (fixed_idx.numel() > 0) {
      const auto fixed_val =
          detail::value_tensor(fixed_values(), free.options());
      full.index_put_({fixed_idx}, fixed_val);
    }

    return full;
  }

  /// @brief Returns a copy of the full tensor with fixed values imposed
  [[nodiscard]] torch::Tensor apply(const torch::Tensor &full) const {
    if (full.numel() != tensor_size_)
      throw std::runtime_error("Full tensor size is incompatible with constraints");

    auto result = full.flatten().clone();
    const auto fixed_idx = detail::int64_tensor(fixed_indices(), full.device());
    if (fixed_idx.numel() > 0) {
      const auto fixed_val =
          detail::value_tensor(fixed_values(), full.options());
      result.index_put_({fixed_idx}, fixed_val);
    }
    return result.view(full.sizes());
  }

private:
  int64_t scalar_dofs_{0};
  short_t value_dim_{0};
  int64_t tensor_size_{0};
  double conflict_tol_{1e-10};
  std::map<int64_t, value_type> fixed_;
};

template <typename Space>
StrongDirichletConstraints(const Space &, double = 1e-10)
    -> StrongDirichletConstraints<typename Space::value_type>;

} // namespace iganet
