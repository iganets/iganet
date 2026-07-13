/**
   @file net/collocation.hpp

   @brief Collocation point helpers

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace iganet {

//  clang-format off
/// @brief Enumerator for the collocation point specifier
enum class collPts : short_t {
  greville               = 0, /*!< Greville points */
  greville_interior      = 1, /*!< Greville points in the interior */
  greville_ref1          = 2, /*!< Greville points, once refined */
  greville_interior_ref1 = 3, /*!< Greville points in the interior, once refined */
  greville_ref2          = 4, /*!< Greville points, twice refined */
  greville_interior_ref2 = 5, /*!< Greville points in the interior, twice refined */
  greville_ref3          = 6, /*!< Greville points, three times refined */
  greville_interior_ref3 = 7, /*!< Greville points in the interior, three times refined */
};
//  clang-format on

template <typename EvalType, typename BoundaryEvalType,
          typename InterfaceEvalType = std::monostate>
struct CollocationData {
  using interior_type = EvalType;
  using boundary_type = BoundaryEvalType;
  using interface_type = InterfaceEvalType;

  EvalType first;
  BoundaryEvalType second;
  std::vector<InterfaceEvalType> interfaces;

  [[nodiscard]] auto &interior() noexcept { return first; }
  [[nodiscard]] const auto &interior() const noexcept { return first; }

  [[nodiscard]] auto &boundary() noexcept { return second; }
  [[nodiscard]] const auto &boundary() const noexcept { return second; }

  [[nodiscard]] bool has_interfaces() const noexcept {
    return !interfaces.empty();
  }
};

namespace detail {

template <typename T>
concept HasBoundaryGrevilleLabel = requires(const T t, const std::string &label) {
  { t.boundary_greville(label) };
};

template <typename T>
concept HasBoundaryGreville = requires(const T t) { { t.boundary_greville() }; };

template <typename T>
concept HasInterfaces = requires(const T t) { { t.interfaces() }; };

template <typename T>
concept HasInterfaceGreville =
    requires(const T t, const typename T::interface_type &interface) {
      { t.interface_greville(interface) };
    };

template <typename T>
concept HasPatchGreville = requires(const T t, bool interior) {
  { t.patch_greville(interior) };
};

template <typename T>
concept HasNSpaces = requires { T::nspaces(); };

template <typename T>
using interface_eval_t =
    std::pair<typename T::interface_type,
              decltype(std::declval<const T>().interface_greville(
                  std::declval<const typename T::interface_type &>()))>;

template <typename T>
using interior_eval_t = decltype(std::declval<const T>().patch_greville(true));

template <typename T>
using boundary_eval_auto_t = decltype(std::declval<const T>().boundary_greville());

template <typename T, bool HasPatch = HasPatchGreville<T>,
          bool HasInterface =
              HasInterfaces<T> && HasInterfaceGreville<T> && HasBoundaryGreville<T>>
struct collocation_type_selector {
  using type = CollocationData<typename T::eval_type, typename T::boundary_eval_type>;
};

template <typename T>
struct collocation_type_selector<T, true, false> {
  using type = CollocationData<interior_eval_t<T>, typename T::boundary_eval_type>;
};

template <typename T>
struct collocation_type_selector<T, true, true> {
  using type = CollocationData<interior_eval_t<T>, boundary_eval_auto_t<T>,
                               interface_eval_t<T>>;
};

template <typename T>
using collocation_type_t = typename collocation_type_selector<T>::type;

template <typename T>
inline auto refined_space(const T &space, int refinement) {
  if (refinement <= 0)
    return space.clone();
  return space.clone().uniform_refine(refinement, -1);
}

template <HasPatchGreville T>
inline const T &refined_space(const T &space, int refinement) {
  if (refinement != 0)
    throw std::runtime_error(
        "Refined collocation is not implemented for MultiPatch collocation yet");
  return space;
}

template <typename T>
inline auto refined_boundary_space(const T &space, int refinement) {
  if (refinement <= 0)
    return space.clone();
  return space.clone().uniform_refine(refinement, -1);
}

template <typename T>
inline auto refined_space_eval(const T &space, bool interior, int refinement) {
  if constexpr (HasPatchGreville<T>) {
    return refined_space(space, refinement).greville(interior);
  } else {
    return refined_space(space.space(), refinement).greville(interior);
  }
}

template <typename T>
inline auto boundary_eval(const T &space, int refinement) {
  if constexpr (HasBoundaryGreville<T>)
    return space.boundary_greville();
  else
    return refined_boundary_space(space.boundary(), refinement).greville();
}

template <typename T>
inline auto interface_eval(const T &space) {
  using result_t = std::vector<interface_eval_t<T>>;
  result_t result;

  if constexpr (HasInterfaces<T> && HasInterfaceGreville<T>) {
    result.reserve(space.ninterfaces());
    for (const auto &interface : space.interfaces())
      result.emplace_back(interface, space.interface_greville(interface));
  }

  return result;
}

template <typename T>
inline int refinement_level(enum collPts collPts) {
  switch (collPts) {
  case collPts::greville:
  case collPts::greville_interior:
    return 0;
  case collPts::greville_ref1:
  case collPts::greville_interior_ref1:
    return 1;
  case collPts::greville_ref2:
  case collPts::greville_interior_ref2:
    return 2;
  case collPts::greville_ref3:
  case collPts::greville_interior_ref3:
    return 3;
  default:
    throw std::runtime_error("Invalid collocation point specifier");
  }
}

inline bool interior_only(enum collPts collPts) {
  switch (collPts) {
  case collPts::greville_interior:
  case collPts::greville_interior_ref1:
  case collPts::greville_interior_ref2:
  case collPts::greville_interior_ref3:
    return true;
  case collPts::greville:
  case collPts::greville_ref1:
  case collPts::greville_ref2:
  case collPts::greville_ref3:
    return false;
  default:
    throw std::runtime_error("Invalid collocation point specifier");
  }
}

template <typename T>
inline collocation_type_t<T> make_collocation_data(enum collPts collPts,
                                                   const T &space) {
  collocation_type_t<T> result;
  const int refinement = refinement_level<T>(collPts);
  const bool interior = interior_only(collPts);

  if constexpr (HasPatchGreville<T>) {
    result.first = refined_space(space, refinement).patch_greville(interior);
    result.second = boundary_eval(space, refinement);
  } else {
    result.first = refined_space(space.space(), refinement).greville(interior);
    result.second = boundary_eval(space, refinement);
  }

  if constexpr (HasInterfaces<T> && HasInterfaceGreville<T>)
    result.interfaces = interface_eval(space);

  return result;
}

} // namespace detail

/// @brief Collocation points helper
/// @{
template <typename> class CollPtsHelper;

template <detail::HasAsTensor CollPts>
class CollPtsHelper<CollPts> {
public:
  using type = detail::collocation_type_t<CollPts>;

private:
  template <typename FunctionSpace, std::size_t... Is>
  static auto collPts_impl(enum collPts collPts, const FunctionSpace &space,
                           std::index_sequence<Is...>) {
    type result;
    const int refinement = detail::refinement_level<FunctionSpace>(collPts);
    const bool interior = detail::interior_only(collPts);

    ((std::get<Is>(result.first) =
          detail::refined_space(space.template space<Is>(), refinement)
              .greville(interior)),
     ...);

    ((std::get<Is>(result.second) = detail::refined_boundary_space(
                                       space.template boundary<Is>().clone(),
                                       refinement)
                                       .greville()),
     ...);

    return result;
  }

public:
  template <typename FunctionSpace>
  static auto collPts(enum collPts collPts, const FunctionSpace &space) {
    if constexpr (!detail::HasNSpaces<FunctionSpace>) {
      return detail::make_collocation_data(collPts, space);
    } else if constexpr (FunctionSpace::nspaces() == 1) {
      return detail::make_collocation_data(collPts, space);
    } else {
      return collPts_impl(collPts, space,
                          std::make_index_sequence<FunctionSpace::nspaces()>{});
    }
  }
};
/// @}

} // namespace iganet
