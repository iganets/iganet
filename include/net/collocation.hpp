/**
   @file net/collocation.hpp

   @brief Isogeometric analysis base class

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

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

/// @brief Collocation points helper
/// @{  
template<typename> class CollPtsHelper;

template <detail::HasAsTensor CollPts>  
class CollPtsHelper<CollPts> {

public:
  /// @brief Type of the collocation points
  using type = std::pair<typename CollPts::eval_type,
                         typename CollPts::boundary_eval_type>;

private:
  /// @brief Returns the collocation points of the index-th function space
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <typename FunctionSpace, std::size_t... Is>
  static auto collPts(enum collPts collPts, const FunctionSpace& space, std::index_sequence<Is...>) {

    type collPts_;

    switch (collPts) {

    case collPts::greville:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) =
            space.template space<Is>().greville(
                /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) =
            space.template boundary<Is>().greville()),
       ...);
      break;

    case collPts::greville_interior:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts_.first) =
            space.template space<Is>().greville(
                /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) =
            space.template boundary<Is>().greville()),
       ...);
      break;

    case collPts::greville_ref1:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) = space
                                          .template space<Is>()
                                          .clone()
                                          .uniform_refine()
                                          .greville(
                                              /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = space
                                           .template boundary<Is>()
                                           .clone()
                                           .uniform_refine()
                                           .greville()),
       ...);
      break;

    case collPts::greville_interior_ref1:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts_.first) = space
                                          .template space<Is>()
                                          .clone()
                                          .uniform_refine()
                                          .greville(
                                              /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = space
                                           .template boundary<Is>()
                                           .clone()
                                           .uniform_refine()
                                           .greville()),
       ...);
      break;

    case collPts::greville_ref2:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) = space
                                          .template space<Is>()
                                          .clone()
                                          .uniform_refine(2, -1)
                                          .greville(
                                              /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = space
                                           .template boundary<Is>()
                                           .clone()
                                           .uniform_refine(2, -1)
                                           .greville()),
       ...);
      break;

    case collPts::greville_interior_ref2:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts_.first) = space
                                          .template space<Is>()
                                          .clone()
                                          .uniform_refine(2, -1)
                                          .greville(
                                              /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = space
                                           .template boundary<Is>()
                                           .clone()
                                           .uniform_refine(2, -1)
                                           .greville()),
       ...);
      break;
      
    case collPts::greville_ref3:
      // Get Greville abscissae inside the domain and at the boundary
      ((std::get<Is>(collPts_.first) = space
                                          .template space<Is>()
                                          .clone()
                                          .uniform_refine(3, -1)
                                          .greville(
                                              /* interior */ false)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = space
                                           .template boundary<Is>()
                                           .clone()
                                           .uniform_refine(3, -1)
                                           .greville()),
       ...);
      break;

    case collPts::greville_interior_ref3:
      // Get Greville abscissae inside the domain
      ((std::get<Is>(collPts_.first) = space
                                          .template space<Is>()
                                          .clone()
                                          .uniform_refine(3, -1)
                                          .greville(
                                              /* interior */ true)),
       ...);

      // Get Greville abscissae at the domain
      ((std::get<Is>(collPts_.second) = space
                                           .template boundary<Is>()
                                           .clone()
                                           .uniform_refine(3, -1)
                                           .greville()),
       ...);
      break;      

    default:
      throw std::runtime_error("Invalid collocation point specifier");
    }

    return collPts_;
  }

public:
  /// @brief Returns the collocation points of the index-th function spaces
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template<typename FunctionSpace>
  static auto collPts(enum collPts collPts, const FunctionSpace& space) {
    if constexpr (FunctionSpace::nspaces() == 1)

      switch (collPts) {

      case collPts::greville:
        return type{
            space.space().greville(/* interior */ false),
            space.boundary().greville()};

      case collPts::greville_interior:
        return type{space.space().greville(/* interior */ true),
                space.boundary().greville()};

      case collPts::greville_ref1:
        return type{
            space.space().clone().uniform_refine().greville(
                /* interior */ false),
            space
                .boundary()
                .clone()
                .uniform_refine()
                .greville()};

      case collPts::greville_interior_ref1:
        return type{
            space.space().clone().uniform_refine().greville(
                /* interior */ true),
            space
                .boundary()
                .clone()
                .uniform_refine()
                .greville()};

      case collPts::greville_ref2:
        return type{space
                    .space()
                    .clone()
                    .uniform_refine(2, -1)
                    .greville(
                        /* interior */ false),
                space
                    .boundary()
                    .clone()
                    .uniform_refine(2, -1)
                    .greville()};

      case collPts::greville_interior_ref2:
        return type{space
                    .space()
                    .clone()
                    .uniform_refine(2, -1)
                    .greville(
                        /* interior */ true),
                space
                    .boundary()
                    .clone()
                    .uniform_refine(2, -1)
                .greville()};

      case collPts::greville_ref3:
        return type{space
                    .space()
                    .clone()
                    .uniform_refine(3, -1)
                    .greville(
                        /* interior */ false),
                space
                    .boundary()
                    .clone()
                    .uniform_refine(3, -1)
                    .greville()};

      case collPts::greville_interior_ref3:
        return type{space
                    .space()
                    .clone()
                    .uniform_refine(3, -1)
                    .greville(
                        /* interior */ true),
                space
                    .boundary()
                    .clone()
                    .uniform_refine(3, -1)
                    .greville()};        

      default:
        throw std::runtime_error("Invalid collocation point specifier");
      }

    else
      return collPts(collPts, space, std::make_index_sequence<type::nspaces()>{});
  }
};
/// @}
  
} // namespace iganet
