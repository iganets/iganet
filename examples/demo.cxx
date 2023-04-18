/**
   @file examples/demo.cxx

   @brief Demonstrator application

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <iostream>

/// @brief IgANet for Poisson's equation
template<typename optimizer_t,
         typename geometry_t,
         typename variable_t>
class poisson : public iganet::IgANet<optimizer_t, geometry_t, variable_t>
{
private:
#if 0
    /// @brief Type of the knot indices of geometry_t type
    using geometry_knot_indices_t =
      decltype(std::declval<geometry_t>().find_knot_indices(std::declval<typename geometry_t::eval_t>()));
    
    /// @brief Type of the knot indices of variable_t type
    using variable_knot_indices_t =
      decltype(std::declval<variable_t>().find_knot_indices(std::declval<typename variable_t::eval_t>()));

    /// @brief Type of the knot indices of boundary_t type
    using boundary_knot_indices_t =
      decltype(std::declval<variable_t>().template find_knot_indices<functionspace::boundary>(std::declval<typename variable_t::boundary_eval_t>()));

    /// @brief Type of the basis functions of geometry_t type
    using geometry_basfunc_t =
      decltype(std::declval<geometry_t>().eval_basfunc(std::declval<typename geometry_t::eval_t>()));
    
    /// @brief Type of the basis functions of variable_t type
    using variable_basfunc_t =
      decltype(std::declval<variable_t>().eval_basfunc(std::declval<typename variable_t::eval_t>()));

    /// @brief Type of the basis functions of boundary_t type
    using boundary_basfunc_t =
      decltype(std::declval<variable_t>().template eval_basfunc<functionspace::boundary>(std::declval<typename variable_t::boundary_eval_t>()));

    /// @brief Type of the coefficient indices of geometry_t type
    using geometry_coeff_indices_t =
      decltype(std::declval<geometry_t>().find_coeff_indices(std::declval<typename geometry_t::eval_t>()));
    
    /// @brief Type of the coefficient indices of variable_t type
    using variable_coeff_indices_t =
      decltype(std::declval<variable_t>().find_coeff_indices(std::declval<typename variable_t::eval_t>()));

    /// @brief Type of the coefficient indices of boundary_t type
    using boundary_coeff_indices_t =
      decltype(std::declval<variable_t>().template find_coeff_indices<functionspace::boundary>(std::declval<typename variable_t::boundary_eval_t>()));

    /// @brief Knot indices of geometry_t type
    geometry_knot_indices_t geo_knot_indices_;

    /// @brief Knot indices of variable_t type
    variable_knot_indices_t ref_knot_indices_;

    /// @brief Knot indices of boundary_t type
    boundary_knot_indices_t bdr_knot_indices_;

    /// @brief Basis functions of geometry_t type
    geometry_basfunc_t geo_basfunc_;

    /// @brief Basis functions of variable_t type
    variable_basfunc_t ref_basfunc_;

    /// @brief Basis functions of boundary_t type
    boundary_basfunc_t bdr_basfunc_;

    /// @brief Coefficient indices of geometry_t type
    geometry_coeff_indices_t geo_coeff_indices_;

    /// @brief Coefficient indices of variable_t type
    variable_coeff_indices_t ref_coeff_indices_;

    /// @brief Coefficient indices of boundary_t type
    boundary_coeff_indices_t bdr_coeff_indices_;
#endif  
public:
  using iganet::IgANet<optimizer_t, geometry_t, variable_t>::IgANet;

  iganet::status get_epoch(int64_t epoch) const override
  {
    std::cout << "Epoch " << std::to_string(epoch) << ": ";
    return iganet::status(0);
  }

#if 0
   // Evaluate constant right-hand side
      auto knot_indices  = ref_.template find_knot_indices<functionspace::interior>(samples.first);
      auto basfunc       = ref_.template eval_basfunc<functionspace::interior>(samples.first, knot_indices);
      auto coeff_indices = ref_.template find_coeff_indices<functionspace::interior>(knot_indices);

      auto rhs = ref_.eval_from_precomputed(basfunc, coeff_indices, samples.first);

      // Evaluate boundary values
      auto bdr_knot_indices  = ref_.template find_knot_indices<functionspace::boundary>(samples.second);
      auto bdr_basfunc       = ref_.template eval_basfunc<functionspace::boundary>(samples.second, bdr_knot_indices);
      auto bdr_coeff_indices = ref_.template find_coeff_indices<functionspace::boundary>(bdr_knot_indices);
            
      auto rhs_bdr = ref_.template eval_from_precomputed<functionspace::boundary>(bdr_basfunc, bdr_coeff_indices, samples.second);
#endif
};

int main()
{
  iganet::init();
  iganet::verbose(std::cout);

  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = double;
  
  using geometry_t  = iganet::S2<iganet::UniformBSpline<real_t, 2, 3, 2>>;
  using variable_t  = iganet::S2<iganet::UniformBSpline<real_t, 2, 3, 2>>;
  
  poisson<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                   {100,100},
                                                   // Activation functions
                                                   {
                                                     {iganet::activation::relu},
                                                     {iganet::activation::relu},
                                                     {iganet::activation::none}
                                                   },
                                                   // Number of B-spline coefficients
                                                   std::tuple(iganet::to_array(5_i64, 4_i64)));

  net.options().max_epoch(1000);
  net.options().min_loss(1e-8);
  
  net.train();
  
  return 0;
}
