/**
   @file examples/iganet.cxx

   @brief Demonstration of the IgANet class

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
public:
  using iganet::IgANet<optimizer_t, geometry_t, variable_t>::IgANet;

  iganet::status get_epoch(int64_t epoch) const override
  {
    std::cout << "Epoch " << std::to_string(epoch) << ": ";
    return iganet::status(0);
  }
};

int main()
{
  iganet::init();
  iganet::verbose(std::cout);

  torch::autograd::AnomalyMode::set_enabled(true);
  
  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t      = float;

  #if 0
  {
    using UniformBSpline_t = iganet::UniformBSpline<real_t, 1, 3>;    
    using geometry_t       = iganet::S1<UniformBSpline_t>;
    using variable_t       = iganet::RT1<UniformBSpline_t>;
    
    poisson<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                     {100,100},
                                                     // Activation functions
                                                     {
                                                       {iganet::activation::relu},
                                                       {iganet::activation::relu},
                                                       {iganet::activation::none}
                                                     },
                                                     // Number of B-spline coefficients
                                                     std::tuple(iganet::to_array(5_i64)));
    
    net.options().max_epoch(1000);
    net.options().min_loss(1e-8);
    
    net.train();
  }
  #endif
  
  {
    using UniformBSpline_t = iganet::UniformBSpline<real_t, 1, 3,3>;    
    using geometry_t       = iganet::S2<UniformBSpline_t>;
    using variable_t       = iganet::RT2<UniformBSpline_t>;
    
    poisson<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                     {100,100},
                                                     // Activation functions
                                                     {
                                                       {iganet::activation::relu},
                                                       {iganet::activation::relu},
                                                       {iganet::activation::none}
                                                     },
                                                     // Number of B-spline coefficients
                                                     std::tuple(iganet::to_array(5_i64, 5_i64)));
    
    net.options().max_epoch(1000);
    net.options().min_loss(1e-8);
    
    net.train();
  }

  {
    using UniformBSpline_t = iganet::UniformBSpline<real_t, 1, 3,3,3>;    
    using geometry_t       = iganet::S3<UniformBSpline_t>;
    using variable_t       = iganet::RT3<UniformBSpline_t>;
    
    poisson<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                     {100,100},
                                                     // Activation functions
                                                     {
                                                       {iganet::activation::relu},
                                                       {iganet::activation::relu},
                                                       {iganet::activation::none}
                                                     },
                                                     // Number of B-spline coefficients
                                                     std::tuple(iganet::to_array(5_i64, 5_i64, 5_i64)));
    
    net.options().max_epoch(1000);
    net.options().min_loss(1e-8);
    
    net.train();
  }

  {
    using UniformBSpline_t = iganet::UniformBSpline<real_t, 1, 3,3,3,3>;    
    using geometry_t       = iganet::S4<UniformBSpline_t>;
    using variable_t       = iganet::RT4<UniformBSpline_t>;
    
    poisson<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                     {100,100},
                                                     // Activation functions
                                                     {
                                                       {iganet::activation::relu},
                                                       {iganet::activation::relu},
                                                       {iganet::activation::none}
                                                     },
                                                     // Number of B-spline coefficients
                                                     std::tuple(iganet::to_array(5_i64, 5_i64, 5_i64, 5_i64)));
    
    net.options().max_epoch(1000);
    net.options().min_loss(1e-8);
    
    net.train();
  }

  return 0;
}
