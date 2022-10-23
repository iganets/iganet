/**
   @file examples/iganet.cxx

   @brief Demonstration of the IgaNet class

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <iostream>

int main()
{
  using real_t      = float;
  using optimizer_t = torch::optim::Adam;

  torch::autograd::AnomalyMode::set_enabled(true);
  
  iganet::init();
  iganet::verbose(std::cout);
  
  {
    iganet::IgANet<real_t, optimizer_t,
                   iganet::UniformBSpline,
                   2> net({100,100}, // Number of neurons per layers
                          {
                            {iganet::activation::relu},
                            {iganet::activation::relu},
                            {iganet::activation::none}
                          },         // Activation functions
                          {5});      // Number of B-spline coefficients

    // Set rhs to x
    net.rhs().transform( [](const std::array<real_t,1> X){ return std::array<real_t,1>{ static_cast<real_t>( X[0] ) }; } );

    // Set left boundary value to 0
    net.bdr().coeffs()[0].accessor<real_t,1>()[0] = 0;
    net.bdr().coeffs()[1].accessor<real_t,1>()[0] = 1;

    net.options().max_epoch(1000);
    net.options().min_loss(1e-8);
    
    net.train();
  }

  return 0;
  
  {
    iganet::IgANet<real_t, optimizer_t,
                   iganet::UniformBSpline,
                   5> net({50,30,70}, // Number of neurons per layers
                          {
                            {iganet::activation::relu},
                            {iganet::activation::relu},
                            {iganet::activation::relu},
                            {iganet::activation::none}
                          },          // Activation functions
                          {6});       // Number of B-spline coefficients
    std::cout << "Saved IgaNet1\n";
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,1> X){ return std::array<real_t,1>{ static_cast<real_t>( X[0]*sin(M_PI*X[0]) ) }; } );
    std::cout << net.sol().eval( iganet::to_tensor<real_t>({0.0, 0.2, 0.5, 1.0}) ) << std::endl;    

    net.save("iganet1.pt");

    iganet::IgANet<real_t, optimizer_t,
                   iganet::UniformBSpline,
                   5> net1;

    net1.load("iganet1.pt");
    // std::cout << "Loaded IgaNet1\n";
    // std::cout << net1 << std::endl;

    // std::cout << (net == net1) << std::endl;
  }

  {
    iganet::IgANet<real_t, optimizer_t, iganet::UniformBSpline,
                   2, 2> net({50,30,70}, // Number of neurons per layers
                             {
                               {iganet::activation::relu},
                               {iganet::activation::relu},
                               {iganet::activation::relu},
                               {iganet::activation::none}
                             },          // Activation functions
                             {3,6});     // Number of B-spline coefficients
    std::cout << "Saved IgaNet2\n";
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,2> X){ return std::array<real_t,1>{ static_cast<real_t>( sin(M_PI*X[0])*sin(M_PI*X[1]) ) }; } );
    std::cout << net.sol().eval( iganet::to_tensorArray<real_t>({0.0, 0.2, 0.5, 1.0},
                                                                {0.0, 0.2, 0.5, 1.0}) ) << std::endl;    

    net.save("iganet2.pt");
  }

  {
    iganet::IgANet<real_t, optimizer_t, iganet::UniformBSpline,
                   5, 5, 5> net({50,30,70}, // Number of neurons per layers
                                {
                                  {iganet::activation::relu},
                                  {iganet::activation::relu},
                                  {iganet::activation::relu},
                                  {iganet::activation::none}
                                },          // Activation functions
                                {6,6,6});   // Number of B-spline coefficients
    std::cout << "Saved IgaNet3\n";
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,3> X){ return std::array<real_t,1>{ static_cast<real_t>( sin(M_PI*X[0])*sin(M_PI*X[1])*sin(M_PI*X[2]) ) }; } );
    std::cout << net.sol().eval( iganet::to_tensorArray<real_t>({0.0, 0.2, 0.5, 1.0},
                                                                {0.0, 0.2, 0.5, 1.0},
                                                                {0.0, 0.2, 0.5, 1.0}) ) << std::endl;
    
    net.save("iganet3.pt");
  }

  {
    iganet::IgANet<real_t, optimizer_t, iganet::UniformBSpline, 
                   5, 5, 5, 5> net({50,30,70}, // Number of neurons per layers
                                   {
                                     {iganet::activation::relu},
                                     {iganet::activation::relu},
                                     {iganet::activation::relu},
                                     {iganet::activation::none}
                                   },          // Activation functions
                                   {6,6,6,6}); // Number of B-spline coefficients
    std::cout << "Saved IgaNet4\n";
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,4> X){ return std::array<real_t,1>{ static_cast<real_t>( sin(M_PI*X[0])*sin(M_PI*X[1])*sin(M_PI*X[2])*sin(M_PI*X[3]) ) }; } );
    std::cout << net.sol().eval( iganet::to_tensorArray<real_t>({0.0, 0.2, 0.5, 1.0},
                                                                {0.0, 0.2, 0.5, 1.0},
                                                                {0.0, 0.2, 0.5, 1.0},
                                                                {0.0, 0.2, 0.5, 1.0}) ) << std::endl;
    
    net.save("iganet4.pt");
  }

  return 0;
}
