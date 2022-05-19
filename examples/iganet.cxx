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
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;
  
  torch::manual_seed(1);

  {
    iganet::IgANet<real_t, iganet::UniformBSpline, optimizer_t,
                   5> net({50,30,70}, // Number of neurons per layers
                          {6});       // Number of B-spline coefficients
    std::cout << "Saved IgaNet1\n";
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,1> X){ return std::array<real_t,1>{ X[0]*sin(M_PI*X[0]) }; } );
    std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5)}).flatten() ) << std::endl;

    net.save("iganet1.pt");

    iganet::IgANet<real_t, iganet::UniformBSpline, optimizer_t,
                   5> net1;

    net1.load("iganet1.pt");
    std::cout << "Loaded IgaNet1\n";
    std::cout << net1 << std::endl;

    std::cout << (net == net1) << std::endl;
    return 0;
  }

  {
    iganet::IgANet<real_t, iganet::UniformBSpline, optimizer_t,
                   5,5> net({50,30,70}, // Number of neurons per layers
                            {6,6});     // Number of B-spline coefficients
    std::cout << "Saved IgaNet2\n";
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,2> X){ return std::array<real_t,1>{ sin(M_PI*X[0])*sin(M_PI*X[1]) }; } );
    std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5), torch::full({1}, 0.5)}).flatten() ) << std::endl;

    net.save("iganet2.pt");
  }

  {
    iganet::IgANet<real_t, iganet::UniformBSpline, optimizer_t,
                   5,5,5> net({50,30,70}, // Number of neurons per layers
                              {6,6,6});   // Number of B-spline coefficients
    std::cout << "Saved IgaNet3\n";
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,3> X){ return std::array<real_t,1>{ sin(M_PI*X[0])*sin(M_PI*X[1])*sin(M_PI*X[2]) }; } );
    std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5), torch::full({1}, 0.5), torch::full({1}, 0.5)}).flatten() ) << std::endl;

    net.save("iganet3.pt");
  }

  {
    iganet::IgANet<real_t, iganet::UniformBSpline, optimizer_t,
                   5,5,5,5> net({50,30,70}, // Number of neurons per layers
                                {6,6,6,6}); // Number of B-spline coefficients
    std::cout << "Saved IgaNet4\n";
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,4> X){ return std::array<real_t,1>{ sin(M_PI*X[0])*sin(M_PI*X[1])*sin(M_PI*X[2])*sin(M_PI*X[3]) }; } );
    std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5), torch::full({1}, 0.5), torch::full({1}, 0.5), torch::full({1}, 0.5)}).flatten() ) << std::endl;

    net.save("iganet4.pt");
  }

  {
    iganet::UniformBSpline<double, 2, 3, 4> bspline({2,3});
    std::cout << "Saved BSpline\n";
    std::cout << bspline << std::endl;
    bspline.save("bspline.pl");
    
    iganet::UniformBSpline<double, 2, 3, 4> bspline1;
    bspline1.load("bspline.pl");
    std::cout << "Loaded BSpline\n";
    std::cout << bspline1 << std::endl;
    
    std::cout << (bspline == bspline1) << std::endl;
  }

  return 0;
}
