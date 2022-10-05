/**
   @file examples/uniform_bsplines.cxx

   @brief Demonstration of the uniform B-spline class

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
  using real_t = double;
  iganet::init();

  {
    // Univariate uniform B-spline of degree 2 with 6 control points in R^1
    iganet::UniformBSpline<double,1,2> bspline({6});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,1>{ xi[0]*xi[0] }; } );

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval( iganet::to_tensor({0.0}) ) << std::endl
              << bspline.eval( iganet::to_tensor({0.5}) ) << std::endl
              << bspline.eval( iganet::to_tensor({1.0}) ) << std::endl;

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval_( iganet::to_tensor({0.0, 0.5, 1.0}) ) << std::endl;
    
    // Plot B-spline
    bspline.plot(50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }

  {
    // Univariate uniform B-spline of degree 2 with 6 control points in R^2
    iganet::UniformBSpline<double,2,2> bspline({6});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,2>{ xi[0]*xi[0],
                                                                                       sin(M_PI*xi[0]) }; } );

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval( iganet::to_tensor({0.0}) ) << std::endl
              << bspline.eval( iganet::to_tensor({0.5}) ) << std::endl
              << bspline.eval( iganet::to_tensor({1.0}) ) << std::endl;

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval_( iganet::to_tensor({0.0, 0.5, 1.0}) ) << std::endl;
    
    // Plot B-spline
    bspline.plot(50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }

  {
    // Univariate uniform B-spline of degree 2 with 6 control points in R^3
    iganet::UniformBSpline<double,3,2> bspline({6});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,3>{ xi[0]*xi[0],
                                                                                       sin(M_PI*xi[0]),
                                                                                       xi[0]          }; } );

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval( iganet::to_tensor({0.0}) ) << std::endl
              << bspline.eval( iganet::to_tensor({0.5}) ) << std::endl
              << bspline.eval( iganet::to_tensor({1.0}) ) << std::endl;

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval_( iganet::to_tensor({0.0, 0.5, 1.0}) ) << std::endl;
    
    // Plot B-spline
    bspline.plot(50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }

  {
    // Bivariate uniform B-spline of degree 3 in xi-direction and 4
    // in eta-direction with 5 x 6 control points in R^2
    iganet::UniformBSpline<double,2,3,4> bspline({5,6});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,2> xi){ return std::array<real_t,2>{(xi[0]+1)*cos(M_PI*xi[1]),
                                                                                      (xi[0]+1)*sin(M_PI*xi[1])}; } );

    // Evaluate B-spline at (xi=0,eta=0), (xi=0.5,eta=0.5), and (xi=1,eta=0.5)
    std::cout << bspline.eval( iganet::to_tensor({0.0, 0.0}) ) << std::endl
              << bspline.eval( iganet::to_tensor({0.5, 0.5}) ) << std::endl
              << bspline.eval( iganet::to_tensor({1.0, 0.5}) ) << std::endl;

    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval_( iganet::to_tensor({0.0, 0.5, 1.0, 0.0, 0.5, 0.5}, {2,3}) ) << std::endl;
    
    // Plot B-spline
    bspline.plot(50,50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }

  {
    // Bivariate uniform B-spline of degree 3 in xi-direction and 4
    // in eta-direction with 5 x 6 control points in R^3
    iganet::UniformBSpline<double,3,3,4> bspline({5,6});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,2> xi){ return std::array<real_t,3>{(xi[0]+1)*cos(M_PI*xi[1]),
                                                                                      (xi[0]+1)*sin(M_PI*xi[1]),
                                                                                       xi[0] }; } );

    // Evaluate B-spline at (xi=0,eta=0), (xi=0.5,eta=0.5), and (xi=1,eta=0.5)
    std::cout << bspline.eval( iganet::to_tensor({0.0, 0.0}) ) << std::endl
              << bspline.eval( iganet::to_tensor({0.5, 0.5}) ) << std::endl
              << bspline.eval( iganet::to_tensor({1.0, 0.5}) ) << std::endl;

    // Plot B-spline
    bspline.plot(50,50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }
}
