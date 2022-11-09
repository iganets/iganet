/**
   @file examples/nonuniform_bsplines.cxx

   @brief Demonstration of the non-uniform B-spline class

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
  std::cout << iganet::verbose;
  using real_t = double;
  iganet::init();

  {
    // Univariate non-uniform B-spline of degree 2 with 6 control points in R^1
    iganet::NonUniformBSpline<real_t,1,2> bspline({{{0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0}}});
    iganet::NonUniformBSpline<real_t,1,2>   color({{{0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0}}});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,1>{ xi[0]*xi[0] }; } );

    // Map colors
    color.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,1>{ xi[0] }; } );
    
    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval( iganet::to_tensorArray<real_t>({0.0, 0.5, 1.0}) ) << std::endl;

    // Plot B-spline
    bspline.plot(50);
    bspline.plot(color, 50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }

  {
    // Univariate non-uniform B-spline of degree 2 with 6 control points in R^2
    iganet::NonUniformBSpline<real_t,2,2> bspline({{{0.0,0.0,0.0,0.25,0.5,0.75,1.0,1.0,1.0}}});
    iganet::NonUniformBSpline<real_t,1,2> color({{{0.0,0.0,0.0,0.25,0.5,0.75,1.0,1.0,1.0}}});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,2>{ xi[0]*xi[0],
                                                                                       sin(M_PI*xi[0]) }; } );

    // Map colors
    color.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,1>{ xi[0] }; } );
    
    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval( iganet::to_tensorArray<real_t>({0.0, 0.5, 1.0}) ) << std::endl;
    
    // Plot B-spline
    bspline.plot(50);
    bspline.plot(color, 50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }

  {
    // Univariate non-uniform B-spline of degree 2 with 6 control points in R^3
    iganet::NonUniformBSpline<real_t,3,2> bspline({{{0.0,0.0,0.0,0.25,0.5,0.75,1.0,1.0,1.0}}});
    iganet::NonUniformBSpline<real_t,1,2> color({{{0.0,0.0,0.0,0.25,0.5,0.75,1.0,1.0,1.0}}});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,3>{ xi[0]*xi[0],
                                                                                       sin(M_PI*xi[0]),
                                                                                       xi[0]          }; } );

    // Map colors
    color.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,1>{ xi[0] }; } );
    
    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval( iganet::to_tensorArray<real_t>({0.0, 0.5, 1.0}) ) << std::endl;

    // Plot B-spline
    bspline.plot(50);
    bspline.plot(color, 50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }

  {
    // Bivariate non-uniform B-spline of degree 3 in xi-direction and 4
    // in eta-direction with 5 x 6 control points in R^2
    iganet::NonUniformBSpline<real_t,2,3,4> bspline({{{0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0},
                                                      {0.0,0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0,1.0}}});
    iganet::NonUniformBSpline<real_t,1,3,4> color({{{0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0},
                                                    {0.0,0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0,1.0}}});
    
    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,2> xi){ return std::array<real_t,2>{(xi[0]+1)*cos(M_PI*xi[1]),
                                                                                      (xi[0]+1)*sin(M_PI*xi[1])}; } );

    // Map colors
    color.transform( [](const std::array<real_t,2> xi){ return std::array<real_t,1>{ xi[0]*xi[1] }; } );
    
    // Evaluate B-spline at xi=0, xi=0.5, and xi=1
    std::cout << bspline.eval( iganet::to_tensorArray<real_t>({0.0, 0.5, 1.0},
                                                              {0.0, 0.5, 0.5}) ) << std::endl;

    // Plot B-spline
    bspline.plot(50, 50);
    bspline.plot(color, 50, 50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }

  {
    // Bivariate non-uniform B-spline of degree 3 in xi-direction and 4
    // in eta-direction with 5 x 6 control points in R^3
    iganet::NonUniformBSpline<real_t,3,3,4> bspline({{{0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0},
                                                      {0.0,0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0,1.0}}});
    iganet::NonUniformBSpline<real_t,1,3,4> color({{{0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0},
                                                    {0.0,0.0,0.0,0.0,0.0,0.5,1.0,1.0,1.0,1.0,1.0}}});

    // Print information
    std::cout << bspline << std::endl;

    // Map control points to phyiscal coordinates
    bspline.transform( [](const std::array<real_t,2> xi){ return std::array<real_t,3>{(xi[0]+1)*cos(M_PI*xi[1]),
                                                                                      (xi[0]+1)*sin(M_PI*xi[1]),
                                                                                       xi[0] }; } );

    // Map colors
    color.transform( [](const std::array<real_t,2> xi){ return std::array<real_t,1>{ xi[0]*xi[1] }; } );
    
    // Evaluate B-spline at (xi=0,eta=0), (xi=0.5,eta=0.5), and (xi=1,eta=0.5)
    std::cout << bspline.eval( iganet::to_tensorArray<real_t>({0.0, 0.5, 1.0},
                                                              {0.0, 0.5, 0.5}) ) << std::endl;
    
    // Plot B-spline
    bspline.plot(50, 50);
    bspline.plot(color, 50, 50);

    // Export B-spline to XML
    std::cout << bspline.to_xml() << std::endl;
  }
}
