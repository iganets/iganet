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

int main()
{
  std::cout << iganet::verbose;
  using real_t = double;
  iganet::init();
  
  //  iganet::FunctionSpace< iganet::UniformBSpline<real_t, 1, 2, 3, 1>,
  //                         iganet::UniformBSpline<real_t, 1, 2, 3> > S( {{10,20, 15}}, {{10,20}} );
  
  //  std::cout << S << std::endl;
  
  iganet::UniformBSpline<real_t, 3, 2, 2> S({20,20});

  // Map control points to phyiscal coordinates
  S.transform( [](const std::array<real_t,2> xi){ return std::array<real_t,3>{ xi[0]*xi[0], xi[0], 2*xi[0] }; } );

  // Print B-spline
  std::cout << S.eval( iganet::to_tensorArray<real_t>({0.0, 0.5, 1.0, 0.2},
                                                      {0.0, 0.5, 0.5, 0.2}) ) << std::endl;
  //std::cout << S << std::endl;

  // Plot B-spline
  //S.plot(50);

  //S.insert_knots( iganet::to_tensorArray<real_t>({0.5, 0.1, 0.3}) );
  //S.uniformRefine(3);

  // Print B-spline
  std::cout << S << std::endl;
  
  return 0;
}
