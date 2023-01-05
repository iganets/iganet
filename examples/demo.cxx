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
  
  iganet::NonUniformBSpline<real_t, 3, 2> S({{{0.0, 0.0, 0.0, 0.1, 0.5, 0.8, 1.0, 1.0, 1.0}}});

  // Map control points to physical coordinates
  S.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,3>{ xi[0], xi[0]*xi[0], 3*xi[0]*xi[0] }; } );
  
  std::cout << "\n----\n" << S << "\n----\n";
  S.plot();
  
  S.insert_knots( iganet::to_tensorArray<real_t>({0.4, 0.4}) );
  S.uniform_refine(2);
  
  std::cout << "\n----\n" << S << "\n----\n";
  S.plot();
  
  return 0;
}
