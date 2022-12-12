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


#include <iganet.hpp>
#include <iostream>

int main()
{
  //std::cout << iganet::verbose;
  using real_t = double;
  iganet::init();

  iganet::RT3<double, iganet::UniformBSpline, 2> S( {{10,10,10}}, {{15,15,15}}, {{5,5,5}}, {{10,10,10}} );

  std::cout << std::get<0>(S) << std::endl;
  std::cout << std::get<1>(S) << std::endl;
  std::cout << std::get<2>(S) << std::endl;
  std::cout << std::get<3>(S) << std::endl;
    
  return 0;
}
