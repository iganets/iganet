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

  using namespace iganet::literals;
  
  using UniformBSpline_t    = iganet::UniformBSpline<real_t, 3, 1>;
  using NonUniformBSpline_t = iganet::NonUniformBSpline<real_t, 3, 1, 1>;
  using FunctionSpace_t     = iganet::FunctionSpace<UniformBSpline_t,
                                                    NonUniformBSpline_t,
                                                    UniformBSpline_t>;
   
  // FunctionSpace_t space(iganet::init::greville,
  //                       iganet::to_array(8, 6_i64),
  //                       iganet::to_array(iganet::to_vector(0.0,0.0,0.5,1.0,1.0),
  //                                        iganet::to_vector(0.0,0.0,0.5,1.0,1.0)),
  //                       iganet::to_array(13, 2_i64)
  //                       );

  iganet::TH1<UniformBSpline_t> s(iganet::to_array(5_i64));
  
  std::cout << s << std::endl;
  
  return 0;
}


// TH1, TH2, TH3, TH4
