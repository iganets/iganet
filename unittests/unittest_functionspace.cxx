/**
   @file unittests/unittest_functionspace.cxx

   @brief Function space unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <filesystem>
#include <iostream>

#include <gtest/gtest.h>

class FunctionSpaceTest
  : public ::testing::Test
{
protected:
  using real_t = double;
  iganet::Options<real_t> options;
};

TEST_F(FunctionSpaceTest, S1_geoDim1_degrees1)
{
  iganet::S1<iganet::UniformBSpline<real_t, 1, 1>> functionspace({2}, iganet::init::greville, options);
  
  auto xi_interior  = iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options);
  
  EXPECT_TRUE(torch::equal(*(functionspace.eval<iganet::functionspace::interior, iganet::deriv::func, false>(xi_interior)[0]),
                           xi_interior[0]));
  
  EXPECT_TRUE(torch::equal(*(functionspace.eval<iganet::functionspace::interior, iganet::deriv::dx,   false>(xi_interior)[0]),
                           torch::ones_like(xi_interior[0])));
  
  EXPECT_TRUE(torch::equal(*(functionspace.eval<iganet::functionspace::interior, iganet::deriv::dx2,  false>(xi_interior)[0]),
                           torch::zeros_like(xi_interior[0])));

  EXPECT_TRUE(torch::equal(*(functionspace.eval<iganet::functionspace::interior, iganet::deriv::func, true>(xi_interior)[0]),
                           xi_interior[0]));
  
  // std::cout << functionspace.eval<iganet::functionspace::interior, iganet::deriv::dx,   true>(xi_interior);
  // std::cout << functionspace.eval<iganet::functionspace::interior, iganet::deriv::dx2,  true>(xi_interior);

  //  auto xi_boundary = std::tuple{ iganet::utils::to_tensorArray<real_t>({0.0}, options),
  //                                 iganet::utils::to_tensorArray<real_t>({0.0}, options) };

  //  auto boundary = functionspace.eval<iganet::functionspace::boundary, iganet::deriv::func, false>(xi_boundary);

  
}

TEST_F(FunctionSpaceTest, S2_geoDim1_degrees1)
{
  iganet::S2<iganet::UniformBSpline<real_t, 1, 1, 1>> functionspace({2, 2}, iganet::init::greville, options);
  
  // auto xi_interior  = iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
  //                                                           {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options);
  
  // std::cout << functionspace.eval<iganet::functionspace::interior, iganet::deriv::func, false>(xi_interior);
  // std::cout << functionspace.eval<iganet::functionspace::interior, iganet::deriv::dx,   false>(xi_interior);
  // std::cout << functionspace.eval<iganet::functionspace::interior, iganet::deriv::dx2,  false>(xi_interior);
  // std::cout << functionspace.eval<iganet::functionspace::interior, iganet::deriv::dy,   false>(xi_interior);
  // std::cout << functionspace.eval<iganet::functionspace::interior, iganet::deriv::dy2,  false>(xi_interior);
  // std::cout << functionspace.eval<iganet::functionspace::interior, iganet::deriv::dxdy, false>(xi_interior);

  // auto xi_boundary  = std::tuple{ iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options),
  //                                 iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options),
  //                                 iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options),
  //                                 iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options) };
  
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::func, false>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dx,   false>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dx2,  false>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dy,   false>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dy2,  false>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dx,   false>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dxdy, false>(xi_boundary);

  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::func, true>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dx,   true>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dx2,  true>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dy,   true>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dy2,  true>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dx,   true>(xi_boundary);
  // std::cout << functionspace.eval<iganet::functionspace::boundary, iganet::deriv::dxdy, true>(xi_boundary);
  
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
