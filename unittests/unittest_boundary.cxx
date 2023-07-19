/**
   @file unittests/unittest_boundary.cxx

   @brief Boundary unittests

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

class BoundaryTest
  : public ::testing::Test
{
protected:
  using real_t = double;
  iganet::Options<real_t> options;
};

TEST_F(BoundaryTest, Boundary_parDim1_geoDim1_degrees2)
{
  using iganet::deriv;
  using BSpline_t = iganet::UniformBSpline<real_t, 1, 2>;
  iganet::Boundary<BSpline_t> boundary({0}, iganet::init::greville, options);

  std::cout << boundary << std::endl;
}

TEST_F(BoundaryTest, Boundary_parDim2_geoDim1_degrees23)
{
  using iganet::deriv;
  using BSpline_t = iganet::UniformBSpline<real_t, 1, 2, 3>;
  iganet::Boundary<BSpline_t> boundary({5, 5});

  std::cout << boundary << std::endl;
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
