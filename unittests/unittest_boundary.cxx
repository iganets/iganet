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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
