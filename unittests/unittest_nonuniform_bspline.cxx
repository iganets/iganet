/**
   @file unittests/unittest_nonuniform_bspline.cxx

   @brief B-Spline unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <filesystem>
#include <iostream>

#include "unittest_splinelib.hpp"
#include <gtest/gtest.h>

TEST(BSpline, NonUniformBSpline_parDim1_geoDim1_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 1, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 1, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, NonUniformBSpline_parDim1_geoDim2_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 2, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 2, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, NonUniformBSpline_parDim1_geoDim3_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 3, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 3, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, NonUniformBSpline_parDim1_geoDim4_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 4, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 4, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim1_degrees12_double)
{
  iganet::NonUniformBSpline<double, 1, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  9);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim2_degrees12_double)
{
  iganet::NonUniformBSpline<double, 2, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  9);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim3_degrees12_double)
{
  iganet::NonUniformBSpline<double, 3, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  9);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim4_degrees12_double)
{
  iganet::NonUniformBSpline<double, 4, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  9);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim1_degrees123_double)
{
  iganet::NonUniformBSpline<double, 1, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(), 45);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim2_degrees123_double)
{
  iganet::NonUniformBSpline<double, 2, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(), 45);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim3_degrees123_double)
{
  iganet::NonUniformBSpline<double, 3, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(), 45);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim4_degrees123_double)
{
  iganet::NonUniformBSpline<double, 4, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(), 45);
}

TEST(BSpline, NonUniformBSpline_parDim4_geoDim1_degrees1234_double)
{
  iganet::NonUniformBSpline<double, 1, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.degree(3),  4);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncoeffs(),270);
}

TEST(BSpline, NonUniformBSpline_parDim4_geoDim2_degrees1234_double)
{
  iganet::NonUniformBSpline<double, 2, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.degree(3),  4);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncoeffs(),270);
}

TEST(BSpline, NonUniformBSpline_parDim4_geoDim3_degrees1234_double)
{
  iganet::NonUniformBSpline<double, 3, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.degree(3),  4);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncoeffs(),270);
}

TEST(BSpline, NonUniformBSpline_parDim4_geoDim4_degrees1234_double)
{
  iganet::NonUniformBSpline<double, 4, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.degree(3),  4);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncoeffs(),270);
}

TEST(BSpline, NonUniformBSpline_read_write_double)
{
  std::filesystem::path filename = std::filesystem::temp_directory_path() / std::to_string(rand());
  iganet::NonUniformBSpline<double, 3, 1, 2, 3> bspline_out( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                               {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                               {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  bspline_out.save(filename.c_str());
  iganet::NonUniformBSpline<double, 3, 1, 2, 3> bspline_in;
  bspline_in.load(filename.c_str());
  std::filesystem::remove(filename);

  EXPECT_EQ( (bspline_in == bspline_out), true);
  EXPECT_EQ( (bspline_in != bspline_out), false);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
