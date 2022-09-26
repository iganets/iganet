/**
   @file unittests/unittest_iganet.cxx

   @brief IgANet unittests

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <iostream>

#include <gtest/gtest.h>

TEST(BSpline, IgaNet_UniformBSpline_1d_double)
{
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::UniformBSpline, 
                 5> net({50,30,70}, // Number of neurons per layers
                        {
                          {iganet::activation::tanh},
                          {iganet::activation::relu},
                          {iganet::activation::sigmoid},
                          {iganet::activation::none}
                        },          // Activation functions
                        {6});       // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 1);
  EXPECT_EQ(net.rhs().parDim(), 1);
  EXPECT_EQ(net.sol().parDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::left>().parDim(),  0);
  EXPECT_EQ(net.bdr().side<iganet::side::right>().parDim(), 0);

  EXPECT_EQ(net.geo().geoDim(), 1);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::left>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::right>().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 5);
  EXPECT_EQ(net.rhs().degree(0), 5);
  EXPECT_EQ(net.sol().degree(0), 5);

  EXPECT_EQ(net.geo().ncoeffs(0), 6);
  EXPECT_EQ(net.rhs().ncoeffs(0), 6);
  EXPECT_EQ(net.sol().ncoeffs(0), 6);
}

TEST(BSpline, IgaNet_UniformBSpline_2d_double)
{
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::UniformBSpline, 
                 3, 5> net({50,30,70}, // Number of neurons per layers
                           {
                             {iganet::activation::tanh},
                             {iganet::activation::relu},
                             {iganet::activation::sigmoid},
                             {iganet::activation::none}
                           },          // Activation functions
                           {4,6});     // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 2);
  EXPECT_EQ(net.rhs().parDim(), 2);
  EXPECT_EQ(net.sol().parDim(), 2);
  
  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 1);

  EXPECT_EQ(net.geo().geoDim(), 2);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.rhs().degree(0), 3);
  EXPECT_EQ(net.sol().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.rhs().degree(1), 5);
  EXPECT_EQ(net.sol().degree(1), 5);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(0), 3);

  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.rhs().ncoeffs(0), 4);
  EXPECT_EQ(net.sol().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.rhs().ncoeffs(1), 6);
  EXPECT_EQ(net.sol().ncoeffs(1), 6);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(0), 4);
}

TEST(BSpline, IgaNet_UniformBSpline_3d_double)
{
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::UniformBSpline,
                 3, 5, 1> net({50,30,70}, // Number of neurons per layers
                              {
                                {iganet::activation::tanh},
                                {iganet::activation::relu},
                                {iganet::activation::sigmoid},
                                {iganet::activation::none}
                              },          // Activation functions
                              {4,6,3});   // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 3);
  EXPECT_EQ(net.rhs().parDim(), 3);
  EXPECT_EQ(net.sol().parDim(), 3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  2);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  2);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().parDim(),  2);

  EXPECT_EQ(net.geo().geoDim(), 3);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().geoDim(),  1);
  
  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.rhs().degree(0), 3);
  EXPECT_EQ(net.sol().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.rhs().degree(1), 5);
  EXPECT_EQ(net.sol().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.rhs().degree(2), 1);
  EXPECT_EQ(net.sol().degree(2), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(0),  3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(1),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(1),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(1), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(1), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(1), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(1),  5);
  
  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.rhs().ncoeffs(0), 4);
  EXPECT_EQ(net.sol().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.rhs().ncoeffs(1), 6);
  EXPECT_EQ(net.sol().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.rhs().ncoeffs(2), 3);
  EXPECT_EQ(net.sol().ncoeffs(2), 3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(0),  4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(1),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(1),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(1), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(1), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(1), 6);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(1),  6);
}

TEST(BSpline, IgaNet_UniformBSpline_4d_double)
{
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::UniformBSpline,
                 3, 5, 1, 4> net({50,30,70}, // Number of neurons per layers
                                {
                                  {iganet::activation::tanh},
                                  {iganet::activation::relu},
                                  {iganet::activation::sigmoid},
                                  {iganet::activation::none}
                                },          // Activation functions
                                {4,6,3,5}); // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 4);
  EXPECT_EQ(net.rhs().parDim(), 4);
  EXPECT_EQ(net.sol().parDim(), 4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().parDim(), 3);

  EXPECT_EQ(net.geo().geoDim(), 4);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().geoDim(), 1);
  
  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.rhs().degree(0), 3);
  EXPECT_EQ(net.sol().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.rhs().degree(1), 5);
  EXPECT_EQ(net.sol().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.rhs().degree(2), 1);
  EXPECT_EQ(net.sol().degree(2), 1);

  EXPECT_EQ(net.geo().degree(3), 4);
  EXPECT_EQ(net.rhs().degree(3), 4);
  EXPECT_EQ(net.sol().degree(3), 4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(0),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().degree(0), 3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(1),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(1),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(1), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(1), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(1), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(1),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().degree(1), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().degree(1), 5);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(2),  4);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(2),  4);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(2), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(2), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(2), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(2),  4);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().degree(2), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().degree(2), 1);
  
  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.rhs().ncoeffs(0), 4);
  EXPECT_EQ(net.sol().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.rhs().ncoeffs(1), 6);
  EXPECT_EQ(net.sol().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.rhs().ncoeffs(2), 3);
  EXPECT_EQ(net.sol().ncoeffs(2), 3);

  EXPECT_EQ(net.geo().ncoeffs(3), 5);
  EXPECT_EQ(net.rhs().ncoeffs(3), 5);
  EXPECT_EQ(net.sol().ncoeffs(3), 5);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(0),  4);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().ncoeffs(0), 4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(1),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(1),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(1), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(1), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(1), 6);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(1),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().ncoeffs(1), 6);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().ncoeffs(1), 6);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(2),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(2),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(2), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(2), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(2), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(2),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().ncoeffs(2), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().ncoeffs(2), 3);
}

TEST(BSpline, IgaNet_NonUniformBSpline_1d_double)
{
  using real_t      = double;
  using optimizer_t = torch::optim::LBFGS;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::NonUniformBSpline, 
                 5> net({50,30,70}, // Number of neurons per layers
                        {
                          {iganet::activation::tanh},
                          {iganet::activation::relu},
                          {iganet::activation::sigmoid},
                          {iganet::activation::none}
                        },          // Activation functions
                        {6});       // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 1);
  EXPECT_EQ(net.rhs().parDim(), 1);
  EXPECT_EQ(net.sol().parDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::left>().parDim(),  0);
  EXPECT_EQ(net.bdr().side<iganet::side::right>().parDim(), 0);

  EXPECT_EQ(net.geo().geoDim(), 1);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::left>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::right>().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 5);
  EXPECT_EQ(net.rhs().degree(0), 5);
  EXPECT_EQ(net.sol().degree(0), 5);

  EXPECT_EQ(net.geo().ncoeffs(0), 6);
  EXPECT_EQ(net.rhs().ncoeffs(0), 6);
  EXPECT_EQ(net.sol().ncoeffs(0), 6);
}

TEST(BSpline, IgaNet_NonUniformBSpline_2d_double)
{
  using real_t      = double;
  using optimizer_t = torch::optim::LBFGS;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::NonUniformBSpline, 
                 3, 5> net({50,30,70}, // Number of neurons per layers
                           {
                             {iganet::activation::tanh},
                             {iganet::activation::relu},
                             {iganet::activation::sigmoid},
                             {iganet::activation::none}
                           },          // Activation functions
                           {4,6});     // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 2);
  EXPECT_EQ(net.rhs().parDim(), 2);
  EXPECT_EQ(net.sol().parDim(), 2);
  
  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 1);

  EXPECT_EQ(net.geo().geoDim(), 2);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.rhs().degree(0), 3);
  EXPECT_EQ(net.sol().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.rhs().degree(1), 5);
  EXPECT_EQ(net.sol().degree(1), 5);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(0), 3);

  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.rhs().ncoeffs(0), 4);
  EXPECT_EQ(net.sol().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.rhs().ncoeffs(1), 6);
  EXPECT_EQ(net.sol().ncoeffs(1), 6);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(0), 4);
}

TEST(BSpline, IgaNet_NonUniformBSpline_3d_double)
{
  using real_t      = double;
  using optimizer_t = torch::optim::LBFGS;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::NonUniformBSpline,
                 3, 5, 1> net({50,30,70}, // Number of neurons per layers
                              {
                                {iganet::activation::tanh},
                                {iganet::activation::relu},
                                {iganet::activation::sigmoid},
                                {iganet::activation::none}
                              },          // Activation functions
                              {4,6,3});   // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 3);
  EXPECT_EQ(net.rhs().parDim(), 3);
  EXPECT_EQ(net.sol().parDim(), 3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  2);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  2);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().parDim(),  2);

  EXPECT_EQ(net.geo().geoDim(), 3);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().geoDim(),  1);
  
  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.rhs().degree(0), 3);
  EXPECT_EQ(net.sol().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.rhs().degree(1), 5);
  EXPECT_EQ(net.sol().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.rhs().degree(2), 1);
  EXPECT_EQ(net.sol().degree(2), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(0),  3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(1),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(1),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(1), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(1), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(1), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(1),  5);
  
  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.rhs().ncoeffs(0), 4);
  EXPECT_EQ(net.sol().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.rhs().ncoeffs(1), 6);
  EXPECT_EQ(net.sol().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.rhs().ncoeffs(2), 3);
  EXPECT_EQ(net.sol().ncoeffs(2), 3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(0),  4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(1),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(1),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(1), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(1), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(1), 6);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(1),  6);
}

TEST(BSpline, IgaNet_NonUniformBSpline_4d_double)
{
  using real_t      = double;
  using optimizer_t = torch::optim::LBFGS;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::NonUniformBSpline,
                 3, 5, 1, 4> net({50,30,70}, // Number of neurons per layers
                                {
                                  {iganet::activation::tanh},
                                  {iganet::activation::relu},
                                  {iganet::activation::sigmoid},
                                  {iganet::activation::none}
                                },          // Activation functions
                                {4,6,3,5}); // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 4);
  EXPECT_EQ(net.rhs().parDim(), 4);
  EXPECT_EQ(net.sol().parDim(), 4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().parDim(), 3);

  EXPECT_EQ(net.geo().geoDim(), 4);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().geoDim(), 1);
  
  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.rhs().degree(0), 3);
  EXPECT_EQ(net.sol().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.rhs().degree(1), 5);
  EXPECT_EQ(net.sol().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.rhs().degree(2), 1);
  EXPECT_EQ(net.sol().degree(2), 1);

  EXPECT_EQ(net.geo().degree(3), 4);
  EXPECT_EQ(net.rhs().degree(3), 4);
  EXPECT_EQ(net.sol().degree(3), 4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(0),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().degree(0), 3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(1),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(1),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(1), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(1), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(1), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(1),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().degree(1), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().degree(1), 5);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(2),  4);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(2),  4);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(2), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(2), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().degree(2), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().degree(2),  4);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().degree(2), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().degree(2), 1);
  
  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.rhs().ncoeffs(0), 4);
  EXPECT_EQ(net.sol().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.rhs().ncoeffs(1), 6);
  EXPECT_EQ(net.sol().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.rhs().ncoeffs(2), 3);
  EXPECT_EQ(net.sol().ncoeffs(2), 3);

  EXPECT_EQ(net.geo().ncoeffs(3), 5);
  EXPECT_EQ(net.rhs().ncoeffs(3), 5);
  EXPECT_EQ(net.sol().ncoeffs(3), 5);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(0),  4);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().ncoeffs(0), 4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(1),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(1),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(1), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(1), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(1), 6);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(1),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().ncoeffs(1), 6);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().ncoeffs(1), 6);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(2),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(2),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(2), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(2), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().ncoeffs(2), 5);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().ncoeffs(2),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().ncoeffs(2), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().ncoeffs(2), 3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
