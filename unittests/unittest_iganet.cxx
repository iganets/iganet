/**
   @file unittests/unittest_iganet.cxx

   @brief IgANet unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <iostream>

#include <gtest/gtest.h>

template<typename optimizer_t,
         typename geometry_t,
         typename variable_t>
class IgANet : public iganet::IgANet<optimizer_t, geometry_t, variable_t>
{
public:
  using iganet::IgANet<optimizer_t, geometry_t, variable_t>::IgANet;
  
  iganet::status get_epoch(int64_t epoch) const override
  {
    std::cout << "Epoch " << std::to_string(epoch) << ": ";
    return iganet::status(0);
  }
};

TEST(BSpline, IgANet_UniformBSpline_1d_double)
{
  using namespace iganet::literals;
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;
  
  using geometry_t  = iganet::S1<iganet::UniformBSpline<real_t, 1, 5>>;
  using variable_t  = iganet::S1<iganet::UniformBSpline<real_t, 1, 5>>;

  IgANet<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                  {50,30,70},
                                                  // Activation functions
                                                  {
                                                    {iganet::activation::tanh},
                                                    {iganet::activation::relu},
                                                    {iganet::activation::sigmoid},
                                                    {iganet::activation::none}
                                                  },
                                                  // Number of B-spline coefficients
                                                  std::tuple(iganet::to_array(6_i64)));
  
  EXPECT_EQ(net.geo().parDim(), 1);
  EXPECT_EQ(net.ref().parDim(), 1);
  EXPECT_EQ(net.out().parDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::left>().parDim(),  0);
  EXPECT_EQ(net.bdr().side<iganet::side::right>().parDim(), 0);

  EXPECT_EQ(net.geo().geoDim(), 1);
  EXPECT_EQ(net.ref().geoDim(), 1);
  EXPECT_EQ(net.out().geoDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::left>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::right>().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 5);
  EXPECT_EQ(net.ref().degree(0), 5);
  EXPECT_EQ(net.out().degree(0), 5);

  EXPECT_EQ(net.geo().ncoeffs(0), 6);
  EXPECT_EQ(net.ref().ncoeffs(0), 6);
  EXPECT_EQ(net.out().ncoeffs(0), 6);
}

TEST(BSpline, IgANet_UniformBSpline_2d_double)
{
  using namespace iganet::literals;
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;
  
  using geometry_t  = iganet::S2<iganet::UniformBSpline<real_t, 2, 3, 5>>;
  using variable_t  = iganet::S2<iganet::UniformBSpline<real_t, 1, 3, 5>>;

  IgANet<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                  {50,30,70},
                                                  // Activation functions
                                                  {
                                                    {iganet::activation::tanh},
                                                    {iganet::activation::relu},
                                                    {iganet::activation::sigmoid},
                                                    {iganet::activation::none}
                                                  },
                                                  // Number of B-spline coefficients
                                                  std::tuple(iganet::to_array(4_i64, 6_i64))); 
  
  EXPECT_EQ(net.geo().parDim(), 2);
  EXPECT_EQ(net.ref().parDim(), 2);
  EXPECT_EQ(net.out().parDim(), 2);
  
  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 1);

  EXPECT_EQ(net.geo().geoDim(), 2);
  EXPECT_EQ(net.ref().geoDim(), 1);
  EXPECT_EQ(net.out().geoDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.ref().degree(0), 3);
  EXPECT_EQ(net.out().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.ref().degree(1), 5);
  EXPECT_EQ(net.out().degree(1), 5);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(0), 3);

  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.ref().ncoeffs(0), 4);
  EXPECT_EQ(net.out().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.ref().ncoeffs(1), 6);
  EXPECT_EQ(net.out().ncoeffs(1), 6);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(0), 4);
}

TEST(BSpline, IgANet_UniformBSpline_3d_double)
{
  using namespace iganet::literals;
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;
  
  using geometry_t  = iganet::S3<iganet::UniformBSpline<real_t, 3, 3, 5, 1>>;
  using variable_t  = iganet::S3<iganet::UniformBSpline<real_t, 1, 3, 5, 1>>;

  IgANet<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                  {50,30,70},
                                                  // Activation functions
                                                  {
                                                    {iganet::activation::tanh},
                                                    {iganet::activation::relu},
                                                    {iganet::activation::sigmoid},
                                                    {iganet::activation::none}
                                                  },
                                                  // Number of B-spline coefficients
                                                  std::tuple(iganet::to_array(4_i64, 6_i64, 3_i64)));
  
  EXPECT_EQ(net.geo().parDim(), 3);
  EXPECT_EQ(net.ref().parDim(), 3);
  EXPECT_EQ(net.out().parDim(), 3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  2);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  2);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().parDim(),  2);

  EXPECT_EQ(net.geo().geoDim(), 3);
  EXPECT_EQ(net.ref().geoDim(), 1);
  EXPECT_EQ(net.out().geoDim(), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().geoDim(),  1);
  
  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.ref().degree(0), 3);
  EXPECT_EQ(net.out().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.ref().degree(1), 5);
  EXPECT_EQ(net.out().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.ref().degree(2), 1);
  EXPECT_EQ(net.out().degree(2), 1);

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
  EXPECT_EQ(net.ref().ncoeffs(0), 4);
  EXPECT_EQ(net.out().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.ref().ncoeffs(1), 6);
  EXPECT_EQ(net.out().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.ref().ncoeffs(2), 3);
  EXPECT_EQ(net.out().ncoeffs(2), 3);

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

TEST(BSpline, IgANet_UniformBSpline_4d_double)
{
  using namespace iganet::literals;
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;
    
  using geometry_t  = iganet::S4<iganet::UniformBSpline<real_t, 4, 3, 5, 1, 4>>;
  using variable_t  = iganet::S4<iganet::UniformBSpline<real_t, 1, 3, 5, 1, 4>>;

  IgANet<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                  {50,30,70},
                                                  // Activation functions
                                                  {
                                                    {iganet::activation::tanh},
                                                    {iganet::activation::relu},
                                                    {iganet::activation::sigmoid},
                                                    {iganet::activation::none}
                                                  },
                                                  // Number of B-spline coefficients
                                                  std::tuple(iganet::to_array(4_i64, 6_i64,
                                                                              3_i64, 5_i64)));
  
  EXPECT_EQ(net.geo().parDim(), 4);
  EXPECT_EQ(net.ref().parDim(), 4);
  EXPECT_EQ(net.out().parDim(), 4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().parDim(), 3);

  EXPECT_EQ(net.geo().geoDim(), 4);
  EXPECT_EQ(net.ref().geoDim(), 1);
  EXPECT_EQ(net.out().geoDim(), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().geoDim(), 1);
  
  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.ref().degree(0), 3);
  EXPECT_EQ(net.out().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.ref().degree(1), 5);
  EXPECT_EQ(net.out().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.ref().degree(2), 1);
  EXPECT_EQ(net.out().degree(2), 1);

  EXPECT_EQ(net.geo().degree(3), 4);
  EXPECT_EQ(net.ref().degree(3), 4);
  EXPECT_EQ(net.out().degree(3), 4);

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
  EXPECT_EQ(net.ref().ncoeffs(0), 4);
  EXPECT_EQ(net.out().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.ref().ncoeffs(1), 6);
  EXPECT_EQ(net.out().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.ref().ncoeffs(2), 3);
  EXPECT_EQ(net.out().ncoeffs(2), 3);

  EXPECT_EQ(net.geo().ncoeffs(3), 5);
  EXPECT_EQ(net.ref().ncoeffs(3), 5);
  EXPECT_EQ(net.out().ncoeffs(3), 5);

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

TEST(BSpline, IgANet_NonUniformBSpline_1d_double)
{
  using namespace iganet::literals;
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;
  
 
  using geometry_t  = iganet::S1<iganet::UniformBSpline<real_t, 1, 5>>;
  using variable_t  = iganet::S1<iganet::NonUniformBSpline<real_t, 1, 5>>;

  IgANet<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                  {50,30,70},
                                                  // Activation functions
                                                  {
                                                    {iganet::activation::tanh},
                                                    {iganet::activation::relu},
                                                    {iganet::activation::sigmoid},
                                                    {iganet::activation::none}
                                                  },
                                                  // Number of B-spline coefficients
                                                  std::tuple(iganet::to_array(6_i64)));
  
  EXPECT_EQ(net.geo().parDim(), 1);
  EXPECT_EQ(net.ref().parDim(), 1);
  EXPECT_EQ(net.out().parDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::left>().parDim(),  0);
  EXPECT_EQ(net.bdr().side<iganet::side::right>().parDim(), 0);

  EXPECT_EQ(net.geo().geoDim(), 1);
  EXPECT_EQ(net.ref().geoDim(), 1);
  EXPECT_EQ(net.out().geoDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::left>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::right>().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 5);
  EXPECT_EQ(net.ref().degree(0), 5);
  EXPECT_EQ(net.out().degree(0), 5);

  EXPECT_EQ(net.geo().ncoeffs(0), 6);
  EXPECT_EQ(net.ref().ncoeffs(0), 6);
  EXPECT_EQ(net.out().ncoeffs(0), 6);
}

TEST(BSpline, IgANet_NonUniformBSpline_2d_double)
{
  using namespace iganet::literals;
  using real_t      = double;
  using optimizer_t = torch::optim::Adam; 
  
  using geometry_t  = iganet::S2<iganet::NonUniformBSpline<real_t, 2, 3, 5>>;
  using variable_t  = iganet::S2<iganet::NonUniformBSpline<real_t, 1, 3, 5>>;

  IgANet<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                  {50,30,70},
                                                  // Activation functions
                                                  {
                                                    {iganet::activation::tanh},
                                                    {iganet::activation::relu},
                                                    {iganet::activation::sigmoid},
                                                    {iganet::activation::none}
                                                  },
                                                  // Number of B-spline coefficients
                                                  std::tuple(iganet::to_array(4_i64, 6_i64)));
  
  EXPECT_EQ(net.geo().parDim(), 2);
  EXPECT_EQ(net.ref().parDim(), 2);
  EXPECT_EQ(net.out().parDim(), 2);
  
  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 1);

  EXPECT_EQ(net.geo().geoDim(), 2);
  EXPECT_EQ(net.ref().geoDim(), 1);
  EXPECT_EQ(net.out().geoDim(), 1);
  
  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.ref().degree(0), 3);
  EXPECT_EQ(net.out().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.ref().degree(1), 5);
  EXPECT_EQ(net.out().degree(1), 5);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().degree(0),  5);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().degree(0), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().degree(0), 3);

  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.ref().ncoeffs(0), 4);
  EXPECT_EQ(net.out().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.ref().ncoeffs(1), 6);
  EXPECT_EQ(net.out().ncoeffs(1), 6);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().ncoeffs(0),  6);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().ncoeffs(0), 4);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().ncoeffs(0), 4);
}

TEST(BSpline, IgANet_NonUniformBSpline_3d_double)
{
  using namespace iganet::literals;
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;
    
  using geometry_t  = iganet::S3<iganet::NonUniformBSpline<real_t, 3, 3, 5, 1>>;
  using variable_t  = iganet::S3<iganet::NonUniformBSpline<real_t, 1, 3, 5, 1>>;

  IgANet<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                  {50,30,70},
                                                  // Activation functions
                                                  {
                                                    {iganet::activation::tanh},
                                                    {iganet::activation::relu},
                                                    {iganet::activation::sigmoid},
                                                    {iganet::activation::none}
                                                  },
                                                  // Number of B-spline coefficients
                                                  std::tuple(iganet::to_array(4_i64, 6_i64,
                                                                              3_i64)));
  
  EXPECT_EQ(net.geo().parDim(), 3);
  EXPECT_EQ(net.ref().parDim(), 3);
  EXPECT_EQ(net.out().parDim(), 3);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  2);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  2);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().parDim(), 2);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().parDim(),  2);

  EXPECT_EQ(net.geo().geoDim(), 3);
  EXPECT_EQ(net.ref().geoDim(), 1);
  EXPECT_EQ(net.out().geoDim(), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().geoDim(),  1);
  
  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.ref().degree(0), 3);
  EXPECT_EQ(net.out().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.ref().degree(1), 5);
  EXPECT_EQ(net.out().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.ref().degree(2), 1);
  EXPECT_EQ(net.out().degree(2), 1);

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
  EXPECT_EQ(net.ref().ncoeffs(0), 4);
  EXPECT_EQ(net.out().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.ref().ncoeffs(1), 6);
  EXPECT_EQ(net.out().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.ref().ncoeffs(2), 3);
  EXPECT_EQ(net.out().ncoeffs(2), 3);

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

TEST(BSpline, IgANet_NonUniformBSpline_4d_double)
{
  using namespace iganet::literals;
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;
    
  using geometry_t  = iganet::S4<iganet::NonUniformBSpline<real_t, 4, 3, 5, 1, 4>>;
  using variable_t  = iganet::S4<iganet::NonUniformBSpline<real_t, 1, 3, 5, 1, 4>>;

  IgANet<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                  {50,30,70},
                                                  // Activation functions
                                                  {
                                                    {iganet::activation::tanh},
                                                    {iganet::activation::relu},
                                                    {iganet::activation::sigmoid},
                                                    {iganet::activation::none}
                                                  },
                                                  // Number of B-spline coefficients
                                                  std::tuple(iganet::to_array(4_i64, 6_i64,
                                                                              3_i64, 5_i64)));
  
  EXPECT_EQ(net.geo().parDim(), 4);
  EXPECT_EQ(net.ref().parDim(), 4);
  EXPECT_EQ(net.out().parDim(), 4);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().parDim(),  3);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().parDim(), 3);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().parDim(), 3);

  EXPECT_EQ(net.geo().geoDim(), 4);
  EXPECT_EQ(net.ref().geoDim(), 1);
  EXPECT_EQ(net.out().geoDim(), 1);

  EXPECT_EQ(net.bdr().side<iganet::side::east>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::west>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::south>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::north>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::front>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::back>().geoDim(),  1);
  EXPECT_EQ(net.bdr().side<iganet::side::stime>().geoDim(), 1);
  EXPECT_EQ(net.bdr().side<iganet::side::etime>().geoDim(), 1);
  
  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.ref().degree(0), 3);
  EXPECT_EQ(net.out().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.ref().degree(1), 5);
  EXPECT_EQ(net.out().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.ref().degree(2), 1);
  EXPECT_EQ(net.out().degree(2), 1);

  EXPECT_EQ(net.geo().degree(3), 4);
  EXPECT_EQ(net.ref().degree(3), 4);
  EXPECT_EQ(net.out().degree(3), 4);

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
  EXPECT_EQ(net.ref().ncoeffs(0), 4);
  EXPECT_EQ(net.out().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.ref().ncoeffs(1), 6);
  EXPECT_EQ(net.out().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.ref().ncoeffs(2), 3);
  EXPECT_EQ(net.out().ncoeffs(2), 3);

  EXPECT_EQ(net.geo().ncoeffs(3), 5);
  EXPECT_EQ(net.ref().ncoeffs(3), 5);
  EXPECT_EQ(net.out().ncoeffs(3), 5);

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
  iganet::init();
  return RUN_ALL_TESTS();
}
