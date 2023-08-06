/**
   @file unittests/unittest_options.cxx

   @brief Options unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>
#include <complex>

#include <gtest/gtest.h>

TEST(Options, Options_default)
{
  iganet::Options<double> options;

  EXPECT_EQ(options.dtype(),         c10::ScalarType::Double);
  EXPECT_EQ(options.device(),        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  EXPECT_EQ(options.layout(),        torch::kStrided);
  EXPECT_EQ(options.requires_grad(), false);
  EXPECT_EQ(options.pinned_memory(), false);
  EXPECT_EQ(options.is_sparse(),     false);  
}

TEST(Options, Options_nondefault)
{
  auto options = iganet::Options<float>{}.device(torch::kCPU).layout(torch::kSparse).requires_grad(true);

  EXPECT_EQ(options.dtype(),         c10::ScalarType::Float);
  EXPECT_EQ(options.device(),        torch::kCPU);
  EXPECT_EQ(options.layout(),        torch::kSparse);
  EXPECT_EQ(options.requires_grad(), true);
  EXPECT_EQ(options.pinned_memory(), false);
  EXPECT_EQ(options.is_sparse(),     true);  
}

TEST(Options, Options_dtype)
{
  EXPECT_EQ(iganet::Options<double>{}.dtype(),       c10::ScalarType::Double);
  EXPECT_EQ(iganet::Options<float>{}.dtype(),        c10::ScalarType::Float);
  EXPECT_EQ(iganet::Options<iganet::half>{}.dtype(), c10::ScalarType::Half);
  EXPECT_EQ(iganet::Options<long>{}.dtype(),         c10::ScalarType::Long);
  EXPECT_EQ(iganet::Options<int>{}.dtype(),          c10::ScalarType::Int);
  EXPECT_EQ(iganet::Options<short>{}.dtype(),        c10::ScalarType::Short);
  EXPECT_EQ(iganet::Options<char>{}.dtype(),         c10::ScalarType::Char);
  EXPECT_EQ(iganet::Options<bool>{}.dtype(),         c10::ScalarType::Bool);

  EXPECT_EQ(iganet::Options<std::complex<double>>{}.dtype(),       c10::ScalarType::ComplexDouble);
  EXPECT_EQ(iganet::Options<std::complex<float>>{}.dtype(),        c10::ScalarType::ComplexFloat);
  EXPECT_EQ(iganet::Options<std::complex<iganet::half>>{}.dtype(), c10::ScalarType::ComplexHalf);
}

TEST(Options, Options_clone)
{
  auto options = iganet::Options<float>{}.device(torch::kCPU).layout(torch::kSparse).requires_grad(false);

  auto options_clone(options);
  
  EXPECT_EQ(options_clone.dtype(),         c10::ScalarType::Float);
  EXPECT_EQ(options_clone.device(),        torch::kCPU);
  EXPECT_EQ(options_clone.layout(),        torch::kSparse);
  EXPECT_EQ(options_clone.requires_grad(), false);
  EXPECT_EQ(options_clone.pinned_memory(), false);
  EXPECT_EQ(options_clone.is_sparse(),     true);
}

TEST(Options, Options_conversion)
{
  auto options = iganet::Options<float>{}.device(torch::kCPU).layout(torch::kSparse).requires_grad(true);

  torch::TensorOptions tensorOptions(options);
  
  EXPECT_EQ(tensorOptions.dtype(),         c10::ScalarType::Float);
  EXPECT_EQ(tensorOptions.device(),        torch::kCPU);
  EXPECT_EQ(tensorOptions.layout(),        torch::kSparse);
  EXPECT_EQ(tensorOptions.requires_grad(), true);
  EXPECT_EQ(tensorOptions.pinned_memory(), false);
  EXPECT_EQ(tensorOptions.is_sparse(),     true);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
