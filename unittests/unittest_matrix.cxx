/**
   @file unittests/unittest_matrix.cxx

   @brief Compile-time unittests

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <iostream>

#include <gtest/gtest.h>

TEST(Matrix, Matrix_2x1_double)
{
  iganet::Matrix<torch::Tensor, 2, 1> A  ( torch::ones({5,5}), 2*torch::ones({5,5}) );
  iganet::Matrix<torch::Tensor, 2, 1> B = {3*torch::ones({5,5}), 4*torch::ones({5,5})};
  
  EXPECT_EQ(A+B,
            (iganet::Matrix<torch::Tensor, 2, 1>( 4*torch::ones({5,5}),
                                                  6*torch::ones({5,5}))) );

  EXPECT_EQ(A-B,
            (iganet::Matrix<torch::Tensor, 2, 1>( -2*torch::ones({5,5}),
                                                  -2*torch::ones({5,5}))) );
  EXPECT_EQ(A.tr(),
            (iganet::Matrix<torch::Tensor, 1, 2>(   torch::ones({5,5}),
                                                  2*torch::ones({5,5}))) );

  EXPECT_EQ(B.tr(),
            (iganet::Matrix<torch::Tensor, 1, 2>( 3*torch::ones({5,5}),
                                                  4*torch::ones({5,5}))) );
  
  EXPECT_EQ(A*B.tr(),
            (iganet::Matrix<torch::Tensor, 2, 2>( 3*torch::ones({5,5}),
                                                  4*torch::ones({5,5}),
                                                  6*torch::ones({5,5}),
                                                  8*torch::ones({5,5}))) );
  
  EXPECT_EQ(A.tr()*B,
            (iganet::Matrix<torch::Tensor, 1, 1>( 11*torch::ones({5,5}))) );

  
  iganet::Matrix<torch::Tensor, 1, 1> C( 5*torch::ones({5,5}) );

  EXPECT_EQ(C.inv(),
            (iganet::Matrix<torch::Tensor, 1, 1>( 0.2*torch::ones({5,5}))) );

  iganet::Matrix<torch::Tensor, 2, 2> D(   torch::ones({5,5}),
                                         2*torch::ones({5,5}),
                                         3*torch::ones({5,5}),
                                         4*torch::ones({5,5}));

  EXPECT_EQ(D.inv(),
            (iganet::Matrix<torch::Tensor, 2, 2>( -2.0*torch::ones({5,5}),
                                                   1.5*torch::ones({5,5}),
                                                   1.0*torch::ones({5,5}),
                                                  -0.5*torch::ones({5,5}))) );

  iganet::Matrix<torch::Tensor, 3, 3> E( 2*torch::ones({5,5}),
                                           torch::ones({5,5}),
                                           torch::zeros({5,5}),
                                           torch::ones({5,5}),
                                         2*torch::ones({5,5}),
                                           torch::ones({5,5}),
                                           torch::zeros({5,5}),
                                           torch::ones({5,5}),
                                         2*torch::ones({5,5}));

  EXPECT_EQ(E.inv(),
            (iganet::Matrix<torch::Tensor, 3, 3>(  0.75*torch::ones({5,5}),
                                                  -0.50*torch::ones({5,5}),
                                                   0.25*torch::ones({5,5}),
                                                  -0.50*torch::ones({5,5}),
                                                        torch::ones({5,5}),
                                                  -0.50*torch::ones({5,5}),
                                                   0.25*torch::ones({5,5}),
                                                  -0.50*torch::ones({5,5}),
                                                   0.75*torch::ones({5,5}))) );
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
