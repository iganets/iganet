/**
   @file unittests/unittest_performance.cxx

   @brief Performance unittests

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <filesystem>
#include <iostream>
#include <chrono>

#include <gtest/gtest.h>

TEST(Performance, MatmulTensorLayout_double)
{
  iganet::core<double> core_;
  
  for (short_t n : {2, 3, 4, 5}) {
    for (int64_t m : {100, 500, 1000, 5000, 10000, 50000, 100000}) {

      { // (n,m) data format
        torch::Tensor a = torch::ones({n,m}, core_.options());
        torch::Tensor b = torch::ones({n,m}, core_.options());
        torch::Tensor c;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<100; i++)
          c = torch::sum(torch::mul(a,b),0);
        auto t2 = std::chrono::high_resolution_clock::now();
        
        std::cout << "("
                  << std::right << std::setw(8) << n << ","
                  << std::right << std::setw(8) << m << ") "
                  << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(n*m*100)
                  << " (ns/entry)";
        
        EXPECT_EQ(c.sizes(), c10::IntArrayRef({m}));
      }
      
      { // (m,n) data format
        torch::Tensor a = torch::ones({m,n}, core_.options());
        torch::Tensor b = torch::ones({m,n}, core_.options());
        torch::Tensor c;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<100; i++)
          c = torch::sum(torch::mul(a,b),1);
        auto t2 = std::chrono::high_resolution_clock::now();

        std::cout << "   ("
                  << std::right << std::setw(8) << m << ","
                  << std::right << std::setw(8) << n << ") "
                  << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(n*m*100)
                  << " (ns/entry)"
                  << std::endl;
        
        EXPECT_EQ(c.sizes(), c10::IntArrayRef({m}));  
      }      
    }
  } 
}

TEST(Performance, UniformBSpline_parDim1_float)
{
  iganet::core<float> core_;
  
  for (int64_t ncoeffs : {10, 100, 1000, 10000}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      { // linear B-Splines
        iganet::UniformBSpline<float, 1, 1> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "   ("
                  << std::right << std::setw(8) << ncoeffs << ","
                  << std::right << std::setw(8) << nsamples << ") "
                  << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)";
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<float, 1, 2> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)";
      }

      { // cubic B-Splines
        iganet::UniformBSpline<float, 1, 3> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)";
      }

      { // quartic B-Splines
        iganet::UniformBSpline<float, 1, 4> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)";
      }
      
      { // quintic B-Splines
        iganet::UniformBSpline<float, 1, 5> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)"
                  << std::endl;
      }
    }
  }
}

TEST(Performance, UniformBSpline_parDim1_double)
{
  iganet::core<double> core_;
  
  for (int64_t ncoeffs : {10, 100, 1000, 10000}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      { // linear B-Splines
        iganet::UniformBSpline<double, 1, 1> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "   ("
                  << std::right << std::setw(8) << ncoeffs << ","
                  << std::right << std::setw(8) << nsamples << ") "
                  << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)";
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<double, 1, 2> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)";
      }

      { // cubic B-Splines
        iganet::UniformBSpline<double, 1, 3> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)";
      }

      { // quartic B-Splines
        iganet::UniformBSpline<double, 1, 4> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)";
      }
      
      { // quintic B-Splines
        iganet::UniformBSpline<double, 1, 5> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)"
                  << std::endl;
      }
    }
  }
}

TEST(Performance, UniformBSpline_parDim2_float)
{
  iganet::core<float> core_;
  
  for (int64_t ncoeffs : {10, 100, 1000, 10000}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      { // linear B-Splines
        iganet::UniformBSpline<float, 1, 1, 1> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "   ("
                  << std::right << std::setw(8) << ncoeffs << ","
                  << std::right << std::setw(8) << nsamples << ") "
                  << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)";
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<float, 1, 2, 2> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)";
      }

      { // cubic B-Splines
        iganet::UniformBSpline<float, 1, 3, 3> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)";
      }

      { // quartic B-Splines
        iganet::UniformBSpline<float, 1, 4, 4> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)";
      }
      
      { // quintic B-Splines
        iganet::UniformBSpline<float, 1, 5,5 > bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                  << " (ns/entry)"
                  << std::endl;
      }
    }
  }
}

TEST(Performance, UniformBSpline_parDim2_double)
{
  iganet::core<double> core_;
  
  for (int64_t ncoeffs : {10, 100, 1000, 10000}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      { // linear B-Splines
        iganet::UniformBSpline<double, 1, 1, 1> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "   ("
                  << std::right << std::setw(8) << ncoeffs << ","
                  << std::right << std::setw(8) << nsamples << ") "
                  << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)";
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<double, 1, 2, 2> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)";
      }

      { // cubic B-Splines
        iganet::UniformBSpline<double, 1, 3, 3> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)";
      }

      { // quartic B-Splines
        iganet::UniformBSpline<double, 1, 4, 4> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)";
      }
      
      { // quintic B-Splines
        iganet::UniformBSpline<double, 1, 5,5 > bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};
        
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i=0; i<10; i++)
          bspline.eval_<iganet::BSplineDeriv::func>(xi);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::right << std::setw(12)
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                  << " (ns/entry)"
                  << std::endl;
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
