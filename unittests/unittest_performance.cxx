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

#include "unittest_splinelib.hpp"
#include <gtest/gtest.h>

#define SPLINELIB

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

namespace unittest {

  template<typename real_t, short_t GeoDim,
           short_t Degree0, iganet::BSplineDeriv deriv>
  auto test_UniformBSpline(int64_t ncoeffs, int64_t nsamples)
  {
    iganet::core<real_t> core_;
    iganet::UniformBSpline<real_t, GeoDim, Degree0> bspline({ncoeffs}, iganet::BSplineInit::linear);
    iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<10; i++)
      bspline.template eval<deriv>(xi);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::right << std::setw(10)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
              << " (ns/entry)";
    
#ifdef SPLINELIB
    if (nsamples == 1) {
      auto splinelib_bspline = to_splinelib_bspline(bspline);
      
      // B-spline evaluation
      using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
      using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
      using Derivative                 = typename decltype(splinelib_bspline)::ParameterSpace_::Derivative_;
      using ScalarDerivative           = typename Derivative::value_type;
      
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<1000; i++)
        splinelib_bspline(ParametricCoordinate
                          {
                            ScalarParametricCoordinate{0.5}
                          },
                          Derivative
                          {
                            ScalarDerivative{(short_t) deriv % 10}
                          });
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                << " (ns/entry)";
    } else {
      std::cout << std::right << std::setw(10)
                << "--"
                << " (ns/entry)";
    }
#endif
  }

  template<typename real_t, short_t GeoDim,
           short_t Degree0, short_t Degree1,
           iganet::BSplineDeriv deriv>
  auto test_UniformBSpline(int64_t ncoeffs, int64_t nsamples)
  {
    iganet::core<real_t> core_;
    iganet::UniformBSpline<real_t, GeoDim, Degree0, Degree1> bspline({ncoeffs,
                                                                      ncoeffs}, iganet::BSplineInit::linear);
    iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options())};
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<10; i++)
      bspline.template eval<deriv>(xi);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::right << std::setw(10)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
              << " (ns/entry)";
    
#ifdef SPLINELIB
    if (nsamples == 1) {
      auto splinelib_bspline = to_splinelib_bspline(bspline);
      
      // B-spline evaluation
      using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
      using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
      using Derivative                 = typename decltype(splinelib_bspline)::ParameterSpace_::Derivative_;
      using ScalarDerivative           = typename Derivative::value_type;
      
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<1000; i++)
        splinelib_bspline(ParametricCoordinate
                          {
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5}
                          },
                          Derivative
                          {
                            ScalarDerivative{ (short_t)deriv    %10},
                            ScalarDerivative{((short_t)deriv/10)%10}
                          });
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                << " (ns/entry)";
    } else {
      std::cout << std::right << std::setw(10)
                << "--"
                << " (ns/entry)";
    }
#endif
  }

  template<typename real_t, short_t GeoDim,
           short_t Degree0, short_t Degree1, short_t Degree2,
           iganet::BSplineDeriv deriv>
  auto test_UniformBSpline(int64_t ncoeffs, int64_t nsamples)
  {
    iganet::core<real_t> core_;
    iganet::UniformBSpline<real_t, GeoDim, Degree0, Degree1, Degree2> bspline({ncoeffs,
                                                                               ncoeffs,
                                                                               ncoeffs}, iganet::BSplineInit::linear);
    iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options())};
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<10; i++)
      bspline.template eval<deriv>(xi);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::right << std::setw(10)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
              << " (ns/entry)";
    
#ifdef SPLINELIB
    if (nsamples == 1) {
      auto splinelib_bspline = to_splinelib_bspline(bspline);
      
      // B-spline evaluation
      using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
      using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
      using Derivative                 = typename decltype(splinelib_bspline)::ParameterSpace_::Derivative_;
      using ScalarDerivative           = typename Derivative::value_type;
      
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<1000; i++)
        splinelib_bspline(ParametricCoordinate
                          {
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5}
                          },
                          Derivative
                          {
                            ScalarDerivative{ (short_t)deriv     %10},
                            ScalarDerivative{((short_t)deriv/ 10)%10},
                            ScalarDerivative{((short_t)deriv/100)%10}
                          });
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                << " (ns/entry)";
    } else {
      std::cout << std::right << std::setw(10)
                << "--"
                << " (ns/entry)";
    }
#endif
  }

  template<typename real_t, short_t GeoDim,
           short_t Degree0, short_t Degree1, short_t Degree2, short_t Degree3,
           iganet::BSplineDeriv deriv>
  auto test_UniformBSpline(int64_t ncoeffs, int64_t nsamples)
  {
    iganet::core<real_t> core_;
    iganet::UniformBSpline<real_t, GeoDim, Degree0, Degree1, Degree2, Degree3> bspline({ncoeffs,
                                                                                        ncoeffs,
                                                                                        ncoeffs,
                                                                                        ncoeffs}, iganet::BSplineInit::linear);
    iganet::TensorArray4 xi = {torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options())};
    
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i=0; i<10; i++)
      bspline.template eval<deriv>(xi);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::right << std::setw(10)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
              << " (ns/entry)";
    
#ifdef SPLINELIB
    if (nsamples == 1) {
      auto splinelib_bspline = to_splinelib_bspline(bspline);
      
      // B-spline evaluation
      using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
      using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
      using Derivative                 = typename decltype(splinelib_bspline)::ParameterSpace_::Derivative_;
      using ScalarDerivative           = typename Derivative::value_type;
      
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<1000; i++)
        splinelib_bspline(ParametricCoordinate
                          {
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5}
                          },
                          Derivative
                          {
                            ScalarDerivative{ (short_t)deriv      %10},
                            ScalarDerivative{((short_t)deriv/  10)%10},
                            ScalarDerivative{((short_t)deriv/ 100)%10},
                            ScalarDerivative{((short_t)deriv/1000)%10}
                          });
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                << " (ns/entry)";
    } else {
      std::cout << std::right << std::setw(10)
                << "--"
                << " (ns/entry)";
    }
#endif
  }
  
} // namespace unittest

TEST(Performance, UniformBSpline_parDim1_float)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      unittest::template test_UniformBSpline<float, 1, 1, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 2, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 3, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 4, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 5, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim1_double)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      unittest::template test_UniformBSpline<double, 1, 1, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 2, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 3, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 4, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 5, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim2_float)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      unittest::template test_UniformBSpline<float, 1, 1, 1, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 2, 2, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 3, 3, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 4, 4, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 5, 5, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim2_double)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      unittest::template test_UniformBSpline<double, 1, 1, 1, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 2, 2, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 3, 3, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 4, 4, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 5, 5, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim3_float)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      unittest::template test_UniformBSpline<float, 1, 1, 1, 1, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 2, 2, 2, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 3, 3, 3, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 4, 4, 4, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 5, 5, 5, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim3_double)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      unittest::template test_UniformBSpline<double, 1, 1, 1, 1, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 2, 2, 2, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 3, 3, 3, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 4, 4, 4, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 5, 5, 5, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim4_float)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      unittest::template test_UniformBSpline<float, 1, 1, 1, 1, 1, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 2, 2, 2, 2, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 3, 3, 3, 3, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 4, 4, 4, 4, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<float, 1, 5, 5, 5, 5, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  } 
}

TEST(Performance, UniformBSpline_parDim4_double)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      unittest::template test_UniformBSpline<double, 1, 1, 1, 1, 1, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 2, 2, 2, 2, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 3, 3, 3, 3, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 4, 4, 4, 4, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      unittest::template test_UniformBSpline<double, 1, 5, 5, 5, 5, iganet::BSplineDeriv::func>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  } 
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
