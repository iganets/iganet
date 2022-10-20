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

TEST(Performance, UniformBSpline_parDim1_float)
{
  iganet::core<float> core_;

  std::cout << std::scientific << std::setprecision(3);
  
  for (int64_t ncoeffs : {10, 100, 1000, 10000}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      { // linear B-Splines
        iganet::UniformBSpline<float, 1, 1> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<float, 1, 2> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // cubic B-Splines
        iganet::UniformBSpline<float, 1, 3> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quartic B-Splines
        iganet::UniformBSpline<float, 1, 4> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quintic B-Splines
        iganet::UniformBSpline<float, 1, 5> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
        std::cout << std::endl;
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

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<double, 1, 2> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // cubic B-Splines
        iganet::UniformBSpline<double, 1, 3> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quartic B-Splines
        iganet::UniformBSpline<double, 1, 4> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quintic B-Splines
        iganet::UniformBSpline<double, 1, 5> bspline({ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(xi[0]);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)"
                    << std::endl;
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
        std::cout << std::endl;
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

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<float, 1, 2, 2> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // cubic B-Splines
        iganet::UniformBSpline<float, 1, 3, 3> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quartic B-Splines
        iganet::UniformBSpline<float, 1, 4, 4> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quintic B-Splines
        iganet::UniformBSpline<float, 1, 5, 5> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)"
                    << std::endl;
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
        std::cout << std::endl;
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

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<double, 1, 2, 2> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // cubic B-Splines
        iganet::UniformBSpline<double, 1, 3, 3> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quartic B-Splines
        iganet::UniformBSpline<double, 1, 4, 4> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quintic B-Splines
        iganet::UniformBSpline<double, 1, 5, 5> bspline({ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()), torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(2, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)"
                    << std::endl;
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
        std::cout << std::endl;
      }
    }
  }
}

TEST(Performance, UniformBSpline_parDim3_float)
{
  iganet::core<float> core_;

  for (int64_t ncoeffs : {10, 100, 1000, 10000}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      { // linear B-Splines
        iganet::UniformBSpline<float, 1, 1, 1, 1> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<float, 1, 2, 2, 2> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // cubic B-Splines
        iganet::UniformBSpline<float, 1, 3, 3, 3> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quartic B-Splines
        iganet::UniformBSpline<float, 1, 4, 4, 4> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quintic B-Splines
        iganet::UniformBSpline<float, 1, 5, 5, 5> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)"
                    << std::endl;
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
        std::cout << std::endl;
      }
    }
  }
}

TEST(Performance, UniformBSpline_parDim3_double)
{
  iganet::core<double> core_;

  for (int64_t ncoeffs : {10, 100, 1000, 10000}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      { // linear B-Splines
        iganet::UniformBSpline<double, 1, 1, 1, 1> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << "   ("
                    << std::right << std::setw(8) << ncoeffs << ","
                    << std::right << std::setw(8) << nsamples << ") "
                    << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quadratic B-Splines
        iganet::UniformBSpline<double, 1, 2, 2, 2> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // cubic B-Splines
        iganet::UniformBSpline<double, 1, 3, 3, 4> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quartic B-Splines
        iganet::UniformBSpline<double, 1, 4, 4, 4> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)";
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
      }

      { // quintic B-Splines
        iganet::UniformBSpline<double, 1, 5, 5, 5> bspline({ncoeffs, ncoeffs, ncoeffs}, iganet::BSplineInit::linear);
        iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options()),
                                   torch::rand(nsamples, core_.options())};

        if (nsamples == 1) {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            bspline.eval<iganet::BSplineDeriv::func>(torch::rand(3, core_.options()));
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                    << " (ns/entry)"
                    << std::endl;
        } else {
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<10; i++)
            bspline.eval_<iganet::BSplineDeriv::func>(xi);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
                    << " (ns/entry)";
        }

#ifdef SPLINELIB
        {
          auto splinelib_bspline = to_splinelib_bspline(bspline);

          // B-spline evaluation
          using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
          using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
          
          auto t1 = std::chrono::high_resolution_clock::now();
          for (int i=0; i<1000; i++)
            splinelib_bspline(ParametricCoordinate
                              {
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5},
                                ScalarParametricCoordinate{0.5}
                              });
          auto t2 = std::chrono::high_resolution_clock::now();
          std::cout << std::right << std::setw(10)
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / float(1000)
                    << " (ns/entry)";
        }
#endif
        std::cout << std::endl;
      }
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
