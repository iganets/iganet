/**
   @file include/bspline.hpp

   @brief Multivariate B-splines

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <exception>
#include <filesystem>
#include <functional>

#include <core.hpp>
#include <blocktensor.hpp>
#include <utils.hpp>

#pragma once

namespace iganet {
  
  /// @brief Enumerator for specifying the initialization of B-spline coefficients
  enum class BSplineInit : short_t
    {
      zeros    = 0, /*!< set coefficient values to zero */
      ones     = 1, /*!< set coefficient values to one */
      linear   = 2, /*!< set coefficient values to \f$0,1,\dots \#\text{coeffs}-1\f$ */
      random   = 3, /*!< set coefficient values to random numbers */
      greville = 4  /*!< set coefficient values to the Greville abscissae */
    };

  /// @brief Enumerator for specifying the derivative of B-spline evaluation
  ///
  /// **Examples**
  ///
  /// * 3d Laplace operator `dx2+dy2+dz2`
  /// * 2d convection operator with time derivative dt+dx+dy`
  enum class BSplineDeriv : short_t
    {
      func   =    0, /*!< function value */
      
      dx     =    1, /*!< first derivative in x-direction  */
      dx1    =    1, /*!< first derivative in x-direction  */
      dx2    =    2, /*!< second derivative in x-direction */
      dx3    =    3, /*!< third derivative in x-direction  */
      dx4    =    4, /*!< fourth derivative in x-direction */
      dy     =   10, /*!< first derivative in y-direction  */
      dy1    =   10, /*!< first derivative in y-direction  */
      dy2    =   20, /*!< second derivative in y-direction */
      dy3    =   30, /*!< third derivative in y-direction  */
      dy4    =   40, /*!< fourth derivative in y-direction */
      dz     =  100, /*!< first derivative in z-direction  */
      dz1    =  100, /*!< first derivative in z-direction  */
      dz2    =  200, /*!< second derivative in z-direction */
      dz3    =  300, /*!< third derivative in z-direction  */
      dz4    =  400, /*!< fourth derivative in z-direction */
      dt     = 1000, /*!< first derivative in t-direction  */
      dt1    = 1000, /*!< first derivative in t-direction  */
      dt2    = 2000, /*!< second derivative in t-direction */
      dt3    = 3000, /*!< third derivative in t-direction  */
      dt4    = 4000, /*!< fourth derivative in t-direction */
      
      dxdy   =   11, /*!< second mixed derivative in x- and y-direction  */
      dx1dy1 =   11, /*!< second mixed derivative in x- and y-direction  */
      dx2dy1 =   12, /*!< third mixed derivative in x- and y-direction   */
      dx3dy1 =   13, /*!< fourth mixed derivative in x- and y-direction  */
      dx4dy1 =   14, /*!< fifth mixed derivative in x- and y-direction   */
      dx1dy2 =   21, /*!< third mixed derivative in x- and y-direction   */
      dx2dy2 =   22, /*!< fourth mixed derivative in x- and y-direction  */
      dx3dy2 =   23, /*!< fifth mixed derivative in x- and y-direction   */
      dx4dy2 =   24, /*!< sixth mixed derivative in x- and y-direction   */
      dx1dy3 =   31, /*!< fourth mixed derivative in x- and y-direction  */
      dx2dy3 =   32, /*!< fifth mixed derivative in x- and y-direction   */
      dx3dy3 =   33, /*!< sixth mixed derivative in x- and y-direction   */
      dx4dy3 =   34, /*!< seventh mixed derivative in x- and y-direction */
      dx1dy4 =   41, /*!< fifth mixed derivative in x- and y-direction   */
      dx2dy4 =   42, /*!< sixth mixed derivative in x- and y-direction   */
      dx3dy4 =   43, /*!< seventh mixed derivative in x- and y-direction */
      dx4dy4 =   44, /*!< eigths mixed derivative in x- and y-direction  */
      
      dxdz   =  101, /*!< second mixed derivative in x- and z-direction  */
      dx1dz1 =  101, /*!< second mixed derivative in x- and z-direction  */
      dx2dz1 =  102, /*!< third mixed derivative in x- and z-direction   */
      dx3dz1 =  103, /*!< fourth mixed derivative in x- and z-direction  */
      dx4dz1 =  104, /*!< fifth mixed derivative in x- and z-direction   */
      dx1dz2 =  201, /*!< third mixed derivative in x- and z-direction   */
      dx2dz2 =  202, /*!< fourth mixed derivative in x- and z-direction  */
      dx3dz2 =  203, /*!< fifth mixed derivative in x- and z-direction   */
      dx4dz2 =  204, /*!< sixth mixed derivative in x- and z-direction   */
      dx1dz3 =  301, /*!< fourth mixed derivative in x- and z-direction  */
      dx2dz3 =  302, /*!< fifth mixed derivative in x- and z-direction   */
      dx3dz3 =  303, /*!< sixth mixed derivative in x- and z-direction   */
      dx4dz3 =  304, /*!< seventh mixed derivative in x- and z-direction */
      dx1dz4 =  401, /*!< fifth mixed derivative in x- and z-direction   */
      dx2dz4 =  402, /*!< sixth mixed derivative in x- and z-direction   */
      dx3dz4 =  403, /*!< seventh mixed derivative in x- and z-direction */
      dx4dz4 =  404, /*!< eigths mixed derivative in x- and z-direction  */
      
      dxdt   = 1001, /*!< second mixed derivative in x- and t-direction  */
      dx1dt1 = 1001, /*!< second mixed derivative in x- and t-direction  */
      dx2dt1 = 1002, /*!< third mixed derivative in x- and t-direction   */
      dx3dt1 = 1003, /*!< fourth mixed derivative in x- and t-direction  */
      dx4dt1 = 1004, /*!< fifth mixed derivative in x- and t-direction   */
      dx1dt2 = 2001, /*!< third mixed derivative in x- and t-direction   */
      dx2dt2 = 2002, /*!< fourth mixed derivative in x- and t-direction  */
      dx3dt2 = 2003, /*!< fifth mixed derivative in x- and t-direction   */
      dx4dt2 = 2004, /*!< sixth mixed derivative in x- and t-direction   */
      dx1dt3 = 3001, /*!< fourth mixed derivative in x- and t-direction  */
      dx2dt3 = 3002, /*!< fifth mixed derivative in x- and t-direction   */
      dx3dt3 = 3003, /*!< sixth mixed derivative in x- and t-direction   */
      dx4dt3 = 3004, /*!< seventh mixed derivative in x- and t-direction */
      dx1dt4 = 4001, /*!< fifth mixed derivative in x- and t-direction   */
      dx2dt4 = 4002, /*!< sixth mixed derivative in x- and t-direction   */
      dx3dt4 = 4003, /*!< seventh mixed derivative in x- and t-direction */
      dx4dt4 = 4004, /*!< eigths mixed derivative in x- and t-direction  */

      dydx   =   11, /*!< second mixed derivative in y- and x-direction  */
      dy1dx1 =   11, /*!< second mixed derivative in y- and x-direction  */
      dy2dx1 =   21, /*!< third mixed derivative in y- and x-direction   */
      dy3dx1 =   31, /*!< fourth mixed derivative in y- and x-direction  */
      dy4dx1 =   41, /*!< fifth mixed derivative in y- and x-direction   */
      dy1dx2 =   12, /*!< third mixed derivative in y- and x-direction   */
      dy2dx2 =   22, /*!< fourth mixed derivative in y- and x-direction  */
      dy3dx2 =   32, /*!< fifth mixed derivative in y- and x-direction   */
      dy4dx2 =   42, /*!< sixth mixed derivative in y- and x-direction   */
      dy1dx3 =   13, /*!< fourth mixed derivative in y- and x-direction  */
      dy2dx3 =   23, /*!< fifth mixed derivative in y- and x-direction   */
      dy3dx3 =   33, /*!< sixth mixed derivative in y- and x-direction   */
      dy4dx3 =   43, /*!< seventh mixed derivative in y- and x-direction */
      dy1dx4 =   14, /*!< fifth mixed derivative in y- and x-direction   */
      dy2dx4 =   24, /*!< sixth mixed derivative in y- and x-direction   */
      dy3dx4 =   34, /*!< seventh mixed derivative in y- and x-direction */
      dy4dx4 =   44, /*!< eigths mixed derivative in y- and x-direction  */

      dydz   =  110, /*!< second mixed derivative in y- and z-direction  */
      dy1dz1 =  110, /*!< second mixed derivative in y- and z-direction  */
      dy2dz1 =  120, /*!< third mixed derivative in y- and z-direction   */
      dy3dz1 =  130, /*!< fourth mixed derivative in y- and z-direction  */
      dy4dz1 =  140, /*!< fifth mixed derivative in y- and z-direction   */
      dy1dz2 =  210, /*!< third mixed derivative in y- and z-direction   */
      dy2dz2 =  220, /*!< fourth mixed derivative in y- and z-direction  */
      dy3dz2 =  230, /*!< fifth mixed derivative in y- and z-direction   */
      dy4dz2 =  240, /*!< sixth mixed derivative in y- and z-direction   */
      dy1dz3 =  310, /*!< fourth mixed derivative in y- and z-direction  */
      dy2dz3 =  320, /*!< fifth mixed derivative in y- and z-direction   */
      dy3dz3 =  330, /*!< sixth mixed derivative in y- and z-direction   */
      dy4dz3 =  340, /*!< seventh mixed derivative in y- and z-direction */
      dy1dz4 =  410, /*!< fifth mixed derivative in y- and z-direction   */
      dy2dz4 =  420, /*!< sixth mixed derivative in y- and z-direction   */
      dy3dz4 =  430, /*!< seventh mixed derivative in y- and z-direction */
      dy4dz4 =  440, /*!< eigths mixed derivative in y- and z-direction  */

      dydt   = 1010, /*!< second mixed derivative in y- and t-direction  */
      dy1dt1 = 1010, /*!< second mixed derivative in y- and t-direction  */
      dy2dt1 = 1020, /*!< third mixed derivative in y- and t-direction   */
      dy3dt1 = 1030, /*!< fourth mixed derivative in y- and t-direction  */
      dy4dt1 = 1040, /*!< fifth mixed derivative in y- and t-direction   */
      dy1dt2 = 2010, /*!< third mixed derivative in y- and t-direction   */
      dy2dt2 = 2020, /*!< fourth mixed derivative in y- and t-direction  */
      dy3dt2 = 2030, /*!< fifth mixed derivative in y- and t-direction   */
      dy4dt2 = 2040, /*!< sixth mixed derivative in y- and t-direction   */
      dy1dt3 = 3010, /*!< fourth mixed derivative in y- and t-direction  */
      dy2dt3 = 3020, /*!< fifth mixed derivative in y- and t-direction   */
      dy3dt3 = 3030, /*!< sixth mixed derivative in y- and t-direction   */
      dy4dt3 = 3040, /*!< seventh mixed derivative in y- and t-direction */
      dy1dt4 = 4010, /*!< fifth mixed derivative in y- and t-direction   */
      dy2dt4 = 4020, /*!< sixth mixed derivative in y- and t-direction   */
      dy3dt4 = 4030, /*!< seventh mixed derivative in y- and t-direction */
      dy4dt4 = 4040, /*!< eigths mixed derivative in y- and t-direction  */

      dzdx   =  101, /*!< second mixed derivative in z- and x-direction  */
      dz1dx1 =  101, /*!< second mixed derivative in z- and x-direction  */
      dz2dx1 =  102, /*!< third mixed derivative in z- and x-direction   */
      dz3dx1 =  103, /*!< fourth mixed derivative in z- and x-direction  */
      dz4dx1 =  104, /*!< fifth mixed derivative in z- and x-direction   */
      dz1dx2 =  201, /*!< third mixed derivative in z- and x-direction   */
      dz2dx2 =  202, /*!< fourth mixed derivative in z- and x-direction  */
      dz3dx2 =  203, /*!< fifth mixed derivative in z- and x-direction   */
      dz4dx2 =  204, /*!< sixth mixed derivative in z- and x-direction   */
      dz1dx3 =  301, /*!< fourth mixed derivative in z- and x-direction  */
      dz2dx3 =  302, /*!< fifth mixed derivative in z- and x-direction   */
      dz3dx3 =  303, /*!< sixth mixed derivative in z- and x-direction   */
      dz4dx3 =  304, /*!< seventh mixed derivative in z- and x-direction */
      dz1dx4 =  401, /*!< fifth mixed derivative in z- and x-direction   */
      dz2dx4 =  402, /*!< sixth mixed derivative in z- and x-direction   */
      dz3dx4 =  403, /*!< seventh mixed derivative in z- and x-direction */
      dz4dx4 =  404, /*!< eigths mixed derivative in z- and x-direction  */

      dzdy   =  110, /*!< second mixed derivative in z- and y-direction  */
      dz1dy1 =  110, /*!< second mixed derivative in z- and y-direction  */
      dz2dy1 =  120, /*!< third mixed derivative in z- and y-direction   */
      dz3dy1 =  130, /*!< fourth mixed derivative in z- and y-direction  */
      dz4dy1 =  140, /*!< fifth mixed derivative in z- and y-direction   */
      dz1dy2 =  210, /*!< third mixed derivative in z- and y-direction   */
      dz2dy2 =  220, /*!< fourth mixed derivative in z- and y-direction  */
      dz3dy2 =  230, /*!< fifth mixed derivative in z- and y-direction   */
      dz4dy2 =  240, /*!< sixth mixed derivative in z- and y-direction   */
      dz1dy3 =  310, /*!< fourth mixed derivative in z- and y-direction  */
      dz2dy3 =  320, /*!< fifth mixed derivative in z- and y-direction   */
      dz3dy3 =  330, /*!< sixth mixed derivative in z- and y-direction   */
      dz4dy3 =  340, /*!< seventh mixed derivative in z- and y-direction */
      dz1dy4 =  410, /*!< fifth mixed derivative in z- and y-direction   */
      dz2dy4 =  420, /*!< sixth mixed derivative in z- and y-direction   */
      dz3dy4 =  430, /*!< seventh mixed derivative in z- and y-direction */
      dz4dy4 =  440, /*!< eigths mixed derivative in z- and y-direction  */

      dzdt   = 1100, /*!< second mixed derivative in z- and t-direction  */
      dz1dt1 = 1100, /*!< second mixed derivative in z- and t-direction  */
      dz2dt1 = 1200, /*!< third mixed derivative in z- and t-direction   */
      dz3dt1 = 1300, /*!< fourth mixed derivative in z- and t-direction  */
      dz4dt1 = 1400, /*!< fifth mixed derivative in z- and t-direction   */
      dz1dt2 = 2100, /*!< third mixed derivative in z- and t-direction   */
      dz2dt2 = 2200, /*!< fourth mixed derivative in z- and t-direction  */
      dz3dt2 = 2300, /*!< fifth mixed derivative in z- and t-direction   */
      dz4dt2 = 2400, /*!< sixth mixed derivative in z- and t-direction   */
      dz1dt3 = 3100, /*!< fourth mixed derivative in z- and t-direction  */
      dz2dt3 = 3200, /*!< fifth mixed derivative in z- and t-direction   */
      dz3dt3 = 3300, /*!< sixth mixed derivative in z- and t-direction   */
      dz4dt3 = 3400, /*!< seventh mixed derivative in z- and t-direction */
      dz1dt4 = 4100, /*!< fifth mixed derivative in z- and t-direction   */
      dz2dt4 = 4200, /*!< sixth mixed derivative in z- and t-direction   */
      dz3dt4 = 4300, /*!< seventh mixed derivative in z- and t-direction */
      dz4dt4 = 4400, /*!< eigths mixed derivative in z- and t-direction  */
      
      dtdx   = 1001, /*!< second mixed derivative in t- and x-direction  */
      dt1dx1 = 1001, /*!< second mixed derivative in t- and x-direction  */
      dt2dx1 = 1002, /*!< third mixed derivative in t- and x-direction   */
      dt3dx1 = 1003, /*!< fourth mixed derivative in t- and x-direction  */
      dt4dx1 = 1004, /*!< fifth mixed derivative in t- and x-direction   */
      dt1dx2 = 2001, /*!< third mixed derivative in t- and x-direction   */
      dt2dx2 = 2002, /*!< fourth mixed derivative in t- and x-direction  */
      dt3dx2 = 2003, /*!< fifth mixed derivative in t- and x-direction   */
      dt4dx2 = 2004, /*!< sixth mixed derivative in t- and x-direction   */
      dt1dx3 = 3001, /*!< fourth mixed derivative in t- and x-direction  */
      dt2dx3 = 3002, /*!< fifth mixed derivative in t- and x-direction   */
      dt3dx3 = 3003, /*!< sixth mixed derivative in t- and x-direction   */
      dt4dx3 = 3004, /*!< seventh mixed derivative in t- and x-direction */
      dt1dx4 = 4001, /*!< fifth mixed derivative in t- and x-direction   */
      dt2dx4 = 4002, /*!< sixth mixed derivative in t- and x-direction   */
      dt3dx4 = 4003, /*!< seventh mixed derivative in t- and x-direction */
      dt4dx4 = 4004, /*!< eigths mixed derivative in t- and x-direction  */

      dtdy   = 1010, /*!< second mixed derivative in t- and y-direction  */
      dt1dy1 = 1010, /*!< second mixed derivative in t- and y-direction  */
      dt2dy1 = 2010, /*!< third mixed derivative in t- and y-direction   */
      dt3dy1 = 3010, /*!< fourth mixed derivative in t- and y-direction  */
      dt4dy1 = 4010, /*!< fifth mixed derivative in t- and y-direction   */
      dt1dy2 = 1020, /*!< third mixed derivative in t- and y-direction   */
      dt2dy2 = 2020, /*!< fourth mixed derivative in t- and y-direction  */
      dt3dy2 = 3020, /*!< fifth mixed derivative in t- and y-direction   */
      dt4dy2 = 4020, /*!< sixth mixed derivative in t- and y-direction   */
      dt1dy3 = 1030, /*!< fourth mixed derivative in t- and y-direction  */
      dt2dy3 = 2030, /*!< fifth mixed derivative in t- and y-direction   */
      dt3dy3 = 3030, /*!< sixth mixed derivative in t- and y-direction   */
      dt4dy3 = 4030, /*!< seventh mixed derivative in t- and y-direction */
      dt1dy4 = 1040, /*!< fifth mixed derivative in t- and y-direction   */
      dt2dy4 = 2040, /*!< sixth mixed derivative in t- and y-direction   */
      dt3dy4 = 3040, /*!< seventh mixed derivative in t- and y-direction */
      dt4dy4 = 4040, /*!< eigths mixed derivative in t- and y-direction  */

      dtdz   = 1100, /*!< second mixed derivative in t- and z-direction  */
      dt1dz1 = 1100, /*!< second mixed derivative in t- and z-direction  */
      dt2dz1 = 2100, /*!< third mixed derivative in t- and z-direction   */
      dt3dz1 = 3100, /*!< fourth mixed derivative in t- and z-direction  */
      dt4dz1 = 4100, /*!< fifth mixed derivative in t- and z-direction   */
      dt1dz2 = 1200, /*!< third mixed derivative in t- and z-direction   */
      dt2dz2 = 2200, /*!< fourth mixed derivative in t- and z-direction  */
      dt3dz2 = 3200, /*!< fifth mixed derivative in t- and z-direction   */
      dt4dz2 = 4200, /*!< sixth mixed derivative in t- and z-direction   */
      dt1dz3 = 1300, /*!< fourth mixed derivative in t- and z-direction  */
      dt2dz3 = 2300, /*!< fifth mixed derivative in t- and z-direction   */
      dt3dz3 = 3300, /*!< sixth mixed derivative in t- and z-direction   */
      dt4dz3 = 4300, /*!< seventh mixed derivative in t- and z-direction */
      dt1dz4 = 1400, /*!< fifth mixed derivative in t- and z-direction   */
      dt2dz4 = 2400, /*!< sixth mixed derivative in t- and z-direction   */
      dt3dz4 = 3400, /*!< seventh mixed derivative in t- and z-direction */
      dt4dz4 = 4400, /*!< eigths mixed derivative in t- and z-direction  */
    };

  inline auto operator+(BSplineDeriv lhs, BSplineDeriv rhs)
  {
    return BSplineDeriv( static_cast<short_t>(lhs)+static_cast<short_t>(rhs) );
  }
  
  /// @brief Tensor-product uniform B-spline (core functionality)
  ///
  /// This class implements the core functionality of all B-spline
  /// classes and serves as base class for (non-)uniform B-splines.
  ///
  /// Mathematically, this class defines a mapping
  ///
  /// \f[
  /// \mathbf{f}:\hat\Omega \mapsto \Omega
  /// \f]
  ///
  /// from the \f$d_\text{par}\f$-dimensional *parametric space*
  /// \f$\hat\Omega=[0,1]^{d_\text{par}}\f$ to the
  /// \f$d_\text{geo}\f$-dimensional *geometric space*
  /// \f$\Omega\subset\mathbb{R}^{d_\text{geo}}\f$.
  ///
  /// This mapping is defined by tensor-product B-spline basis
  /// functions
  ///
  /// \f[  
  /// B_I(\boldsymbol{\xi}) = \bigotimes_{d=1}^{d_\text{par}} B_{i_d,p_d}(\xi_d)
  /// \f]
  ///
  /// and the control points
  ///
  /// \f[  
  /// \mathbf{c}_I = \mathbf{c}_{i_1,i_2,\dots, i_{d_\text{par}}} \in
  /// \mathbb{R}^{d_\text{geo}}.  
  /// \f]
  ///
  /// Here, \f$i_d\f$ are the local numbers of the univariate
  /// B-splines \f$\left(B_{i_d,p_d}\right)_{i_d=1}^{n_d}\f$ in the
  /// \f$d\f$-th parametric dimension, \f$p_d\f$ is the respective
  /// *degree*, and \f$n_d\f$ is the number of univariate B-splines in
  /// the \f$d\f$-th direction. Moreover, \f$0\le \xi_{i_d}\le 1\f$ is
  /// the parametric value at which the B-spline is evaluated. The
  /// multivariate B-spline function is defined as follows
  ///
  /// \f[
  /// \mathbf{f}(\boldsymbol{\xi}) = \sum_{I=1}^N B_I(\boldsymbol{\xi}) \mathbf{c}_I
  /// \f]
  ///
  /// Here and below we adopt the vector notation \f$\boldsymbol{\xi}
  /// = \left(\xi_1,\xi_2,\dots,\xi_{d_\text{par}}\right)^\top\f$ and
  /// combine multiple local indices
  /// \f$i_1,i_2,\dots,i_{d_\text{par}}\f$ of univariate B-spline
  /// basis functions into the global index \f$1\le I \le N\f$ with
  /// \f$N=n_1\cdot n_2\cdot\dots\cdot n_{d_\text{par}}\f$ denoting
  /// the total number of multivariate B-splines.
  ///
  /// This class implements B-spline functions and their derivatives
  /// for 1, 2, 3, and 4 parametric dimensions. The univariate
  /// B-splines are uniquely determined by their knot vectors
  ///
  /// \f[
  /// \left(t_{i_d}\right)_{i_d=1}^{n_d+p_d+1}
  /// \f]
  ///
  /// with \f$0\le t_{i_d}\le 1\f$ and \f$t_{i_d}\le t_{i_d+1}\f$ for
  /// all \f$i_d\f$, that is, the knot vectors are given by a
  /// non-decreasing sequence of values in the interval \f$[0,1]\f$
  /// with the possibility that knot values are repeated.
  ///
  /// This class implements the evaluation of B-splines and their
  /// derivatives as explained in Chapters 2 and 3 from \cite Lyche:2011.
  ///
  /// @note C++ uses 0-based indexing so that all of the above
  /// formulas need to be shifted by -1. Moreover, all vectors,
  /// matrices, and tensors are implemented as `torch::Tensor`
  /// objects and hence adopt Torch's local-to-global mapping. It is
  /// therefore imperative to always use Torch's indexing
  /// functionality to extract sub-tensors.
  
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  class UniformBSplineCore : public core<real_t> {
  protected:
    /// @brief Dimension of the parametric space
    /// \f$\hat\Omega=[0,1]^{d_\text{par}}\f$
    static constexpr const short_t parDim_ = sizeof...(Degrees);

    /// @brief Dimension of the geometric space
    /// \f$\Omega\subset\mathbb{R}^{d_\text{geo}}\f$
    static constexpr const short_t geoDim_ = GeoDim;

    /// @brief Array storing the degrees
    /// \f$\left(p_d\right)_{d=1}^{d_\text{par}}\f$
    static constexpr const std::array<short_t, parDim_> degrees_ = {Degrees...};

    /// @brief Array storing the knot vectors
    /// \f$\left(\left(t_{i_d}\right)_{i_d=1}^{n_d+p_d+1}\right)_{d=1}^{d_\text{par}}\f$
    std::array<torch::Tensor, parDim_> knots_;

    /// @brief Array storing the sizes of the knot vectors
    /// \f$\left(n_d+p_d+1\right)_{d=1}^{d_\text{par}}\f$
    std::array<int64_t, parDim_> nknots_;

    /// @brief Array storing the coefficients of the control net
    /// \f$\left(\mathbf{c}_{i_d}\right)_{i_d=1}^{n_d}\f$,
    /// \f$\mathbf{c}_{i_d}\in\mathbb{R}^{d_\text{geo}}\f$
    std::array<torch::Tensor, geoDim_> coeffs_;

    /// @brief Array storing the sizes of the coefficients of the
    /// control net \f$\left(n_d\right)_{d=1}^{d_\text{par}}\f$
    std::array<int64_t, parDim_> ncoeffs_;

    /// @brief LibTorch constants
    /// @{
    const torch::Tensor one_, zero_;
    /// @}

  public:
    using value_type = real_t;
    
    /// @brief Default constructor
    UniformBSplineCore()
        : core<real_t>(),
          one_(torch::ones(1, core<real_t>::options_)),
          zero_(torch::zeros(1, core<real_t>::options_)) {}

    /// @brief Constructor for equidistant knot vectors
    UniformBSplineCore(const std::array<int64_t, parDim_>& ncoeffs,
                       BSplineInit init = BSplineInit::zeros)
        : core<real_t>(),
          ncoeffs_(ncoeffs),
          one_(torch::ones(1, core<real_t>::options_)),
          zero_(torch::zeros(1, core<real_t>::options_))
    {
      for (short_t i = 0; i < parDim_; ++i) {

        // Check that open knot vector can be created
        if (degrees_[i] > ncoeffs_[i] + 1)
          throw std::runtime_error("Not enough coefficients to create open knot vector");

        // Create open uniform knot vector
        std::vector<real_t> kv;

        for (int64_t j = 0; j < degrees_[i]; ++j)
          kv.push_back(static_cast<real_t>(0));

        for (int64_t j = 0; j < ncoeffs[i] - degrees_[i] + 1; ++j)
          kv.push_back(static_cast<real_t>(j / real_t(ncoeffs[i] - degrees_[i])));

        for (int64_t j = 0; j < degrees_[i]; ++j)
          kv.push_back(static_cast<real_t>(1));

        if (core<real_t>::options_.device() == torch::kCPU)
          knots_[i] = torch::from_blob(static_cast<real_t *>(kv.data()),
                                       kv.size(), core<real_t>::options_).clone();
        else
          knots_[i] = torch::from_blob(static_cast<real_t *>(kv.data()),
                                       kv.size(), core<real_t>::options_.device(torch::kCPU))
            .to(core<real_t>::options_.device());
        
        // Store the size of the knot vector
        nknots_[i] = knots_[i].size(0);
      }

      // Initialize coefficients
      init_coeffs(init);
    }

    /// @brief Returns the parametric dimension
    inline static constexpr short_t parDim()
    {
      return parDim_;
    }

    /// @brief Returns the geometric dimension
    inline static constexpr short_t geoDim()
    {
      return geoDim_;
    }

    /// @brief Returns a constant reference to the array of degrees
    inline static constexpr const std::array<short_t, parDim_>& degrees()
    {
      return degrees_;
    }

    /// @brief Returns a constant reference to the degree in the
    /// \f$i\f$-th dimension
    inline static constexpr const short_t& degree(short_t i)
    {
      assert(i >= 0 && i < parDim_);
      return degrees_[i];
    }

    /// @brief Returns a constant reference to the array of knot
    /// vectors
    inline const std::array<torch::Tensor, parDim_>& knots() const
    {
      return knots_;
    }

    /// @brief Returns a constant reference to the knot vector in the
    /// \f$i\f$-th dimension
    inline const torch::Tensor& knots(short_t i) const {
      assert(i >= 0 && i < parDim_);
      return knots_[i];
    }

    /// @brief Returns a non-constant reference to the array of knot
    /// vectors
    inline std::array<torch::Tensor, parDim_>& knots()
    {
      return knots_;
    }

    /// @brief Returns a non-constant reference to the knot vector in
    /// the \f$i\f$-th dimension
    inline torch::Tensor& knots(short_t i)
    {
      assert(i >= 0 && i < parDim_);
      return knots_[i];
    }

    /// @brief Returns a constant reference to the array of knot
    /// vector dimensions
    inline const std::array<int64_t, parDim_>& nknots() const
    {
      return nknots_;
    }

    /// @brief Returns the dimension of the knot vector in the
    /// \f$i\f$-th dimension
    inline int64_t nknots(short_t i) const
    {
      assert(i >= 0 && i < parDim_);
      return nknots_[i];
    }

    /// @brief Returns a constant reference to the array of
    /// coefficients.
    inline const auto& coeffs() const
    {
      return coeffs_;
    }

    /// @brief Returns a constant reference to the coefficients in the
    /// \f$i\f$-th dimension.
    inline const auto& coeffs(short_t i) const
    {
      assert(i >= 0 && i < geoDim_);
      return coeffs_[i];
    }

    /// @brief Returns a non-constant reference to the array of
    /// coefficients
    inline auto& coeffs()
    {
      return coeffs_;
    }

    /// @brief Returns a non-constant reference to the coefficients in
    /// the \f$i\f$-th dimension
    inline auto& coeffs(short_t i)
    {
      assert(i >= 0 && i < geoDim_);
      return coeffs_[i];
    }

    /// @brief Returns the total number of coefficients
    inline int64_t ncoeffs() const
    {
      int64_t s = 1;
      for (short_t i = 0; i < parDim_; ++i)
        s *= ncoeffs(i);
      return s;
    }

    /// @brief Returns the total number of coefficients in the
    /// \f$i\f$-th direction
    inline int64_t ncoeffs(short_t i) const
    {
      assert(i >= 0 && i < parDim_);
      return ncoeffs_[i];
    }

    /// @brief Returns the Greville abscissae
    inline std::array<torch::Tensor, geoDim_> greville() const
    {
      std::array<torch::Tensor, geoDim_> coeffs;

      // Fill coefficients with the tensor-product of Greville
      // abscissae values per univariate dimension
      for (short_t i = 0; i < geoDim_; ++i) {
        coeffs[i] = torch::ones(1, core<real_t>::options_);

        for (short_t j = 0; j < parDim_; ++j) {
          if (i == j) {
            auto greville_ = torch::zeros(ncoeffs_[j], core<real_t>::options_);
            auto greville = greville_.template accessor<real_t, 1>();
            auto knots = knots_[j].template accessor<real_t, 1>();
            for (int64_t k = 0; k < ncoeffs_[j]; ++k) {
              for (short_t l = 1; l <= degrees_[j]; ++l)
                greville[k] += knots[k + l];
              greville[k] /= degrees_[j];
            }
            coeffs[i] = torch::kron(greville_, coeffs[i]);
          } else
            coeffs[i] = torch::kron(torch::ones(ncoeffs_[j],
                                                core<real_t>::options_), coeffs[i]);
        }
      }

      return coeffs;
    }

    /// @brief Returns the value of the B-spline object in the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for univariate B-splines
    /// (i.e. \f$d_\text{par}=1\f$)
    ///
    /// @param[in] basfunc Value(s) of the multivariate B-spline basis
    ///                    functions evaluated at the point(s) `xi`
    ///
    /// @param[in] idx Indices where to evaluate the coefficients
    ///
    /// @param[in] numeval Number of evaluation points
    ///
    /// @param[in] sizes Dimension of the result
    ///
    /// @result Value(s) of the univariate B-spline object
    inline auto eval_from_precomputed(const torch::Tensor& basfunc,
                                      const torch::Tensor& idx,
                                      int64_t numeval, torch::IntArrayRef sizes) const
    {
      if constexpr (geoDim_ > 1) {
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, idx).view({-1, numeval}),
                                   0).view(sizes));
        return result;
      } else
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(basfunc,
                                                      coeffs(0).index_select(0, idx).view({-1, numeval}),
                                                      0).view(sizes));
    }
    
    /// @brief Returns the value of the B-spline object in the point `xi`
    ///
    /// This implementation follows the procedure described in
    /// Chapters 2 and 3 of \cite Lyche:2011.
    ///
    /// @anchor BSplineEvaluation **Algorithm: B-spline evaluation**
    ///
    /// 1. Determine the indices
    ///    \f$(i_d)_{d=1}^{d_\text{par}}\f$ of the knot spans such that
    ///
    ///    \f[
    ///      \boldsymbol{\xi} = \left(\xi_1, \dots, \xi_{d_\text{par}}\right)^\top \in
    ///      \bigotimes_{d=1}^{d_\text{par}}
    ///      [t_{i_d}, t_{i_d+1}).
    ///    \f]
    ///
    /// 2. Evaluate the vectors of univariate B-spline basis functions (or
    ///    their derivatives) that are non-zero at \f$\boldsymbol{\xi}\f$
    ///
    ///    \f[
    ///      D^{r_d}\mathbf{B}_d =
    ///      \left( D^{r_d} B_{i_d-p_d,p_d}, \dots, D^{r_d} B_{i_d,p_d} \right)^\top,
    ///    \f]
    ///
    ///    where \f$ p_d \f$ is the degree of the \f$d\f$-th
    ///    univariate B-spline and \f$ r_d \f$ denotes the requested
    ///    derivative in the \f$d\f$-direction.
    ///   
    /// 3. Multiply the tensor-product of the above row vectors by the
    ///    column vector of control points
    ///
    ///    \f[
    ///    \left( \bigotimes_{d=1}^{d_\text{par}} D^{r_d}\mathbf{B}_d \right) \cdot \mathbf{c}_\mathcal{J},
    ///    \f]
    ///
    ///    where \f$\mathcal{J}\f$ is the subset of global indices
    ///    that belong to the coefficients
    ///
    ///    \f[
    ///    \mathbf{c}_{i_1-p_1:i_1,\dots,i_\text{par}-p_\text{par}:i_\text{par}}
    ///    \f]
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the multivariate B-spline object
    ///
    /// @result Value(s) of the multivariate B-spline evaluated at the point(s) `xi`
    ///
    /// @{
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi) const
    {
      return eval<deriv>(TensorArray1({xi}));
    }
          
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const std::array<torch::Tensor, parDim_>& xi) const
    {
      if constexpr (parDim_ == 0)
        return coeffs_[0];
      else
        return eval<deriv>(xi, eval_knot_indices(xi));
    }
    /// @}
    
    /// @brief Returns the value of the univariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for univariate B-splines
    /// (i.e. \f$d_\text{par}=1\f$)
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the univariate B-spline object
    ///
    /// @param[in] idx Knot indices where to evaluate the univariate B-spline object
    ///
    /// @result Value(s) of the univariate B-spline evaluated at the point(s) `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const TensorArray1& xi, const TensorArray1& idx) const
    {
      assert(parDim_ == 1 &&
             xi[0].sizes() == idx[0].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc = eval_basfunc<deriv>(xi, idx);
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, eval_coeff_indices(idx)).view({-1, xi[0].numel()}),
                                   0).view(xi[0].sizes()));
        return result;
        } else
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(eval_basfunc<deriv>(xi, idx),
                                                      coeffs(0).index_select(0, eval_coeff_indices(idx)).view({-1, xi[0].numel()}),
                                                      0).view(xi[0].sizes()));
    }

    /// @brief Returns the value of the univariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for univariate B-splines
    /// (i.e. \f$d_\text{par}=1\f$)
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the univariate B-spline object
    ///
    /// @param[in] idx Knot indices where to evaluate the univariate B-spline object
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the univariate B-spline object
    ///
    /// @result Value(s) of the univariate B-spline evaluated at the point(s) `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const TensorArray1& xi, const TensorArray1& idx,
                     const torch::Tensor& coeff_idx) const
    {
      assert(parDim_ == 1 &&
             xi[0].sizes() == idx[0].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc = eval_basfunc<deriv>(xi, idx);
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, coeff_idx).view({-1, xi[0].numel()}),
                                   0).view(xi[0].sizes()));
        return result;
        } else
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(eval_basfunc<deriv>(xi, idx),
                                                      coeffs(0).index_select(0, coeff_idx).view({-1, xi[0].numel()}),
                                                      0).view(xi[0].sizes()));
    }
    
    /// @brief Returns the value of the bivariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for bivariate B-splines
    /// (i.e. \f$d_\text{par}=2\f$)
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the bivariate B-spline object
    ///
    /// @param[in] idx Knot indices where to evaluate the bivariate B-spline object
    ///
    /// @result Value(s) of the bivariate B-spline evaluated at the point(s) `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const TensorArray2& xi, const TensorArray2& idx) const
    {
      assert(parDim_ == 2 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[0].sizes() ==  xi[1].sizes());
      
      if constexpr (geoDim_ > 1) {       
        auto basfunc = eval_basfunc<deriv>(xi, idx);
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, eval_coeff_indices(idx)).view({-1, xi[0].numel()}),
                                   0).view(xi[0].sizes()));
        return result;
      } else
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(eval_basfunc<deriv>(xi, idx),
                                                      coeffs(0).index_select(0, eval_coeff_indices(idx)).view({-1, xi[0].numel()}),
                                                      0).view(xi[0].sizes()));
    }

    /// @brief Returns the value of the bivariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for bivariate B-splines
    /// (i.e. \f$d_\text{par}=2\f$)
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the bivariate B-spline object
    ///
    /// @param[in] idx Knot indices where to evaluate the bivariate B-spline object
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the bivariate B-spline object
    ///
    /// @result Value(s) of the bivariate B-spline evaluated at the point(s) `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const TensorArray2& xi, const TensorArray2& idx,
                     const torch::Tensor& coeff_idx) const
    {
      assert(parDim_ == 2 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[0].sizes() ==  xi[1].sizes());
      
      if constexpr (geoDim_ > 1) {       
        auto basfunc = eval_basfunc<deriv>(xi, idx);
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, coeff_idx).view({-1, xi[0].numel()}),
                                   0).view(xi[0].sizes()));
        return result;
      } else
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(eval_basfunc<deriv>(xi, idx),
                                                      coeffs(0).index_select(0, coeff_idx).view({-1, xi[0].numel()}),
                                                      0).view(xi[0].sizes()));
    }

    /// @brief Returns the value of the trivariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for bivariate B-splines
    /// (i.e. \f$d_\text{par}=3\f$)
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the trivariate B-spline object
    ///
    /// @param[in] idx Knot indices where to evaluate the trivariate B-spline object
    ///
    /// @result Value(s) of the trivariate B-spline evaluated at the point(s) `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const TensorArray3& xi, const TensorArray3& idx) const
    {
      assert(parDim_ == 3 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[0].sizes() ==  xi[1].sizes() &&
             xi[1].sizes() ==  xi[2].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc = eval_basfunc<deriv>(xi, idx);
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, eval_coeff_indices(idx)).view({-1, xi[0].numel()}),
                                   0).view(xi[0].sizes()));
        return result;
      } else        
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(eval_basfunc<deriv>(xi, idx),
                                                      coeffs(0).index_select(0, eval_coeff_indices(idx)).view({-1, xi[0].numel()}),
                                                      0).view(xi[0].sizes()));
    }

    /// @brief Returns the value of the trivariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for bivariate B-splines
    /// (i.e. \f$d_\text{par}=3\f$)
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the trivariate B-spline object
    ///
    /// @param[in] idx Knot indices where to evaluate the trivariate B-spline object
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the trivariate B-spline object
    ///
    /// @result Value(s) of the trivariate B-spline evaluated at the point(s) `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const TensorArray3& xi, const TensorArray3& idx,
                     const torch::Tensor& coeff_idx) const
    {
      assert(parDim_ == 3 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[0].sizes() ==  xi[1].sizes() &&
             xi[1].sizes() ==  xi[2].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc = eval_basfunc<deriv>(xi, idx);
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, coeff_idx).view({-1, xi[0].numel()}),
                                   0).view(xi[0].sizes()));
        return result;
      } else        
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(eval_basfunc<deriv>(xi, idx),
                                                      coeffs(0).index_select(0, coeff_idx).view({-1, xi[0].numel()}),
                                                      0).view(xi[0].sizes()));
    }

    /// @brief Returns the value of the quartvariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for bivariate B-splines
    /// (i.e. \f$d_\text{par}=4\f$)
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the quartvariate B-spline object
    ///
    /// @param[in] idx Knot indices where to evaluate the quartvariate B-spline object
    ///
    /// @result Value(s) of the quartvariate B-spline evaluated at the point(s) `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const TensorArray4& xi, const TensorArray4& idx) const
    {
      assert(parDim_ == 4 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[3].sizes() == idx[3].sizes() &&
             xi[0].sizes() ==  xi[1].sizes() &&
             xi[1].sizes() ==  xi[2].sizes() &&
             xi[2].sizes() ==  xi[3].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc = eval_basfunc<deriv>(xi, idx);
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, eval_coeff_indices(idx)).view({-1, xi[0].numel()}),
                                   0).view(xi[0].sizes()));
        return result;
      } else        
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(eval_basfunc<deriv>(xi, idx),
                                                      coeffs(0).index_select(0, eval_coeff_indices(idx)).view({-1, xi[0].numel()}),
                                                      0).view(xi[0].sizes()));
    }

    /// @brief Returns the value of the quartvariate B-spline object in
    /// the points `xi`
    ///
    /// This function implements steps 2-3 of algorithm \ref
    /// BSplineEvaluation for bivariate B-splines
    /// (i.e. \f$d_\text{par}=4\f$)
    ///
    /// @tparam deriv Composition of derivative indicators of type \ref BSplineDeriv
    ///
    /// @param[in] xi Point(s) where to evaluate the quartvariate B-spline object
    ///
    /// @param[in] idx Knot indices where to evaluate the quartvariate B-spline object
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the quartvariate B-spline object
    ///
    /// @result Value(s) of the quartvariate B-spline evaluated at the point(s) `xi`
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const TensorArray4& xi, const TensorArray4& idx,
                     const torch::Tensor& coeff_idx) const
    {
      assert(parDim_ == 4 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[3].sizes() == idx[3].sizes() &&
             xi[0].sizes() ==  xi[1].sizes() &&
             xi[1].sizes() ==  xi[2].sizes() &&
             xi[2].sizes() ==  xi[3].sizes());
      
      if constexpr (geoDim_ > 1) {
        auto basfunc = eval_basfunc<deriv>(xi, idx);
        BlockTensor<torch::Tensor, 1, geoDim_> result;
        for (std::size_t i = 0; i < geoDim_; ++i)
          result.set(i, dotproduct(basfunc,
                                   coeffs(i).index_select(0, coeff_idx).view({-1, xi[0].numel()}),
                                   0).view(xi[0].sizes()));
        return result;
      } else        
        return
          BlockTensor<torch::Tensor, 1, 1>(dotproduct(eval_basfunc<deriv>(xi, idx),
                                                      coeffs(0).index_select(0, coeff_idx).view({-1, xi[0].numel()}),
                                                      0).view(xi[0].sizes()));
    }

    /// @brief Returns the indices of knot spans containing `xi`
    ///
    /// This function returns the indices
    /// \f$(i_d)_{d=1}^{d_\text{par}}\f$ of the knot spans such that
    ///
    /// \f[
    ///   \boldsymbol{\xi} \in [t_{i_1}, t_{i_1+1}) \times [t_{i_2}, t_{i_2+1})
    ///   \times \dots \times [t_{i_{d_\text{par}}}, t_{i_{d_\text{par}}+1}).
    /// \f]
    ///
    /// The indices are returned as `std::array<torch::Tensor,
    /// parDim_>` in the same order as provided in `xi`
    /// @{
    inline auto eval_knot_indices(const TensorArray1& xi) const
    {
      assert(parDim_ == 1);
      return TensorArray1({
          torch::min(torch::full_like(xi[0], ncoeffs_[0]-1, core<real_t>::options_),
                     torch::floor(xi[0] * (ncoeffs_[0] - degrees_[0]) + degrees_[0])).to(torch::kInt64)
        });
    }
      
    inline auto eval_knot_indices(const TensorArray2& xi) const
    {
      assert(parDim_ == 2);
      return TensorArray2({
          torch::min(torch::full_like(xi[0], ncoeffs_[0]-1, core<real_t>::options_),
                     torch::floor(xi[0] * (ncoeffs_[0] - degrees_[0]) + degrees_[0])).to(torch::kInt64),
          torch::min(torch::full_like(xi[1], ncoeffs_[1]-1, core<real_t>::options_),
                     torch::floor(xi[1] * (ncoeffs_[1] - degrees_[1]) + degrees_[1])).to(torch::kInt64)
        });
    }

    inline auto eval_knot_indices(const TensorArray3& xi) const
    {
      assert(parDim_ == 3);
      return TensorArray3({
          torch::min(torch::full_like(xi[0], ncoeffs_[0]-1, core<real_t>::options_),
                     torch::floor(xi[0] * (ncoeffs_[0] - degrees_[0]) + degrees_[0])).to(torch::kInt64),
          torch::min(torch::full_like(xi[1], ncoeffs_[1]-1, core<real_t>::options_),
                     torch::floor(xi[1] * (ncoeffs_[1] - degrees_[1]) + degrees_[1])).to(torch::kInt64),
          torch::min(torch::full_like(xi[2], ncoeffs_[2]-1, core<real_t>::options_),
                     torch::floor(xi[2] * (ncoeffs_[2] - degrees_[2]) + degrees_[2])).to(torch::kInt64)
        });
    }

    inline auto eval_knot_indices(const TensorArray4& xi) const
    {
      assert(parDim_ == 4);
      return TensorArray4({
          torch::min(torch::full_like(xi[0], ncoeffs_[0]-1, core<real_t>::options_),
                     torch::floor(xi[0] * (ncoeffs_[0] - degrees_[0]) + degrees_[0])).to(torch::kInt64),
          torch::min(torch::full_like(xi[1], ncoeffs_[1]-1, core<real_t>::options_),
                     torch::floor(xi[1] * (ncoeffs_[1] - degrees_[1]) + degrees_[1])).to(torch::kInt64),
          torch::min(torch::full_like(xi[2], ncoeffs_[2]-1, core<real_t>::options_),
                     torch::floor(xi[2] * (ncoeffs_[2] - degrees_[2]) + degrees_[2])).to(torch::kInt64),
          torch::min(torch::full_like(xi[3], ncoeffs_[3]-1, core<real_t>::options_),
                     torch::floor(xi[3] * (ncoeffs_[3] - degrees_[3]) + degrees_[3])).to(torch::kInt64)
        });
    }
    /// @}

    /// @brief Returns the indices of the coefficients corresponding to the knot indices `idx`
    /// @{
    inline auto eval_coeff_indices(const TensorArray1& idx) const
    {
      assert(parDim_ == 1);
      return VSlice(idx[0].flatten(), -degrees_[0], 1);
    }

    inline auto eval_coeff_indices(const TensorArray2& idx) const
    {
      assert(parDim_ == 2);
      return VSlice(TensorArray2({idx[0].flatten(), idx[1].flatten()}),
                    std::array<int64_t, 2>{-degrees_[0], -degrees_[1]},
                    std::array<int64_t, 2>{1, 1},
                    ncoeffs(0));
    }

    inline auto eval_coeff_indices(const TensorArray3& idx) const
    {
      assert(parDim_ == 3);
      return VSlice(TensorArray3({idx[0].flatten(), idx[1].flatten(), idx[2].flatten()}),
                    std::array<int64_t, 3>{-degrees_[0], -degrees_[1], -degrees_[2]},
                    std::array<int64_t, 3>{1, 1, 1},
                    std::array<int64_t, 2>{ncoeffs(0), ncoeffs(1)});
    }

    inline auto eval_coeff_indices(const TensorArray4& idx) const
    {
      assert(parDim_ == 4);
      return VSlice(TensorArray4({idx[0].flatten(), idx[1].flatten(), idx[2].flatten(), idx[3].flatten()}),
                    std::array<int64_t, 4>{-degrees_[0], -degrees_[1], -degrees_[2], -degrees_[3]},
                    std::array<int64_t, 4>{1, 1, 1, 1},
                    std::array<int64_t, 3>{ncoeffs(0), ncoeffs(1), ncoeffs(2)});
    }
    /// @}

    /// @brief Returns the vector of multivariate B-spline basis
    /// functions (or their derivatives) evaluated in the point `xi`
    /// @{
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_basfunc(const torch::Tensor& xi) const
    {
      return eval_basfunc<deriv>(TensorArray1({xi}));
    }
          
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_basfunc(const std::array<torch::Tensor, parDim_>& xi) const
    {
      if constexpr (parDim_ == 0)
                     return torch::ones_like(coeffs_[0]);
      else
        return eval_basfunc<deriv>(xi, eval_knot_indices(xi));
    }
    
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_basfunc(const TensorArray1& xi, const TensorArray1& idx) const
    {
      assert(parDim_ == 1 &&
             xi[0].sizes() == idx[0].sizes());
      return
        eval_prefactor<degrees_[0], (short_t) deriv % 10>() *
        eval_univariate<degrees_[0], 0, (short_t) deriv % 10>( xi[0].flatten(),
                                                              idx[0].flatten());
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_basfunc(const TensorArray2& xi, const TensorArray2& idx) const
    {
      assert(parDim_ == 2 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[0].sizes() ==  xi[1].sizes());
      return
        eval_prefactor<degrees_[0],  (short_t)deriv    %10>() *
        eval_prefactor<degrees_[1], ((short_t)deriv/10)%10>() *
        kronproduct(eval_univariate<degrees_[1], 1, ((short_t)deriv/10)%10>( xi[1].flatten(),
                                                                            idx[1].flatten()),
                    eval_univariate<degrees_[0], 0,  (short_t)deriv    %10>( xi[0].flatten(),
                                                                            idx[0].flatten()),
                    0);
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_basfunc(const TensorArray3& xi, const TensorArray3& idx) const
    {
      assert(parDim_ == 3 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[0].sizes() ==  xi[1].sizes() &&
             xi[1].sizes() ==  xi[2].sizes());
      return
        eval_prefactor<degrees_[0],  (short_t)deriv     %10>() *
        eval_prefactor<degrees_[1], ((short_t)deriv/ 10)%10>() *
        eval_prefactor<degrees_[2], ((short_t)deriv/100)%10>() *
        kronproduct(eval_univariate<degrees_[2], 2, ((short_t)deriv/100)%10>( xi[2].flatten(),
                                                                             idx[2].flatten()),
                    kronproduct(eval_univariate<degrees_[1], 1, ((short_t)deriv/10)%10>( xi[1].flatten(),
                                                                                        idx[1].flatten()),
                                eval_univariate<degrees_[0], 0,  (short_t)deriv    %10>( xi[0].flatten(),
                                                                                        idx[0].flatten()),
                                0),                      
                    0);
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_basfunc(const TensorArray4& xi, const TensorArray4& idx) const
    {
      assert(parDim_ == 4 &&
             xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[3].sizes() == idx[3].sizes() &&
             xi[0].sizes() ==  xi[1].sizes() &&
             xi[1].sizes() ==  xi[2].sizes() &&
             xi[2].sizes() ==  xi[3].sizes());
      return
        eval_prefactor<degrees_[0],  (short_t)deriv      %10>() *
        eval_prefactor<degrees_[1], ((short_t)deriv/  10)%10>() *
        eval_prefactor<degrees_[2], ((short_t)deriv/ 100)%10>() *
        eval_prefactor<degrees_[3], ((short_t)deriv/1000)%10>() *
        kronproduct(kronproduct(eval_univariate<degrees_[3], 3, ((short_t)deriv/1000)%10>( xi[3].flatten(),
                                                                                          idx[3].flatten()),
                                eval_univariate<degrees_[2], 2, ((short_t)deriv/ 100)%10>( xi[2].flatten(),
                                                                                          idx[2].flatten()),
                                0),
                    kronproduct(eval_univariate<degrees_[1], 1, ((short_t)deriv/  10)%10>( xi[1].flatten(),
                                                                                          idx[1].flatten()),
                                eval_univariate<degrees_[0], 0,  (short_t)deriv      %10>( xi[0].flatten(),
                                                                                          idx[0].flatten()),
                                0),                      
                    0);
    }
    /// @}
    
    /// @brief Transforms the coefficients based on the given mapping
    inline UniformBSplineCore& 
    transform(const std::function<std::array<real_t, geoDim_>(const std::array<real_t, parDim_>& )> transformation)
    {
      static_assert(parDim_ <= 4, "Unsupported parametric dimension");

      // 1D
      if constexpr (parDim_ == 1) {
#pragma omp parallel for simd
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          auto c = transformation(std::array<real_t, 1>{i / real_t(ncoeffs_[0] - 1)});
          for (short_t d = 0; d < geoDim_; ++d)
            coeffs_[d].detach()[i] = c[d];
        }
      }

        // 2D
      else if constexpr (parDim_ == 2) {
#pragma omp parallel for simd collapse(2)
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
            auto c = transformation(std::array<real_t, 2>{i / real_t(ncoeffs_[0] - 1),
                                                          j / real_t(ncoeffs_[1] - 1)});
            for (short_t d = 0; d < geoDim_; ++d)
              coeffs_[d].detach()[j * ncoeffs_[0] +
                                  i] = c[d];
          }
        }
      }

        // 3D
      else if constexpr (parDim_ == 3) {
#pragma omp parallel for simd collapse(3)
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
            for (int64_t k = 0; k < ncoeffs_[2]; ++k) {
              auto c = transformation(std::array<real_t, 3>{i / real_t(ncoeffs_[0] - 1),
                                                            j / real_t(ncoeffs_[1] - 1),
                                                            k / real_t(ncoeffs_[2] - 1)});
              for (short_t d = 0; d < geoDim_; ++d)
                coeffs_[d].detach()[k * ncoeffs_[0] * ncoeffs_[1] +
                                    j * ncoeffs_[0] +
                                    i] = c[d];
            }
          }
        }
      }

        // 4D
      else if constexpr (parDim_ == 4) {
#pragma omp parallel for simd collapse(4)
        for (int64_t i = 0; i < ncoeffs_[0]; ++i) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j) {
            for (int64_t k = 0; k < ncoeffs_[2]; ++k) {
              for (int64_t l = 0; l < ncoeffs_[3]; ++l) {
                auto c = transformation(std::array<real_t, 4>{i / real_t(ncoeffs_[0] - 1),
                                                              j / real_t(ncoeffs_[1] - 1),
                                                              k / real_t(ncoeffs_[2] - 1),
                                                              l / real_t(ncoeffs_[3] - 1)});
                for (short_t d = 0; d < geoDim_; ++d)
                  coeffs_[d].detach()[l * ncoeffs_[0] * ncoeffs_[1] * ncoeffs_[2] +
                                      k * ncoeffs_[0] * ncoeffs_[1] +
                                      j * ncoeffs_[0] +
                                      i] = c[d];
              }
            }
          }
        }
      } else {
        throw std::runtime_error("Unsupported parametric dimension");
      }

      return *this;
    }

    /// @brief Returns the B-spline object as XML string
    std::string to_xml() const
    {
      std::stringstream ss;

      // Write preamble and knot vectors
      ss << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
         << "<xml>\n";

      // 1D parametric dimension
      if constexpr (parDim_ == 1) {
        ss << " <Geometry type=\"BSpline\">\n"
           << "  <Basis type=\"BSplineBasis\">\n"
           << "   <KnotVector degree=\"" << degrees_[0] << "\">";
        auto knots = knots_[0].template accessor<real_t, 1>();
        for (int64_t i = 0; i < nknots_[0]; ++i)
          ss << knots[i] << " ";
        ss << "</KnotVector>\n"
           << "  </Basis>\n";
      }

        // >1D parametric dimension
      else {
        ss << " <Geometry type=\"TensorBSpline" << parDim_ << "\" id=\"0\">\n"
           << "  <Basis type=\"TensorBSplineBasis" << parDim_ << "\">\n";
        for (short_t i = 0; i < parDim_; ++i) {
          ss << "   <Basis type=\"BSplineBasis\" index=\"" << i << "\">\n"
             << "    <KnotVector degree=\"" << degrees_[i] << "\">";
          auto knots = knots_[i].template accessor<real_t, 1>();
          for (int64_t j = 0; j < nknots_[i]; ++j)
            ss << knots[j] << " ";
          ss << "</KnotVector>\n"
             << "   </Basis>\n";
        }
        ss << "  </Basis>\n";
      }

      // Write coefficients
      ss << "  <coefs geoDim=\"" << geoDim_ << "\">\n";

      // 1D geometric dimension
      if constexpr (geoDim_ == 1) {
        auto coeffs0 = coeffs_[0].template accessor<real_t, 1>();
        for (int64_t i = 0; i < ncoeffs(); ++i)
          ss << "   " << coeffs() << "\n";
      }

        // 2D geometric dimension
      else if constexpr (geoDim_ == 2) {
        auto coeffs0 = coeffs_[0].template accessor<real_t, 1>();
        auto coeffs1 = coeffs_[1].template accessor<real_t, 1>();
        if constexpr (parDim_ == 1) {
          for (int64_t i = 0; i < ncoeffs(); ++i)
            ss << "   " << coeffs() << "\n";
        } else if constexpr (parDim_ == 2) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j)
            for (int64_t i = 0; i < ncoeffs_[0]; ++i)
              ss << "   " << coeffs0[i * ncoeffs_[1] + j]
                 << " " << coeffs1[i * ncoeffs_[1] + j] << "\n";
        } else if constexpr (parDim_ == 3) {
          for (int64_t k = 0; k < ncoeffs_[2]; ++k)
            for (int64_t j = 0; j < ncoeffs_[1]; ++j)
              for (int64_t i = 0; i < ncoeffs_[0]; ++i)
                ss << "   " << coeffs0[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs1[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k] << "\n";
        }
      }

        // 3D geometric dimension
      else if constexpr (geoDim_ == 3) {
        auto coeffs0 = coeffs_[0].template accessor<real_t, 1>();
        auto coeffs1 = coeffs_[1].template accessor<real_t, 1>();
        auto coeffs2 = coeffs_[2].template accessor<real_t, 1>();
        if constexpr (parDim_ == 1) {
          for (int64_t i = 0; i < ncoeffs(); ++i)
            ss << "   " << coeffs() << "\n";
        } else if constexpr (parDim_ == 2) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j)
            for (int64_t i = 0; i < ncoeffs_[0]; ++i)
              ss << "   " << coeffs0[i * ncoeffs_[1] + j]
                 << " " << coeffs1[i * ncoeffs_[1] + j]
                 << " " << coeffs2[i * ncoeffs_[1] + j] << "\n";
        } else if constexpr (parDim_ == 3) {
          for (int64_t k = 0; k < ncoeffs_[2]; ++k)
            for (int64_t j = 0; j < ncoeffs_[1]; ++j)
              for (int64_t i = 0; i < ncoeffs_[0]; ++i)
                ss << "   " << coeffs0[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs1[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs2[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k] << "\n";
        }
      }

        // 4D geometric dimension
      else if constexpr (geoDim_ == 4) {
        auto coeffs0 = coeffs_[0].template accessor<real_t, 1>();
        auto coeffs1 = coeffs_[1].template accessor<real_t, 1>();
        auto coeffs2 = coeffs_[2].template accessor<real_t, 1>();
        auto coeffs3 = coeffs_[3].template accessor<real_t, 1>();
        if constexpr (parDim_ == 1) {
          for (int64_t i = 0; i < ncoeffs(); ++i)
            ss << "   " << coeffs() << "\n";
        } else if constexpr (parDim_ == 2) {
          for (int64_t j = 0; j < ncoeffs_[1]; ++j)
            for (int64_t i = 0; i < ncoeffs_[0]; ++i)
              ss << "   " << coeffs0[i * ncoeffs_[1] + j]
                 << " " << coeffs1[i * ncoeffs_[1] + j]
                 << " " << coeffs2[i * ncoeffs_[1] + j]
                 << " " << coeffs3[i * ncoeffs_[1] + j] << "\n";
        } else if constexpr (parDim_ == 3) {
          for (int64_t k = 0; k < ncoeffs_[2]; ++k)
            for (int64_t j = 0; j < ncoeffs_[1]; ++j)
              for (int64_t i = 0; i < ncoeffs_[0]; ++i)
                ss << "   " << coeffs0[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs1[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs2[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k]
                   << " " << coeffs3[i * ncoeffs_[1] * ncoeffs_[2] + j * ncoeffs_[1] + k] << "\n";
        } else if constexpr (parDim_ == 4) {
          for (int64_t l = 0; l < ncoeffs_[3]; ++l)
            for (int64_t k = 0; k < ncoeffs_[2]; ++k)
              for (int64_t j = 0; j < ncoeffs_[1]; ++j)
                for (int64_t i = 0; i < ncoeffs_[0]; ++i)
                  ss << "   " << coeffs0[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[1] * ncoeffs_[2] +
                                         k * ncoeffs_[1] + l]
                     << " " << coeffs1[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[1] * ncoeffs_[2] +
                                       k * ncoeffs_[1] + l]
                     << " " << coeffs2[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[1] * ncoeffs_[2] +
                                       k * ncoeffs_[1] + l]
                     << " " << coeffs3[i * ncoeffs_[1] * ncoeffs_[2] * ncoeffs_[3] + j * ncoeffs_[1] * ncoeffs_[2] +
                                       k * ncoeffs_[1] + l] << "\n";
        }
      } else {
        throw std::runtime_error("Unsupported parametric dimension");
      }

      ss << "  </coefs>\n"
         << " </Geometry>\n"
         << "</xml>\n";

      return ss.str();
    }

    /// @brief Saves the B-spline to file
    inline void save(const std::string& filename,
                     const std::string& key = "bspline") const
    {
      torch::serialize::OutputArchive archive;
      write(archive, key).save_to(filename);
    }

    /// @brief Loads the B-spline from file
    inline void load(const std::string& filename,
                     const std::string& key = "bspline")
    {
      torch::serialize::InputArchive archive;
      archive.load_from(filename);
      read(archive, key);
    }

    /// @brief Writes the B-spline into a torch::serialize::OutputArchive object
    inline torch::serialize::OutputArchive& write(torch::serialize::OutputArchive& archive,
                                                  const std::string& key = "bspline") const
    {
      archive.write(key + ".parDim", torch::full({1}, parDim_));
      archive.write(key + ".geoDim", torch::full({1}, geoDim_));

      for (short_t i = 0; i < parDim_; ++i)
        archive.write(key + ".degree[" + std::to_string(i) + "]", torch::full({1}, degrees_[i]));

      for (short_t i = 0; i < parDim_; ++i)
        archive.write(key + ".nknots[" + std::to_string(i) + "]", torch::full({1}, nknots_[i]));

      for (short_t i = 0; i < parDim_; ++i)
        archive.write(key + ".knots[" + std::to_string(i) + "]", knots_[i]);

      for (short_t i = 0; i < parDim_; ++i)
        archive.write(key + ".ncoeffs[" + std::to_string(i) + "]", torch::full({1}, ncoeffs_[i]));

      for (short_t i = 0; i < geoDim_; ++i)
        archive.write(key + ".coeffs[" + std::to_string(i) + "]", coeffs_[i]);

      return archive;
    }

    /// @brief Reads the B-spline from a torch::serialize::InputArchive object
    inline torch::serialize::InputArchive& read(torch::serialize::InputArchive& archive,
                                                const std::string& key = "bspline")
    {
      torch::Tensor tensor;

      archive.read(key + ".parDim", tensor);
      if (tensor.item<int64_t>() != parDim_)
        throw std::runtime_error("parDim mismatch");

      archive.read(key + ".geoDim", tensor);
      if (tensor.item<int64_t>() != geoDim_)
        throw std::runtime_error("geoDim mismatch");

      for (short_t i = 0; i < parDim_; ++i) {
        archive.read(key + ".degree[" + std::to_string(i) + "]", tensor);
        if (tensor.item<int64_t>() != degrees_[i])
          throw std::runtime_error("degrees mismatch");
      }

      for (short_t i = 0; i < parDim_; ++i) {
        archive.read(key + ".nknots[" + std::to_string(i) + "]", tensor);
        nknots_[i] = tensor.item<int64_t>();
      }

      for (short_t i = 0; i < parDim_; ++i)
        archive.read(key + ".knots[" + std::to_string(i) + "]", knots_[i]);

      for (short_t i = 0; i < parDim_; ++i) {
        archive.read(key + ".ncoeffs[" + std::to_string(i) + "]", tensor);
        ncoeffs_[i] = tensor.item<int64_t>();
      }

      for (short_t i = 0; i < geoDim_; ++i)
        archive.read(key + ".coeffs[" + std::to_string(i) + "]", coeffs_[i]);

      return archive;
    }

    /// @brief Returns true if both B-spline objects are the same
    bool operator==(const UniformBSplineCore& other) const
    {
      bool result(true);

      result *= (parDim_ == other.parDim());
      result *= (geoDim_ == other.geoDim());

      for (short_t i = 0; i < parDim_; ++i)
        result *= (degree(i) == other.degree(i));

      for (short_t i = 0; i < parDim_; ++i)
        result *= (nknots(i) == other.nknots(i));

      for (short_t i = 0; i < parDim_; ++i)
        result *= (ncoeffs(i) == other.ncoeffs(i));

      for (short_t i = 0; i < parDim_; ++i)
        result *= torch::all(knots(i) == other.knots(i)).template item<bool>();

      for (short_t i = 0; i < parDim_; ++i)
        result *= torch::all(coeffs(i) == other.coeffs(i)).template item<bool>();

      return result;
    }

    /// @brief Returns true if both B-spline objects are different
    bool operator!=(const UniformBSplineCore& other) const {
      return !(*this==other); // Do not change this to (*this != other) is it does not work
    }

  private:
    /// @brief Computes the prefactor \f$p_d!/(p_d-r_d)! = p_d \cdots (p_d-r_d+1)\f$
    template<int64_t degree, int64_t deriv, int64_t terminal=degree-deriv>
    int64_t constexpr eval_prefactor() const
    {
      if constexpr (degree > terminal)
        return degree * eval_prefactor<degree-1, deriv, terminal>();
      else
        return 1;
    }
    
  protected:
    /// @brief Initializes the B-spline coefficients
    inline void init_coeffs(BSplineInit init)
    {
      switch (init) {

        case (BSplineInit::zeros): {

          // Fill coefficients with zeros
          for (short_t i = 0; i < geoDim_; ++i) {
            int64_t size = ncoeffs();
            coeffs_[i] = torch::zeros(size, core<real_t>::options_);
          }
          break;
        }

        case (BSplineInit::ones): {

          // Fill coefficients with ones
          for (short_t i = 0; i < geoDim_; ++i) {
            int64_t size = ncoeffs();
            coeffs_[i] = torch::ones(size, core<real_t>::options_);
          }
          break;
        }

        case (BSplineInit::linear): {

          // Fill coefficients with the tensor-product of linearly
          // increasing values between 0 and 1 per univariate dimension
          for (short_t i = 0; i < geoDim_; ++i) {
            coeffs_[i] = torch::ones(1, core<real_t>::options_);

            for (short_t j = 0; j < parDim_; ++j) {
              if (i == j)
                coeffs_[i] = torch::kron(torch::linspace(static_cast<real_t>(0),
                                                         static_cast<real_t>(1),
                                                         ncoeffs_[j],
                                                         core<real_t>::options_),
                                         coeffs_[i]);
              else
                coeffs_[i] = torch::kron(torch::ones(ncoeffs_[j],
                                                     core<real_t>::options_),
                                         coeffs_[i]);
            }
          }
          break;
        }

        case (BSplineInit::random): {

          // Fill coefficients with random values
          for (short_t i = 0; i < geoDim_; ++i) {

            int64_t size = 1;
            for (short_t j = 0; j < parDim_; ++j)
              size *= ncoeffs_[j];

            coeffs_[i] = torch::rand(size, core<real_t>::options_);
          }
          break;
        }

        case (BSplineInit::greville): {

          // Fill coefficients with the tensor-product of Greville
          // abscissae values per univariate dimension
          for (short_t i = 0; i < geoDim_; ++i) {
            coeffs_[i] = torch::ones(1, core<real_t>::options_);

            for (short_t j = 0; j < parDim_; ++j) {
              if (i == j) {
                auto greville_ = torch::zeros(ncoeffs_[j], core<real_t>::options_);
                auto greville = greville_.template accessor<real_t, 1>();
                auto knots = knots_[j].template accessor<real_t, 1>();
                for (int64_t k = 0; k < ncoeffs_[j]; ++k) {
                  for (short_t l = 1; l <= degrees_[j]; ++l)
                    greville[k] += knots[k + l];
                  greville[k] /= degrees_[j];
                }
                coeffs_[i] = torch::kron(greville_, coeffs_[i]);
              } else
                coeffs_[i] = torch::kron(torch::ones(ncoeffs_[j],
                                                     core<real_t>::options_), coeffs_[i]);
            }
          }
          break;
        }

        default:
          throw std::runtime_error("Unsupported BSplineInit option");
      }
    }

    /// @brief Returns the vector of univariate B-spline basis
    /// functions (or their derivatives) evaluated in the point `xi`
    ///
    /// This function implements step 2 of algorithm \ref
    /// BSplineEvaluation, that is, it evaluates the vector of
    /// univariate B-spline basis functions (or their derivatives)
    /// that are non-zero at \f$\xi_d \in [t_{i_d}, t_{i_d+1})\f$
    ///
    /// \f[
    ///   D^{r_d}\mathbf{B}_d(\xi_d)
    ///   = \left( D^{r_d} B_{i_d-p_d,p_d}(\xi_d), \dots, D^{r_d} B_{i_d,p_d}(\xi_d) \right)^\top,
    /// \f]
    ///
    /// where \f$ p_d \f$ is the degree of the \f$d\f$-th univariate
    /// B-spline and \f$ r_d \f$ denotes the requested derivative in
    /// the \f$d\f$-direction.
    ///
    /// According to the procedure described in Chapters 2 and 3 of
    /// \cite Lyche:2011 this can be accomplished by the following
    /// expression
    ///
    /// \f[
    ///   D^{r_d}\mathbf{B}_d(\xi_d)
    ///   = \frac{p_d!}{(p_d-r_d)!}\mathbf{R}_1(\xi_d)\cdot \cdots \cdot \mathbf{R}_{p_d-r_d}(\xi_d)
    ///     D\mathbf{R}_{p_d-r_d+1}\cdot \cdots \cdot D\mathbf{R}_{p_d}(\xi_d),
    /// \f]
    ///
    /// where (cf. Equation (2.20) in \cite Lyche:2011)
    ///
    /// \f[
    ///   \mathbf{R}_k(\xi_d) =
    ///   \begin{pmatrix}
    ///     \frac{t_{i_p+1} - \xi_d}{t_{i_p+1} - t_{i_p+1-k}} & \frac{\xi_d - t_{i_p+1-k}}{t_{i_p+1} - t_{i_p+1-k}} & 0 & \cdots & 0 \\
    ///     0 & \frac{t_{i_p+2} - \xi_d}{t_{i_p+2} - t_{i_p+2-k}} & \frac{\xi_d - t_{i_p+2-k}}{t_{i_p+2} - t_{i_p+1-k}} & \cdots & 0 \\
    ///     \vdots & \vdots & \ddots & \ddots & \vdots \\
    ///     0 & 0 & \cdots & \frac{t_{i_p+k} - \xi_d}{t_{i_p+k} - t_{i_p}} & \frac{\xi_d - t_{i_p}}{t_{i_p+k} - t_{i_p}}
    ///   \end{pmatrix}
    /// \f]
    ///
    /// and (cf. Equation (3.30) in \cite Lyche:2011)
    ///
    /// \f[
    ///   D\mathbf{R}_k(\xi_d) =
    ///   \begin{pmatrix}
    ///     \frac{-1}{t_{i_p+1} - t_{i_p+1-k}} & \frac{1}{t_{i_p+1} - t_{i_p+1-k}} & 0 & \cdots & 0 \\
    ///     0 & \frac{-1}{t_{i_p+2} - t_{i_p+2-k}} & \frac{1}{t_{i_p+2} - t_{i_p+1-k}} & \cdots & 0 \\
    ///     \vdots & \vdots & \ddots & \ddots & \vdots \\
    ///     0 & 0 & \cdots & \frac{-1}{t_{i_p+k} - t_{i_p}} & \frac{1}{t_{i_p+k} - t_{i_p}}
    ///   \end{pmatrix}.
    /// \f]
    ///
    /// To improve computational efficiency, the prefactor
    ///
    /// \f[
    ///    \frac{p_d!}{(p_d-r_d)!}=p_d \cdots (p_d-r_d+1)
    /// \f]
    ///
    /// is computed as
    /// compile-time expression by the eval_prefactor() function.
    ///
    /// Moreover, the above expression for
    /// \f$D^{r_d}\mathbf{B}_d(\xi_d)\f$ is evaluated as described in
    /// Algorithm 2.22 (R-vector version) in \cite Lyche:2011) and its
    /// generalization to derivatives, respectively.
    ///
    /// The algorithm goes as follows:
    ///
    /// 1. \f$\mathbf{b} = 1\f$
    ///
    /// 2. For \f$k = 1, \dots, p_d-r_d\f$
    ///
    ///    1. \f$\mathbf{t}_1 = \left(t_{i_d-k+1},\dots,t_{i_d}\right)\f$
    ///
    ///    2. \f$\mathbf{t}_2 = \left(t_{i_d+1},\dots,t_{i_d+k}\right)\f$
    ///
    ///    3. \f$\mathbf{w}   = \left(\xi_d-\mathbf{t}_1\right)\div\left(\mathbf{t}_2-\mathbf{t}_1\right)\f$
    ///
    ///    4. \f$\mathbf{b}   = \left[\left(1-\mathbf{w}\right)\odot\mathbf{b}, 0\right]
    ///                       + \left[0, \mathbf{w}\odot\mathbf{b}\right]\f$
    ///
    /// 3. For \f$k = p_d-r_d+1, \dots, p_d\f$
    ///
    ///    1. \f$\mathbf{t}_1 = \left(t_{i_d-k+1},\dots,t_{i_d}\right)\f$
    ///
    ///    2. \f$\mathbf{t}_2 = \left(t_{i_d+1},\dots,t_{i_d+k}\right)\f$
    ///
    ///    3. \f$\mathbf{w}   = 1\div\left(\mathbf{t}_2-\mathbf{t}_1\right)\f$
    ///
    ///    4. \f$\mathbf{b}   = \left[-\mathbf{w}\odot\mathbf{b}, 0\right]
    ///                       + \left[0, \mathbf{w}\odot\mathbf{b}\right]\f$
    ///
    /// where \f$\div\f$ and \f$\odot\f$ denote the element-wise
    /// division and multiplication of vectors, respectively.
    template<short_t degree, short_t dim, short_t deriv>
    inline auto eval_univariate(const torch::Tensor& xi, const torch::Tensor& idx) const
    {
      assert(xi.sizes() == idx.sizes());

      if constexpr (deriv > degree) {
          return torch::zeros({degree+1, xi.numel()}, core<real_t>::options_);
        }
      else {
        // Algorithm 2.22 from \cite Lyche:2011
        torch::Tensor b = torch::ones({xi.numel()}, core<real_t>::options_);
        
        // Calculate R_k, k = 1, ..., p_d-r_d
        for (short_t k=1; k<= degree-deriv; ++k) {

          // Instead of calculating t1 and t2 we calculate t1 and t21=(t2-t1)
          auto t1  = knots_[dim].index_select(0, VSlice(idx, -k+1,   1) );
          auto t21 = knots_[dim].index_select(0, VSlice(idx,    1, k+1) ) - t1;

          // We handle the special case 0/0:=0 by first creating a
          // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
          // we do not have to take the absolute value as t2 >= t1.
          auto mask = (t21 < std::numeric_limits<real_t>::epsilon()).to(dtype<real_t>());

          // Instead of computing (xi-t1)/(t2-t1) which is prone to
          // yielding 0/0 we compute (xi-t1-mask)/(t2-t1-mask) which
          // equals the original expression if the mask is 0, i.e.,
          // t2-t1 >= eps and 1 otherwise since t1 <= xi < t2.          
          auto w  = torch::div(xi.repeat(k)-t1-mask, t21-mask);

          // Calculate the vector of B-splines evaluated at xi
          b = torch::cat({ torch::mul(torch::ones_like(w, core<real_t>::options_)-w, b),
              torch::zeros_like(xi, core<real_t>::options_) }, 0)
            + torch::cat({ torch::zeros_like(xi, core<real_t>::options_),
                torch::mul(w, b) }, 0);
        }

        // Calculate DR_k, k = p_d-r_d+1, ..., p_d
        for (short_t k=degree-deriv+1; k<=degree; ++k) {
          
          // Instead of calculating t1 and t2 we calculate t1 and t21=(t2-t1)
          auto t21 = knots_[dim].index_select(0, VSlice(idx,    1, k+1) )
            -        knots_[dim].index_select(0, VSlice(idx, -k+1,   1) );

          // We handle the special case 0/0:=0 by first creating a
          // mask that is 1 if t2-t1 < eps and 0 otherwise. Note that
          // we do not have to take the absolute value as t2 >= t1.
          auto mask = (t21 < std::numeric_limits<real_t>::epsilon()).to(dtype<real_t>());

          // Instead of computing 1/(t2-t1) which is prone to yielding
          // 0/0 we compute (1-mask)/(t2-t1-mask) which equals the
          // original expression if the mask is 0, i.e., t2-t1 >= eps
          // and 1 otherwise since t1 <= xi < t2.
          auto w  = torch::div(torch::ones_like(t21, core<real_t>::options_)-mask, t21-mask);

          // Calculate the vector of B-splines evaluated at xi
          b = torch::cat({ torch::mul(-w, b),
              torch::zeros_like(xi, core<real_t>::options_) }, 0)
            + torch::cat({ torch::zeros_like(xi, core<real_t>::options_),
                torch::mul(w, b) }, 0);
        }
        
        return b.view({degree+1, xi.numel()});
      }
    }
  };

  /// @brief Serializes a B-spline object
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  inline torch::serialize::OutputArchive& operator<<(torch::serialize::OutputArchive& archive,
                                                     const UniformBSplineCore<real_t, GeoDim, Degrees...>& obj)
  {
    return obj.write(archive);
  }

  /// @brief De-serializes a B-spline object
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  inline torch::serialize::InputArchive& operator>>(torch::serialize::InputArchive& archive,
                                                    UniformBSplineCore<real_t, GeoDim, Degrees...>& obj)
  {
    return obj.read(archive);
  }

  /// @brief Tensor-product non-uniform B-spline (core functionality)
  ///
  /// This class extends the base class UniformBSplineCore to
  /// non-uniform B-splines. Like its base class it only implements
  /// the core functionality of non-uniform B-splines
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  class NonUniformBSplineCore : public UniformBSplineCore<real_t, GeoDim, Degrees...>
  {
  private:
    using Base = UniformBSplineCore<real_t, GeoDim, Degrees...>;    
    
  public:
    /// @brief Constructor for equidistant knot vectors
    using UniformBSplineCore<real_t, GeoDim, Degrees...>::UniformBSplineCore;

    /// @brief Constructor for non-equidistant knot vectors
    NonUniformBSplineCore(std::array<std::vector<real_t>, Base::parDim_> kv,
                          BSplineInit init = BSplineInit::zeros)
      : Base(std::array<int64_t, Base::parDim_>{Degrees...}, init)
    {
      for (short_t i=0; i<Base::parDim_; ++i) {

        // Check that knot vector has enough (n+p+1) entries
        if (2*Base::degrees_[i]>kv[i].size()-2)
          throw std::runtime_error("Knot vector is too short for an open knot vector (n+p+1 > 2*(p+1))");

        if (core<real_t>::options_.device() == torch::kCPU)
          Base::knots_[i] = torch::from_blob(static_cast<real_t*>(kv[i].data()),
                                             kv[i].size(), core<real_t>::options_).clone();
        else
          Base::knots_[i] = torch::from_blob(static_cast<real_t *>(kv[i].data()),
                                             kv[i].size(), core<real_t>::options_.device(torch::kCPU))
            .to(core<real_t>::options_.device());

        // Store the size of the knot vector
        Base::nknots_[i] = Base::knots_[i].size(0);

        // Store the size of the coefficient vector
        Base::ncoeffs_[i] = Base::nknots_[i]-Base::degrees_[i]-1;
      }

      // Initialize coefficients
      Base::init_coeffs(init);
    }

    /// @brief Returns the value of the multivariate B-spline object in the point `xi`
    /// @{
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi) const
    {
      return eval<deriv>(TensorArray1({xi}));
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const std::array<torch::Tensor, Base::parDim_>& xi) const
    {
      if constexpr (Base::parDim_ == 0)
        return Base::coeffs_[0];
      else
        return Base::template eval<deriv>(xi, eval_knot_indices(xi));
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const std::array<torch::Tensor, Base::parDim_>& xi,
                     const std::array<torch::Tensor, Base::parDim_>& idx) const
    {
      if constexpr (Base::parDim_ == 0)
        return Base::coeffs_[0];
      else
        return Base::template eval<deriv>(xi, idx);
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const std::array<torch::Tensor, Base::parDim_>& xi,
                     const std::array<torch::Tensor, Base::parDim_>& idx,
                     const torch::Tensor& coeff_idx) const
    {
      if constexpr (Base::parDim_ == 0)
        return Base::coeffs_[0];
      else
        return Base::template eval<deriv>(xi, idx, coeff_idx);
    }
    /// @}
    
    /// @brief Returns the indices of knot spans containing `xi`
    ///
    /// This function returns the indices
    /// \f$(i_d)_{d=1}^{d_\text{par}}\f$ of the knot spans such that
    ///
    /// \f[
    ///   \boldsymbol{\xi} \in [t_{i_1}, t_{i_1+1}) \times [t_{i_2}, t_{i_2+1})
    ///   \times \dots \times [t_{i_{d_\text{par}}}, t_{i_{d_\text{par}}+1}).
    /// \f]
    ///
    /// The indices are returned as `std::array<torch::Tensor,
    /// parDim_>` in the same order as provided in `xi`
    inline auto eval_knot_indices(const TensorArray1& xi) const
    {
      assert(Base::parDim_ == 1);
      
      auto nnz0 = Base::knots_[0].repeat({xi[0].numel(), 1}) > xi[0].flatten().view({-1, 1});
      return TensorArray1({
                           torch::remainder(std::get<1>(((nnz0.cumsum(1) == 1) & nnz0).max(1))-1,
                                            Base::nknots_[0]-Base::degrees_[0]-1).view(xi[0].sizes())
                          });
    }

    /// @brief Returns the indices of knot spans containing `xi`    
    inline auto eval_knot_indices(const TensorArray2& xi) const
    {
      assert(Base::parDim_ == 2);
      
      auto nnz0 = Base::knots_[0].repeat({xi[0].numel(), 1}) > xi[0].flatten().view({-1, 1});
      auto nnz1 = Base::knots_[1].repeat({xi[1].numel(), 1}) > xi[1].flatten().view({-1, 1});
      return TensorArray2({
                           torch::remainder(std::get<1>(((nnz0.cumsum(1) == 1) & nnz0).max(1))-1,
                                            Base::nknots_[0]-Base::degrees_[0]-1).view(xi[0].sizes()),
                           torch::remainder(std::get<1>(((nnz1.cumsum(1) == 1) & nnz1).max(1))-1,
                                            Base::nknots_[1]-Base::degrees_[1]-1).view(xi[1].sizes())
                          });
    }

    /// @brief Returns the indices of knot spans containing `xi`    
    inline auto eval_knot_indices(const TensorArray3& xi) const
    {
      assert(Base::parDim_ == 3);
      
      auto nnz0 = Base::knots_[0].repeat({xi[0].numel(), 1}) > xi[0].flatten().view({-1, 1});
      auto nnz1 = Base::knots_[1].repeat({xi[1].numel(), 1}) > xi[1].flatten().view({-1, 1});
      auto nnz2 = Base::knots_[2].repeat({xi[2].numel(), 1}) > xi[2].flatten().view({-1, 1});
      return TensorArray3({
                           torch::remainder(std::get<1>(((nnz0.cumsum(1) == 1) & nnz0).max(1))-1,
                                            Base::nknots_[0]-Base::degrees_[0]-1).view(xi[0].sizes()),
                           torch::remainder(std::get<1>(((nnz1.cumsum(1) == 1) & nnz1).max(1))-1,
                                            Base::nknots_[1]-Base::degrees_[1]-1).view(xi[1].sizes()),
                           torch::remainder(std::get<1>(((nnz2.cumsum(1) == 1) & nnz2).max(1))-1,
                                            Base::nknots_[2]-Base::degrees_[2]-1).view(xi[2].sizes())
                          });
    }

    /// @brief Returns the indices of knot spans containing `xi`    
    inline auto eval_knot_indices(const TensorArray4& xi) const
    {
      assert(Base::parDim_ == 4);
      
      auto nnz0 = Base::knots_[0].repeat({xi[0].numel(), 1}) > xi[0].flatten().view({-1, 1});
      auto nnz1 = Base::knots_[1].repeat({xi[1].numel(), 1}) > xi[1].flatten().view({-1, 1});
      auto nnz2 = Base::knots_[2].repeat({xi[2].numel(), 1}) > xi[2].flatten().view({-1, 1});
      auto nnz3 = Base::knots_[3].repeat({xi[3].numel(), 1}) > xi[3].flatten().view({-1, 1});
      return TensorArray4({
                           torch::remainder(std::get<1>(((nnz0.cumsum(1) == 1) & nnz0).max(1))-1,
                                            Base::nknots_[0]-Base::degrees_[0]-1).view(xi[0].sizes()),
                           torch::remainder(std::get<1>(((nnz1.cumsum(1) == 1) & nnz1).max(1))-1,
                                            Base::nknots_[1]-Base::degrees_[1]-1).view(xi[1].sizes()),
                           torch::remainder(std::get<1>(((nnz2.cumsum(1) == 1) & nnz2).max(1))-1,
                                            Base::nknots_[2]-Base::degrees_[2]-1).view(xi[2].sizes()),
                           torch::remainder(std::get<1>(((nnz3.cumsum(1) == 1) & nnz3).max(1))-1,
                                            Base::nknots_[3]-Base::degrees_[3]-1).view(xi[3].sizes())
                          });
    }
  };

  /// @brief B-spline (common high-level functionality)
  ///
  /// This class implements some high-level common functionality of
  /// all B-spline classes, e.g., plotting which rely on low-level
  /// functionality that is implemented differently for uniform and
  /// non-uniform B-spline. C++ suggests to use virtual methods for
  /// this purpose and implement the common functionality in a base
  /// class. However, this is not performant for low-level
  /// functionality, e.g., point-wise function evaluation which is
  /// called repeatedly. Moreover, virtual methods do not work with
  /// templated functions, which is why we implement high-level common
  /// functionality here and 'inject' the core functionality by
  /// deriving from a particular base class.
  template<typename real_t, typename BSplineCore>
  class BSplineCommon : public BSplineCore
  {
  public:
    using BSplineCore::BSplineCore;

    /// @brief Returns a block-tensor with the divergence of the
    /// B-spline object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the divergence
    ///
    /// @result Block-tensor with the divergence with respect to the
    /// parametric variables `xi`
    /// \f[
    ///     \nabla_{\boldsymbol{\xi}} \cdot \mathbf{u}
    ///        =
    ///     \text{trace} ( J_{\boldsymbol{\xi}}(u) )
    ///        =
    ///     \frac{\partial u_0}{\partial \xi_0} +
    ///     \frac{\partial u_1}{\partial \xi_1} +
    ///        \dots
    ///     \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
    /// \f]
    ///
    /// @note This function can only be applied to B-spline objects with
    /// equal parametric and geometric multiplicity.
    /// @{
    auto div(torch::Tensor& xi) const
    {
      return div(TensorArray1({xi}));
    }
    
    inline auto div(const std::array<torch::Tensor, BSplineCore::parDim_>& xi) const
    {
      return div(xi, BSplineCore::eval_knot_indices(xi));
    }
    /// @}
    
    /// @brief Returns a block-tensor with the divergence of the
    /// B-spline object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the divergence
    ///
    /// @param[in] idx Knot indices where to evaluate the divergence
    ///
    /// @result Block-tensor with the divergence with respect to
    /// the parametric variables
    /// \f[
    ///     \nabla_{\boldsymbol{\xi}} \cdot \mathbf{u}
    ///        =
    ///     \text{trace} ( J_{\boldsymbol{\xi}}(u) )
    ///        =
    ///     \frac{\partial u_0}{\partial \xi_0} +
    ///     \frac{\partial u_1}{\partial \xi_1} +
    ///        \dots
    ///     \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
    /// \f]
    ///
    /// @note This function can only be applied to B-spline objects with
    /// equal parametric and geometric multiplicity.
    inline auto div(const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                    const std::array<torch::Tensor, BSplineCore::parDim_>& idx) const
    {
      return div(xi, idx, BSplineCore::eval_coeff_indices(idx));
    }
    
    /// @brief Returns a block-tensor with the divergence of the
    /// B-spline object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the divergence
    ///
    /// @param[in] idx Knot indices where to evaluate the divergence
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the divergence
    ///
    /// @result Block-tensor with the divergence of with respect to
    /// the parametric variables
    /// \f[
    ///     \nabla_{\boldsymbol{\xi}} \cdot \mathbf{u}
    ///        =
    ///     \text{trace} ( J_{\boldsymbol{\xi}}(u) )
    ///        =
    ///     \frac{\partial u_0}{\partial \xi_0} +
    ///     \frac{\partial u_1}{\partial \xi_1} +
    ///        \dots
    ///     \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
    /// \f]
    ///
    /// @note This function can only be applied to B-spline objects with
    /// equal parametric and geometric multiplicity.
    ///
    /// @{
    inline auto div(const TensorArray1& xi, const TensorArray1& idx,
                    const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes());
      if constexpr (BSplineCore::parDim_ == 1)
        
        return BlockTensor<torch::Tensor, 1, 1>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx)[0]);
      
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
    
    inline auto div(const TensorArray2& xi, const TensorArray2& idx,
                    const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes());

      if constexpr (BSplineCore::parDim_ == 2)
      
        return BlockTensor<torch::Tensor, 1, 1>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx)[0] +
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx)[1]);
       
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto div(const TensorArray3& xi, const TensorArray3& idx,
                    const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes());
    
      if constexpr (BSplineCore::parDim_ == 3)
      
        return BlockTensor<torch::Tensor, 1, 1>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx)[0] +
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx)[1] +
           BSplineCore::template eval<BSplineDeriv::dz>(xi, idx, coeff_idx)[2]);
    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto div(const TensorArray4& xi, const TensorArray4& idx,
                    const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[3].sizes() == idx[3].sizes());
    
      if constexpr (BSplineCore::parDim_ == 4)
      
        return BlockTensor<torch::Tensor, 1, 1>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx)[0] +
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx)[1] +
           BSplineCore::template eval<BSplineDeriv::dz>(xi, idx, coeff_idx)[2] +
           BSplineCore::template eval<BSplineDeriv::dt>(xi, idx, coeff_idx)[3]);
    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
    /// @}

    /// @brief Returns a block-tensor with the divergence of the
    /// B-spline object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the divergence
    ///
    /// @result Block-tensor with the divergence with respect to the
    /// parametric variables    
    /// \f[
    ///     \nabla_{\mathbf{x}} \cdot \mathbf{u}
    ///        =
    ///     \text{trace} ( J_{\mathbf{x}}(u) )
    ///        =
    ///     \frac{\partial u_0}{\partial x_0} +
    ///     \frac{\partial u_1}{\partial x_1} +
    ///     \frac{\partial u_{d_\text{geo}}}{\partial x_{d_\text{par}}}
    /// \f]
    ///
    /// @{
    template<typename Geometry_t>
    auto idiv(const Geometry_t& G, torch::Tensor& xi)
    {
      return idiv(G, TensorArray1({xi}));
    }
  
    template<typename Geometry_t>
    inline auto idiv(const Geometry_t& G,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& xi) const
    {
      return idiv(G, xi, eval_knot_indices(xi), G.eval_knot_indices(xi));
    }
    /// @}

    /// @brief Returns a block-tensor with the divergence of the
    /// B-spline object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the divergence
    ///
    /// @param[in] idx Knot indices where to evaluate the divergence
    ///
    /// @param[in] idx_G Knot indices where to evaluate the Jacobian of `G`
    ///
    /// @result Block-tensor with the divergence with respect to the
    /// physical variables
    /// \f[
    ///     \nabla_{\mathbf{x}} \cdot \mathbf{u}
    ///        =
    ///     \text{trace} ( J_{\mathbf{x}}(u) )
    ///        =
    ///     \frac{\partial u_0}{\partial x_0} +
    ///     \frac{\partial u_1}{\partial x_1} +
    ///     \frac{\partial u_{d_\text{geo}}}{\partial x_{d_\text{par}}}
    /// \f]
    template<typename Geometry_t>
    inline auto idiv(const Geometry_t G,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& idx,
                     const std::array<torch::Tensor, Geometry_t::parDim()>& idx_G) const
    {
      return idiv(G, xi, idx, eval_coeff_indices(idx),
                  idx_G, G.eval_coeff_indices(idx_G));
    }

    /// @brief Returns a block-tensor with the divergence of the
    /// B-spline object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the divergence
    ///
    /// @param[in] idx Knot indices where to evaluate the divergence
    ///
    /// @param[in] idx_G Knot indices where to evaluate the Jacobian of `G`
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the divergence
    ///
    /// @param[in] coeff_idx_G Coefficient indices where to evaluate the Jacobian of `G`
    ///
    /// @result Block-tensor with the divergence with respect to the
    /// physical variables
    /// \f[
    ///     \nabla_{\mathbf{x}} \cdot \mathbf{u}
    ///        =
    ///     \text{trace} ( J_{\mathbf{x}}(u) )
    ///        =
    ///     \frac{\partial u_0}{\partial x_0} +
    ///     \frac{\partial u_1}{\partial x_1} +
    ///     \frac{\partial u_{d_\text{geo}}}{\partial x_{d_\text{par}}}
    /// \f]
    template<typename Geometry_t>
    inline auto idiv(const Geometry_t& G,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& idx,
                     const torch::Tensor& coeff_idx, 
                     const std::array<torch::Tensor, Geometry_t::parDim()>& idx_G,
                     const torch::Tensor& coeff_idx_G) const
    {
      return BSplineCore::ijac(xi, idx, coeff_idx, idx_G, coeff_idx_G).trace();
    }
    
    /// @brief Returns a block-tensor with the gradient of the B-spline
    /// object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the gradient
    ///
    /// @result Block-tensor with the gradient with respect to the
    /// parametric variables `xi`
    /// \f[
    ///     \nabla_{\boldsymbol{\xi}}u
    ///        =
    ///     \left(\frac{\partial u}{\partial \xi_0},
    ///           \frac{\partial u}{\partial \xi_1},
    ///           \dots
    ///           \frac{\partial u}{\partial \xi_{d_\text{par}}}\right)
    /// \f]
    ///
    /// @note This function can only be applied to B-spline objects with
    /// geometric multiplicity 1, i.e. scalar fields.
    ///
    /// @{
    auto grad(torch::Tensor& xi) const
    {
      return grad(TensorArray1({xi}));
    }
    
    inline auto grad(const std::array<torch::Tensor, BSplineCore::parDim_>& xi) const
    {
      return grad(xi, BSplineCore::eval_knot_indices(xi));
    }
    /// @}
    
    /// @brief Returns a block-tensor with the gradient of the
    /// B-spline object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the gradient
    ///
    /// @param[in] idx Knot indices where to evaluate the gradient
    ///
    /// @result Block-tensor with the gradient with respect to
    /// the parametric variables
    /// \f[
    ///     \nabla_{\boldsymbol{\xi}}u
    ///        =
    ///     \left(\frac{\partial u}{\partial \xi_0},
    ///           \frac{\partial u}{\partial \xi_1},
    ///           \dots
    ///           \frac{\partial u}{\partial \xi_{d_\text{par}}}\right)
    /// \f]
    ///
    /// @note This function can only be applied to B-spline objects with
    /// geometric multiplicity 1, i.e. scalar fields.
    inline auto grad(const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& idx) const
    {
      return grad(xi, idx, BSplineCore::eval_coeff_indices(idx));
    }
    
    /// @brief Returns a block-tensor with the gradient of the
    /// B-spline object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the gradient
    ///
    /// @param[in] idx Knot indices where to evaluate the gradient
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the gradient
    ///
    /// @result Block-tensor with the gradient with respect to the
    /// parametric variables    
    /// \f[
    ///     \nabla_{\boldsymbol{\xi}}u
    ///        =
    ///     \left(\frac{\partial u}{\partial \xi_0},
    ///           \frac{\partial u}{\partial \xi_1},
    ///           \dots
    ///           \frac{\partial u}{\partial \xi_{d_\text{par}}}\right)
    /// \f]
    ///
    /// @note This function can only be applied to B-spline objects with
    /// geometric multiplicity 1, i.e. scalar fields.
    ///
    /// @{
    inline auto grad(const TensorArray1& xi, const TensorArray1& idx,
                     const torch::Tensor& coeff_idx) const
    {
      static_assert(BSplineCore::geoDim_ == 1,
                    "grad(.) requires 1D variable, use jac(.) instead");
      assert(xi[0].sizes() == idx[0].sizes());
      if constexpr (BSplineCore::parDim_ == 1)
        
        return BlockTensor<torch::Tensor, 1, 1>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx));
      
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto grad(const TensorArray2& xi, const TensorArray2& idx,
                     const torch::Tensor& coeff_idx) const
    {
      static_assert(BSplineCore::geoDim_ == 1,
                    "grad(.) requires 1D variable, use jac(.) instead");
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes());

      if constexpr (BSplineCore::parDim_ == 2)
      
        return BlockTensor<torch::Tensor, 1, 2>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx));
       
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto grad(const TensorArray3& xi, const TensorArray3& idx,
                     const torch::Tensor& coeff_idx) const
    {
      static_assert(BSplineCore::geoDim_ == 1,
                    "grad(.) requires 1D variable, use jac(.) instead");
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes());
    
      if constexpr (BSplineCore::parDim_ == 3)
      
        return BlockTensor<torch::Tensor, 1, 3>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dz>(xi, idx, coeff_idx));
    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto grad(const TensorArray4& xi, const TensorArray4& idx,
                     const torch::Tensor& coeff_idx) const
    {
      static_assert(BSplineCore::geoDim_ == 1,
                    "grad(.) requires 1D variable, use jac(.) instead");
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[3].sizes() == idx[3].sizes());
    
      if constexpr (BSplineCore::parDim_ == 4)
      
        return BlockTensor<torch::Tensor, 1, 4>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dz>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dt>(xi, idx, coeff_idx));
    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
    /// @}

    /// @brief Returns a block-tensor with the gradient of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the gradient
    ///
    /// @result Block-tensor with the gradient with respect to the
    /// parametric variables    
    /// \f[
    ///     \nabla_{\mathbf{x}} u
    ///        =
    ///     \nabla_{\boldsymbol{\xi}} u \, J_{\boldsymbol{\xi}}(G)^{-T} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi})
    /// \f]
    ///
    /// @{
    template<typename Geometry_t>
    auto igrad(const Geometry_t& G, torch::Tensor& xi)
    {
      return igrad(G, TensorArray1({xi}));
    }
  
    template<typename Geometry_t>
    inline auto igrad(const Geometry_t& G,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& xi) const
    {
      return igrad(G, xi, BSplineCore::eval_knot_indices(xi), G.eval_knot_indices(xi));
    }
    /// @}

    /// @brief Returns a block-tensor with the gradient of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the gradient
    ///
    /// @param[in] idx Knot indices where to evaluate the gradient
    ///
    /// @param[in] idx_G Knot indices where to evaluate Jacobian of `G`
    ///
    /// @result Block-tensor with the gradient with respect to the
    /// physical variables    
    /// \f[
    ///     \nabla_{\mathbf{x}} u
    ///        =
    ///     \nabla_{\boldsymbol{\xi}} u \, J_{\boldsymbol{\xi}}(G)^{-T} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi}) 
    /// \f]
    template<typename Geometry_t>
    inline auto igrad(const Geometry_t G,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& idx,
                      const std::array<torch::Tensor, Geometry_t::parDim()>& idx_G) const
    {
      return igrad(G, xi, idx, BSplineCore::eval_coeff_indices(idx),
                   idx_G, G.eval_coeff_indices(idx_G));
    }

    /// @brief Returns a block-tensor with the gradient of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the gradient
    ///
    /// @param[in] idx Knot indices where to evaluate the gradient
    ///
    /// @param[in] idx_G Knot indices where to evaluate the Jacobian of `G`
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the gradient
    ///
    /// @param[in] coeff_idx_G Coefficient indices where to evaluate the Jacobian of `G`
    ///
    /// @result Block-tensor with the gradient with respect to the
    /// physical variables    
    /// \f[
    ///     \nabla_{\mathbf{x}} u
    ///        =
    ///     \nabla_{\boldsymbol{\xi}} u \, J_{\boldsymbol{\xi}}(G)^{-T} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi}) 
    /// \f]
    template<typename Geometry_t>
    inline auto igrad(const Geometry_t& G,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& idx,
                      const torch::Tensor& coeff_idx, 
                      const std::array<torch::Tensor, Geometry_t::parDim()>& idx_G,
                      const torch::Tensor& coeff_idx_G) const
    {
      return grad(xi, idx, coeff_idx) * G.jac(xi, idx_G, coeff_idx_G).ginv();
    }

    /// @brief Returns a block-tensor with the Hessian of the B-spline
    /// object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the Hessian
    ///
    /// @result Block-tensor with the Hessian with respect to the
    /// parametric variables `xi`
    /// \f[
    ///     H_{\boldsymbol{\xi}}(u)
    ///        =
    ///     \begin{bmatrix}
    ///           \frac{\partial^2 u}{\partial^2 \xi_0}&
    ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_{d_\text{par}}}\\
    ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_0}&
    ///           \frac{\partial^2 u}{\partial^2 \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_{d_\text{par}}}\\
    ///           \vdots& \vdots & \ddots & \vdots\\
    ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_0}&
    ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial^2 \xi_{d_\text{par}}}
    ///     \end{bmatrix}
    /// \f]
    ///
    /// @note If the B-spline object has geometric dimension larger
    /// then one then all Hessian matrices are returned as slices of a
    /// rank-3 tensor.
    ///
    /// @{
    auto hess(torch::Tensor& xi) const
    {
      return hess(TensorArray1({xi}));
    }
    
    inline auto hess(const std::array<torch::Tensor, BSplineCore::parDim_>& xi) const
    {
      return hess(xi, BSplineCore::eval_knot_indices(xi));
    }
    /// @}
    
    /// @brief Returns a block-tensor with the Hessian of the
    /// B-spline object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the gradient
    ///
    /// @param[in] idx Knot indices where to evaluate the Hessian
    ///
    /// @result Block-tensor with the Hessian with respect to
    /// the parametric variables
    /// \f[
    ///     H_{\boldsymbol{\xi}}(u)
    ///        =
    ///     \begin{bmatrix}
    ///           \frac{\partial^2 u}{\partial^2 \xi_0}&
    ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_{d_\text{par}}}\\
    ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_0}&
    ///           \frac{\partial^2 u}{\partial^2 \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_{d_\text{par}}}\\
    ///           \vdots& \vdots & \ddots & \vdots\\
    ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_0}&
    ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial^2 \xi_{d_\text{par}}}
    ///     \end{bmatrix}
    /// \f]
    ///
    /// @note If the B-spline object has geometric dimension larger
    /// then one then all Hessian matrices are returned as slices of a
    /// rank-3 tensor.
    inline auto hess(const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& idx) const
    {
      return hess(xi, idx, BSplineCore::eval_coeff_indices(idx));
    }
    
    /// @brief Returns a block-tensor with the Hessian of the
    /// B-spline object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the Hessian
    ///
    /// @param[in] idx Knot indices where to evaluate the Hessian
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the Hessian
    ///
    /// @result Block-tensor with the Hessian with respect to the
    /// parametric variables
    /// \f[
    ///     H_{\boldsymbol{\xi}}(u)
    ///        =
    ///     \begin{bmatrix}
    ///           \frac{\partial^2 u}{\partial^2 \xi_0}&
    ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial \xi_0\partial \xi_{d_\text{par}}}\\
    ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_0}&
    ///           \frac{\partial^2 u}{\partial^2 \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial \xi_1\partial \xi_{d_\text{par}}}\\
    ///           \vdots& \vdots & \ddots & \vdots\\
    ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_0}&
    ///           \frac{\partial^2 u}{\partial \xi_{d_\text{par}}\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial^2 u}{\partial^2 \xi_{d_\text{par}}}
    ///     \end{bmatrix}
    /// \f]
    ///
    ///
    /// @note If the B-spline object has geometric dimension larger
    /// then one then all Hessian matrices are returned as slices of a
    /// rank-3 tensor.
    ///
    /// @{
    inline auto hess(const TensorArray1& xi, const TensorArray1& idx,
                     const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes());
      if constexpr (BSplineCore::parDim_ == 1)        
        return BlockTensor<torch::Tensor, 1, 1, BSplineCore::geoDim_>
          (BSplineCore::template eval<BSplineDeriv::dx2>(xi, idx, coeff_idx));      
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto hess(const TensorArray2& xi, const TensorArray2& idx,
                     const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes());

      if constexpr (BSplineCore::parDim_ == 2)     
        return BlockTensor<torch::Tensor, 2, BSplineCore::geoDim_, 2>
          (BSplineCore::template eval<BSplineDeriv::dx2 >(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dxdy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dydx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy2 >(xi, idx, coeff_idx)).reorder_ikj();       
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto hess(const TensorArray3& xi, const TensorArray3& idx,
                     const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes());
    
      if constexpr (BSplineCore::parDim_ == 3)     
        return BlockTensor<torch::Tensor, 3, BSplineCore::geoDim_, 3>
          (BSplineCore::template eval<BSplineDeriv::dx2 >(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dxdy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dxdz>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dydx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy2> (xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dydz>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dzdx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dzdy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dz2 >(xi, idx, coeff_idx)).reorder_ikj();    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto hess(const TensorArray4& xi, const TensorArray4& idx,
                     const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[3].sizes() == idx[3].sizes());
    
      if constexpr (BSplineCore::parDim_ == 4)      
        return BlockTensor<torch::Tensor, 4, BSplineCore::geoDim_, 4>
          (BSplineCore::template eval<BSplineDeriv::dx2 >(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dxdy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dxdz>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dxdt>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dydx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy2 >(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dydz>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dydt>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dzdx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dzdy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dz2 >(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dzdt>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dtdx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dtdy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dtdz>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dt2 >(xi, idx, coeff_idx)).reorder_ikj();    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
    /// @}

    /// @brief Returns a block-tensor with the Hessian of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the Hessian
    ///
    /// @result Block-tensor with the Hessian with respect to the
    /// parametric variables
    /// \f[
    ///     H_{\mathbf{x}}(u)
    ///        =
    ///     J_{\boldsymbol{\xi}}(G)^{-T}
    ///     \left(
    ///       H_\boldsymbol{\xi}(u)
    ///       -
    ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
    ///     \right)
    ///     J_{\boldsymbol{\xi}}(G)^{-1} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi})
    /// \f]    
    ///
    /// @{
    template<typename Geometry_t>
    auto ihess(const Geometry_t& G, torch::Tensor& xi)
    {
      return ihess(G, TensorArray1({xi}));
    }
  
    template<typename Geometry_t>
    inline auto ihess(const Geometry_t& G,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& xi) const
    {
      return ihess(G, xi, BSplineCore::eval_knot_indices(xi), G.eval_knot_indices(xi));
    }
    /// @}

    /// @brief Returns a block-tensor with the Hessian of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the Hessian
    ///
    /// @param[in] idx Knot indices where to evaluate the Hessian
    ///
    /// @param[in] idx_G Knot indices where to evaluate Jacobian of `G`
    ///
    /// @result Block-tensor with the Hessian with respect to the
    /// physical variables
    /// \f[
    ///     H_{\mathbf{x}}(u)
    ///        =
    ///     J_{\boldsymbol{\xi}}(G)^{-T}
    ///     \left(
    ///       H_\boldsymbol{\xi}(u)
    ///       -
    ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
    ///     \right)
    ///     J_{\boldsymbol{\xi}}(G)^{-1} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi})
    /// \f]    
    template<typename Geometry_t>
    inline auto ihess(const Geometry_t G,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& idx,
                      const std::array<torch::Tensor, Geometry_t::parDim()>& idx_G) const
    {
      return ihess(G, xi, idx, BSplineCore::eval_coeff_indices(idx),
                   idx_G, G.eval_coeff_indices(idx_G));
    }

    /// @brief Returns a block-tensor with the Hessian of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the Hessian
    ///
    /// @param[in] idx Knot indices where to evaluate the Hessian
    ///
    /// @param[in] idx_G Knot indices where to evaluate the Jacobian of `G`
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the Hessian
    ///
    /// @param[in] coeff_idx_G Coefficient indices where to evaluate the Jacobian of `G`
    ///
    /// @result Block-tensor with the Hessian with respect to the
    /// physical variables    
    /// \f[
    ///     H_{\mathbf{x}}(u)
    ///        =
    ///     J_{\boldsymbol{\xi}}(G)^{-T}
    ///     \left(
    ///       H_\boldsymbol{\xi}(u)
    ///       -
    ///       \sum_k \nabla_{\mathbf{x},k}u H_{\boldsymbol{\xi}}(G_k)
    ///     \right)
    ///     J_{\boldsymbol{\xi}}(G)^{-1} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi})
    /// \f]    
    template<typename Geometry_t>
    inline auto ihess(const Geometry_t& G,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                      const std::array<torch::Tensor, BSplineCore::parDim_>& idx,
                      const torch::Tensor& coeff_idx, 
                      const std::array<torch::Tensor, Geometry_t::parDim()>& idx_G,
                      const torch::Tensor& coeff_idx_G) const
    {
      
      auto hessu  = hess(xi, idx, coeff_idx).slice(0);

      {
        auto igradG = igrad(G, xi, idx, coeff_idx, idx_G, coeff_idx_G);
        auto hessG  = G.hess(xi, idx_G, coeff_idx_G);
        assert(igradG.cols() == hessG.slices());
        for (short_t k=0; k<hessG.slices(); ++k)
          hessu -= igradG(0,k)*hessG.slice(k);
      }

      auto jacInv = G.jac(xi, idx_G, coeff_idx_G).ginv();
      return jacInv.tr() * hessu * jacInv;
    }

    /// @brief Returns a block-tensor with the Jacobian of the B-spline
    /// object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the Jacobian
    ///
    /// @result Block-tensor with the Jacobian with respect to the
    /// parametric variables    
    /// \f[
    ///     J_{\boldsymbol{\xi}}(u)
    ///        =
    ///     \begin{bmatrix}
    ///           \frac{\partial u_0}{\partial \xi_0}&
    ///           \frac{\partial u_0}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_0}{\partial \xi_{d_\text{par}}}\\
    ///           \frac{\partial u_1}{\partial \xi_0}&
    ///           \frac{\partial u_1}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_1}{\partial \xi_{d_\text{par}}}\\
    ///           \vdots& \vdots & \ddots & \vdots\\
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_0}&
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
    ///     \end{bmatrix}
    /// \f]
    ///
    /// @{
    auto jac(torch::Tensor& xi)
    {
      return jac(TensorArray1({xi}));
    }
  
    inline auto jac(const std::array<torch::Tensor, BSplineCore::parDim_>& xi) const
    {
      return jac(xi, BSplineCore::eval_knot_indices(xi));
    }
    /// @}

    /// @brief Returns a block-tensor with the Jacobian of the B-spline
    /// object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the Jacobian
    ///
    /// @param[in] idx Knot indices where to evaluate the Jacobian
    ///
    /// @result Block-tensor with the Jacobian with respect to the
    /// parametric variables   
    /// \f[
    ///     J_{\boldsymbol{\xi}}(u)
    ///        =
    ///     \begin{bmatrix}
    ///           \frac{\partial u_0}{\partial \xi_0}&
    ///           \frac{\partial u_0}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_0}{\partial \xi_{d_\text{par}}}\\
    ///           \frac{\partial u_1}{\partial \xi_0}&
    ///           \frac{\partial u_1}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_1}{\partial \xi_{d_\text{par}}}\\
    ///           \vdots& \vdots & \ddots & \vdots\\
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_0}&
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
    ///     \end{bmatrix}
    /// \f]
    inline auto jac(const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                    const std::array<torch::Tensor, BSplineCore::parDim_>& idx) const
    {
      return jac(xi, idx, BSplineCore::eval_coeff_indices(idx));
    }

    /// @brief Returns a block-tensor with the Jacobian of the B-spline
    /// object with respect to the parametric variables
    ///
    /// @param[in] xi Point(s) where to evaluate the Jacobian
    ///
    /// @param[in] idx Knot indices where to evaluate the Jacobian
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the Jacobian
    ///
    /// @result Block-tensor with the Jacobian with respect to the
    /// parametric variables    
    /// \f[
    ///     J_{\boldsymbol{\xi}}(u)
    ///        =
    ///     \begin{bmatrix}
    ///           \frac{\partial u_0}{\partial \xi_0}&
    ///           \frac{\partial u_0}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_0}{\partial \xi_{d_\text{par}}}\\
    ///           \frac{\partial u_1}{\partial \xi_0}&
    ///           \frac{\partial u_1}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_1}{\partial \xi_{d_\text{par}}}\\
    ///           \vdots& \vdots & \ddots & \vdots\\
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_0}&
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_1}&
    ///           \dots&
    ///           \frac{\partial u_{d_\text{geo}}}{\partial \xi_{d_\text{par}}}
    ///     \end{bmatrix}
    /// \f]
    ///
    /// @note Since the B-spline evaluation function computes the
    /// specified derivatives for all variables simultaneously we
    /// compute the transpose of the Jacobian and return its
    /// tranposed, hence, the Jacobian.    
    /// @{
    inline auto jac(const TensorArray1& xi, const TensorArray1& idx,
                    const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes());
    
      if constexpr (BSplineCore::parDim_ == 1)
      
        return BlockTensor<torch::Tensor, 1, BSplineCore::geoDim_>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx)).tr();
    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
  
    inline auto jac(const TensorArray2& xi, const TensorArray2& idx,
                    const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes());

      if constexpr (BSplineCore::parDim_ == 2)
      
        return BlockTensor<torch::Tensor, 2, BSplineCore::geoDim_>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx)).tr();
    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto jac(const TensorArray3& xi, const TensorArray3& idx,
                    const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes());

      if constexpr (BSplineCore::parDim_ == 3)
      
        return BlockTensor<torch::Tensor, 3, BSplineCore::geoDim_>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dz>(xi, idx, coeff_idx)).tr();
    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }

    inline auto jac(const TensorArray4& xi, const TensorArray4& idx,
                    const torch::Tensor& coeff_idx) const
    {
      assert(xi[0].sizes() == idx[0].sizes() &&
             xi[1].sizes() == idx[1].sizes() &&
             xi[2].sizes() == idx[2].sizes() &&
             xi[3].sizes() == idx[3].sizes());

      if constexpr (BSplineCore::parDim_ == 4)
      
        return BlockTensor<torch::Tensor, 4, BSplineCore::geoDim_>
          (BSplineCore::template eval<BSplineDeriv::dx>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dy>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dz>(xi, idx, coeff_idx),
           BSplineCore::template eval<BSplineDeriv::dt>(xi, idx, coeff_idx)).tr();
    
      else
        throw std::runtime_error("Unsupported parametric dimension");
    }
    /// @}

    /// @brief Returns a block-tensor with the Jacobian of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the Jacobian
    ///
    /// @result Block-tensor with the Jacobian with respect to the
    /// parametric variables    
    /// \f[
    ///     J_{\mathbf{x}}(u)
    ///        =
    ///     J_{\boldsymbol{\xi}}(u) \, J_{\boldsymbol{\xi}}(G)^{-T} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi})
    /// \f]
    ///
    /// @{
    template<typename Geometry_t>
    auto ijac(const Geometry_t& G, torch::Tensor& xi)
    {
      return ijac(G, TensorArray1({xi}));
    }
  
    template<typename Geometry_t>
    inline auto ijac(const Geometry_t& G,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& xi) const
    {
      return ijac(G, xi, BSplineCore::eval_knot_indices(xi), G.eval_knot_indices(xi));
    }
    /// @}

    /// @brief Returns a block-tensor with the Jacobian of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the Jacobian
    ///
    /// @param[in] idx Knot indices where to evaluate the Jacobian
    ///
    /// @param[in] idx_G Knot indices where to evaluate the Jacobian of `G`
    ///
    /// @result Block-tensor with the Jacobian with respect to the
    /// physical variables    
    /// \f[
    ///     J_{\mathbf{x}}(u)
    ///        =
    ///     J_{\boldsymbol{\xi}}(u) \, J_{\boldsymbol{\xi}}(G)^{-T} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi})
    /// \f]
    template<typename Geometry_t>
    inline auto ijac(const Geometry_t G,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& idx,
                     const std::array<torch::Tensor, Geometry_t::parDim()>& idx_G) const
    {
      return ijac(G, xi, idx, BSplineCore::eval_coeff_indices(idx),
                  idx_G, G.eval_coeff_indices(idx_G));
    }

    /// @brief Returns a block-tensor with the Jacobian of the B-spline
    /// object with respect to the physical variables
    ///
    /// @tparam Geometry_t Type of the geometry B-spline object
    ///
    /// @param[in] G B-spline geometry object
    ///
    /// @param[in] xi Point(s) where to evaluate the Jacobian
    ///
    /// @param[in] idx Knot indices where to evaluate the Jacobain
    ///
    /// @param[in] idx_G Knot indices where to evaluate the Jacobian of `G`
    ///
    /// @param[in] coeff_idx Coefficient indices where to evaluate the Jacobian
    ///
    /// @param[in] coeff_idx_G Coefficient indices where to evaluate the Jacobian of `G`
    ///
    /// @result Block-tensor with the Jacobian with respect to the
    /// physical variables    
    /// \f[
    ///     J_{\mathbf{x}}(u)
    ///        =
    ///     J_{\boldsymbol{\xi}}(u) \, J_{\boldsymbol{\xi}}(G)^{-T} ,
    ///     \quad
    ///     \mathbf{x} = G(\boldsymbol{\xi})
    /// \f]
    template<typename Geometry_t>
    inline auto ijac(const Geometry_t& G,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& xi,
                     const std::array<torch::Tensor, BSplineCore::parDim_>& idx,
                     const torch::Tensor& coeff_idx, 
                     const std::array<torch::Tensor, Geometry_t::parDim()>& idx_G,
                     const torch::Tensor& coeff_idx_G) const
    {
      return jac(xi, idx, coeff_idx) * G.jac(xi, idx_G, coeff_idx_G).ginv();
    }  
    
    /// Plots the B-spline object using matplotlibcpp
    inline auto plot(int64_t res0=10, int64_t res1=10, int64_t res2=10) const
    {
      return plot(*this, res0, res1, res2);
    }

    /// Plots the B-spline object using matplotlibcpp
    template<typename BSplineCore_t>
    inline auto plot(const BSplineCommon<real_t, BSplineCore_t>& color,
                     int64_t res0=10, int64_t res1=10, int64_t res2=10) const
    {
      static_assert(BSplineCore::parDim() == BSplineCore_t::parDim(),
                    "Parametric dimensions must match");

      if ((void*)this != (void*)&color && BSplineCore_t::geoDim() > 1)
        throw std::runtime_error("BSpline for coloring must have geoDim=1");

      if constexpr(BSplineCore::parDim()==1 && BSplineCore::geoDim()==1) {

        //
        // mapping: [0,1] -> R^1
        //

        matplot::vector_1d Xfine(res0, 0.0);
        matplot::vector_1d Yfine(res0, 0.0);

        auto Coords = BSplineCore::eval(torch::linspace(0, 1, res0));
        auto XAccessor = Coords(0).template accessor<real_t,1>();
        
#pragma omp parallel for simd
        for (int64_t i=0; i<res0; ++i)
          Xfine[i] = XAccessor[i];

        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
            auto Color = color.eval(torch::linspace(0, 1, res0));
            auto CAccessor = Color(0).template accessor<real_t,1>();
            
#pragma omp parallel for simd
            for (int64_t i=0; i<res0; ++i)
              Yfine[i] = CAccessor[i];
          }
        }

        // Plotting ...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
            matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          }
        } else {
          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);

          auto xAccessor = BSplineCore::coeffs(0).template accessor<real_t,1>();

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = xAccessor[i];
          }

          matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot(X, Y, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R");
        matplot::xlabel("x");
        matplot::ylabel("y");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim_==1 && BSplineCore::geoDim_==2) {

        //
        // mapping: [0,1] -> R^2
        //

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim()==1) {
            matplot::vector_2d Xfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Yfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Zfine(1, matplot::vector_1d(res0, 0.0));

            auto Coords = BSplineCore::eval(torch::linspace(0, 1, res0));
            auto XAccessor = Coords(0).template accessor<real_t,1>();
            auto YAccessor = Coords(1).template accessor<real_t,1>();

            auto Color = color.eval(torch::linspace(0, 1, res0));
            auto CAccessor = Color(0).template accessor<real_t,1>();
            
#pragma omp parallel for simd
            for (int64_t i=0; i<res0; ++i) {              
              Xfine[0][i] = XAccessor[i];
              Yfine[0][i] = YAccessor[i];
              Zfine[0][i] = CAccessor[i];
            }
            matplot::view(2);
            matplot::mesh(Xfine, Yfine, Zfine);
          }
        } else {
          matplot::vector_1d Xfine(res0, 0.0);
          matplot::vector_1d Yfine(res0, 0.0);

          auto Coords = BSplineCore::eval(torch::linspace(0, 1, res0));
          auto XAccessor = Coords(0).template accessor<real_t,1>();
          auto YAccessor = Coords(1).template accessor<real_t,1>();
          
#pragma omp parallel for simd
          for (int64_t i=0; i<res0; ++i) {
            Xfine[i] = XAccessor[i];
            Yfine[i] = YAccessor[i];
          }

          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);

          auto xAccessor = BSplineCore::coeffs(0).template accessor<real_t,1>();
          auto yAccessor = BSplineCore::coeffs(1).template accessor<real_t,1>();

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = xAccessor[i];
            Y[i] = yAccessor[i];
          }

          matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot(X, Y, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R^2");
        matplot::xlabel("x");
        matplot::ylabel("y");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim()==1 && BSplineCore::geoDim()==3) {

        //
        // mapping: [0,1] -> R^3
        //

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim()==1) {
            matplot::vector_2d Xfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Yfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Zfine(1, matplot::vector_1d(res0, 0.0));
            matplot::vector_2d Cfine(1, matplot::vector_1d(res0, 0.0));

            auto Coords = BSplineCore::eval(torch::linspace(0, 1, res0));
            auto XAccessor = Coords(0).template accessor<real_t,1>();
            auto YAccessor = Coords(1).template accessor<real_t,1>();
            auto ZAccessor = Coords(2).template accessor<real_t,1>();

            auto Color = color.eval(torch::linspace(0, 1, res0));
            auto CAccessor = Color(0).template accessor<real_t,1>();
            
#pragma omp parallel for simd
            for (int64_t i=0; i<res0; ++i) {
              Xfine[0][i] = XAccessor[i];
              Yfine[0][i] = YAccessor[i];
              Zfine[0][i] = ZAccessor[i];
              Cfine[0][i] = CAccessor[i];
            }

            matplot::mesh(Xfine, Yfine, Zfine, Cfine);
          }
        } else {
          matplot::vector_1d Xfine(res0, 0.0);
          matplot::vector_1d Yfine(res0, 0.0);
          matplot::vector_1d Zfine(res0, 0.0);

          auto Coords = BSplineCore::eval(torch::linspace(0, 1, res0));
          auto XAccessor = Coords(0).template accessor<real_t,1>();
          auto YAccessor = Coords(1).template accessor<real_t,1>();
          auto ZAccessor = Coords(2).template accessor<real_t,1>();
          
#pragma omp parallel for simd
          for (int64_t i=0; i<res0; ++i) {
            Xfine[i] = XAccessor[i];
            Yfine[i] = YAccessor[i];
            Zfine[i] = ZAccessor[i];
          }

          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Z(BSplineCore::ncoeffs(0), 0.0);

          auto xAccessor = BSplineCore::coeffs(0).template accessor<real_t,1>();
          auto yAccessor = BSplineCore::coeffs(1).template accessor<real_t,1>();
          auto zAccessor = BSplineCore::coeffs(2).template accessor<real_t,1>();

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = xAccessor[i];
            Y[i] = yAccessor[i];
            Z[i] = zAccessor[i];
          }

          matplot::plot3(Xfine, Yfine, Zfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot3(X, Y, Z, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R^3");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim()==2 && BSplineCore::geoDim()==2) {

        //
        // mapping: [0,1]^2 -> R^2
        //

        matplot::vector_2d Xfine(res1, matplot::vector_1d(res0, 0.0));
        matplot::vector_2d Yfine(res1, matplot::vector_1d(res0, 0.0));
        matplot::vector_2d Zfine(res1, matplot::vector_1d(res0, 0.0));

        std::array<torch::Tensor,2> meshgrid = convert<2>(torch::meshgrid({torch::linspace(0, 1, res0),
                                                                           torch::linspace(0, 1, res1)}, "xy"));
        auto Coords = BSplineCore::eval(meshgrid);
        auto XAccessor = Coords(0).template accessor<real_t,2>();
        auto YAccessor = Coords(1).template accessor<real_t,2>();

#pragma omp parallel for simd collapse(2)
        for (int64_t i=0; i<res0; ++i)
          for (int64_t j=0; j<res1; ++j) {
            Xfine[j][i] = XAccessor[j][i];
            Yfine[j][i] = YAccessor[j][i];
          }
        
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim()==1) {            
            auto Color = color.eval(meshgrid);
            auto CAccessor = Color(0).template accessor<real_t,2>();
            
#pragma omp parallel for simd collapse(2)
            for (int64_t i=0; i<res0; ++i)
              for (int64_t j=0; j<res1; ++j)
                Zfine[j][i] = CAccessor[j][i];
          }
        }

        // Plotting...
        if ((void*)this != (void*)&color && BSplineCore_t::geoDim()==1) {
          matplot::view(2);
          matplot::colormap(matplot::palette::hsv());
          matplot::mesh(Xfine, Yfine, Zfine)->palette_map_at_surface(true).face_alpha(0.7);
        } else {
          matplot::view(2);
          matplot::vector_2d X(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Y(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Z(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));

          auto xAccessor = BSplineCore::coeffs(0).template accessor<real_t,1>();
          auto yAccessor = BSplineCore::coeffs(1).template accessor<real_t,1>();

#pragma omp parallel for simd collapse(2)
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i)
            for (int64_t j=0; j<BSplineCore::ncoeffs(1); ++j) {
              X[j][i] = xAccessor[j*BSplineCore::ncoeffs(0) + i];
              Y[j][i] = yAccessor[j*BSplineCore::ncoeffs(0) + i];
            }

          matplot::colormap(matplot::palette::winter());
          matplot::mesh(Xfine, Yfine, Zfine);
          matplot::hold(matplot::on);
          matplot::surf(X, Y, Z)->palette_map_at_surface(true).face_alpha(0);
          matplot::hold(matplot::off);
        }
        matplot::title("BSpline: [0,1]^2 -> R^2");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim()==2 && BSplineCore::geoDim()==3) {

        ///
        // mapping: [0,1]^2 -> R^3
        ///

        matplot::vector_2d Xfine(res1, matplot::vector_1d(res0, 0.0));
        matplot::vector_2d Yfine(res1, matplot::vector_1d(res0, 0.0));
        matplot::vector_2d Zfine(res1, matplot::vector_1d(res0, 0.0));

        std::array<torch::Tensor,2> meshgrid = convert<2>(torch::meshgrid({torch::linspace(0, 1, res0),
                                                                           torch::linspace(0, 1, res1)}, "xy"));
        auto Coords = BSplineCore::eval(meshgrid);
        auto XAccessor = Coords(0).template accessor<real_t,2>();
        auto YAccessor = Coords(1).template accessor<real_t,2>();
        auto ZAccessor = Coords(2).template accessor<real_t,2>();
        
#pragma omp parallel for simd collapse(2)
        for (int64_t i=0; i<res0; ++i)
          for (int64_t j=0; j<res1; ++j) {
            Xfine[j][i] = XAccessor[j][i];
            Yfine[j][i] = YAccessor[j][i];
            Zfine[j][i] = ZAccessor[j][i];            
          }

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim()==1) {            
            matplot::vector_2d Cfine(res1, matplot::vector_1d(res0, 0.0));

            auto Color = color.eval(meshgrid);
            auto CAccessor = Color(0).template accessor<real_t,2>();
            
#pragma omp parallel for simd collapse(2)
            for (int64_t i=0; i<res0; ++i)
              for (int64_t j=0; j<res1; ++j) {
                Cfine[j][i] = CAccessor[j][i];                
              }
            matplot::colormap(matplot::palette::hsv());
            matplot::mesh(Xfine, Yfine, Zfine, Cfine);
          }
        }
        else {
          matplot::vector_2d X(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Y(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Z(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));

          auto xAccessor = BSplineCore::coeffs(0).template accessor<real_t,1>();
          auto yAccessor = BSplineCore::coeffs(1).template accessor<real_t,1>();
          auto zAccessor = BSplineCore::coeffs(2).template accessor<real_t,1>();

#pragma omp parallel for simd collapse(2)
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i)
            for (int64_t j=0; j<BSplineCore::ncoeffs(1); ++j) {
              X[j][i] = xAccessor[j*BSplineCore::ncoeffs(0) + i];
              Y[j][i] = yAccessor[j*BSplineCore::ncoeffs(0) + i];
              Z[j][i] = zAccessor[j*BSplineCore::ncoeffs(0) + i];
            }

          matplot::colormap(matplot::palette::winter());
          matplot::mesh(Xfine, Yfine, Zfine);

          matplot::hold(matplot::on);
          matplot::surf(X, Y, Z)->palette_map_at_surface(true).face_alpha(0);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1]^2 -> R^3");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else
        throw std::runtime_error("Unsupported combination of parametric/geometric dimensions");
    }

    /// Returns a string representation of the BSplineCommon object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << BSplineCore::name()
         << "(\n  parDim=" << BSplineCore::parDim_
         << ", geoDim=" << BSplineCore::geoDim_

         << ", degrees=";
      for (short_t i=0; i<BSplineCore::parDim_-1; ++i)
        os << BSplineCore::degree(i) << "x";
      if (BSplineCore::parDim_ > 0)
        os << BSplineCore::degree(BSplineCore::parDim_-1);
      else
        os << 0;

      os << ", knots=";
      for (short_t i=0; i<BSplineCore::parDim_-1; ++i)
        os << BSplineCore::nknots(i) << "x";
      if (BSplineCore::parDim_ > 0)
        os << BSplineCore::nknots(BSplineCore::parDim_-1);
      else
        os << 0;
          
      os << ", coeffs=";
      for (short_t i=0; i<BSplineCore::parDim_-1; ++i)
        os << BSplineCore::ncoeffs(i) << "x";
      if (BSplineCore::parDim_ > 0)
        os << BSplineCore::ncoeffs(BSplineCore::parDim_-1);
      else
        os << 1;

      if (is_verbose(os)) {
        os << "\nknots = ";
        if (BSplineCore::parDim_ > 0)
          os << BSplineCore::knots();
        else
          os << "{}";
        os << "\ncoeffs = "
           << BSplineCore::coeffs();
      }
      
      os << "\n)";
    }
  };

  /// Tensor-product uniform B-spline
  template<typename real_t, short_t geoDim, short_t... Degrees>
  using UniformBSpline = BSplineCommon<real_t, UniformBSplineCore<real_t, geoDim, Degrees...>>;

  /// Print (as string) a UniformBSpline object
  template<typename real_t, short_t geoDim, short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const UniformBSpline<real_t, geoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

  /// Tensor-product non-uniform B-spline
  template<typename real_t, short_t geoDim, short_t... Degrees>
  using NonUniformBSpline = BSplineCommon<real_t, NonUniformBSplineCore<real_t, geoDim, Degrees...>>;

  /// Print (as string) a UniformBSpline object
  template<typename real_t, short_t geoDim, short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const NonUniformBSpline<real_t, geoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }
} // namespace iganet
