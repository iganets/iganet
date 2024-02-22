/**
   @file examples/iganet_fitting_simple.cxx

   @brief Demonstration of IgANet function fitting

   This example demonstrates how to implement a simple IgANet to fit a
   given function on a square geometry. In contrast to the example
   iganet_fitting.cxx this examples does not make use of pre-computed
   indices and coefficients and might therefore be slower.
   
   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

/// @brief Specialization of the abstract IgANet class for function fitting
template <typename Optimizer, typename GeometryMap, typename Variable>
class fitting
  : public iganet::IgANet<Optimizer, GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<Optimizer, GeometryMap, Variable>::IgANet;

  /// @brief Initializes the epoch
  ///
  /// @param[in] epoch Epoch number
  ///
  /// @param[in] status Status flag
  iganet::status epoch(int64_t epoch) override {
    std::cout << "Epoch " << std::to_string(epoch) << ": ";

    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    return
      (epoch == 0
       ? iganet::status::inputs
       + iganet::status::variable_collPts
       : iganet::status::none);
  }

  /// @brief Computes the loss function
  ///
  /// @param[in] outputs Output of the network
  ///
  /// @param[in] geometryMap_collPts Sampling points for the geometry
  ///
  /// @param[in] variable_collPts Sampling points for the variable
  ///
  /// @param[in] epoch Epoch number
  ///
  /// @param[in] status Status flag
  torch::Tensor
  loss(const torch::Tensor &outputs,
       const typename Base::geometryMap_collPts_type &geometryMap_collPts,
       const typename Base::variable_collPts_type &variable_collPts,
       int64_t epoch, iganet::status status) override {

    // Cast the network output (a raw tensor) into the proper
    // function-space format, i.e. B-spline objects for the interior
    // and boundary parts that can be evaluated.
    Base::u_.from_tensor(outputs, false);
    
    // Evaluate the loss function
    return torch::mse_loss(*Base::u_.eval(variable_collPts.first)[0],
                           *Base::f_.eval(variable_collPts.first)[0]);
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = double;

  // Geometry: Bi-linear B-spline function space S2 (geoDim = 2, p = q = 1)
  using geometry_t = iganet::S2<iganet::UniformBSpline<real_t, 2, 1, 1>>;

  // Variable: Bi-quadratic B-spline function space S2 (geoDim = 1, p = q = 2)
  using variable_t = iganet::S2<iganet::UniformBSpline<real_t, 1, 2, 2>>;

  fitting<optimizer_t, geometry_t, variable_t>
    net( // Number of neurons per layers
         {50, 50, 50, 50, 50}
         ,
         // Activation functions
         {{iganet::activation::sigmoid},
          {iganet::activation::sigmoid},
          {iganet::activation::sigmoid},
          {iganet::activation::sigmoid},
          {iganet::activation::sigmoid},
          {iganet::activation::none}}
         ,
         // Number of B-spline coefficients of the geometry, just [0,1] x [0,1]
         std::tuple(iganet::utils::to_array(2_i64, 2_i64))
         ,
         // Number of B-spline coefficients of the variable
         std::tuple(iganet::utils::to_array(7_i64, 7_i64))
         );
  
  // Impose solution value for supervised training (not right-hand side)
  net.f().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{static_cast<real_t>(sin(M_PI * xi[0]) * sin(M_PI * xi[1]))};
  });

  net.options().max_epoch(1000);
  net.options().min_loss(1e-8);
  net.train();


  std::cout << net.variable_collPts(0).first << std::endl;
  
#ifdef IGANET_WITH_MATPLOT
  net.G().plot(net.u(), 50, 50);
#endif
  
  return 0;
}
