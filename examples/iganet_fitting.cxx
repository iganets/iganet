/**
   @file examples/iganet_fitting.cxx

   @brief Demonstration of IgANet function fitting

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

/// @brief IgANet for function fitting
template <typename Optimizer, typename GeometryMap, typename Variable>
class fitting
  : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
    public iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  /// @brief Type of the customizable class
  using Customizable =
    iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable>;

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

    return (epoch == 0
            ? iganet::status::inputs + iganet::status::geometryMap_collPts +
            iganet::status::variable_collPts
            : iganet::status::inputs);
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

    std::cout << "LOSS\n";
    
    std::cout << geometryMap_collPts.first << std::endl;

    std::cout << variable_collPts.first << std::endl;
    
    // Update indices and precompute basis functions for geometry
    if (status & iganet::status::geometryMap_collPts) {
      Customizable::geometryMap_interior_knot_indices_ =
        Base::G_
        .template find_knot_indices<iganet::functionspace::interior>(
                                                                     geometryMap_collPts.first);
      Customizable::geometryMap_interior_coeff_indices_ =
        Base::G_
        .template find_coeff_indices<iganet::functionspace::interior>(
                                                                      Customizable::geometryMap_interior_knot_indices_);

      Customizable::geometryMap_boundary_knot_indices_ =
        Base::G_
        .template find_knot_indices<iganet::functionspace::boundary>(
                                                                     geometryMap_collPts.second);
      Customizable::geometryMap_boundary_coeff_indices_ =
        Base::G_
        .template find_coeff_indices<iganet::functionspace::boundary>(
                                                                      Customizable::geometryMap_boundary_knot_indices_);
    }

    // Update indices and precompute basis functions for variable
    if (status & iganet::status::variable_collPts) {
      Customizable::variable_interior_knot_indices_ =
        Base::f_
        .template find_knot_indices<iganet::functionspace::interior>(
                                                                     variable_collPts.first);
      Customizable::variable_interior_coeff_indices_ =
        Base::f_
        .template find_coeff_indices<iganet::functionspace::interior>(
                                                                      Customizable::variable_interior_knot_indices_);

      Customizable::variable_boundary_knot_indices_ =
        Base::f_
        .template find_knot_indices<iganet::functionspace::boundary>(
                                                                     variable_collPts.second);
      Customizable::variable_boundary_coeff_indices_ =
        Base::f_
        .template find_coeff_indices<iganet::functionspace::boundary>(
                                                                      Customizable::variable_boundary_knot_indices_);
    }

    // Evaluate loss function
    return torch::mse_loss(*Base::u_.eval(variable_collPts.first)[0],
                           *Base::f_.eval(variable_collPts.first)[0]);
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = float;

  using geometry_t = iganet::S2<iganet::UniformBSpline<real_t, 2, 2, 2>>;
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
         // Number of B-spline coefficients of the geometry
         std::tuple(iganet::utils::to_array(7_i64, 7_i64))
         ,
         // Number of B-spline coefficients of the variable
         std::tuple(iganet::utils::to_array(7_i64, 7_i64))
         );
  
  // Deform geometry
  net.G().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 2>{(xi[0] + 1) * cos(static_cast<real_t>(M_PI) * xi[1]),
                                 (xi[0] + 1) * sin(static_cast<real_t>(M_PI) * xi[1])};
  });

  // Impose solution value for supervised training (not right-hand side)
  net.f().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{static_cast<real_t>(sin(M_PI * xi[0]) * sin(M_PI * xi[1]))};
  });

  // Impose boundary values
  net.f().boundary().template side<1>().transform(
                                                         [](const std::array<real_t, 1> xi) {
                                                           return std::array<real_t, 1>{0.0};
                                                         });

  net.f().boundary().template side<2>().transform(
                                                         [](const std::array<real_t, 1> xi) {
                                                           return std::array<real_t, 1>{0.0};
                                                         });

  net.f().boundary().template side<3>().transform(
                                                         [](const std::array<real_t, 1> xi) {
                                                           return std::array<real_t, 1>{0.0};
                                                         });

  net.f().boundary().template side<4>().transform(
                                                         [](const std::array<real_t, 1> xi) {
                                                           return std::array<real_t, 1>{0.0};
                                                         });

  net.options().max_epoch(1000);
  net.options().min_loss(1e-8);
  net.train();

  net.G().plot(net.u(), 50, 50);
  
  return 0;
}
