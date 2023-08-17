/**
   @file examples/demo.cxx

   @brief Demonstrator application

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

/// @brief IgANet for Poisson's equation
template<typename optimizer_t,
         typename geometry_t,
         typename variable_t>
class poisson : public iganet::IgANet<optimizer_t, geometry_t, variable_t>,
                public iganet::IgANetCustomizable<optimizer_t, geometry_t, variable_t>
{
public:
  bool pde;

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<optimizer_t, geometry_t, variable_t>;

  /// @brief Type of the customizable class
  using Customizable = iganet::IgANetCustomizable<optimizer_t, geometry_t, variable_t>;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<optimizer_t, geometry_t, variable_t>::IgANet;

  /// @brief Initializes the epoch
  iganet::status epoch(int64_t epoch) override
  {
    std::cout << "Epoch " << std::to_string(epoch) << ": ";

    return (epoch == 0
            ? iganet::status::inputs + iganet::status::geometry_samples + iganet::status::variable_samples
            : iganet::status::inputs);
  }

  /// @brief Computes the loss function
  torch::Tensor loss(const torch::Tensor& outputs,
                     const typename Base::geometry_samples_type& geometry_samples,
                     const typename Base::variable_samples_type& variable_samples,
                     int64_t epoch, iganet::status status) override
  {
    // Update indices and precompute basis functions for geometry
    if (status & iganet::status::geometry_samples) {
      Customizable::geometry_interior_knot_indices_  =
        Base::geometry_.template find_knot_indices<iganet::functionspace::interior>(geometry_samples.first);      
      Customizable::geometry_interior_coeff_indices_ =
        Base::geometry_.template find_coeff_indices<iganet::functionspace::interior>(Customizable::geometry_interior_knot_indices_);
      
      Customizable::geometry_boundary_knot_indices_  =
        Base::geometry_.template find_knot_indices<iganet::functionspace::boundary>(geometry_samples.second);
      Customizable::geometry_boundary_coeff_indices_ =
        Base::geometry_.template find_coeff_indices<iganet::functionspace::boundary>(Customizable::geometry_boundary_knot_indices_);
    }

    // Update indices and precompute basis functions for variable
    if (status & iganet::status::variable_samples) {
      Customizable::variable_interior_knot_indices_  =
        Base::variable_.template find_knot_indices<iganet::functionspace::interior>(variable_samples.first);
      Customizable::variable_interior_coeff_indices_ =
        Base::variable_.template find_coeff_indices<iganet::functionspace::interior>(Customizable::variable_interior_knot_indices_);

      Customizable::variable_boundary_knot_indices_  =
        Base::variable_.template find_knot_indices<iganet::functionspace::boundary>(variable_samples.second);
      Customizable::variable_boundary_coeff_indices_ =
        Base::variable_.template find_coeff_indices<iganet::functionspace::boundary>(Customizable::variable_boundary_knot_indices_);
    }

    // Evaluate right-hand side in the interior
    Base::outputs_.from_tensor(outputs, false);
    auto rhs = Base::variable_.eval(variable_samples.first);

    // Evaluate solution at the boundary
    auto bdr_pred = Base::outputs_.template eval<iganet::functionspace::boundary>(variable_samples.second);
    auto bdr_cond = Base::variable_.template eval<iganet::functionspace::boundary>(variable_samples.second);

    // Evaluate boundary losses
    auto loss_bdr0 = torch::mse_loss(*std::get<0>(bdr_pred)[0], *std::get<0>(bdr_cond)[0]);
    auto loss_bdr1 = torch::mse_loss(*std::get<1>(bdr_pred)[0], *std::get<1>(bdr_cond)[0]);
    auto loss_bdr2 = torch::mse_loss(*std::get<2>(bdr_pred)[0], *std::get<2>(bdr_cond)[0]);
    auto loss_bdr3 = torch::mse_loss(*std::get<3>(bdr_pred)[0], *std::get<3>(bdr_cond)[0]);

    // std::cout << loss_bdr0.template item<double>() << ", "
    //           << loss_bdr1.template item<double>() << ", "
    //           << loss_bdr2.template item<double>() << ", "
    //           << loss_bdr3.template item<double>() << std::endl;

    // Evaluate loss function
    if (!pde)
      return torch::mse_loss(*Base::outputs_.eval(variable_samples.first)[0], *rhs[0])
        + 0*(loss_bdr0 + loss_bdr1 + loss_bdr2 + loss_bdr3);

    // Evaluate pde loss
    auto sol_ilaplace = Base::outputs_.ihess(Base::geometry_, variable_samples.first);
    //auto loss_pde     = torch::mse_loss(*sol_ilaplace[0] + *sol_ilaplace[3], *rhs[0]);

    //return loss_pde + 0*(loss_bdr0 + loss_bdr1 + loss_bdr2 + loss_bdr3);
  }
};

int main()
{
  iganet::init();
  iganet::verbose(std::cout);
  
  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = double;

  using bspline_t   = iganet::UniformBSpline<real_t, 1, 2, 2>;
  using geometry_t  = iganet::S2<iganet::UniformBSpline<real_t, 2, 2, 2>>;
  using variable_t  = iganet::S2<iganet::UniformBSpline<real_t, 1, 2, 2>>;


  bspline_t sol( iganet::utils::to_array(7_i64, 7_i64) );
  
  sol.transform( [](const std::array<real_t,2> xi)
  {
    return std::array<real_t,1>{(xi[0]+1)*cos(M_PI*xi[1])};
  } );
  
  std::cout << sol << std::endl;

  sol.to<float>();

  std::cout << sol << std::endl;
  
  exit(0);
  
  poisson<optimizer_t, geometry_t, variable_t> net(// Number of neurons per layers
                                                   {50,50,50,50,50},
                                                   // Activation functions
                                                   {
                                                     {iganet::activation::sigmoid},
                                                     {iganet::activation::sigmoid},
                                                     {iganet::activation::sigmoid},
                                                     {iganet::activation::sigmoid},
                                                     {iganet::activation::sigmoid},
                                                     {iganet::activation::none}
                                                   },
                                                   // Number of B-spline coefficients
                                                   std::tuple(iganet::utils::to_array(7_i64, 7_i64)));

  // deform geometry
  net.geometry().transform( [](const std::array<real_t,2> xi)
  {
    return std::array<real_t,2>{(xi[0]+1)*cos(M_PI*xi[1]),
                                (xi[0]+1)*sin(M_PI*xi[1])};
  } );
  
  // impose solution value for supervised training (not right-hand side)
  net.variable().transform( [](const std::array<real_t,2> xi)
  {
    return std::array<real_t,1>{ sin(M_PI*xi[0]) * sin(M_PI*xi[1]) };
  } );

  // boundary values
  net.variable().boundary().template side<1>().transform( [](const std::array<real_t,1> xi)
  {
    return std::array<real_t,1>{ 0.0 };
  } );

  net.variable().boundary().template side<2>().transform( [](const std::array<real_t,1> xi)
  {
    return std::array<real_t,1>{ 0.0 };
  } );

  net.variable().boundary().template side<3>().transform( [](const std::array<real_t,1> xi)
  {
    return std::array<real_t,1>{ 0.0 };
  } );

  net.variable().boundary().template side<4>().transform( [](const std::array<real_t,1> xi)
  {
    return std::array<real_t,1>{ 0.0 };
  } );

  net.options().max_epoch(1000);
  net.options().min_loss(1e-8);

  net.train();

  net.geometry().plot(net.outputs(), 50, 50);

  net.variable().transform( [](const std::array<real_t,2> xi)
  {
    return std::array<real_t,1>{ -2.0 * M_PI * M_PI * sin(M_PI*xi[0]) * sin(M_PI*xi[1]) };
  } );
  
  net.pde = true;
  net.options().max_epoch(1000);
  net.options().min_loss(1e-10);
  net.train();

  net.geometry().plot(net.outputs(), 50, 50);

  return 0;
}
