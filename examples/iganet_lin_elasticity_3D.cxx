#include <iganet.h>
#include <iostream>

using namespace iganet::literals;

/// @brief Specialization of the IgANet class for linear elasticity in 3D
template <typename Optimizer, typename GeometryMap, typename Variable>
class linear_elasticity : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
                          public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  typename Base::variable_collPts_type collPts_;
  Variable ref_;

  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_;

  // material properties
  double E_; // Young's modulus
  double nu_; // Poisson's ratio

public:
  /// @brief Constructor
  template <typename... Args>
  linear_elasticity(double E, double nu, std::vector<int64_t> &&layers,
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        E_(E), nu_(nu), ref_(iganet::utils::to_array(4_i64, 4_i64, 4_i64)) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

  /// @brief Initializes the epoch
  bool epoch(int64_t epoch) override {
    // print epoch number
    std::cout << "Epoch: " << epoch << std::endl;

    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville_ref1);

      var_knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      var_coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_);

      G_knot_indices_ =
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      G_coeff_indices_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              G_knot_indices_);

      return true;
    } else
      return false;
  }

  /// @brief Computes the loss function
  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

    Base::u_.from_tensor(outputs);

    // calculation of the second derivative of the displacements
    auto hessian_coll = Base::u_.ihess(Base::G_, collPts_.first, var_knot_indices_, var_coeff_indices_, G_knot_indices_, G_coeff_indices_);
    auto hessian_ux = hessian_coll[0];
    auto hessian_uy = hessian_coll[1];
    auto hessian_uz = hessian_coll[2];

    // partial derivatives of the displacements - each variable has 216 entries of the collPts
    auto ux_xx = *(hessian_ux[0]);
    auto ux_xy = *(hessian_ux[1]);
    auto ux_xz = *(hessian_ux[2]);
    auto ux_yx = *(hessian_ux[3]);
    auto ux_yy = *(hessian_ux[4]);
    auto ux_yz = *(hessian_ux[5]);
    auto ux_zx = *(hessian_ux[6]);
    auto ux_zy = *(hessian_ux[7]);
    auto ux_zz = *(hessian_ux[8]);

    auto uy_xx = *(hessian_uy[0]);
    auto uy_xy = *(hessian_uy[1]);
    auto uy_xz = *(hessian_uy[2]);
    auto uy_yx = *(hessian_uy[3]);
    auto uy_yy = *(hessian_uy[4]);
    auto uy_yz = *(hessian_uy[5]);
    auto uy_zx = *(hessian_uy[6]);
    auto uy_zy = *(hessian_uy[7]);
    auto uy_zz = *(hessian_uy[8]);

    auto uz_xx = *(hessian_uz[0]);
    auto uz_xy = *(hessian_uz[1]);
    auto uz_xz = *(hessian_uz[2]);
    auto uz_yx = *(hessian_uz[3]);
    auto uz_yy = *(hessian_uz[4]);
    auto uz_yz = *(hessian_uz[5]);
    auto uz_zx = *(hessian_uz[6]);
    auto uz_zy = *(hessian_uz[7]);
    auto uz_zz = *(hessian_uz[8]);

    // calculation of lame parameters
    double lambda = (E_ * nu_) / ((1 + nu_) * (1 - 2 * nu_));
    double mu = E_ / (2 * (1 + nu_));

    // pre-allocation of the results
    torch::Tensor results_x = torch::zeros({216});
    torch::Tensor results_y = torch::zeros({216});
    torch::Tensor results_z = torch::zeros({216});
    torch::Tensor zeros = torch::stack({results_x, results_y, results_z}, /*dim=*/1);

    // calculation of the divergence of the stress tensor
    for (int i = 0; i < 216; ++i) {

        // x-dircection
        results_x[i] = lambda * (ux_xx[i] + uy_xy[i] + uz_xz[i]) +
                          2 * mu * ux_xx[i] + mu * ux_yy[i] + mu * ux_zz[i] +
                          mu * (uy_xy[i] + uz_xz[i]);

        // y-direction
        results_y[i] = mu * (uy_xx[i] + ux_yx[i]) +
                          lambda * (ux_yx[i] + uy_yy[i] + uz_yz[i]) +
                          2 * mu * uy_yy[i] + mu * uy_zz[i];

        // z-direction
        results_z[i] = mu * (uz_xx[i] + ux_zx[i]) +
                          lambda * (ux_xz[i] + uy_yz[i] + uz_zz[i]) +
                          mu * uz_yy[i] + 2 * mu * uz_zz[i];
    }
    
    // create a tensor of the divergence of the stress tensor
    torch::Tensor div_stress = torch::stack({results_x, results_y, results_z}, /*dim=*/1);
    
    // evaluation at the boundary
    auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
    auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);
    
    // print the result of the loss function
    std::cout << torch::mse_loss(div_stress, zeros) +
           10e1 * torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
           10e1 * torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) << std::endl;

    // div_stress is compared to zero since "divergence*sigma = 0" is the governing equation
    return torch::mse_loss(div_stress, zeros) +
           10e1 * torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
           10e1 * torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]);
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  // user inputs for material properties
  double E, nu;
  E = 210;
  nu = 0.4;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using real_t = double;

  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 3, 1, 1, 1>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 3, 2, 2, 2>>;

  linear_elasticity<optimizer_t, geometry_t, variable_t>
      net(E, nu, // Material properties
          {20, 20},
          // Activation functions
          {{iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry
          std::tuple(iganet::utils::to_array(2_i64, 2_i64, 2_i64)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(4_i64, 4_i64, 4_i64)));

  // // initial values for later comparison
  // auto anfangswert = net.u().as_tensor();
  // std::cout << anfangswert << std::endl;

  // // imposing rhs f is not necessary, since 0
  // net.f().transform([=](const std::array<real_t, 3> xi) {
  //   return std::array<real_t, 3>{0.0, 0.0, 0.0};
  // });

  // Impose boundary conditions (Dirichlet BCs)
  net.ref().boundary().template side<1>().transform(
      [](const std::array<real_t, 2> xi) {
        return std::array<real_t, 3>{0.0, 0.0, 0.0};
      });
  
 // Impose boundary conditions (Dirichlet BCs)
  net.ref().boundary().template side<2>().transform(
      [](const std::array<real_t, 2> xi) {
        return std::array<real_t, 3>{2.0, 0, 0};
      });

  // // MERLE BC SETTING
  // // SIDE 1 - x value = 0
  // net.ref().template boundary<0>().template side<1>().transform(
  //     [](const std::array<real_t, 2> xi) {
  //         return std::array<real_t, 1>{0.0 };
  //     });

  // // SIDE 6 - x value = 0
  // net.ref().template boundary<0>().template side<6>().transform(
  //     [](const std::array<real_t, 2> xi) {
  //         return std::array<real_t, 1>{10.0 };
  //     });

  // Set maximum number of epochs
  net.options().max_epoch(35);

  // Set tolerance for the loss functions
  net.options().min_loss(1e-8);

  // Start time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Train network
  net.train();

  // Stop time measurement
  auto t2 = std::chrono::high_resolution_clock::now();
  iganet::Log(iganet::log::info)
      << "Training took "
      << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
             .count()
      << " seconds\n";

#ifdef IGANET_WITH_MATPLOT
  // // Plot the solution
  // net.G().plot(net.u(), net.collPts().first, json)->show();

  // // Plot the difference between the exact and predicted solutions
  // net.G().plot(net.ref().abs_diff(net.u()), net.collPts().first, json)->show();
#endif

  // final values of the solution
  auto endwert = net.u();

  // auto inner_values = net.u().template eval<iganet::functionspace::interior>(net.collPts().first);
  auto outer_values = net.u().template eval<iganet::functionspace::boundary>(net.collPts().second);
  // std::cout << inner_values << std::endl;
  std::cout << outer_values << std::endl;

  iganet::finalize();
  return 0;
}
