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

  // material properties - lame's parameters
  double lambda_;
  double mu_;

public:
  /// @brief Constructor
  template <typename... Args>
  linear_elasticity(double lambda, double mu, std::vector<int64_t> &&layers,
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        lambda_(lambda), mu_(mu), ref_(iganet::utils::to_array(4_i64, 4_i64, 4_i64)) {}

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

    // partial derivatives of the displacements - each variable has 216 entries of the collPts
    auto& ux_xx = *(hessian_coll[0][0]);
    auto& ux_xy = *(hessian_coll[0][1]);
    auto& ux_xz = *(hessian_coll[0][2]);
    auto& ux_yx = *(hessian_coll[0][3]);
    auto& ux_yy = *(hessian_coll[0][4]);
    auto& ux_yz = *(hessian_coll[0][5]);
    auto& ux_zx = *(hessian_coll[0][6]);
    auto& ux_zy = *(hessian_coll[0][7]);
    auto& ux_zz = *(hessian_coll[0][8]);

    auto& uy_xx = *(hessian_coll[1][0]);
    auto& uy_xy = *(hessian_coll[1][1]);
    auto& uy_xz = *(hessian_coll[1][2]);
    auto& uy_yx = *(hessian_coll[1][3]);
    auto& uy_yy = *(hessian_coll[1][4]);
    auto& uy_yz = *(hessian_coll[1][5]);
    auto& uy_zx = *(hessian_coll[1][6]);
    auto& uy_zy = *(hessian_coll[1][7]);
    auto& uy_zz = *(hessian_coll[1][8]);

    auto& uz_xx = *(hessian_coll[2][0]);
    auto& uz_xy = *(hessian_coll[2][1]);
    auto& uz_xz = *(hessian_coll[2][2]);
    auto& uz_yx = *(hessian_coll[2][3]);
    auto& uz_yy = *(hessian_coll[2][4]);
    auto& uz_yz = *(hessian_coll[2][5]);
    auto& uz_zx = *(hessian_coll[2][6]);
    auto& uz_zy = *(hessian_coll[2][7]);
    auto& uz_zz = *(hessian_coll[2][8]);

    // pre-allocation of the results
    torch::Tensor results_x = torch::zeros({216});
    torch::Tensor results_y = torch::zeros({216});
    torch::Tensor results_z = torch::zeros({216});
    torch::Tensor zeros = torch::stack({results_x, results_y, results_z}, /*dim=*/1);

    // calculation of the divergence of the stress tensor
    for (int i = 0; i < 216; ++i) {

        // x-direction
        results_x[i] = lambda_ * (ux_xx[i] + uy_xy[i] + uz_xz[i]) +
                          2 * mu_ * ux_xx[i] + mu_ * ux_yy[i] + mu_ * ux_zz[i] +
                          mu_ * (uy_xy[i] + uz_xz[i]);

        // y-direction
        results_y[i] = mu_ * (uy_xx[i] + ux_yx[i]) +
                          lambda_ * (ux_yx[i] + uy_yy[i] + uz_yz[i]) +
                          2 * mu_ * uy_yy[i] + mu_ * uy_zz[i];

        // z-direction
        results_z[i] = mu_ * (uz_xx[i] + ux_zx[i]) +
                          lambda_ * (ux_xz[i] + uy_yz[i] + uz_zz[i]) +
                          mu_ * uz_yy[i] + 2 * mu_ * uz_zz[i];
    }
    
    // create a tensor of the divergence of the stress tensor
    torch::Tensor div_stress = torch::stack({results_x, results_y, results_z}, /*dim=*/1);
    
    // evaluation at the boundary
    auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
    auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

    // calculation of the loss function for double-sided constraint solid
    // div_stress is compared to zero since "divergence*sigma = 0" is the governing equation
    torch::Tensor loss = torch::mse_loss(div_stress, zeros) +
           10e1 * torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
           10e1 * torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]) +
           10e1 * torch::mse_loss(*std::get<0>(u_bdr)[2], *std::get<0>(bdr)[2]) +
           10e1 * torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) +
           10e1 * torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]) +
           10e1 * torch::mse_loss(*std::get<1>(u_bdr)[2], *std::get<1>(bdr)[2]);

    // print loss
    std::cout << loss << std::endl;

    return loss;
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;
  json["res2"] = 50;

  // user inputs for material properties
  double E, nu;
  E = 210;
  nu = 0.4;
  // calculation of lame parameters
  double lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
  double mu = E / (2 * (1 + nu));

  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using real_t = double;

  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 3, 2, 2, 2>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 3, 2, 2, 2>>;

  linear_elasticity<optimizer_t, geometry_t, variable_t>
      net(// Material properties
          lambda, mu, 
          // Number of neurons per layer
          {100, 100},
          // Activation functions
          {{iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry
          std::tuple(iganet::utils::to_array(4_i64, 4_i64, 4_i64)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(4_i64, 4_i64, 4_i64)));

  // // imposing rhs f is not necessary, since 0
  // net.f().transform([=](const std::array<real_t, 3> xi) {
  //   return std::array<real_t, 3>{0.0, 0.0, 0.0};
  // });

  // BC SIDE WEST
  net.ref().boundary().template side<1>().transform<1>(
      [](const std::array<real_t, 2> &xi) {
          return std::array<real_t, 1>{0.0}; 
      },
      std::array<iganet::short_t, 1>{0} 
  );

  net.ref().boundary().template side<1>().transform<1>(
      [](const std::array<real_t, 2> &xi) {
          return std::array<real_t, 1>{0.0}; 
      },
      std::array<iganet::short_t, 1>{1} 
  );

  net.ref().boundary().template side<1>().transform<1>(
      [](const std::array<real_t, 2> &xi) {
          return std::array<real_t, 1>{0.0}; 
      },
      std::array<iganet::short_t, 1>{2} 
  );

  // BC SIDE EAST
  net.ref().boundary().template side<2>().transform<1>(
    [](const std::array<real_t, 2> &xi) {
        return std::array<real_t, 1>{2.0};
    },
    std::array<iganet::short_t, 1>{0}
  );

  net.ref().boundary().template side<2>().transform<1>(
    [](const std::array<real_t, 2> &xi) {
        return std::array<real_t, 1>{0.0};
    },
    std::array<iganet::short_t, 1>{1}
  );

  net.ref().boundary().template side<2>().transform<1>(
    [](const std::array<real_t, 2> &xi) {
        return std::array<real_t, 1>{0.0};
    },
    std::array<iganet::short_t, 1>{2}
  );

  // Set maximum number of epochs
  net.options().max_epoch(60);

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
  // Plot the solution
  // net.G().space().plot(net.u().space(), net.collPts().first, json)->show();
  // net.G().space().plot(net.collPts().first, json)->show();
  // // Plot the difference between the exact and predicted solutions
  // net.G().plot(net.ref().abs_diff(net.u()), net.collPts().first, json)->show();
#endif

#ifdef IGANET_WITH_GISMO
  // transform network output into g+smo-compatible objects
  auto G_gismo = net.G().space().to_gismo(); // geometry of the domain
  auto u_gismo = net.u().space().to_gismo(); // displacement field

  // setting material properties
  real_t youngsModulus = 210.0;
  real_t poissonsRatio = 0.4;

  // creating a multi-patch object for the geometry
  gsMultiPatch<real_t> geometry;
  // adding the geometry as a patch
  geometry.addPatch(G_gismo);
  // creating a multi-basis object for the geometry
  gsMultiBasis<> basis(geometry);

  // creating boundary condition variable
  gsBoundaryConditions<real_t> bcInfo;

  // setting Dirichlet boundary conditions (west: no displacement, east: displacement of 2.0 in x-direction)
  for (int d = 0; d < 3; d++) {
      bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, gsConstantFunction<real_t>(0.0, 3), d);
  }
  bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, gsConstantFunction<real_t>(2.0, 3), 0);
  bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, gsConstantFunction<real_t>(0.0, 3), 1);
  bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, gsConstantFunction<real_t>(0.0, 3), 2);

  // setting body force to zero for this calculation
  gsConstantFunction<real_t> body_force(0.,0.,0., 3);
  gsInfo << "Geometry dimension: " << geometry.parDim() << "\n";

  // initializing the elasticity assembler with the material behavior 
  gsElasticityAssembler<real_t> assembler(geometry, basis, bcInfo, body_force);
  assembler.options().setReal("YoungsModulus", youngsModulus);
  assembler.options().setReal("PoissonsRatio", poissonsRatio);
  assembler.options().setInt("DirichletValues", dirichlet::l2Projection);

  // assembling the system
  assembler.assemble();

  // solving the system
  gsSparseSolver<>::CGDiagonal solver;
  gsMatrix<real_t> solution;
  solver.compute(assembler.matrix());
  solution = solver.solve(assembler.rhs());

  // creating a multi-patch object for the solution
  gsMultiPatch<real_t> solution_patch;
  // constructing the solution
  assembler.constructSolution(solution, assembler.allFixedDofs(), solution_patch);
  
  // creating a mesh object for the control net
  gsMesh<real_t> controlNetMesh;
  // loading the control net of our geometry into the mesh object
  geometry.patch(0).controlNet(controlNetMesh);

  // create collection matrix for all the control points (gismo)
  gsMatrix<real_t> controlPoints(64, 3);
  // create collection matrix for all the displacements (gismo)
  gsMatrix<real_t> displacements(64, 3);

  // printing the position and displacement of the control points
  for (int i = 0; i < controlNetMesh.numVertices(); ++i) {
      controlPoints(i, 0) = controlNetMesh.vertex(i)(0);
      controlPoints(i, 1) = controlNetMesh.vertex(i)(1);
      controlPoints(i, 2) = controlNetMesh.vertex(i)(2);

      gsMatrix<real_t> point(3, 1);
      point(0, 0) = controlPoints(i, 0);
      point(1, 0) = controlPoints(i, 1);
      point(2, 0) = controlPoints(i, 2);

      auto displacement = solution_patch.patch(0).eval(point);
      displacements(i, 0) = displacement(0);
      displacements(i, 1) = displacement(1);
      displacements(i, 2) = displacement(2);

      // // printing gismo control points and displacements
      // gsInfo << "Control Point " << std::setw(2) << i 
      //       << " Position: (" << std::setw(5) << controlPoints(i, 0)
      //       << ", " << std::setw(5) << controlPoints(i, 1)
      //       << ", " << std::setw(5) << controlPoints(i, 2) << ")"
      //       << "  Displacement: (" << std::setw(12) << displacements(i, 0)
      //       << ", " << std::setw(12) << displacements(i, 1)
      //       << ", " << std::setw(12) << displacements(i, 2) << ")\n";
  }
#endif

  // // final values of the solution
  // auto final_displacement = net.u().space();
  // std::cout << endwert << std::endl;

  // auto inner_values = net.u().template eval<iganet::functionspace::interior>(net.collPts().first);
  // auto outer_values = net.u().template eval<iganet::functionspace::boundary>(net.collPts().second);
  // std::cout << inner_values << std::endl;
  // std::cout << outer_values << std::endl;

  // applying displacement to the geometry
  // net.G().space().operator+=(net.u().space());
  // std::cout << net.G().space() << std::endl;

  // printing final displacement
  // std::cout << net.u().space() << std::endl;
  
  // final values of the geometry
  // std::cout << net.G().as_tensor() << std::endl;


  at::Tensor geo_as_tensor = net.G().as_tensor();
  at::Tensor displ_as_tensor = net.u().as_tensor();

  // creating collection matrix for all the control points (iganet)
  gsMatrix<real_t> net_controlPoints(64, 3);
  // creating collection matrix for all the displacements (iganet)
  gsMatrix<real_t> net_displacements(64, 3);

  // filling the collection matrices with the values from the tensors
  for (int i = 0; i < 64; ++i) {
      double x = geo_as_tensor[i].item<double>();          
      double y = geo_as_tensor[i + 64].item<double>();
      double z = geo_as_tensor[i + 128].item<double>();
      net_controlPoints(i, 0) = x;
      net_controlPoints(i, 1) = y;
      net_controlPoints(i, 2) = z;

      double ux = displ_as_tensor[i].item<double>();
      double uy = displ_as_tensor[i + 64].item<double>();
      double uz = displ_as_tensor[i + 128].item<double>();
      net_displacements(i, 0) = ux;
      net_displacements(i, 1) = uy;
      net_displacements(i, 2) = uz;
  }

  // // printing net control points and displacements
  // for (size_t i = 0; i < net_controlPoints.rows(); ++i) {
  //     std::cout << "Control Point " << std::setw(2) << i 
  //               << " Position: (" << std::setw(5) << net_controlPoints(i, 0)
  //               << ", " << std::setw(5) << net_controlPoints(i, 1)
  //               << ", " << std::setw(5) << net_controlPoints(i, 2) << ")"
  //               << "  Displacement: (" << std::setw(12) << net_displacements(i, 0)
  //               << ", " << std::setw(12) << net_displacements(i, 1)
  //               << ", " << std::setw(12) << net_displacements(i, 2) << ")" << std::endl;
  // }

  // GISMO SOLUTION - printing the new position of the control points
  std::cout << "Neue CPs von Gismo:\n"
            << controlPoints + displacements << std::endl;
  // NET SOLUTION - printing the new position of the control points 
  std::cout << "Neue CPs von IgANet:\n"
            << net_controlPoints + net_displacements << std::endl;

  iganet::finalize();
  return 0;
}
