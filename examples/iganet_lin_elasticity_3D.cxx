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

  // gismo solution
  gsMatrix<double> gs_displacements_;
  
  // supervised learning (true) or unsupervised learning (false)
  bool supervised_learning_ = false;

public:
  /// @brief Constructor
  template <typename... Args>
  linear_elasticity(double lambda, double mu, bool supervised_learning, 
                    gsMatrix<double> gs_displacements, std::vector<int64_t> &&layers,
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        lambda_(lambda), mu_(mu), supervised_learning_(supervised_learning), 
        gs_displacements_(std::move(gs_displacements)), ref_(iganet::utils::to_array(4_i64, 4_i64, 4_i64)) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

  /// @brief GISMO workflow
  static std::tuple<gsMatrix<double>, gsMatrix<double>> RunGismoSimulation() {
    int degree = 2;       
    int numCtrlPts = 4;

    // initialize control points and displacements
    gsMatrix<double> gs_controlPoints(numCtrlPts * numCtrlPts * numCtrlPts, 3);
    gsMatrix<double> gs_displacements(numCtrlPts * numCtrlPts * numCtrlPts, 3);

    // create knot vectors
    gsKnotVector<double> knot_vector_u(0.0, 1.0, 1, degree+1);
    gsKnotVector<double> knot_vector_v(0.0, 1.0, 1, degree+1);
    gsKnotVector<double> knot_vector_w(0.0, 1.0, 1, degree+1);

    // create control points
    std::vector<double> ctrlValues = {0.0, 0.25, 0.75, 1.0};
    gsMatrix<double> control_points(numCtrlPts * numCtrlPts * numCtrlPts, 3);

    // systematic placement of control points
    int index = 0;
    for (int k = 0; k < numCtrlPts; ++k) {         // Z-Koordinate zuerst
        for (int j = 0; j < numCtrlPts; ++j) {     // Y-Koordinate als nächstes
            for (int i = 0; i < numCtrlPts; ++i) { // X-Koordinate zuletzt
                control_points(index, 0) = ctrlValues[i];  // x-Koordinate
                control_points(index, 1) = ctrlValues[j];  // y-Koordinate
                control_points(index, 2) = ctrlValues[k];  // z-Koordinate
                ++index;
            }
        }
    }

    // create geometry
    gsTensorBSpline<3, double> geometry(knot_vector_u, knot_vector_v, knot_vector_w, control_points);

    // create multipatch and add the geometry
    gsMultiPatch<double> multiPatch;
    multiPatch.addPatch(geometry);
    gsMultiBasis<> basis(multiPatch);

    // boundary conditions
    gsBoundaryConditions<double> bcInfo;
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 3), 0);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 3), 1);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 3), 2);    
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, 
            gsConstantFunction<double>(2.0, 3), 0);
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 3), 1);
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 3), 2);

    // body force (currently set to zero)
    gsConstantFunction<double> body_force(0.,0.,0., 3);

    // initialize the elasticity assembler
    gsElasticityAssembler<double> assembler(geometry, basis, bcInfo, body_force);
    assembler.options().setReal("YoungsModulus", 210.0);
    assembler.options().setReal("PoissonsRatio", 0.4);
    assembler.options().setInt("DirichletValues", dirichlet::l2Projection);
    assembler.assemble();

    // solve the system
    gsSparseSolver<>::CGDiagonal solver;
    gsMatrix<double> solution;
    solver.compute(assembler.matrix());
    solution = solver.solve(assembler.rhs());

    // create a multipatch object for the solution
    gsMultiPatch<double> solution_patch;
    assembler.constructSolution(solution, assembler.allFixedDofs(), solution_patch);
    
    // creating a mesh object for the control net
    gsMesh<double> controlNetMesh;
    geometry.controlNet(controlNetMesh);

    // create collection matrices for all the control points and displacements
    gs_controlPoints.resize(controlNetMesh.numVertices(), 3);
    gs_displacements.resize(controlNetMesh.numVertices(), 3);

    for (int i = 0; i < controlNetMesh.numVertices(); ++i) {
        gs_controlPoints(i, 0) = controlNetMesh.vertex(i)(0);
        gs_controlPoints(i, 1) = controlNetMesh.vertex(i)(1);
        gs_controlPoints(i, 2) = controlNetMesh.vertex(i)(2);

        gsMatrix<double> point(3, 1);
        point(0, 0) = gs_controlPoints(i, 0);
        point(1, 0) = gs_controlPoints(i, 1);
        point(2, 0) = gs_controlPoints(i, 2);

        auto displacement = solution_patch.patch(0).eval(point);
        gs_displacements(i, 0) = displacement(0);
        gs_displacements(i, 1) = displacement(1);
        gs_displacements(i, 2) = displacement(2);
    }   

    // return the control points and displacements
    return std::make_tuple(gs_controlPoints, gs_displacements);
  }

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
    torch::Tensor loss; 

    // calculation of the second derivative of the gs_displacements
    auto hessian_coll = Base::u_.ihess(Base::G_, collPts_.first, var_knot_indices_, 
          var_coeff_indices_, G_knot_indices_, G_coeff_indices_);

    // partial derivatives of the gs_displacements - each variable has 216 entries of the collPts
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

    // UNSUPERVISED LEARNING (default)
    if (supervised_learning_ == false) {
        // calculation of the loss function for double-sided constraint solid
        // div_stress is compared to zero since "divergence*sigma = 0" is the governing equation
        loss = torch::mse_loss(div_stress, zeros) +
              10e1 * torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
              10e1 * torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]) +
              10e1 * torch::mse_loss(*std::get<0>(u_bdr)[2], *std::get<0>(bdr)[2]) +
              10e1 * torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) +
              10e1 * torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]) +
              10e1 * torch::mse_loss(*std::get<1>(u_bdr)[2], *std::get<1>(bdr)[2]);
    }

    // SUPERVISED LEARNING
    else if (supervised_learning_ == true) {
        torch::Tensor modified_outputs = outputs * 1.0;     
        // Create net_displacements from slices of modified_outputs
        torch::Tensor net_displacements = torch::stack({
            modified_outputs.slice(0, 0, 64),
            modified_outputs.slice(0, 64, 128),
            modified_outputs.slice(0, 128, 192)
        }, 1);

        // create new tensor with requires_grad=true for training
        auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);
        // dimensions of the matrix
        int rows_gs = gs_displacements_.rows();
        int cols_gs = gs_displacements_.cols();
        // transforming matrix into row vector
        std::vector<double> data_gs(rows_gs * cols_gs);
        // writing data column-wise in matrix
        for (int col = 0; col < cols_gs; ++col) {
            for (int row = 0; row < rows_gs; ++row) {
                data_gs[row * cols_gs + col] = gs_displacements_(row, col);
            }
        }

        // creating tensor from the transformed data
        torch::Tensor torch_gs_displacements = torch::from_blob(data_gs.data(), 
                {rows_gs, cols_gs}, options).clone();

        // superivsed learning loss
        loss = torch::mse_loss(net_displacements, torch_gs_displacements) 
                + torch::mse_loss(div_stress, zeros);
    }

    else {
        throw std::runtime_error("Invalid value for supervised_learning_");
    }

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

  // USER INPUTS
  double E = 210;
  double nu = 0.4;
  int max_epoch = 20;
  double min_loss = 1e-8;
  bool supervised_learning = false;

  // calculation of lame parameters
  double lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
  double mu = E / (2 * (1 + nu));

  using real_t = double;
  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 3, 2, 2, 2>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 3, 2, 2, 2>>;
  using linear_elasticity_t = linear_elasticity<optimizer_t, geometry_t, variable_t>;

  gsMatrix<double> gs_controlPoints;
  gsMatrix<double> gs_displacements;
  std::tie(gs_controlPoints, gs_displacements) = linear_elasticity_t::RunGismoSimulation();

  linear_elasticity_t
      net(// simulation parameters
          lambda, mu, supervised_learning, gs_displacements,
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
  net.options().max_epoch(max_epoch);

  // Set tolerance for the loss functions
  net.options().min_loss(min_loss);

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

  // PROCESSING NETWORK OUTPUT FOR SPLINEPY

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

  // GISMO SOLUTION - printing the new position of the control points
  std::cout << "Neue CPs von Gismo:\n"
            << gs_controlPoints + gs_displacements << std::endl;
  // NET SOLUTION - printing the new position of the control points 
  std::cout << "Neue CPs von IgANet:\n"
            << net_controlPoints + net_displacements << std::endl;

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

  iganet::finalize();
  return 0;
}
