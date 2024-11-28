#include <iganet.h>
#include <iostream>

using namespace iganet::literals;

/// @brief Specialization of the IgANet class for linear elasticity in 2D
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
        gs_displacements_(std::move(gs_displacements)), ref_(iganet::utils::to_array(8_i64, 8_i64)) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

  /// @brief GISMO workflow
  static std::tuple<gsMatrix<double>, gsMatrix<double>> RunGismoSimulation(int64_t nrCtrlPts, int degree) {

    // initialize control points and displacements
    gsMatrix<double> gs_controlPoints(nrCtrlPts * nrCtrlPts, 2);
    gsMatrix<double> gs_displacements(nrCtrlPts * nrCtrlPts, 2);

    // create knot vectors
    gsKnotVector<double> knot_vector_u(0.0, 1.0, nrCtrlPts-degree-1, degree+1);
    gsKnotVector<double> knot_vector_v(0.0, 1.0, nrCtrlPts-degree-1, degree+1);

    // create control points
    std::vector<double> ctrlValues = {0.0};
    for(int i = 1; i < nrCtrlPts; ++i) {
        ctrlValues.push_back(i * 1.0 / (nrCtrlPts - 1));
    }
    gsMatrix<double> control_points(nrCtrlPts * nrCtrlPts, 2);

    // systematic placement of control points
    int index = 0;
    for (int j = 0; j < nrCtrlPts; ++j) {
        for (int i = 0; i < nrCtrlPts; ++i) {
            control_points(index, 0) = ctrlValues[i];
            control_points(index, 1) = ctrlValues[j];
            ++index;
        }
    }

    // create geometry
    gsTensorBSpline<2, double> geometry(knot_vector_u, knot_vector_v, control_points);

    // create multipatch and add the geometry
    gsMultiPatch<double> multiPatch;
    multiPatch.addPatch(geometry);
    gsMultiBasis<> basis(multiPatch);

    // boundary conditions
    gsBoundaryConditions<double> bcInfo;
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 2), 0);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 2), 1);
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, 
            gsConstantFunction<double>(2.0, 2), 0);
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 2), 1);

    // body force (currently set to zero)
    gsConstantFunction<double> body_force(0., 0., 2);

    // initialize the elasticity assembler
    gsElasticityAssembler<double> assembler(geometry, basis, bcInfo, body_force);
    assembler.options().setReal("YoungsModulus", 210.0);
    assembler.options().setReal("PoissonsRatio", 0.4);
    assembler.assemble();

    // solve the system
    gsSparseSolver<>::CGDiagonal solver;
    gsMatrix<double> solution;
    solver.compute(assembler.matrix());
    solution = solver.solve(assembler.rhs());

    // create a multipatch object for the solution
    gsMultiPatch<double> solution_patch;
    assembler.constructSolution(solution, assembler.allFixedDofs(), solution_patch);

    // create a mesh object for the control net
    gsMesh<double> controlNetMesh;
    geometry.controlNet(controlNetMesh);

    // create collection matrices for all the control points and displacements
    gs_controlPoints.resize(controlNetMesh.numVertices(), 2);
    gs_displacements.resize(controlNetMesh.numVertices(), 2);
    gsMatrix<double> point(2, 1);

    for (int i = 0; i < controlNetMesh.numVertices(); ++i) {
        gs_controlPoints(i, 0) = controlNetMesh.vertex(i)(0);
        gs_controlPoints(i, 1) = controlNetMesh.vertex(i)(1);

        point(0, 0) = gs_controlPoints(i, 0);
        point(1, 0) = gs_controlPoints(i, 1);

        auto displacement = solution_patch.patch(0).eval(point);
        gs_displacements(i, 0) = displacement(0);
        gs_displacements(i, 1) = displacement(1);
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
    // number of DOFs
    int dofs = outputs.size(0);

    // calculation of the second derivative of the gs_displacements
    auto hessian_coll = Base::u_.ihess(Base::G_, collPts_.first, var_knot_indices_, 
            var_coeff_indices_, G_knot_indices_, G_coeff_indices_);

    // partial derivatives of the gs_displacements - each variable has 36 entries of the collPts
    auto& ux_xx = *(hessian_coll[0][0]);
    auto& ux_xy = *(hessian_coll[0][1]);
    auto& ux_yx = *(hessian_coll[0][2]);
    auto& ux_yy = *(hessian_coll[0][3]);

    auto& uy_xx = *(hessian_coll[1][0]);
    auto& uy_xy = *(hessian_coll[1][1]);
    auto& uy_yx = *(hessian_coll[1][2]);
    auto& uy_yy = *(hessian_coll[1][3]);

    // pre-allocation of the results
    torch::Tensor results_x = torch::zeros({hessian_coll[0][0]->size(0)});
    torch::Tensor results_y = torch::zeros({hessian_coll[0][0]->size(0)});
    torch::Tensor zeros = torch::stack({results_x, results_y}, /*dim=*/1);

    // calculation of the divergence of the stress tensor
    for (int i = 0; i < hessian_coll[0][0]->size(0); ++i) {

        // x-direction
        results_x[i] = (lambda_ + 2 * mu_) * ux_xx[i] + mu_ * ux_yy[i] + (lambda_ + mu_) * uy_xy[i];

        // y-direction
        results_y[i] = mu_ * uy_xx[i] + (lambda_ + 2 * mu_) * uy_yy[i] + (lambda_ + mu_) * ux_xy[i];
    }
    
    // create a tensor of the divergence of the stress tensor
    torch::Tensor div_stress = torch::stack({results_x, results_y}, /*dim=*/1);
    
    // evaluation at the boundary
    auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
    auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

    // UNSUPERVISED LEARNING (default)
    if (supervised_learning_ == false) {
        // calculation of the loss function for double-sided constraint solid
        // div_stress is compared to 0 since "divergence*sigma = 0" is the governing equation
        loss = torch::mse_loss(div_stress, zeros) +
            10e1 * torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
            10e1 * torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]) +
            10e1 * torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) +
            10e1 * torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]);
    }
    // SUPERVISED LEARNING
    else if (supervised_learning_ == true) {
        torch::Tensor modified_outputs = outputs * 1.0;
        // Create net_displacements from slices of modified_outputs
        torch::Tensor net_displacements = torch::stack({
            modified_outputs.slice(0, 0, dofs/2),
            modified_outputs.slice(0, dofs/2, dofs),
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
        // supervised learning loss
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

  // USER INPUTS
  double E = 210;
  double nu = 0.4;
  int max_epoch = 100;
  double min_loss = 1e-8;
  bool supervised_learning = false;
  int64_t nrCtrlPts = 8; // in each direction
  int degree = 2;

  // calculation of lame parameters
  double lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu));
  double mu = E / (2 * (1 + nu));

  using real_t = double;
  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, 2, 2>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, 2, 2>>;
  using linear_elasticity_t = linear_elasticity<optimizer_t, geometry_t, variable_t>;

  gsMatrix<double> gs_controlPoints;
  gsMatrix<double> gs_displacements;
  std::tie(gs_controlPoints, gs_displacements) = linear_elasticity_t::RunGismoSimulation(nrCtrlPts, degree);
  
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
          std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(nrCtrlPts, nrCtrlPts)));

  // // imposing rhs f is not necessary, since 0
  // net.f().transform([=](const std::array<real_t, 3> xi) {
  //   return std::array<real_t, 3>{0.0, 0.0, 0.0};
  // });

  // BC SIDE WEST
  net.ref().boundary().template side<1>().transform<1>(
      [](const std::array<real_t, 1> &xi) {
          return std::array<real_t, 1>{0.0}; 
      },
      std::array<iganet::short_t, 1>{0} 
  );

  net.ref().boundary().template side<1>().transform<1>(
      [](const std::array<real_t, 1> &xi) {
          return std::array<real_t, 1>{0.0}; 
      },
      std::array<iganet::short_t, 1>{1} 
  );

  // BC SIDE EAST
  net.ref().boundary().template side<2>().transform<1>(
    [](const std::array<real_t, 1> &xi) {
        return std::array<real_t, 1>{1.0};
    },
    std::array<iganet::short_t, 1>{0}
  );

  net.ref().boundary().template side<2>().transform<1>(
    [](const std::array<real_t, 1> &xi) {
        return std::array<real_t, 1>{0.0};
    },
    std::array<iganet::short_t, 1>{1}
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
  gsMatrix<real_t> net_controlPoints(nrCtrlPts * nrCtrlPts, 2);
  // creating collection matrix for all the displacements (iganet)
  gsMatrix<real_t> net_displacements(nrCtrlPts * nrCtrlPts, 2);

  // filling the collection matrices with the values from the tensors
  for (int i = 0; i < nrCtrlPts * nrCtrlPts; ++i) {
      double x = geo_as_tensor[i].item<double>();          
      double y = geo_as_tensor[i + nrCtrlPts * nrCtrlPts].item<double>();
      net_controlPoints(i, 0) = x;
      net_controlPoints(i, 1) = y;

      double ux = displ_as_tensor[i].item<double>();
      double uy = displ_as_tensor[i + nrCtrlPts * nrCtrlPts].item<double>();
      net_displacements(i, 0) = ux;
      net_displacements(i, 1) = uy;
  }

  // GISMO SOLUTION - printing the new position of the control points
  std::cout << "New CPs from Gismo:\n"
            << gs_controlPoints + gs_displacements << std::endl;
  // NET SOLUTION - printing the new position of the control points 
  std::cout << "New CPs from IgANet:\n"
            << net_controlPoints + net_displacements << std::endl;

  iganet::finalize();
  return 0;
}
