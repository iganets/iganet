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

  // Material properties
  double E_; // Young's modulus

public:
  /// @brief Constructor
  template <typename... Args>
  linear_elasticity(double E, std::vector<int64_t> &&layers,
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        E_(E), ref_(iganet::utils::to_array(4_i64)) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

  /// @brief Initializes the epoch
  bool epoch(int64_t epoch) override {
    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville);

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

    // Compute strain
    auto lapl = Base::u_.ilapl(Base::G_, collPts_.first);
    
    int nr_of_eval_collPts = std::get<0>(collPts_.first).size(0);

    // Compute the derivative of the stress
    auto der_stress = E_ * *(lapl[0]);

    auto u_bdr = Base::u_.eval(iganet::utils::to_tensorArray({0.0, 1.0}));

    auto bdr = ref_.eval(iganet::utils::to_tensorArray({0.0, 1.0}));

    torch::Tensor zeros = torch::zeros({nr_of_eval_collPts}, torch::dtype(torch::kDouble));

    torch::Tensor loss = torch::mse_loss(der_stress, zeros) +
            1e1 * torch::mse_loss(*u_bdr[0], *bdr[0]);

    std::cout << loss << std::endl;

    return loss;

  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  // User inputs for material properties and external forces
  double E = 210;

  nlohmann::json json;
  json["res0"] = 50;
  // json["res1"] = 50;
  // json["res2"] = 50;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using real_t = double;

  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 1, 2>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 1, 2>>;

  linear_elasticity<optimizer_t, geometry_t, variable_t>
      net(E, // Material properties
          {120, 120},
          // Activation functions
          {{iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry
          std::tuple(iganet::utils::to_array(8_i64)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(8_i64)));

  // // Applying rhs to every point xi
  // net.f().transform([=](const std::array<real_t, 1> xi) {
  //   return std::array<real_t, 1>{0.0};
  // });

  // Impose boundary conditions (Dirichlet BCs)
  net.ref().boundary().template side<1>().transform(
      [](const std::array<real_t, 0> xi) {
        return std::array<real_t, 1>{0.0};
      }
      );
  
    // Impose boundary conditions (Dirichlet BCs)
  net.ref().boundary().template side<2>().transform(
      [](const std::array<real_t, 0> xi) {
        return std::array<real_t, 1>{1.0};
      });

  // Set maximum number of epochs
  net.options().max_epoch(20);

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
  // net.G().space().plot(net.u().space(), net.collPts().first, json)->show();
  // // Plot the difference between the exact and predicted solutions
  // net.G().plot(net.ref().abs_diff(net.u()), net.collPts().first, json)->show();
#endif

#ifdef IGANET_WITH_GISMO
   // transform network output into g+smo-compatible objects
  auto G_gismo = net.G().space().to_gismo(); // geometry of the domain
  auto u_gismo = net.u().space().to_gismo(); // displacement field

  // setting material properties
  real_t youngsModulus = 210.0;

  // creating a multi-patch object for the geometry
  gsMultiPatch<real_t> geometry;
  // adding the geometry as a patch
  geometry.addPatch(G_gismo);
  // creating a multi-basis object for the geometry
  gsMultiBasis<> basis(geometry);

  // creating boundary condition variable
  gsBoundaryConditions<real_t> bcInfo;

  // setting dirichlet boundary conditions
  bcInfo.addCondition(0, boundary::left, condition_type::dirichlet, gsConstantFunction<real_t> (0.0, 1));
  bcInfo.addCondition(0, boundary::right, condition_type::dirichlet, gsConstantFunction<real_t> (1.0, 1));

  // setting body force to zero for this calculation
  gsConstantFunction<real_t> body_force(0., 1);
  gsInfo << "Geometry dimension: " << geometry.parDim() << "\n";

  // initializing the elasticity assembler with the material behavior 
  gsElasticityAssembler<real_t> assembler(geometry, basis, bcInfo, body_force);
  assembler.options().setReal("YoungsModulus", youngsModulus);
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

#endif
  

  net.G().space().operator+=(net.u().space());
  // std::cout << net.G() << std::endl;

  iganet::finalize();
  return 0;
}
