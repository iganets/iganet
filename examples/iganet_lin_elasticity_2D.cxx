#include <iganet.h>
#include <iostream>
#include <fstream>

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

  // simulation parameter
  double MAX_EPOCH_;

  // gismo solution
  gsMatrix<double> gsDisplacements_;

  // supervised learning (true) or unsupervised learning (false)
  bool SUPERVISED_LEARNING_ = false;

  // json path
  static constexpr const char* JSON_PATH = "/home/obergue/Documents/pytest/splinepy/results.json";

public:
  /// @brief Constructor
  template <typename... Args>
  linear_elasticity(double lambda, double mu, bool SUPERVISED_LEARNING, double MAX_EPOCH,
                    gsMatrix<double> gsDisplacements, std::vector<int64_t> &&layers,
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        lambda_(lambda), mu_(mu), SUPERVISED_LEARNING_(SUPERVISED_LEARNING), MAX_EPOCH_(MAX_EPOCH),
        gsDisplacements_(std::move(gsDisplacements)), ref_(iganet::utils::to_array(8_i64, 8_i64)) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }
  
  static void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
    
    // create json object
    nlohmann::json jsonData;

    // try to read the JSON data from the file
    try {
        std::ifstream json_file_in(JSON_PATH);
        if (json_file_in.is_open()) {
            json_file_in >> jsonData;
            json_file_in.close();
        } else {
            std::cerr << "Warning: Could not open file for reading: " << JSON_PATH << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading JSON file: " << JSON_PATH << ". Exception: " << e.what() << "\n";
    }


    // add new data to the JSON object
    try {
        jsonData[key] = data;
    } catch (const std::exception& e) {
        std::cerr << "Error adding key to JSON object: " << e.what() << "\n";
        return;
    }

    // write the JSON data to the file
    try {
        std::ofstream json_file_out(JSON_PATH);
        if (json_file_out.is_open()) {
            json_file_out << jsonData.dump(1);
            json_file_out.close();
        } else {
            std::cerr << "Error: Could not open file for writing: " << JSON_PATH << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing JSON file: " << JSON_PATH << ". Exception: " << e.what() << "\n";
    }
  }

  /// @brief GISMO workflow
  static std::tuple<gsMatrix<double>, gsMatrix<double>, gsMatrix<double>> RunGismoSimulation(int64_t NR_CTRL_PTS, int DEGREE) {

    // initialize control points and displacements
    gsMatrix<double> gsCtrlPts(NR_CTRL_PTS * NR_CTRL_PTS, 2);
    gsMatrix<double> gsDisplacements(NR_CTRL_PTS * NR_CTRL_PTS, 2);

    // create knot vectors
    gsKnotVector<double> knotVector_u(0.0, 1.0, NR_CTRL_PTS-DEGREE-1, DEGREE+1);
    gsKnotVector<double> knotVector_v(0.0, 1.0, NR_CTRL_PTS-DEGREE-1, DEGREE+1);

     // create control points in the according distribution 
    std::vector<double> ctrlValues{0.0};
    if (NR_CTRL_PTS < DEGREE) {
        throw std::invalid_argument("Number of control points must be at least " 
                                    + std::to_string(DEGREE+1) + ".");    }
    double gap = 1.0 / (NR_CTRL_PTS - DEGREE);
    ctrlValues.push_back(gap/DEGREE);
    double settingNumber = NR_CTRL_PTS - 4;
    double settingStart =  1.0 / (settingNumber + 1);
    double setter = settingStart;
    // loop to create the inner control points
    for (int i = 1; i <= settingNumber; ++i) {
        // arrange settingNumber control points around the center
        ctrlValues.push_back(setter);
        setter += settingStart;
    }
    ctrlValues.push_back(1.0-gap/DEGREE);
    ctrlValues.push_back(1.0);

    if ((settingNumber >= 4) && (ctrlValues[1]*DEGREE != ctrlValues[2])) {
        for (int j = 2; j <= settingNumber+2; ++j) {
            ctrlValues[j] = ctrlValues[j-1] + gap;
        }
    }

    gsMatrix<double> controlPoints(NR_CTRL_PTS * NR_CTRL_PTS, 2); 
    // systematic placement of control points
    int index = 0;
    for (int j = 0; j < NR_CTRL_PTS; ++j) {
        for (int i = 0; i < NR_CTRL_PTS; ++i) {
            controlPoints(index, 0) = ctrlValues[i];
            controlPoints(index, 1) = ctrlValues[j];
            ++index;
        }
    }

    // create geometry
    gsTensorBSpline<2, double> geometry(knotVector_u, knotVector_v, controlPoints);

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
            gsConstantFunction<double>(1.0, 2), 0);
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 2), 1);

    // body force (currently set to zero)
    gsConstantFunction<double> bodyForce(0., 0., 2);

    // initialize the elasticity assembler
    gsElasticityAssembler<double> assembler(geometry, basis, bcInfo, bodyForce);
    assembler.options().setReal("YoungsModulus", 210.0);
    assembler.options().setReal("PoissonsRatio", 0.4);
    assembler.assemble();

    // solve the system
    gsSparseSolver<>::CGDiagonal solver;
    gsMatrix<double> solution;
    solver.compute(assembler.matrix());
    solution = solver.solve(assembler.rhs());

    // create a multipatch object for the solution
    gsMultiPatch<double> solutionPatch;
    assembler.constructSolution(solution, assembler.allFixedDofs(), solutionPatch);

    // create a mesh object for the control net
    gsMesh<double> controlNetMesh;
    geometry.controlNet(controlNetMesh);

    // create collection matrices for all the control points and displacements
    gsCtrlPts.resize(controlNetMesh.numVertices(), 2);
    gsDisplacements.resize(controlNetMesh.numVertices(), 2);
    gsMatrix<double> point(2, 1);

    for (int i = 0; i < controlNetMesh.numVertices(); ++i) {
        gsCtrlPts(i, 0) = controlNetMesh.vertex(i)(0);
        gsCtrlPts(i, 1) = controlNetMesh.vertex(i)(1);

        point(0, 0) = gsCtrlPts(i, 0);
        point(1, 0) = gsCtrlPts(i, 1);

        auto displacement = solutionPatch.patch(0).eval(point);
        gsDisplacements(i, 0) = displacement(0);
        gsDisplacements(i, 1) = displacement(1);
    }

    // create a piecewise function for the stresses
    gsPiecewiseFunction<double> stressFunction;

    // calculet von Mises stresses (cauchy form)
    assembler.constructCauchyStresses(solutionPatch, stressFunction, stress_components::von_mises);

    // allocate a matrix for the von Mises stresses at every control point
    gsMatrix<double> gsStresses(gsCtrlPts.rows(), 1);

    // loop all control points
    for (int i = 0; i < gsCtrlPts.rows(); ++i) {
        gsMatrix<double> point(2, 1);
        point(0, 0) = gsCtrlPts(i, 0);
        point(1, 0) = gsCtrlPts(i, 1);

        const auto &segment = stressFunction.piece(0); // patch index 0 (bc only 1 patch)

        // eval the von Mises stress at the control point
        gsMatrix<double> stressValue(1, 1);
        segment.eval_into(point, stressValue);

        // collect all von Mises stresses
        gsStresses(i, 0) = stressValue(0, 0);
       
    }
    // return the control points, displacements and stresses
    return std::make_tuple(gsCtrlPts, gsDisplacements, gsStresses);
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
    torch::Tensor totalLoss; 
    torch::Tensor elastLoss;
    torch::Tensor tracLoss;
    torch::Tensor bcLoss;
    torch::Tensor gsLoss;

    // number of DOFs (every CP has 2 DOFs)
    int dofs = outputs.size(0);

    // TRACTION FREE BOUNDARY CONDITIONS

    // // reduce collPts by every first and last point of the upper and lower edge (points at the corners)
    // at::Tensor reduced_collPts_second = std::get<0>(collPts_.second)[0].slice(0, 1, std::get<0>(collPts_.second)[0].size(0) - 1);
    // std::array<at::Tensor, 2ul> secCollPtsUpper = {reduced_collPts_second, torch::ones(reduced_collPts_second.size(0))};    
    // std::array<at::Tensor, 2ul> secCollPtsLower = {reduced_collPts_second, torch::zeros(reduced_collPts_second.size(0))};
    std::array<at::Tensor, 2ul> secCollPtsUpper = {std::get<0>(collPts_.second)[0], torch::ones(std::get<0>(collPts_.second)[0].size(0))};
    std::array<at::Tensor, 2ul> secCollPtsLower = {std::get<0>(collPts_.second)[0], torch::zeros(std::get<0>(collPts_.second)[0].size(0))};
    std::array<at::Tensor, 2ul> secollPts = {torch::cat({secCollPtsUpper[0], secCollPtsLower[0]}, 0), torch::cat({secCollPtsUpper[1], secCollPtsLower[1]}, 0)};

    auto jacobianBoundary = Base::u_.ijac(Base::G_, secollPts);

    auto ux_x = *jacobianBoundary[0];
    auto ux_y = *jacobianBoundary[1];
    auto uy_x = *jacobianBoundary[2];
    auto uy_y = *jacobianBoundary[3];

    torch::Tensor tractionFreeX = torch::zeros({secollPts[0].size(0)});
    torch::Tensor tractionFreeY = torch::zeros({secollPts[0].size(0)});
    torch::Tensor boundaryZeros = torch::stack({tractionFreeX, tractionFreeY}, /*dim=*/1);
    
    for(int i=0; i<secollPts[0].size(0); ++i) {
        // traction-free condition for linear elasticity
        tractionFreeX[i] = mu_ * (uy_x[i] + ux_y[i]);
        tractionFreeY[i] = lambda_ * ux_x[i] + (lambda_ + 2 * mu_) * uy_y[i];
        
        // traction-free condition for laplace equation
        // tractionFreeX[i] = ux_y[i];
        // tractionFreeY[i] = uy_y[i];
    }
    torch::Tensor tractionFree = torch::stack({tractionFreeX, tractionFreeY}, /*dim=*/1);
 
    // LINEAR ELASTICITY EQUATION
  
    // calculation of the second derivatives of the displacements (u)
    auto hessianColl = Base::u_.ihess(Base::G_, collPts_.first, var_knot_indices_, 
            var_coeff_indices_, G_knot_indices_, G_coeff_indices_);

    // partial derivatives of the displacements (u)
    auto& ux_xx = *(hessianColl[0][0]);
    auto& ux_xy = *(hessianColl[0][1]);
    auto& ux_yx = *(hessianColl[0][2]);
    auto& ux_yy = *(hessianColl[0][3]);

    auto& uy_xx = *(hessianColl[1][0]);
    auto& uy_xy = *(hessianColl[1][1]);
    auto& uy_yx = *(hessianColl[1][2]);
    auto& uy_yy = *(hessianColl[1][3]);

    // pre-allocation of the results

    torch::Tensor divStressX = torch::zeros({hessianColl[0][0]->size(0)});
    torch::Tensor divStressY = torch::zeros({hessianColl[0][0]->size(0)});
    torch::Tensor divZeros = torch::stack({divStressX, divStressY}, /*dim=*/1);

    // calculation of the divergence of the stress tensor
    for (int i = 0; i < hessianColl[0][0]->size(0); ++i) {

        // x-direction
        divStressX[i] = (lambda_ + 2 * mu_) * ux_xx[i] + mu_ * ux_yy[i] + (lambda_ + mu_) * uy_xy[i];

        // y-direction
        divStressY[i] = mu_ * uy_xx[i] + (lambda_ + 2 * mu_) * uy_yy[i] + (lambda_ + mu_) * ux_xy[i];
        
        // Laplace equation for testing
        // results_x[i] = ux_xx[i] + ux_yy[i];
        // results_y[i] = uy_xx[i] + uy_yy[i];
    }
    
    // create a tensor of the divergence of the stress tensor
    torch::Tensor divStress = torch::stack({divStressX, divStressY}, /*dim=*/1);
    
    // evaluation at the boundary
    auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
    auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

    // UNSUPERVISED LEARNING (default)
    if (SUPERVISED_LEARNING_ == false) {
        int bcWeight = 10e8;
        // calculation of the loss function for double-sided constraint solid
        // divStress is compared to 0 since "divergence*sigma = 0" is the governing equation
        elastLoss = torch::mse_loss(divStress, divZeros);
        tracLoss = torch::mse_loss(tractionFree, boundaryZeros);
        bcLoss = bcWeight *   ( torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
                                torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]) +
                                torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) +
                                torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));

        // calculate the total loss
        totalLoss = elastLoss + bcLoss + tracLoss;

        // print the loss values
        std::cout   << std::setw(11) << totalLoss.item<double>()        << " = "
           << "EL " << std::setw(11) << elastLoss.item<double>()        << " + "
           << "TL " << std::setw(11) << tracLoss.item<double>()         << " + "
           << "BL " << std::setw(11) << bcLoss.item<double>()/bcWeight  << " * 1e" 
           << static_cast<int>(std::log10(bcWeight)) << std::endl;
    }

    // SUPERVISED LEARNING
    else if (SUPERVISED_LEARNING_ == true) {
        torch::Tensor modifiedOutputs = outputs * 1.0;
        // Create netDisplacements_ from slices of modifiedOutputs
        torch::Tensor netDisplacements_ = torch::stack({
            modifiedOutputs.slice(0, 0, dofs/2),
            modifiedOutputs.slice(0, dofs/2, dofs),
        }, 1);
        // create new tensor with requires_grad=true for training
        auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);
        // dimensions of the matrix
        int gsRows = gsDisplacements_.rows();
        int gsCols = gsDisplacements_.cols();
        // transforming matrix into row vector
        std::vector<double> data_gs(gsRows * gsCols);
        // writing data column-wise in matrix
        for (int col = 0; col < gsCols; ++col) {
            for (int row = 0; row < gsRows; ++row) {
                data_gs[row * gsCols + col] = gsDisplacements_(row, col);
            }
        }
        // creating tensor from the transformed data
        torch::Tensor torchGsDisplacements = torch::from_blob(data_gs.data(), 
                {gsRows, gsCols}, options).clone();
        // supervised learning loss
        gsLoss      = torch::mse_loss(netDisplacements_, torchGsDisplacements);
        tracLoss    = torch::mse_loss(tractionFree, boundaryZeros);
        elastLoss   = torch::mse_loss(divStress, divZeros);
        bcLoss      = torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
                      torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]) +
                      torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) +
                      torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]);
       
        // calculate the total loss
        totalLoss = gsLoss + tracLoss + elastLoss + bcLoss;

        // print the loss values
        std::cout   << std::setw(11) << totalLoss.item<double>() << " = "
           << "EL " << std::setw(11) << elastLoss.item<double>() << " + "
           << "TL " << std::setw(11) << tracLoss.item<double>()  << " + "
           << "BL " << std::setw(11) << bcLoss.item<double>()    << " + "
           << "GL " << std::setw(11) << gsLoss.item<double>()    << std::endl;
    }

    else {
        throw std::runtime_error("Invalid value for SUPERVISED_LEARNING_");
    }

    // POSTPROCESSING PREPARATION - WRITING DATA TO JSON FILE

    // only calculate this at the end of the simulation
    if (epoch == MAX_EPOCH_ - 1) {
        
        // STRESS CALCULATION

        // calculate the jacobian of the displacements (u) at the collocation points
        auto jacobian = Base::u_.ijac(Base::G_, collPts_.first, var_knot_indices_, 
            var_coeff_indices_, G_knot_indices_, G_coeff_indices_);
        
        auto ux_x = *jacobian[0];
        auto ux_y = *jacobian[1];
        auto uy_x = *jacobian[2];
        auto uy_y = *jacobian[3];

        // allocate the stress tensor
        torch::Tensor sigma_xx = torch::zeros({hessianColl[0][0]->size(0)});
        torch::Tensor sigma_xy = torch::zeros({hessianColl[0][0]->size(0)});
        torch::Tensor sigma_yy = torch::zeros({hessianColl[0][0]->size(0)}); 
        torch::Tensor sigma_vm = torch::zeros({hessianColl[0][0]->size(0)});   

        // create json object for the stresses
        nlohmann::json netStresses_j = nlohmann::json::array();

        // calculate the stress tensor
        for (int i = 0; i < hessianColl[0][0]->size(0); ++i) {
            sigma_xx[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * ux_x[i];
            sigma_xy[i] = mu_ * (uy_x[i] + ux_y[i]);
            sigma_yy[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * uy_y[i];
            
            // calculate von mises stress
            sigma_vm[i] = sqrt(sigma_xx[i] * sigma_xx[i] + sigma_yy[i] * 
                            sigma_yy[i] - sigma_xx[i] * sigma_yy[i] + 3 * sigma_xy[i] * sigma_xy[i]);

            // add the von mises stress to the json object
            netStresses_j.push_back({sigma_vm[i].item<double>()});
        }
        // write the von mises stresses to the json file
        appendToJsonFile("netStresses", netStresses_j);

        // CALCULATE THE NEW POSITION OF THE COLLPTS

        // create a tensor of the collocation points
        at::Tensor collPtsFirstAsTensor = torch::stack({std::get<0>(collPts_.first), std::get<1>(collPts_.first)}, 1);
        auto displacementOfCollPts = Base::u_.eval(collPts_.first);
        at::Tensor displacementAsTensor = torch::stack({*(displacementOfCollPts[0]), *(displacementOfCollPts[1]) }, 1);

        // create json objects for the collocation points' reference and deformed position
        nlohmann::json collPtsFirst_j = nlohmann::json::array();
        nlohmann::json collPtsFirstAfterDisplacement_j = nlohmann::json::array();
        for (int i = 0; i < collPtsFirstAsTensor.size(0); ++i) {
            collPtsFirst_j.push_back({collPtsFirstAsTensor[i][0].item<double>(), collPtsFirstAsTensor[i][1].item<double>()});
            collPtsFirstAfterDisplacement_j.push_back({collPtsFirstAsTensor[i][0].item<double>() + displacementAsTensor[i][0].item<double>(), 
                                                                collPtsFirstAsTensor[i][1].item<double>() + displacementAsTensor[i][1].item<double>()});
        }
        // write the collocation points' original position to the json file
        appendToJsonFile("collPtsFirstAsTensor", collPtsFirst_j);
        // write the collocation points' new position to the json file
        appendToJsonFile("collPtsFirstAfterDisplacementAsTensor", collPtsFirstAfterDisplacement_j);

        // WRITING DIVERGENCE OF THE STRESS TENSOR TO JSON FILE

        nlohmann::json netDivergenceX_j = nlohmann::json::array();
        nlohmann::json netDivergenceY_j = nlohmann::json::array();

        for (int i = 0; i < divStressX.size(0); ++i) {
            netDivergenceX_j.push_back({divStressX[i].item<double>()});
            netDivergenceY_j.push_back({divStressY[i].item<double>()});
        }

        // write the divergence of the stress tensor to the json file
        appendToJsonFile("netDivergenceX", netDivergenceX_j);
        appendToJsonFile("netDivergenceY", netDivergenceY_j);
    }
    return totalLoss;
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  // USER INPUTS
  double YOUNG_MODULUS = 210;
  double POISSON_RATIO = 0.4;
  int MAX_EPOCH = 120;
  double MIN_LOSS = 1e-8;
  bool SUPERVISED_LEARNING = false;
  int64_t NR_CTRL_PTS = 8; // in each direction
  constexpr int DEGREE = 2;

  // calculation of lame parameters
  double lambda = (YOUNG_MODULUS * POISSON_RATIO) / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  double mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO));

  using real_t = double;
  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
  using linear_elasticity_t = linear_elasticity<optimizer_t, geometry_t, variable_t>;

  gsMatrix<double> gsCtrlPts;
  gsMatrix<double> gsDisplacements;
  gsMatrix<double> gsStresses;
  std::tie(gsCtrlPts, gsDisplacements, gsStresses) = linear_elasticity_t::RunGismoSimulation(NR_CTRL_PTS, DEGREE);

  linear_elasticity_t
      net(// simulation parameters
          lambda, mu, SUPERVISED_LEARNING, MAX_EPOCH, gsDisplacements,
          // Number of neurons per layer
          {25, 25},
          // Activation functions
          {{iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry
          std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)));

  // // imposing rhs f is not necessary, since 0
  // net.f().transform([=](const std::array<real_t, 3> xi) {
  //   return std::array<real_t, 3>{0.0, 0.0, 0.0};
  // });

  // BC SIDE WEST
  net.ref().boundary().template side<1>().transform<1>(
      [](const std::array<real_t, 1> &xi) {
          return std::array<real_t, 1>{0.0}; 
      },
      std::array<iganet::short_t, 1>{0} // 0 for x-direction
  );

  net.ref().boundary().template side<1>().transform<1>(
      [](const std::array<real_t, 1> &xi) {
          return std::array<real_t, 1>{0.0}; 
      },
      std::array<iganet::short_t, 1>{1} // 1 for y-direction
  );

  // BC SIDE EAST
  net.ref().boundary().template side<2>().transform<1>(
    [](const std::array<real_t, 1> &xi) {
        return std::array<real_t, 1>{1.0};
    },
    std::array<iganet::short_t, 1>{0} // 0 for x-direction
  );

  net.ref().boundary().template side<2>().transform<1>(
    [](const std::array<real_t, 1> &xi) {
        return std::array<real_t, 1>{0.0};
    },
    std::array<iganet::short_t, 1>{1} // 1 for y-direction
  );

  // Set maximum number of epochs
  net.options().max_epoch(MAX_EPOCH);

  // Set tolerance for the loss functions
  net.options().min_loss(MIN_LOSS);

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

  at::Tensor geometryAsTensor = net.G().as_tensor();
  at::Tensor displacementAsTensor = net.u().as_tensor();
  
  // creating collection matrix for all the control points (iganet)
  gsMatrix<real_t> netCtrlPts(NR_CTRL_PTS * NR_CTRL_PTS, 2);
  // creating collection matrix for all the displacements (iganet)
  gsMatrix<real_t> netDisplacements(NR_CTRL_PTS * NR_CTRL_PTS, 2);

  // filling the collection matrices with the values from the tensors
  for (int i = 0; i < NR_CTRL_PTS * NR_CTRL_PTS; ++i) {
      double x = geometryAsTensor[i].item<double>();          
      double y = geometryAsTensor[i + NR_CTRL_PTS * NR_CTRL_PTS].item<double>();
      netCtrlPts(i, 0) = x;
      netCtrlPts(i, 1) = y;

      double ux = displacementAsTensor[i].item<double>();
      double uy = displacementAsTensor[i + NR_CTRL_PTS * NR_CTRL_PTS].item<double>();
      netDisplacements(i, 0) = ux;
      netDisplacements(i, 1) = uy;
  }

//   // GISMO SOLUTION - printing the new position of the control points
//   std::cout << "New CPs from Gismo:\n"
//             << gsCtrlPts + gsDisplacements << std::endl;
//   // NET SOLUTION - printing the new position of the control points 
//   std::cout << "\n\nNew CPs from IgANet:\n"
//             << netCtrlPts + netDisplacements << std::endl;

  // deformed position of the control points
  gsMatrix<double> displacedGsCtrlPts = gsCtrlPts + gsDisplacements;
  gsMatrix<double> displacedNetCtrlPts = netCtrlPts + netDisplacements;
  
  // json objects for the deformed positions of the control points
  nlohmann::json displacedGsCtrlPts_j = nlohmann::json::array();
  nlohmann::json displacedNetCtrlPts_j = nlohmann::json::array();
  nlohmann::json gsStresses_j = nlohmann::json::array();

  // write data from the matrices to the json objects
  for (int i = 0; i < displacedGsCtrlPts.rows(); ++i) {
      // new control points Gismo
      displacedGsCtrlPts_j.push_back({displacedGsCtrlPts(i, 0), displacedGsCtrlPts(i, 1)});
  
      // new control points IgANet
      displacedNetCtrlPts_j.push_back({displacedNetCtrlPts(i, 0), displacedNetCtrlPts(i, 1)});

      // write the von Mises stresses to the json object (calculated at the beginning of the main function)
      gsStresses_j.push_back({gsStresses(i, 0)});
  }

  // write data to the json file
  linear_elasticity_t::appendToJsonFile("gsCtrlPts", displacedGsCtrlPts_j);
  linear_elasticity_t::appendToJsonFile("netCtrlPts", displacedNetCtrlPts_j);
  linear_elasticity_t::appendToJsonFile("gsStresses", gsStresses_j);
  
  iganet::finalize();
  return 0;
}
