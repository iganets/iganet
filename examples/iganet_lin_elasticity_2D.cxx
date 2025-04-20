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
  typename Base::variable_collPts_type interiorCollPts_;

  int nrCollPts_;
  Variable ref_;

  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_interior_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_interior_;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_boundary_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_boundary_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_interior_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_interior_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_boundary_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_boundary_;

  // material properties - lame's parameters
  double lambda_;
  double mu_;

  // simulation parameters
  int MAX_EPOCH_;
  double MIN_LOSS_;
  int64_t NR_CTRL_PTS_;
  std::vector<int> TFBC_SIDES_;
  std::string JSON_PATH_;
  std::vector<std::tuple<int, double, double>> FORCE_SIDES_;
  std::vector<std::tuple<int, double, double>> DIRI_SIDES_;
  bool SUPERVISED_LEARNING_;

public:
  /// @brief Constructor
  template <typename... Args>
  linear_elasticity(double lambda, double mu, bool SUPERVISED_LEARNING, int MAX_EPOCH, 
                    double MIN_LOSS, std::vector<int> TFBC_SIDES,
                    std::vector<std::tuple<int, double, double>> FORCE_SIDES,
                    std::vector<std::tuple<int, double, double>> DIRI_SIDES, 
                    int64_t NR_CTRL_PTS, std::string JSON_PATH, std::vector<int64_t> &&layers, 
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        lambda_(lambda), mu_(mu), SUPERVISED_LEARNING_(SUPERVISED_LEARNING), MAX_EPOCH_(MAX_EPOCH), 
        MIN_LOSS_(MIN_LOSS), TFBC_SIDES_(TFBC_SIDES), FORCE_SIDES_(FORCE_SIDES), 
        DIRI_SIDES_(DIRI_SIDES), NR_CTRL_PTS_(NR_CTRL_PTS), JSON_PATH_(std::move(JSON_PATH)), 
        ref_(iganet::utils::to_array(NR_CTRL_PTS, NR_CTRL_PTS)) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the interior collocation points
  auto const &interiorCollPts() const { return interiorCollPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }
  
  /// @brief Writes data to a JSON file
  void appendToJsonFile(const std::string& key, const nlohmann::json& data) {
    
    // create json object
    nlohmann::json jsonData;

    // try to read the JSON data from the file
    try {
        std::ifstream json_file_in(JSON_PATH_);
        if (json_file_in.is_open()) {
            json_file_in >> jsonData;
            json_file_in.close();
        } else {
            std::cerr << "Warning: Could not open file for reading: " 
                      << JSON_PATH_ << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading JSON file: " << JSON_PATH_ 
                  << ". Exception: " << e.what() << "\n";
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
        std::ofstream json_file_out(JSON_PATH_);
        if (json_file_out.is_open()) {
            json_file_out << jsonData.dump(1);
            json_file_out.close();
        } else {
            std::cerr << "Error: Could not open file for writing: " 
                      << JSON_PATH_ << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error writing JSON file: " << JSON_PATH_ 
                  << ". Exception: " << e.what() << "\n";
    }
  }

  /// @brief helper function to load the displacements from a JSON file
  torch::Tensor loadDisplacements() {
      // create options for the tensor
      auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);
  
      // open the JSON file
      std::ifstream file(JSON_PATH_);
      if (!file.is_open()) {
          throw std::runtime_error("Could not open file: " + JSON_PATH_);
      }
  
      // parse the JSON file
      nlohmann::json jsonData;
      file >> jsonData;
      file.close();
  
      // extract the matlabDisplacements array
      auto matlabDisplacements_j = jsonData["matlabDisplacements"];
      int numCtrlPts = matlabDisplacements_j.size();
  
      // create a tensor for the displacements
      torch::Tensor matlabDisplacements = torch::empty({numCtrlPts, 2}, options);
  
      // fill the tensor with data from the JSON file
      for (int i = 0; i < numCtrlPts; ++i) {
          matlabDisplacements[i][0] = matlabDisplacements_j[i][0].get<double>();
          matlabDisplacements[i][1] = matlabDisplacements_j[i][1].get<double>();
      }
  
      return matlabDisplacements;
  }

  /// @brief helper function to calculate the Greville abscissae
  static std::vector<double> computeGrevilleAbscissae
    (const gsKnotVector<double>& knotVector, int degree, int numCtrlPts) {
      std::vector<double> greville(numCtrlPts, 0.0);
      
      for (int i = 0; i < numCtrlPts; ++i) {
          double sum = 0.0;
          for (int j = i + 1; j <= i + degree; ++j) {
              sum += knotVector.at(j);
          }
          greville[i] = sum / degree;
      }
      return greville;
  }

  /// @brief GISMO workflow
  static std::tuple<gsMatrix<double>, gsMatrix<double>, gsMatrix<double>> RunGismoSimulation(
        int64_t NR_CTRL_PTS, int DEGREE, double YOUNG_MODULUS, double POISSON_RATIO,
        const std::vector<std::tuple<int, double, double>>& DIRI_SIDES,
        const std::vector<std::tuple<int, double, double>>& FORCE_SIDES,
        const std::pair<double, double>& BODY_FORCE) {
    
    // initialize the reference control points and the calculated displacements & stresses
    gsMatrix<double> gsCtrlPts(NR_CTRL_PTS * NR_CTRL_PTS, 2);
    gsMatrix<double> gsDisplacements(NR_CTRL_PTS * NR_CTRL_PTS, 2);
    gsMatrix<double> gsStresses(gsCtrlPts.rows(), 1); //von Mises stresses (only one component)

    // create knot vectors
    gsKnotVector<double> knotVector_u(0.0, 1.0, NR_CTRL_PTS - DEGREE - 1, DEGREE + 1);
    gsKnotVector<double> knotVector_v(0.0, 1.0, NR_CTRL_PTS - DEGREE - 1, DEGREE + 1);
    
    // calculation of the Greville points
    std::vector<double> grevilleU = computeGrevilleAbscissae(knotVector_u, DEGREE, NR_CTRL_PTS);
    std::vector<double> grevilleV = computeGrevilleAbscissae(knotVector_v, DEGREE, NR_CTRL_PTS);
    
    // systematic placement of control points according to greville abscissae
    int index = 0;
    for (int j = 0; j < NR_CTRL_PTS; ++j) {
        for (int i = 0; i < NR_CTRL_PTS; ++i) {
            gsCtrlPts(index, 0) = grevilleU[i];
            gsCtrlPts(index, 1) = grevilleV[j];
            ++index;
        }
    }

    // create geometry
    gsTensorBSpline<2, double> geometry(knotVector_u, knotVector_v, gsCtrlPts);

    // create multipatch and add the geometry
    gsMultiPatch<double> multiPatch;
    multiPatch.addPatch(geometry);
    gsMultiBasis<> basis(multiPatch);

    // helper to map 1-4 to gs boundary enums
    auto getGsBoundarySide = [](int side) -> boundary::side {
        switch (side) {
            case 1: return boundary::west;
            case 2: return boundary::east;
            case 3: return boundary::south;
            case 4: return boundary::north;
            default:
                throw std::invalid_argument("Invalid side number (must be 1 to 4)");
        }
    };

    // define boundary conditions
    gsBoundaryConditions<double> bcInfo;

    // Dirichlet BCs
    for (const auto& d : DIRI_SIDES) {
        int side = std::get<0>(d);
        double xVal = std::get<1>(d);
        double yVal = std::get<2>(d);
        auto gsSide = getGsBoundarySide(side);

        bcInfo.addCondition(0, gsSide, condition_type::dirichlet, 
                            gsConstantFunction<double>(xVal, 2), 0);
        bcInfo.addCondition(0, gsSide, condition_type::dirichlet, 
                            gsConstantFunction<double>(yVal, 2), 1);
    }

    // Neumann (Traction) BCs
    for (const auto& f : FORCE_SIDES) {
        int side = std::get<0>(f);
        double tx = std::get<1>(f);
        double ty = std::get<2>(f);
        auto gsSide = getGsBoundarySide(side);

        gsFunctionExpr<> traction(std::to_string(tx), std::to_string(ty), 2);
        bcInfo.addCondition(0, gsSide, condition_type::neumann, traction);
    }
    
    // body force
    gsConstantFunction<double> bodyForce(BODY_FORCE.first, BODY_FORCE.second, 2);

    // initialize the elasticity assembler
    gsElasticityAssembler<double> assembler(geometry, basis, bcInfo, bodyForce);
    assembler.options().setReal("YoungsModulus", YOUNG_MODULUS);
    assembler.options().setReal("PoissonsRatio", POISSON_RATIO);
    assembler.assemble();

    // solve the system
    gsSparseSolver<>::CGDiagonal solver;
    gsMatrix<double> solution;
    solver.compute(assembler.matrix());
    solution = solver.solve(assembler.rhs());

    // create a multipatch object for the solution
    gsMultiPatch<double> solutionPatch;
    assembler.constructSolution(solution, assembler.allFixedDofs(), solutionPatch);

    // create a piecewise function for the stresses
    gsPiecewiseFunction<double> stressFunction;

    // calculate von Mises stresses (cauchy form)
    assembler.constructCauchyStresses(solutionPatch, stressFunction, 
                                      stress_components::von_mises);

    // loop all control points
    for (int i = 0; i < gsCtrlPts.rows(); ++i) {
        // create temp point
        gsMatrix<double> point(2, 1);
        point(0, 0) = gsCtrlPts(i, 0);
        point(1, 0) = gsCtrlPts(i, 1);

        // DISPLACEMENT EVALUATION
        auto displacement = solutionPatch.patch(0).eval(point);
        gsDisplacements(i, 0) = displacement(0);
        gsDisplacements(i, 1) = displacement(1);
        
        // STRESS EVALUATION
        const auto &segment = stressFunction.piece(0); // patch index 0 (bc only 1 patch)
        gsMatrix<double> stressValue(1, 1);
        segment.eval_into(point, stressValue);
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
      collPts_ = Base::variable_collPts(iganet::collPts::greville);
      interiorCollPts_ = Base::variable_collPts(iganet::collPts::greville_interior);
      
      // WARNING, only works for equal number of control points in x and y direction
      nrCollPts_ = static_cast<int>(std::sqrt(std::get<0>(collPts_)[0].size(0)));
      torch::Tensor collPtsCoeffs = std::get<0>(collPts_)[0].slice(0, 0, nrCollPts_);
      nlohmann::json collPtsCoeffs_j = nlohmann::json::array();
      for (int i = 0; i < collPtsCoeffs.size(0); ++i) {
          collPtsCoeffs_j.push_back({collPtsCoeffs[i].item<double>()});
      }
      appendToJsonFile("collPtsCoeffsRef1", collPtsCoeffs_j);
      appendToJsonFile("nrCollPtsRef1", {nrCollPts_});
      

      var_knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      var_coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_);

      var_knot_indices_interior_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
                interiorCollPts_.first);
      var_coeff_indices_interior_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_interior_);

      G_knot_indices_ =
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      G_coeff_indices_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              G_knot_indices_);

      G_knot_indices_interior_ = 
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              interiorCollPts_.first);
      G_coeff_indices_interior_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              G_knot_indices_interior_);

      return true;
    } else
      return false;
  }

  /// @brief Computes the loss function
  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

    // create u_ from the training's outputs
    Base::u_.from_tensor(outputs);

    // pre-allocation of the loss values
    torch::Tensor totalLoss; 
    torch::Tensor elastLoss;
    std::optional<torch::Tensor> bcLoss;
    std::optional<torch::Tensor> tfbcLoss;
    std::optional<torch::Tensor> gsLoss;
    std::optional<torch::Tensor> forceLoss;

    // pre-allocation of the tensors for the traction boundary conditions
    std::optional<torch::Tensor> forceValues;
    std::optional<torch::Tensor> targetForce;
    std::optional<torch::Tensor> tractionFreeValues;
    std::optional<torch::Tensor> tractionZeros;

    // TRACTION BOUNDARY CONDITIONS
  
    // only calculate the traction-free boundary conditions if there are any
    if (!TFBC_SIDES_.empty() || !FORCE_SIDES_.empty())
    {   
        
        // intersecCtr is used to determine an intersection of dirichlet/force and trac.free sides
        static std::vector<int> intersecCtr(0);
        // allocate tensors for the traction-free boundary conditions
        static std::array<torch::Tensor, 2ul> tractionCollPts;
        // collect sides of traction-free and force BCs
        std::vector<int> neumannSides;

        // collect sides of Dirichlet or force BCs
        std::vector<int> diriOrForceSides;
        for (const auto& tuple : DIRI_SIDES_) {
            // extract only the int-values from DIRI_SIDES_
            diriOrForceSides.push_back(std::get<0>(tuple));
        }       
        
        // add the two vectors of force- and traction-free-BCs
        neumannSides.reserve(TFBC_SIDES_.size() + FORCE_SIDES_.size());
        neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
        // add the force sides to the neumannSides and diriOrForceSides
        for (const auto& force : FORCE_SIDES_) {
            // add the force sides to the neumannSides
            neumannSides.push_back(std::get<0>(force));
            // add the force sides to the diriOrForceSides
            diriOrForceSides.push_back(std::get<0>(force));
        }

        // calculate the tractionCollocationPoints once in the beginning of the simulation
        if (epoch == 0 && intersecCtr.empty()) {
            // allocate tensors for the traction-free boundary conditions
            std::vector<torch::Tensor> tractionCollPtsX;
            std::vector<torch::Tensor> tractionCollPtsY;

            // evaluate the boundary points depending on traction-free sides
            for (int side : neumannSides) {
                if (side == 1) {
                    // check if diriOrForceSides has only side 3 as side
                    if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                        != diriOrForceSides.end() &&
                        std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                        == diriOrForceSides.end()) {     

                        at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}));
                        tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1));
                        // 1 collPt has to be removed
                        intersecCtr.push_back(1);
                    }
                    // check if diriOrForceSides has only side 4 as side
                    else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                            == diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                            != diriOrForceSides.end()) {
        
                        at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}));
                        tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 0, -1));
                        // 1 collPt has to be removed
                        intersecCtr.push_back(1);
                    }
                    // check if diriOrForceSides has side 3 and side 4
                    else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                            != diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                            != diriOrForceSides.end()) {
                        
                        at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 2}));  
                        tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                        // 2 collPts have to be removed
                        intersecCtr.push_back(2);
                    }
                    else {
                        tractionCollPtsX.push_back(torch::zeros(nrCollPts_));
                        tractionCollPtsY.push_back(std::get<0>(collPts_.second)[0]);
                        // no collPt has to be removed
                        intersecCtr.push_back(0);
                    }
                }
                else if (side == 2) {
                    // check if diriOrForceSides has only side 3 as side
                    if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                        != diriOrForceSides.end() &&
                        std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                        == diriOrForceSides.end()) {    

                        at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}));
                        tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1));
                        // 1 collPt has to be removed
                        intersecCtr.push_back(1);
                    }
                    // check if diriOrForceSides has only side 4 as side
                    else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                            == diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                            != diriOrForceSides.end()) {

                        at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}));
                        tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 0, -1));
                        // 1 collPt has to be removed
                        intersecCtr.push_back(1);
                    }
                    // check if diriOrForceSides has side 3 and side 4
                    else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 3) 
                            != diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 4) 
                            != diriOrForceSides.end()) {

                        at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 2}));
                        tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                        // 2 collPts have to be removed
                        intersecCtr.push_back(2);
                    }
                    else {
                        tractionCollPtsX.push_back(torch::ones(nrCollPts_));
                        tractionCollPtsY.push_back(std::get<0>(collPts_.second)[0]);
                        // no collPt has to be removed
                        intersecCtr.push_back(0);
                    }
                    
                }
                else if (side == 3) {
                    // check if diriOrForceSides has only side 1 as side
                    if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                        != diriOrForceSides.end() &&
                        std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                        == diriOrForceSides.end()) {   

                        at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                        tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}));
                        // 1 collPt has to be removed
                        intersecCtr.push_back(1);
                    }
                    // check if diriOrForceSides has only side 2 as side
                    else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                            == diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                            != diriOrForceSides.end()) {   

                        at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 0, -1));
                        tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}));
                        // 1 collPt has to be removed
                        intersecCtr.push_back(1);
                    }
                    // check if diriOrForceSides has side 1 and side 2
                    else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                            != diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                            != diriOrForceSides.end()) {   

                        at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1, -1));
                        tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 2}));
                        // 2 collPts have to be removed
                        intersecCtr.push_back(2);
                    }
                    else {
                        tractionCollPtsX.push_back(std::get<0>(collPts_.second)[0]);
                        tractionCollPtsY.push_back(torch::zeros(nrCollPts_));
                        // no collPt has to be removed
                        intersecCtr.push_back(0);
                    }
                }
                else if (side == 4) {
                    // check if diriOrForceSides has only side 1 as side
                    if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                        != diriOrForceSides.end() &&
                        std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                        == diriOrForceSides.end()) {   

                        at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                        tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}));
                        // 1 collPt has to be removed
                        intersecCtr.push_back(1);
                    }
                    // check if diriOrForceSides has only side 2 as side
                    else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                            == diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                            != diriOrForceSides.end()) {   

                        at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 0, -1));
                        tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}));
                        // 1 collPt has to be removed
                        intersecCtr.push_back(1);
                    }
                    // check if diriOrForceSides has side 1 and side 2
                    else if (std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 1) 
                            != diriOrForceSides.end() &&
                            std::find(diriOrForceSides.begin(), diriOrForceSides.end(), 2) 
                            != diriOrForceSides.end()) {   

                        at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                        tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1, -1));
                        tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 2}));
                        // 2 collPts have to be removed
                        intersecCtr.push_back(2);
                    }
                    else {
                        tractionCollPtsX.push_back(std::get<0>(collPts_.second)[0]);
                        tractionCollPtsY.push_back(torch::ones(nrCollPts_));
                        // no collPt has to be removed
                        intersecCtr.push_back(0);
                    }
                }
                
                else {
                    throw std::invalid_argument("Side for traction BC has to be 1, 2, 3 or 4.");
                }
            }
            
            // merge the tensors to get a (nrTractionCollPts, 2) tensor
            if (!tractionCollPtsX.empty() && !tractionCollPtsY.empty()) {
                tractionCollPts = {
                    torch::cat(tractionCollPtsX, 0), 
                    torch::cat(tractionCollPtsY, 0)
                };
            } 
            var_knot_indices_boundary_ =
                Base::f_.template find_knot_indices<iganet::functionspace::interior>(
                tractionCollPts);
            var_coeff_indices_boundary_ =
                Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
                var_knot_indices_boundary_);
            G_knot_indices_boundary_ =
                Base::G_.template find_knot_indices<iganet::functionspace::interior>(
                    tractionCollPts);
            G_coeff_indices_boundary_ =
            Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
                G_knot_indices_boundary_);
        }  

        // calculate the jacobian of the affected boundary points
        auto jacobianBoundary = Base::u_.ijac(Base::G_, tractionCollPts, 
            var_knot_indices_boundary_, var_coeff_indices_boundary_,
            G_knot_indices_boundary_, G_coeff_indices_boundary_);
        auto ux_x = *jacobianBoundary[0];
        auto ux_y = *jacobianBoundary[1];
        auto uy_x = *jacobianBoundary[2];
        auto uy_y = *jacobianBoundary[3];

        // allocate tensors for the traction-free boundary conditions (tfbc)
        torch::Tensor tractionValuesX = torch::zeros({tractionCollPts[0].size(0)});
        torch::Tensor tractionValuesY = torch::zeros({tractionCollPts[0].size(0)});
        // calculate the traction values at the boundary points
        int pointCtr = 0;
        int sideCtr = 0; 

        for (int side : neumannSides) {
            int n_vals = nrCollPts_ - intersecCtr[sideCtr];

            for (int i = 0; i < n_vals; ++i) {
                int idx = pointCtr + i;

                if (side == 1) {
                    tractionValuesX[idx] =  - lambda_ * (ux_x[idx] + uy_y[idx]) 
                                            - 2 * mu_ * ux_x[idx];
                    tractionValuesY[idx] =  - mu_ * (uy_x[idx] + ux_y[idx]);
                }
                else if (side == 2) {
                    tractionValuesX[idx] = lambda_ * (ux_x[idx] + uy_y[idx]) 
                                           + 2 * mu_ * ux_x[idx];
                    tractionValuesY[idx] = mu_ * (uy_x[idx] + ux_y[idx]);
                }
                else if (side == 3) {
                    tractionValuesX[idx] =  - mu_ * (uy_x[idx] + ux_y[idx]);
                    tractionValuesY[idx] =  - lambda_ * (ux_x[idx] + uy_y[idx]) 
                                            - 2 * mu_ * uy_y[idx];
                }
                else if (side == 4) {
                    tractionValuesX[idx] = mu_ * (uy_x[idx] + ux_y[idx]);
                    tractionValuesY[idx] = lambda_ * (ux_x[idx] + uy_y[idx]) 
                                           + 2 * mu_ * uy_y[idx];
                }
            }

            pointCtr += n_vals;
            sideCtr++;
        }

        // merge the traction tensors of x- and y-directions
        torch::Tensor tractionValues = torch::stack({tractionValuesX, tractionValuesY}, 1);

        if (!FORCE_SIDES_.empty()) {
            // calculate total cutlength from last forceSize entries of intersecCtr
            int cutlength = 0;
            int forceSize = FORCE_SIDES_.size();
            for (int i = static_cast<int>(intersecCtr.size()) - forceSize; 
                     i < static_cast<int>(intersecCtr.size()); ++i) {
                cutlength += nrCollPts_ - intersecCtr[i];
            }
            // separate traction-free and force parts
            tractionFreeValues.emplace
                (tractionValues.slice(0, 0, tractionValues.size(0) - cutlength));
            tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));
            forceValues.emplace(tractionValues.slice(0, tractionValues.size(0) -
                                cutlength, tractionValues.size(0)));
            targetForce.emplace(torch::zeros_like(*forceValues));
            // fill in the known force values
            int offset = 0;
            int startIdx = static_cast<int>(intersecCtr.size()) - forceSize;
            for (size_t i = 0; i < FORCE_SIDES_.size(); ++i) {
                int reducedPts = nrCollPts_ - intersecCtr[startIdx + i];
                auto rowSlice = (*targetForce).slice(0, offset, offset + reducedPts);
                rowSlice.slice(1, 0, 1).fill_(std::get<1>(FORCE_SIDES_[i]));  // x-value
                rowSlice.slice(1, 1, 2).fill_(std::get<2>(FORCE_SIDES_[i]));  // y-value
                offset += reducedPts;
            }       
        }
        else {
            // set the traction-free values
            tractionFreeValues.emplace(tractionValues);
            // set the target values to zero
            tractionZeros.emplace(torch::zeros_like(*tractionFreeValues));
        }

    }

    // LINEAR ELASTICITY EQUATION

    // calculation of the second derivatives of the displacements (u)
    auto hessianColl = Base::u_.ihess(Base::G_, interiorCollPts_.first, 
        var_knot_indices_interior_, var_coeff_indices_interior_,
        G_knot_indices_interior_, G_coeff_indices_interior_);

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

    // calculation of the divergence of the stress tensor
    for (int i = 0; i < hessianColl[0][0]->size(0); ++i) {

        // x-direction
        divStressX[i] = (lambda_ + 2 * mu_) * ux_xx[i] + 
                        mu_ * ux_yy[i] + (lambda_ + mu_) * uy_xy[i];

        // y-direction
        divStressY[i] = mu_ * uy_xx[i] + (lambda_ + 2 * mu_) * uy_yy[i] + 
                        (lambda_ + mu_) * ux_xy[i];
        
    }
    
    // create a tensor of the divergence of the stress tensor
    torch::Tensor divStress = torch::stack({divStressX, divStressY}, /*dim=*/1);

    // BODY FORCE

    // evaluate the reference body force f at all interior collocation points
    auto f = Base::f_.eval(interiorCollPts_.first);

    torch::Tensor bodyForce = torch::stack({*f[0], *f[1]}, /*dim=*/1).to(torch::kFloat32);

    // UNSUPERVISED LEARNING (default)
    if (SUPERVISED_LEARNING_ == false) {

        // create command line output variable for all the different losses
        std::ostringstream singleLossOutput;

        // calculation of the loss function for double-sided constraint solid
        // div(sigma) + f = 0 --> div(sigma) = -f
        elastLoss = torch::mse_loss(divStress, bodyForce);
        
        // add the elasticity loss to the total loss
        totalLoss = elastLoss;

        // add the elasticity loss to the cmd-output variable
        singleLossOutput << "EL " << std::setw(11) << elastLoss.item<double>();

        // only consider traction-free-bc (tfbc) loss if tfbcs are applied
        if (!TFBC_SIDES_.empty()) {
            tfbcLoss = torch::mse_loss(*tractionFreeValues, *tractionZeros);
            totalLoss += *tfbcLoss;
            singleLossOutput << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
        }

        // only consider force loss if force is applied
        if (!FORCE_SIDES_.empty()) {
            forceLoss = torch::mse_loss(*forceValues, *targetForce);
            totalLoss += *forceLoss;
            singleLossOutput << " + FL " << std::setw(11) << (*forceLoss).item<double>();
        }

        // only consider BC loss if dirichlet BCs are applied
        if (!DIRI_SIDES_.empty()) {
            // add a BC weight for penalization of the training
            int bcWeight = 1e7;
            // initialize bcLoss variable
            bcLoss = torch::tensor(0.0);

            // evaluation of the displacements at the boundary points
            auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
            // evaluation of the displacements at the reference boundary points
            auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

            // loop through all dirichlet sides
            for (const auto& side : DIRI_SIDES_) {
                int sideNr = std::get<0>(side);
                
                switch (sideNr) {
                    case 1: 
                        *bcLoss += bcWeight * 
                            (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                             torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                        break;
                    case 2:
                        *bcLoss += bcWeight * 
                            (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                             torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                        break;
                    case 3:
                        *bcLoss += bcWeight * 
                            (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                             torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                        break;
                    case 4:
                        *bcLoss += bcWeight * 
                            (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
                             torch::mse_loss(*std::get<3>(u_bdr)[1], *std::get<3>(bdr)[1]));
                        break;
                    default:
                        std::cerr << "Error: Invalid side number for Dirichlet BC!" << std::endl;
                }
            }
            totalLoss += *bcLoss;
            singleLossOutput << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight 
                             << " * 1e" << static_cast<int>(std::log10(bcWeight));
        }

        // print the loss values
        std::cout << std::setw(11) << 
            totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
    }
    
    // SUPERVISED LEARNING
    else if (SUPERVISED_LEARNING_ == true) {

        // create command line output variable for all the different losses
        std::ostringstream singleLossOutput;
    
        // preprocess the outputs for comparison with the matlab solution
        torch::Tensor modifiedOutputs = outputs * 1.0;
    
        // create netDisplacements_ from slices of modifiedOutputs
        torch::Tensor netDisplacements_ = torch::stack({
            modifiedOutputs.slice(0, 0, outputs.size(0) / 2),
            modifiedOutputs.slice(0, outputs.size(0) / 2, outputs.size(0)),
        }, 1);

        // load the displacements from the matlab solution
        torch::Tensor matlabDisplacements_ = loadDisplacements();

        // supervised loss: MSE gegen matlab-Kontrollpunkte
        gsLoss = 1e9 * torch::mse_loss(netDisplacements_, matlabDisplacements_);

        // calculation of the loss function for double-sided constraint solid
        // div(sigma) + f = 0 --> div(sigma) = -f
        elastLoss = torch::mse_loss(divStress, bodyForce);

        // add the elasticity loss and supervised loss to the total loss
        totalLoss = *gsLoss + elastLoss;

        // add the elasticity and supervised losses to the cmd-output variable
        singleLossOutput << "GL " << std::setw(11) << (*gsLoss).item<double>()
                        << " + EL " << std::setw(11) << elastLoss.item<double>();

        // only consider traction-free-bc (tfbc) loss if tfbcs are applied
        if (!TFBC_SIDES_.empty()) {
            tfbcLoss = torch::mse_loss(*tractionFreeValues, *tractionZeros);
            totalLoss += *tfbcLoss;
            singleLossOutput << " + TL " << std::setw(11) << (*tfbcLoss).item<double>();
        }

        // only consider force loss if force is applied
        if (!FORCE_SIDES_.empty()) {
            forceLoss = torch::mse_loss(*forceValues, *targetForce);
            totalLoss += *forceLoss;
            singleLossOutput << " + FL " << std::setw(11) << (*forceLoss).item<double>();
        }

        // only consider BC loss if dirichlet BCs are applied
        if (!DIRI_SIDES_.empty()) {
            // add a BC weight for penalization of the training
            int bcWeight = 1e0;
            // initialize bcLoss variable
            bcLoss = torch::tensor(0.0);

            // evaluation of the displacements at the boundary points
            auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
            // evaluation of the displacements at the reference boundary points
            auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

            // loop through all dirichlet sides
            for (const auto& side : DIRI_SIDES_) {
                int sideNr = std::get<0>(side);

                switch (sideNr) {
                    case 1:
                        *bcLoss += bcWeight * 
                            (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                             torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                        break;
                    case 2:
                        *bcLoss += bcWeight * 
                            (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                             torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                        break;
                    case 3:
                        *bcLoss += bcWeight * 
                            (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                             torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                        break;
                    case 4:
                        *bcLoss += bcWeight * 
                            (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
                             torch::mse_loss(*std::get<3>(u_bdr)[1], *std::get<3>(bdr)[1]));
                        break;
                    default:
                        std::cerr << "Error: Invalid side number for Dirichlet BC!" << std::endl;
                }
            }
            totalLoss += *bcLoss;
            singleLossOutput << " + BL " << std::setw(11) << (*bcLoss).item<double>() / bcWeight 
                            << " * 1e" << static_cast<int>(std::log10(bcWeight));
        }

        // print the loss values
        std::cout << std::setw(11) << 
            totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
    }

    else {
        throw std::runtime_error("Invalid value for SUPERVISED_LEARNING_");
    }

    // POSTPROCESSING PREPARATION - WRITING DATA TO JSON FILE

    // only calculate this at the end of the simulation
    if ((epoch == MAX_EPOCH_ - 1) || (totalLoss.item<double>() <= MIN_LOSS_)) {
        
        // STRESS CALCULATION

        // calculate the jacobian of the displacements (u) at the collocation points
        auto jacobian = Base::u_.ijac(Base::G_, collPts_.first, var_knot_indices_, 
            var_coeff_indices_, G_knot_indices_, G_coeff_indices_);
        
        auto ux_x = *jacobian[0];
        auto ux_y = *jacobian[1];
        auto uy_x = *jacobian[2];
        auto uy_y = *jacobian[3];

        // allocate the stress tensor
        torch::Tensor sigma_xx = torch::zeros({jacobian[0]->size(0)});
        torch::Tensor sigma_xy = torch::zeros({jacobian[0]->size(0)});
        torch::Tensor sigma_yy = torch::zeros({jacobian[0]->size(0)}); 
        torch::Tensor sigma_vm = torch::zeros({jacobian[0]->size(0)});   

        torch::Tensor epsilon_xx = torch::zeros({jacobian[0]->size(0)});
        torch::Tensor epsilon_yy = torch::zeros({jacobian[0]->size(0)});
        torch::Tensor poisson_re = torch::zeros({jacobian[0]->size(0)});

        // create json object for the stresses
        nlohmann::json netVmStresses_j = nlohmann::json::array();
        nlohmann::json netXStresses_j = nlohmann::json::array();
        nlohmann::json netYStresses_j = nlohmann::json::array();
        nlohmann::json netPoisson_j = nlohmann::json::array();

        // calculate the stress tensor
        for (int i = 0; i < jacobian[0]->size(0); ++i) {
            // calculate the stress values for all collocation points
            sigma_xx[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * ux_x[i];
            sigma_xy[i] = mu_ * (uy_x[i] + ux_y[i]);
            sigma_yy[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * uy_y[i];
            
            // calculate von mises stress at the collocation points
            sigma_vm[i] = sqrt(sigma_xx[i] * sigma_xx[i] + sigma_yy[i] * sigma_yy[i] 
                             - sigma_xx[i] * sigma_yy[i] + sigma_xy[i] * sigma_xy[i] * 3);
            
            // calculate the strains at the collocation points
            epsilon_xx[i] = (lambda_ + mu_) / (mu_ * (3 * lambda_ + 2 * mu_)) * 
                (sigma_xx[i] - lambda_ / (2 * (lambda_ + mu_)) * sigma_yy[i]);
            epsilon_yy[i] = (lambda_ + mu_) / (mu_ * (3 * lambda_ + 2 * mu_)) * 
                (sigma_yy[i] - lambda_ / (2 * (lambda_ + mu_)) * sigma_xx[i]);

            // calculate the actual poisson ratio at the collocation points
            // poisson_re[i] = ( - ( ux_x[i] * (mu_ * (3 * lambda_ + 2 * mu_)) / 
            //                      (lambda_ + mu_) - sigma_xx[i] ) / sigma_yy[i] );
            //                   - ( uy_y[i] * (mu_ * (3 * lambda_ + 2 * mu_)) / 
            //                      (lambda_ + mu_) - sigma_yy[i] ) / sigma_xx[i] ) / 2;
            // poisson_re[i] = - ( uy_y[i] * (mu_ * (3 * lambda_ + 2 * mu_)) / 
            //                      (lambda_ + mu_) - sigma_yy[i] ) / sigma_xx[i];

            // only valid for load in x-direction
            poisson_re[i] = - epsilon_yy[i] / epsilon_xx[i];
            
            // add the stresses to the json objects
            netVmStresses_j.push_back({sigma_vm[i].item<double>()});
            netXStresses_j.push_back({sigma_xx[i].item<double>()});
            netYStresses_j.push_back({sigma_yy[i].item<double>()});
            // add the poisson ratio to the json object
            netPoisson_j.push_back({poisson_re[i].item<double>()});
        }

        // write the stresses and poisson ratios to the json file
        appendToJsonFile("netVmStresses", netVmStresses_j);
        appendToJsonFile("netXStresses", netXStresses_j);
        appendToJsonFile("netYStresses", netYStresses_j);
        appendToJsonFile("netPoisson", netPoisson_j);

        // CALCULATE THE NEW POSITION OF THE COLLPTS

        // create a tensor of the collocation points
        torch::Tensor collPtsFirstAsTensor = torch::stack(
            {std::get<0>(collPts_.first), std::get<1>(collPts_.first)}, 1);
        auto displacementOfCollPts = Base::u_.eval(collPts_.first);
        torch::Tensor displacementAsTensor = torch::stack(
            {*(displacementOfCollPts[0]), *(displacementOfCollPts[1]) }, 1);

        // create json objects for the collocation points' reference and displaced position
        nlohmann::json collPtsFirst_j = nlohmann::json::array();
        nlohmann::json collPtsFirstDispl_j = nlohmann::json::array();
        for (int i = 0; i < collPtsFirstAsTensor.size(0); ++i) {
            // reference position of the collocation points
            collPtsFirst_j.push_back({collPtsFirstAsTensor[i][0].item<double>(), 
                                      collPtsFirstAsTensor[i][1].item<double>()});
            // new position of the collocation points
            collPtsFirstDispl_j.push_back({collPtsFirstAsTensor[i][0].item<double>() + 
                                           displacementAsTensor[i][0].item<double>(), 
                                           collPtsFirstAsTensor[i][1].item<double>() + 
                                           displacementAsTensor[i][1].item<double>()});
        }
        // write the collocation points' original position to the json file
        appendToJsonFile("collPtsFirstAsTensor", collPtsFirst_j);
        // write the collocation points' new position to the json file
        appendToJsonFile("collPtsFirstAfterDisplacementAsTensor", collPtsFirstDispl_j);

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

  // ------- USER INPUTS ------- //

  // material parameters
  double YOUNG_MODULUS = 210.0;
  double POISSON_RATIO = 0.25;

  // simulation parameters
  int MAX_EPOCH = 100;
  double MIN_LOSS = 1e-8;
  bool SUPERVISED_LEARNING = false;
  std::string JSON_PATH = "/home/obergue/Documents/pytest/splinepy/results_2D.json";
  
  // reference simulation parameters
  bool RUN_REF_SIM = false;
  int NR_CTRL_PTS_REF = 100;
  int DEGREE_REF = 3;

  // spline parameters
  int64_t NR_CTRL_PTS = 8;  // in each direction 
  constexpr int DEGREE = 3; // for geometry and variable

  // boundary conditions
  std::vector<std::tuple<int, double, double>> FORCE_SIDES = {
    //   {2, 50.0,  0.0},   // {side, x-traction, y-traction}
    };
  std::vector<std::tuple<int, double, double>> DIRI_SIDES = {
      {1, 0.0,  0.0},       // {side, x-displ, y-displ}
      {2, 1.0,  0.0},
    };
  std::vector<int> TFBC_SIDES = {3,4}; // {sides}

  // body force (constant over the whole domain)
  std::pair<double, double> BODY_FORCE = {0.0, 0.0}; // {fx, fy}

  // --------------------------- //


  // calculation of lame parameters
  double lambda = (YOUNG_MODULUS * POISSON_RATIO) / 
                  ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
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
  std::tie(gsCtrlPts, gsDisplacements, gsStresses) = 
    linear_elasticity_t::RunGismoSimulation(NR_CTRL_PTS, DEGREE, 
        YOUNG_MODULUS, POISSON_RATIO, DIRI_SIDES, FORCE_SIDES, BODY_FORCE);
    
  linear_elasticity_t
    net(// simulation parameters
        lambda, mu, SUPERVISED_LEARNING, MAX_EPOCH, MIN_LOSS, 
        TFBC_SIDES, FORCE_SIDES, DIRI_SIDES, NR_CTRL_PTS, JSON_PATH,
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

  if (RUN_REF_SIM) {
    gsMatrix<double> gsRefCtrlPts;
    gsMatrix<double> gsRefDisplacements;
    gsMatrix<double> gsRefStresses;

    std::tie(gsRefCtrlPts, gsRefDisplacements, gsRefStresses) = 
    linear_elasticity_t::RunGismoSimulation(NR_CTRL_PTS_REF, DEGREE, 
        YOUNG_MODULUS, POISSON_RATIO, DIRI_SIDES, FORCE_SIDES, BODY_FORCE);

    gsMatrix<double> displacedGsRefCtrlPts = gsRefCtrlPts + gsRefDisplacements;
    nlohmann::json displacedGsRefCtrlPts_j = nlohmann::json::array();
    nlohmann::json gsRefStresses_j = nlohmann::json::array();
    nlohmann::json gsRefDisplacements_j = nlohmann::json::array();
    nlohmann::json gsRefOriginalCtrlPts_j = nlohmann::json::array();

    // write G+Smo reference data from the matrices to the json objects
    for (int i = 0; i < displacedGsRefCtrlPts.rows(); ++i) {
        // new control points G+Smo
        displacedGsRefCtrlPts_j.push_back(
            {displacedGsRefCtrlPts(i, 0), displacedGsRefCtrlPts(i, 1)});
        // write the von Mises stresses to the json object
        gsRefStresses_j.push_back({gsRefStresses(i, 0)});
        // write the displacements to the json object
        gsRefDisplacements_j.push_back(
            {gsRefDisplacements(i, 0), gsRefDisplacements(i, 1)});
        // original control points G+Smo
        gsRefOriginalCtrlPts_j.push_back({gsRefCtrlPts(i, 0), gsRefCtrlPts(i, 1)});
    }
    net.appendToJsonFile("gsRefCtrlPts", displacedGsRefCtrlPts_j);
    net.appendToJsonFile("gsRefDegree", DEGREE_REF);
    net.appendToJsonFile("gsRefStresses", gsRefStresses_j);
    net.appendToJsonFile("gsRefDisplacements", gsRefDisplacements_j);
    net.appendToJsonFile("gsRefOriginalCtrlPts", gsRefOriginalCtrlPts_j);
  }

  // imposing body force
  net.f().transform([=](const std::array<real_t, 2> xi) {
    return std::array<real_t, 2>{BODY_FORCE.first, BODY_FORCE.second};
  });

  // get the coefficients of the control points
  torch::Tensor ctrlPtsCoeffs = net.G().as_tensor().slice(0, 0, NR_CTRL_PTS);
  nlohmann::json ctrlPtsCoeffs_j = nlohmann::json::array();
  for (int i = 0; i < NR_CTRL_PTS; ++i) {
      ctrlPtsCoeffs_j.push_back({ctrlPtsCoeffs[i].item<double>()});
  }
  net.appendToJsonFile("ctrlPtsCoeffs", ctrlPtsCoeffs_j);

  // run through all DIRI_SIDES
  for (const auto& side : DIRI_SIDES) {
    int sideNr = std::get<0>(side);
    double xDispl = std::get<1>(side);
    double yDispl = std::get<2>(side);

    switch (sideNr) {
        case 1:
            net.ref().boundary().side<1>().transform<1>(
                [=](const std::array<real_t, 1> &xi) {
                    return std::array<real_t, 1>{xDispl};
                },
                std::array<iganet::short_t, 1>{0} 
            );
            net.ref().boundary().side<1>().transform<1>(
                [=](const std::array<real_t, 1> &xi) {
                    return std::array<real_t, 1>{yDispl};
                },
                std::array<iganet::short_t, 1>{1}
            );
            break;
        case 2:
            net.ref().boundary().side<2>().transform<1>(
                [=](const std::array<real_t, 1> &xi) {
                    return std::array<real_t, 1>{xDispl};
                },
                std::array<iganet::short_t, 1>{0} 
            );
            net.ref().boundary().side<2>().transform<1>(
                [=](const std::array<real_t, 1> &xi) {
                    return std::array<real_t, 1>{yDispl};
                },
                std::array<iganet::short_t, 1>{1}
            );
            break;
        case 3:
            net.ref().boundary().side<3>().transform<1>(
                [=](const std::array<real_t, 1> &xi) {
                    return std::array<real_t, 1>{xDispl};
                },
                std::array<iganet::short_t, 1>{0} 
            );
            net.ref().boundary().side<3>().transform<1>(
                [=](const std::array<real_t, 1> &xi) {
                    return std::array<real_t, 1>{yDispl};
                },
                std::array<iganet::short_t, 1>{1}
            );
            break;
        case 4:
            net.ref().boundary().side<4>().transform<1>(
                [=](const std::array<real_t, 1> &xi) {
                    return std::array<real_t, 1>{xDispl};
                },
                std::array<iganet::short_t, 1>{0} 
            );
            net.ref().boundary().side<4>().transform<1>(
                [=](const std::array<real_t, 1> &xi) {
                    return std::array<real_t, 1>{yDispl};
                },
                std::array<iganet::short_t, 1>{1}
            );
            break;
        default:
            std::cerr << "Error: Invalid side number " << sideNr << std::endl;
    }
}

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

  // get the geometry and displacement as tensors
  torch::Tensor geometryAsTensor = net.G().as_tensor();
  torch::Tensor displacementAsTensor = net.u().as_tensor();
  
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
  nlohmann::json gsDisplacements_j = nlohmann::json::array();
  nlohmann::json gsOriginalCtrlPts_j = nlohmann::json::array();

  // write G+Smo data from the matrices to the json objects
  for (int i = 0; i < displacedGsCtrlPts.rows(); ++i) {
        // new control points G+Smo
        displacedGsCtrlPts_j.push_back({displacedGsCtrlPts(i, 0), displacedGsCtrlPts(i, 1)});
        // write the vM stresses to the json object (calc. in beginning of the main function)
        gsStresses_j.push_back({gsStresses(i, 0)});
        // write the displacements to the json object
        gsDisplacements_j.push_back({gsDisplacements(i, 0), gsDisplacements(i, 1)});
        // original control points G+Smo
        gsOriginalCtrlPts_j.push_back({gsCtrlPts(i, 0), gsCtrlPts(i, 1)});
  }
 
  // write net data from the matrices to the json objects
  for (int i = 0; i < displacedNetCtrlPts.rows(); ++i) {
      // new control points IgANet
      displacedNetCtrlPts_j.push_back({displacedNetCtrlPts(i, 0), displacedNetCtrlPts(i, 1)});
  }

  // write data to the json file
  net.appendToJsonFile("gsCtrlPts", displacedGsCtrlPts_j);
  net.appendToJsonFile("netCtrlPts", displacedNetCtrlPts_j);
  net.appendToJsonFile("gsStresses", gsStresses_j);
  net.appendToJsonFile("gsDisplacements", gsDisplacements_j);
  net.appendToJsonFile("degree", DEGREE);
  net.appendToJsonFile("gsOriginalCtrlPts", gsOriginalCtrlPts_j);
  
  iganet::finalize();
  return 0;
}
