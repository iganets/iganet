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

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_;

  // material properties - lame's parameters
  double lambda_;
  double mu_;

  // simulation parameter
  double MAX_EPOCH_;
  double MIN_LOSS_;
  std::vector<int> TFBC_SIDES_;
  std::vector<std::tuple<int, double, double>> FORCE_SIDES_;
  std::vector<std::tuple<int, double, double>> DIRI_SIDES_;

  // gismo solution
  gsMatrix<double> gsDisplacements_;

  // supervised learning (true) or unsupervised learning (false)
  bool SUPERVISED_LEARNING_ = false;

  // json path
  static constexpr const char* JSON_PATH = "/home/obergue/Documents/pytest/splinepy/results_2D.json";

public:
  /// @brief Constructor
  template <typename... Args>
  linear_elasticity(double lambda, double mu, bool SUPERVISED_LEARNING, double MAX_EPOCH, 
                    double MIN_LOSS, std::vector<int> TFBC_SIDES,
                    std::vector<std::tuple<int, double, double>> FORCE_SIDES,
                    std::vector<std::tuple<int, double, double>> DIRI_SIDES, 
                    gsMatrix<double> gsDisplacements, std::vector<int64_t> &&layers,
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        lambda_(lambda), mu_(mu), SUPERVISED_LEARNING_(SUPERVISED_LEARNING), MAX_EPOCH_(MAX_EPOCH), 
        MIN_LOSS_(MIN_LOSS), TFBC_SIDES_(TFBC_SIDES), FORCE_SIDES_(FORCE_SIDES), DIRI_SIDES_(DIRI_SIDES),
        gsDisplacements_(std::move(gsDisplacements)), ref_(iganet::utils::to_array(8_i64, 8_i64)) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the interior collocation points
  auto const &interiorCollPts() const { return interiorCollPts_; }

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

  /// @brief helper function to calculate the Greville abscissae
  static std::vector<double> computeGrevilleAbscissae(const gsKnotVector<double>& knotVector, int degree, int numCtrlPts) {
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
  static std::tuple<gsMatrix<double>, gsMatrix<double>, gsMatrix<double>> 
    RunGismoSimulation(int64_t NR_CTRL_PTS, int DEGREE, double YOUNG_MODULUS, double POISSON_RATIO) {
    
    // initialize control points and displacements
    gsMatrix<double> gsCtrlPts(NR_CTRL_PTS * NR_CTRL_PTS, 2);
    gsMatrix<double> gsDisplacements(NR_CTRL_PTS * NR_CTRL_PTS, 2);

    // create knot vectors
    gsKnotVector<double> knotVector_u(0.0, 1.0, NR_CTRL_PTS - DEGREE - 1, DEGREE + 1);
    gsKnotVector<double> knotVector_v(0.0, 1.0, NR_CTRL_PTS - DEGREE - 1, DEGREE + 1);
    
    // calculation of the Greville points
    std::vector<double> grevilleU = computeGrevilleAbscissae(knotVector_u, DEGREE, NR_CTRL_PTS);
    std::vector<double> grevilleV = computeGrevilleAbscissae(knotVector_v, DEGREE, NR_CTRL_PTS);

    // initialize control points matrix
    gsMatrix<double> controlPoints(NR_CTRL_PTS * NR_CTRL_PTS, 2); 
    
    // systematic placement of control points according to greville abscissae
    int index = 0;
    for (int j = 0; j < NR_CTRL_PTS; ++j) {
        for (int i = 0; i < NR_CTRL_PTS; ++i) {
            controlPoints(index, 0) = grevilleU[i];
            controlPoints(index, 1) = grevilleV[j];
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
    // dirichlet BCs
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 2), 0);
    bcInfo.addCondition(0, boundary::west, condition_type::dirichlet, 
            gsConstantFunction<double>(0.0, 2), 1);
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet,
            gsConstantFunction<double>(1.0, 2), 0);
    bcInfo.addCondition(0, boundary::east, condition_type::dirichlet,
            gsConstantFunction<double>(0.0, 2), 1);

    // traction BCs
    // gsFunctionExpr<> tractionWest("0.0", "0.0", 2);
    // gsFunctionExpr<> tractionEast("100.0", "0.0", 2);
    // gsFunctionExpr<> tractionSouth("0.0", "-50.0", 2);
    // gsFunctionExpr<> tractionNorth("0.0", "50.0", 2);

    // bcInfo.addCondition(0, boundary::west, condition_type::neumann, tractionWest);
    // bcInfo.addCondition(0, boundary::east, condition_type::neumann, tractionEast);
    // bcInfo.addCondition(0, boundary::south, condition_type::neumann, tractionSouth);
    // bcInfo.addCondition(0, boundary::north, condition_type::neumann, tractionNorth);

    // body force (currently set to zero)
    gsConstantFunction<double> bodyForce(0., 0., 2);

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
      interiorCollPts_ = Base::variable_collPts(iganet::collPts::greville_interior_ref1);
      
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
        // add the two vectors of force- and traction-free-BCs
        std::vector<int> neumannSides;
        neumannSides.reserve(TFBC_SIDES_.size() + FORCE_SIDES_.size());
        neumannSides.insert(neumannSides.end(), TFBC_SIDES_.begin(), TFBC_SIDES_.end());
        for (const auto& force : FORCE_SIDES_) {
            neumannSides.push_back(std::get<0>(force));
        }
        
        // extract only the int-values from DIRI_SIDES_
        std::vector<int> diriSidesInt;
        for (const auto& tuple : DIRI_SIDES_) {
            diriSidesInt.push_back(std::get<0>(tuple));
        }       

        // allocate tensors for the traction-free boundary conditions
        std::vector<torch::Tensor> tractionCollPtsX;
        std::vector<torch::Tensor> tractionCollPtsY;
        
        int diriCtr = 1;

        // evaluate the boundary points depending on traction-free sides
        for (int side : neumannSides) {
            if (side == 1) {
                // check if diri_sides has only side 3 as side
                if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 3) != diriSidesInt.end() &&
                    std::find(diriSidesInt.begin(), diriSidesInt.end(), 4) == diriSidesInt.end()) {     

                    at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1));
                }
                // check if diri_sides has only side 4 as side
                else if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 3) == diriSidesInt.end() &&
                         std::find(diriSidesInt.begin(), diriSidesInt.end(), 4) != diriSidesInt.end()) {
       
                    at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 1}));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 0, -1));
                }
                // check if diri_sides has side 3 and side 4
                else if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 3) != diriSidesInt.end() &&
                         std::find(diriSidesInt.begin(), diriSidesInt.end(), 4) != diriSidesInt.end()) {
                    
                    at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(torch::zeros({nrCollPts_ - 2}));  
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                    diriCtr=2;

                }
                else {
                    tractionCollPtsX.push_back(torch::zeros(nrCollPts_));
                    tractionCollPtsY.push_back(std::get<0>(collPts_.second)[0]);
                }
            }
            else if (side == 2) {
                // check if diri_sides has only side 3 as side
                if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 3) != diriSidesInt.end() &&
                    std::find(diriSidesInt.begin(), diriSidesInt.end(), 4) == diriSidesInt.end()) {    

                    at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1));
                }
                // check if diri_sides has only side 4 as side
                else if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 3) == diriSidesInt.end() &&
                         std::find(diriSidesInt.begin(), diriSidesInt.end(), 4) != diriSidesInt.end()) {

                    at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 1}));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 0, -1));
                }
                // check if diri_sides has side 3 and side 4
                else if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 3) != diriSidesInt.end() &&
                         std::find(diriSidesInt.begin(), diriSidesInt.end(), 4) != diriSidesInt.end()) {

                    at::Tensor collPtsY_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(torch::ones({nrCollPts_ - 2}));
                    tractionCollPtsY.push_back(collPtsY_tensor.slice(0, 1, -1));
                    diriCtr=2;
                }
                else {
                    tractionCollPtsX.push_back(torch::ones(nrCollPts_));
                    tractionCollPtsY.push_back(std::get<0>(collPts_.second)[0]);
                }
                
            }
            else if (side == 3) {
                // check if diri_sides has only side 1 as side
                if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 1) != diriSidesInt.end() &&
                    std::find(diriSidesInt.begin(), diriSidesInt.end(), 2) == diriSidesInt.end()) {   

                    at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}));
                }
                // check if diri_sides has only side 2 as side
                else if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 1) == diriSidesInt.end() &&
                         std::find(diriSidesInt.begin(), diriSidesInt.end(), 2) != diriSidesInt.end()) {   

                    at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 0, -1));
                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 1}));
                }
                // check if diri_sides has side 1 and side 2
                else if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 1) != diriSidesInt.end() &&
                         std::find(diriSidesInt.begin(), diriSidesInt.end(), 2) != diriSidesInt.end()) {   

                    at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1, -1));
                    tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 2}));
                    diriCtr=2;
                }
                else {
                    tractionCollPtsX.push_back(std::get<0>(collPts_.second)[0]);
                    tractionCollPtsY.push_back(torch::zeros(nrCollPts_));
                }
            }
            else if (side == 4) {
                // check if diri_sides has only side 1 as side
                if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 1) != diriSidesInt.end() &&
                    std::find(diriSidesInt.begin(), diriSidesInt.end(), 2) == diriSidesInt.end()) {   

                    at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1));
                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}));
                }
                // check if diri_sides has only side 2 as side
                else if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 1) == diriSidesInt.end() &&
                         std::find(diriSidesInt.begin(), diriSidesInt.end(), 2) != diriSidesInt.end()) {   

                    at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 0, -1));
                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 1}));
                }
                // check if diri_sides has side 1 and side 2
                else if (std::find(diriSidesInt.begin(), diriSidesInt.end(), 1) != diriSidesInt.end() &&
                         std::find(diriSidesInt.begin(), diriSidesInt.end(), 2) != diriSidesInt.end()) {   

                    at::Tensor collPtsX_tensor = std::get<0>(collPts_.second)[0];
                    tractionCollPtsX.push_back(collPtsX_tensor.slice(0, 1, -1));
                    tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 2}));
                    diriCtr=2;
                }
                else {
                    tractionCollPtsX.push_back(std::get<0>(collPts_.second)[0]);
                    tractionCollPtsY.push_back(torch::ones(nrCollPts_));
                }
            }
            
            else {
                throw std::invalid_argument("Side for traction BC has to be 1, 2, 3 or 4.");
            }
        }
        // merge the tensors to get a (nrTractionCollPts, 2) tensor
        std::array<torch::Tensor, 2ul> tractionCollPts;
        if (!tractionCollPtsX.empty() && !tractionCollPtsY.empty()) {
            tractionCollPts = {
                torch::cat(tractionCollPtsX, 0), 
                torch::cat(tractionCollPtsY, 0)
            };
        }   

        // calculate the derivatives at the considered boundary points                                     
        auto jacobianBoundary = Base::u_.ijac(Base::G_, tractionCollPts);
        auto ux_x = *jacobianBoundary[0];
        auto ux_y = *jacobianBoundary[1];
        auto uy_x = *jacobianBoundary[2];
        auto uy_y = *jacobianBoundary[3];

        // allocate tensors for the traction-free boundary conditions (tfbc)
        torch::Tensor tractionValuesX = torch::zeros({tractionCollPts[0].size(0)});
        torch::Tensor tractionValuesY = torch::zeros({tractionCollPts[0].size(0)});

        // calculate the traction values at the boundary points
        int64_t ctr = 0;
        for (int side : neumannSides) {
            for(int i=ctr*(nrCollPts_-diriCtr); i<(ctr+1)*(nrCollPts_-diriCtr); ++i) {
                // traction-free condition for linear elasticity
                if (side == 1) {
                    tractionValuesX[i] =  - lambda_ * (ux_x[i] + uy_y[i]) - 2 * mu_ * ux_x[i];
                    tractionValuesY[i] =  - mu_ * (uy_x[i] + ux_y[i]);
                }
                else if (side == 2) {
                    tractionValuesX[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * ux_x[i];
                    tractionValuesY[i] = mu_ * (uy_x[i] + ux_y[i]);
                }
                else if (side == 3) {
                    tractionValuesX[i] =  - mu_ * (uy_x[i] + ux_y[i]);
                    tractionValuesY[i] =  - lambda_ * (ux_x[i] + uy_y[i]) - 2 * mu_ * uy_y[i];
                }
                else if (side == 4) {
                    tractionValuesX[i] = mu_ * (uy_x[i] + ux_y[i]);
                    tractionValuesY[i] = lambda_ * (ux_x[i] + uy_y[i]) + 2 * mu_ * uy_y[i];
                }
            }
            ctr++;
        }
        ctr = 0;

        // merge the traction tensors of x- and y-directions
        torch::Tensor tractionValues = torch::stack({tractionValuesX, tractionValuesY}, 1);


        // WARNING: FORCE SIDES DOESN'T WORK CORRECTLY AT THE MOMENT

        if (!FORCE_SIDES_.empty()) {
            // cut the tensors to separate between the values of the traction-free sides and the force sides
            int cutlength = FORCE_SIDES_.size() * (nrCollPts_-diriCtr);
            // traction values for the traction-free sides
            tractionFreeValues.emplace(tractionValues.slice(0, 0, tractionValues.size(0)-cutlength)); 
            // target values (0) for the traction-free sides
            tractionZeros.emplace(torch::zeros_like(*tractionFreeValues)); 
            // force values for the force sides
            forceValues.emplace(tractionValues.slice(0, tractionValues.size(0)-cutlength, tractionValues.size(0)));
            // target values for the force sides
            targetForce.emplace(torch::zeros_like(*forceValues));

            for (const auto& side : FORCE_SIDES_) {
                // setting x-values (first column)
                (*targetForce).slice(0, ctr * (nrCollPts_-diriCtr), (ctr + 1) * (nrCollPts_-diriCtr))   // take first few rows
                            .slice(1, 0, 1)                                                             // only take first column
                            .fill_(std::get<1>(side));                                                  // fill with x-force value

                // setting y-values (second column)
                (*targetForce).slice(0, ctr * (nrCollPts_-diriCtr), (ctr + 1) * (nrCollPts_-diriCtr))   // take first few rows
                            .slice(1, 1, 2)                                                             // only take second column
                            .fill_(std::get<2>(side));                                                  // fill with y-force value
                ctr++;
            }
            ctr = 0;
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
    auto hessianColl = Base::u_.ihess(Base::G_, interiorCollPts_.first);

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
        divStressX[i] = (lambda_ + 2 * mu_) * ux_xx[i] + mu_ * ux_yy[i] + (lambda_ + mu_) * uy_xy[i];

        // y-direction
        divStressY[i] = mu_ * uy_xx[i] + (lambda_ + 2 * mu_) * uy_yy[i] + (lambda_ + mu_) * ux_xy[i];
        
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
            int bcWeight = 1e9;
            // initialize bcLoss variable
            bcLoss = torch::tensor(0.0);

            // evaluation of the displacements at the boundary points
            auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
            // evaluation of the displacements at the reference boundary points
            auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

            // loop through all dirichlet sides
            for (const auto& side : DIRI_SIDES_) {
                int sideNr = std::get<0>(side)-1;
                
                switch (sideNr) {
                    case 0: 
                        *bcLoss += bcWeight * (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                                              torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                        break;
                    case 1:
                        *bcLoss += bcWeight * (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                                              torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                        break;
                    case 2:
                        *bcLoss += bcWeight * (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                                              torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                        break;
                    case 3:
                        *bcLoss += bcWeight * (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
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
        std::cout << std::setw(11) << totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
    }
    
    // SUPERVISED LEARNING
    else if (SUPERVISED_LEARNING_ == true) {
        
        // create command line output variable for all the different losses
        std::ostringstream singleLossOutput;

        // preprocess the outputs for comparison with the G+Smo solution
        torch::Tensor modifiedOutputs = outputs * 1.0;

        // create netDisplacements_ from slices of modifiedOutputs
        torch::Tensor netDisplacements_ = torch::stack({
            modifiedOutputs.slice(0, 0, outputs.size(0) / 2),
            modifiedOutputs.slice(0, outputs.size(0) / 2, outputs.size(0)),
        }, 1);

        // create new tensor with requires_grad=true for training
        auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

        // transforming matrix into row vector for tensor creation
        int gsRows = gsDisplacements_.rows();
        int gsCols = gsDisplacements_.cols();
        std::vector<double> data_gs(gsRows * gsCols);

        for (int col = 0; col < gsCols; ++col) {
            for (int row = 0; row < gsRows; ++row) {
                data_gs[row * gsCols + col] = gsDisplacements_(row, col);
            }
        }

        // ´creating tensor from the transformed data
        torch::Tensor torchGsDisplacements = torch::from_blob(data_gs.data(),
            {gsRows, gsCols}, options).clone();

        // calculation of the supervised loss
        gsLoss = torch::mse_loss(netDisplacements_, torchGsDisplacements);

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
            int bcWeight = 1e9;
            // initialize bcLoss variable
            bcLoss = torch::tensor(0.0);

            // evaluation of the displacements at the boundary points
            auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(collPts_.second);
            // evaluation of the displacements at the reference boundary points
            auto bdr = ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

            // loop through all dirichlet sides
            for (const auto& side : DIRI_SIDES_) {
                int sideNr = std::get<0>(side) - 1;

                switch (sideNr) {
                    case 0:
                        *bcLoss += bcWeight * (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                        break;
                    case 1:
                        *bcLoss += bcWeight * (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                        break;
                    case 2:
                        *bcLoss += bcWeight * (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                        break;
                    case 3:
                        *bcLoss += bcWeight * (torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]) + 
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
        std::cout << std::setw(11) << totalLoss.item<double>() << " = " << singleLossOutput.str() << std::endl;
    }

    else {
        throw std::runtime_error("Invalid value for SUPERVISED_LEARNING_");
    }

    // POSTPROCESSING PREPARATION - WRITING DATA TO JSON FILE

    // only calculate this at the end of the simulation
    if ((epoch == MAX_EPOCH_ - 1) || (totalLoss.item<double>() <= MIN_LOSS_)) {
        
        // TRACTION FREE BOUNDARY CONDITIONS

        nlohmann::json tractionX = nlohmann::json::array();
        nlohmann::json tractionY = nlohmann::json::array();

        for (int i = 0; i < (*tractionFreeValues).size(0); ++i) {
            tractionX.push_back({(*tractionFreeValues)[i][0].item<double>()});
            tractionY.push_back({(*tractionFreeValues)[i][1].item<double>()});
        }
        appendToJsonFile("tractionXRef1", tractionX);
        appendToJsonFile("tractionYRef1", tractionY);

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
        torch::Tensor collPtsFirstAsTensor = torch::stack({std::get<0>(collPts_.first), std::get<1>(collPts_.first)}, 1);
        auto displacementOfCollPts = Base::u_.eval(collPts_.first);
        torch::Tensor displacementAsTensor = torch::stack({*(displacementOfCollPts[0]), *(displacementOfCollPts[1]) }, 1);

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
  int MAX_EPOCH = 150;
  double MIN_LOSS = 1e-8;
  bool SUPERVISED_LEARNING = false;

  // spline parameters
  int64_t NR_CTRL_PTS = 8;  // in each direction 
  constexpr int DEGREE = 4;

  // boundary conditions
  std::vector<std::tuple<int, double, double>> FORCE_SIDES = {
    //   {2, 100.0,  0.0},     // {side, x-traction, y-traction}
    };
  std::vector<std::tuple<int, double, double>> DIRI_SIDES = {
      {1, 0.0,  0.0},       // {side, x-displ, y-displ}
      {2, 1.0,  0.0},
    };
  std::vector<int> TFBC_SIDES = {3, 4}; // {sides}
    
  // --------------------------- //


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
  std::tie(gsCtrlPts, gsDisplacements, gsStresses) = 
    linear_elasticity_t::RunGismoSimulation(NR_CTRL_PTS, DEGREE, YOUNG_MODULUS, POISSON_RATIO);

  linear_elasticity_t
      net(// simulation parameters
          lambda, mu, SUPERVISED_LEARNING, MAX_EPOCH, MIN_LOSS, 
          TFBC_SIDES, FORCE_SIDES, DIRI_SIDES, gsDisplacements,
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

  // imposing body force
  net.f().transform([=](const std::array<real_t, 2> xi) {
    // body force {f_x, f_y}
    return std::array<real_t, 2>{0, 0};
  });

  // get the coefficients of the control points
  torch::Tensor ctrlPtsCoeffs = net.G().as_tensor().slice(0, 0, NR_CTRL_PTS);
  nlohmann::json ctrlPtsCoeffs_j = nlohmann::json::array();
  for (int i = 0; i < NR_CTRL_PTS; ++i) {
      ctrlPtsCoeffs_j.push_back({ctrlPtsCoeffs[i].item<double>()});
  }
  linear_elasticity_t::appendToJsonFile("ctrlPtsCoeffs", ctrlPtsCoeffs_j);

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

  // write G+Smo data from the matrices to the json objects
  for (int i = 0; i < displacedGsCtrlPts.rows(); ++i) {
      // new control points G+Smo
      displacedGsCtrlPts_j.push_back({displacedGsCtrlPts(i, 0), displacedGsCtrlPts(i, 1)});
      // write the von Mises stresses to the json object (calculated at the beginning of the main function)
      gsStresses_j.push_back({gsStresses(i, 0)});
  }
  
  // write net data from the matrices to the json objects
  for (int i = 0; i < displacedNetCtrlPts.rows(); ++i) {
      // new control points IgANet
      displacedNetCtrlPts_j.push_back({displacedNetCtrlPts(i, 0), displacedNetCtrlPts(i, 1)});
  }

  // write data to the json file
  linear_elasticity_t::appendToJsonFile("gsCtrlPts", displacedGsCtrlPts_j);
  linear_elasticity_t::appendToJsonFile("netCtrlPts", displacedNetCtrlPts_j);
  linear_elasticity_t::appendToJsonFile("gsStresses", gsStresses_j);
  linear_elasticity_t::appendToJsonFile("degree", DEGREE);
  
  iganet::finalize();
  return 0;
}
