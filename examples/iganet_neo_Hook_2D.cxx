#include <iganet.h>
#include <iostream>
#include <fstream>

using namespace iganet::literals;
using namespace gismo;

/// @brief Specialization of the IgANet class for non linear neo-Hookean material
template <typename Optimizer, typename GeometryMap, typename Variable>
class neo_Hook : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
                          public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  typename Base::variable_collPts_type collPts_;
  typename Base::variable_collPts_type interiorCollPts_;
  std::array<torch::Tensor, 2ul> tractionCollPts_;

  int nrCollPts_;
  Variable ref_;

  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_tf_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_tf_;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_interior_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_interior_;

  typename Customizable::variable_interior_knot_indices_type var_knot_indices_boundary_;
  typename Customizable::variable_interior_coeff_indices_type var_coeff_indices_boundary_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_tf_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_tf_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_interior_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_interior_;

  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_boundary_;
  typename Customizable::geometryMap_interior_coeff_indices_type G_coeff_indices_boundary_;

  // material properties - lame's parameters
  double lambda_;
  double mu_;

  // simulation parameter
  double MAX_EPOCH_;
  double MIN_LOSS_;
  std::vector<std::tuple<int, double, double>> DIRI_SIDES_;

  // another set of scaling parameters
  double DIRICHLET_SCALING = 1.0;
  at::Tensor previous_loss = torch::tensor(-100.0);
  at::Tensor scaling_threshold = torch::tensor(MIN_LOSS_ * 10.0);

  // json path
  static constexpr const char* JSON_PATH = "/home/chg/Programming/PythonNet_IGA/Network_V1/NeuralNet/results.json";

public:
  /// @brief Constructor
  template <typename... Args>
  neo_Hook(double lambda, double mu, double MAX_EPOCH, 
                    double MIN_LOSS,
                    std::vector<std::tuple<int, double, double>> DIRI_SIDES, 
                    std::vector<int64_t> &&layers,
                    std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        lambda_(lambda), mu_(mu), MAX_EPOCH_(MAX_EPOCH), 
        MIN_LOSS_(MIN_LOSS), DIRI_SIDES_(DIRI_SIDES), 
        ref_(iganet::utils::to_array(8_i64, 8_i64)) {
            this->initialize_dirichlet_boundaries();
        }

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the interior collocation points
  auto const &interiorCollPts() const { return interiorCollPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

  void initialize_dirichlet_boundaries() {
    // run through all DIRI_SIDES and registr respective lambdas
    for (const auto& side : this->DIRI_SIDES_) {
        int sideNr = std::get<0>(side);
        double xDispl = std::get<1>(side);
        double yDispl = std::get<2>(side);

        switch (sideNr) {
            case 1:
                this->ref_.boundary().template side<1>().template transform<1>(
                    [this, xDispl](auto const& xi) {
                        return std::array<double, 1>{DIRICHLET_SCALING * xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                this->ref_.boundary().template side<1>().template transform<1>(
                    [this, yDispl](auto const& xi) {
                        return std::array<double, 1>{DIRICHLET_SCALING * yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 2:
                this->ref_.boundary().template side<2>().template transform<1>(
                    [this, xDispl](auto const& xi) {
                        return std::array<double, 1>{DIRICHLET_SCALING * xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                    this->ref_.boundary().template side<2>().template transform<1>(
                    [this, yDispl](auto const& xi) {
                        return std::array<double, 1>{DIRICHLET_SCALING * yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 3:
                this->ref_.boundary().template side<3>().template transform<1>(
                    [this, xDispl](auto const& xi) {
                        return std::array<double, 1>{DIRICHLET_SCALING * xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                    this->ref_.boundary().template side<3>().template transform<1>(
                    [this, yDispl](auto const& xi) {
                        return std::array<double, 1>{DIRICHLET_SCALING * yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            case 4:
                this->ref_.boundary().template side<4>().template transform<1>(
                    [this, xDispl](auto const& xi) {
                        return std::array<double, 1>{DIRICHLET_SCALING * xDispl};
                    },
                    std::array<iganet::short_t, 1>{0} 
                );
                    this->ref_.boundary().template side<4>().template transform<1>(
                    [this, yDispl](auto const& xi) {
                        return std::array<double, 1>{DIRICHLET_SCALING * yDispl};
                    },
                    std::array<iganet::short_t, 1>{1}
                );
                break;
            default:
                std::cerr << "Error: Invalid side number " << sideNr << std::endl;
        }
    }
  }
  
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

  void write_result() {
    // write geometry and solution spline data to file
    appendToJsonFile("G", Base::G_.template to_json());
    appendToJsonFile("u", Base::u_.template to_json());
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

    // WARNING! Neumann Loss is hardcoded right now for top and bottom boundary
      std::vector<torch::Tensor> tractionCollPtsX;
      std::vector<torch::Tensor> tractionCollPtsY;
      at::Tensor collPts_temp = std::get<0>(collPts_.second)[0];
      tractionCollPtsX.push_back(collPts_temp.slice(0, 1, -1));
      tractionCollPtsY.push_back(torch::zeros({nrCollPts_ - 2}));
      tractionCollPtsX.push_back(collPts_temp.slice(0, 1, -1));
      tractionCollPtsY.push_back(torch::ones({nrCollPts_ - 2}));
      tractionCollPts_ = {
        torch::cat(tractionCollPtsX, 0), 
        torch::cat(tractionCollPtsY, 0)};


      torch::Tensor collPtsCoeffs = std::get<0>(collPts_)[0].slice(0, 0, nrCollPts_);
      nlohmann::json collPtsCoeffs_j = nlohmann::json::array();
      for (int i = 0; i < collPtsCoeffs.size(0); ++i) {
          collPtsCoeffs_j.push_back({collPtsCoeffs[i].item<double>()});
      }

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

    var_knot_indices_tf_ =
        Base::f_.template find_knot_indices<iganet::functionspace::interior>(
        tractionCollPts_);
    var_coeff_indices_tf_ =
        Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
        var_knot_indices_tf_);
    G_knot_indices_tf_ =
        Base::G_.template find_knot_indices<iganet::functionspace::interior>(
            tractionCollPts_);
    G_coeff_indices_tf_ =
    Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
        G_knot_indices_tf_);

      return true;
    } else if (epoch == MAX_EPOCH_-1) {
        // write geometry and solution spline data to file
        appendToJsonFile("G", Base::G_.template to_json());
        appendToJsonFile("u", Base::u_.template to_json());
    }
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
  

    // Elasticity Loss

    // calculate the jacobian of the displacements (u) at the collocation points
    auto jacobian = Base::u_.ijac(Base::G_, interiorCollPts_.first, 
        var_knot_indices_interior_, var_coeff_indices_interior_,
        G_knot_indices_interior_, G_coeff_indices_interior_);
    
    auto& u1_x = jacobian(0);
    auto& u1_y = jacobian(1);
    auto& u2_x = jacobian(2);
    auto& u2_y = jacobian(3);

    // calculation of the second derivatives of the displacements (u)
    auto hessianColl = Base::u_.ihess(Base::G_, interiorCollPts_.first, 
        var_knot_indices_interior_, var_coeff_indices_interior_,
        G_knot_indices_interior_, G_coeff_indices_interior_);

    // partial derivatives of the displacements (u)
    auto& u1_xx = hessianColl(0,0,0);
    auto& u1_xy = hessianColl(0,1,0);
    auto& u1_yx = hessianColl(1,0,0);
    auto& u1_yy = hessianColl(1,1,0);

    auto& u2_xx = hessianColl(0,0,1);
    auto& u2_xy = hessianColl(0,1,1);
    auto& u2_yx = hessianColl(1,0,1);
    auto& u2_yy = hessianColl(1,1,1);

    double beta = 2.0;
    // Divergence of first Kirchhoff stress tensor
    auto J = 1.0 + u1_x + u2_y + u1_x*u2_y - u1_y*u2_x;
    auto J_sp = torch::nn::functional::softplus(J, torch::nn::functional::SoftplusFuncOptions()
        .beta(beta)
        .threshold(20.0));
    auto J_x = u1_xx + u2_yx + u1_xx*u2_y +u1_x*u2_yx - u1_yx*u2_x - u1_y*u2_xx;
    auto J_y = u1_xy + u2_yy + u1_xy*u2_y +u1_x*u2_yy - u1_yy*u2_x - u1_y*u2_xy;
    auto J_sp_x = J_x * torch::sigmoid(beta * J);
    auto J_sp_y = J_y * torch::sigmoid(beta * J);
    auto A = (lambda_ * torch::log1p(J_sp - 1) - mu_) / J_sp;
    auto A_x = (lambda_ + mu_ - lambda_*torch::log1p(J_sp - 1)) * J_sp_x / (J_sp*J_sp);
    auto A_y = (lambda_ + mu_ - lambda_*torch::log1p(J_sp - 1)) * J_sp_y / (J_sp*J_sp);

    // First derivatives of first Piola Kirchhoff stress tensor
    auto P11_x = mu_*u1_xx + A_x + A_x*u2_y + A*u2_yx;
    auto P12_y = mu_*u1_yy - A_y*u2_x - A*u2_xy;
    auto P21_x = mu_*u2_xx - A_x*u1_y - A*u1_yx;
    auto P22_y = mu_*u2_yy + A_y + A_y*u1_x + A*u1_xy;

    // Divergence of first PK tensor
    auto divPX = P11_x + P12_y;
    auto divPY = P21_x + P22_y;
    auto divP  = torch::stack({divPX, divPY}, 1);

    // BODY FORCE

    // evaluate the reference body force f at all interior collocation points
    auto f = Base::f_.eval(interiorCollPts_.first);

    auto bodyForce = torch::stack({*f[0], *f[1]}, 1).to(divP.dtype());

    // create command line output variable for all the different losses
    std::ostringstream singleLossOutput;

    // calculation of the loss function for double-sided constraint solid
    // div(sigma) + f = 0 --> div(sigma) = -f
    elastLoss = torch::mse_loss(divP, bodyForce);
    
    // add the elasticity loss to the total loss
    totalLoss = elastLoss;

    // add the elasticity loss to the cmd-output variable
    singleLossOutput << "EL " << std::setw(11) << elastLoss.item<double>();


    // ------------------------------ Neumann Loss ------------------------------------
    // WARNING! Neumann Loss is hardcoded right now for top and bottom boundary

    // calculate the jacobian of the displacements (u) at the collocation points
    auto jacobian_tf = Base::u_.ijac(Base::G_, tractionCollPts_, 
        var_knot_indices_tf_, var_coeff_indices_tf_,
        G_knot_indices_tf_, G_coeff_indices_tf_);
    
    auto& u1_x_tf = jacobian_tf(0);
    auto& u1_y_tf = jacobian_tf(1);
    auto& u2_x_tf = jacobian_tf(2);
    auto& u2_y_tf = jacobian_tf(3);

    // calculation of the second derivatives of the displacements (u)
    auto hessianColl_tf = Base::u_.ihess(Base::G_, tractionCollPts_, 
        var_knot_indices_tf_, var_coeff_indices_tf_,
        G_knot_indices_tf_, G_coeff_indices_tf_);

    // partial derivatives of the displacements (u)
    auto& u1_xx_tf = hessianColl_tf(0,0,0);
    auto& u1_xy_tf = hessianColl_tf(0,1,0);
    auto& u1_yx_tf = hessianColl_tf(1,0,0);
    auto& u1_yy_tf = hessianColl_tf(1,1,0);

    auto& u2_xx_tf = hessianColl_tf(0,0,1);
    auto& u2_xy_tf = hessianColl_tf(0,1,1);
    auto& u2_yx_tf = hessianColl_tf(1,0,1);
    auto& u2_yy_tf = hessianColl_tf(1,1,1);

    // Divergence of first Kirchhoff stress tensor
    auto J_tf = 1.0 + u1_x_tf + u2_y_tf + u1_x_tf*u2_y_tf - u1_y_tf*u2_x_tf;
    auto J_sp_tf = torch::nn::functional::softplus(J_tf, torch::nn::functional::SoftplusFuncOptions()
        .beta(beta)
        .threshold(20.0));
    auto J_x_tf = u1_xx_tf + u2_yx_tf + u1_xx_tf*u2_y_tf +u1_x_tf*u2_yx_tf - u1_yx_tf*u2_x_tf - u1_y_tf*u2_xx_tf;
    auto J_y_tf = u1_xy_tf + u2_yy_tf + u1_xy_tf*u2_y_tf +u1_x_tf*u2_yy_tf - u1_yy_tf*u2_x_tf - u1_y_tf*u2_xy_tf;
    auto J_sp_x_tf = J_x_tf * torch::sigmoid(beta * J_tf);
    auto J_sp_y_tf = J_y_tf * torch::sigmoid(beta * J_tf);
    auto A_tf = (lambda_ * torch::log1p(J_sp_tf - 1) - mu_) / J_sp_tf;
    auto A_x_tf = (lambda_ + mu_ - lambda_*torch::log1p(J_sp_tf - 1)) * J_sp_x_tf / (J_sp_tf*J_sp_tf);
    auto A_y_tf = (lambda_ + mu_ - lambda_*torch::log1p(J_sp_tf - 1)) * J_sp_y_tf / (J_sp_tf*J_sp_tf);

    // Components of first Piola Kirchhoff stress tensor
    auto P11 = mu_ * (1 + u1_x_tf) + A_tf * (1 + u2_y_tf);
    auto P12 = mu_ * u1_y_tf - A_tf * u2_x_tf; 
    auto P21 = mu_ * u2_x_tf - A_tf * u1_y_tf;
    auto P22 = mu_ * (1 + u2_y_tf) + A_tf * (1 + u1_x_tf);

    totalLoss += torch::mse_loss(P12, torch::zeros_like(P12));
    totalLoss += torch::mse_loss(P22, torch::zeros_like(P22));




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
                    *bcLoss += bcWeight * (torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<0>(u_bdr)[1], *std::get<0>(bdr)[1]));
                    break;
                case 2:
                    *bcLoss += bcWeight * (torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<1>(u_bdr)[1], *std::get<1>(bdr)[1]));
                    break;
                case 3:
                    *bcLoss += bcWeight * (torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) + 
                                            torch::mse_loss(*std::get<2>(u_bdr)[1], *std::get<2>(bdr)[1]));
                    break;
                case 4:
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

    if ((torch::abs(this->previous_loss - totalLoss) < this->scaling_threshold).item<bool>() && (this->DIRICHLET_SCALING < 1.0)) {
        this->DIRICHLET_SCALING += 0.02;
        this->initialize_dirichlet_boundaries();
        //this->reset_solver();
        std::cout << "\n\nIncreased Dirichlet scaling factor\n\n" << std::endl;
    }
    this->previous_loss = totalLoss;
    return totalLoss;
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  // ------- USER INPUTS ------- //

  // material parameters
  double YOUNG_MODULUS = 10;
  double POISSON_RATIO = 0.3;

  // simulation parameters
  int MAX_EPOCH = 10000;
  double MIN_LOSS = 5e-4;
  bool SUPERVISED_LEARNING = false;
  bool RUN_REF_SIM = false;

  // spline parameters
  int64_t NR_CTRL_PTS = 6;  // in each direction 
  constexpr int DEGREE = 5;

  // boundary conditions

  std::vector<std::tuple<int, double, double>> DIRI_SIDES = {
      {1, 0.0,  0.0},       // {side, x-displ, y-displ}
      {2, 2.0,  0.0},
    };
    
  // --------------------------- //


  // calculation of lame parameters
  double lambda = (YOUNG_MODULUS * POISSON_RATIO) / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO));
  double mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO));

  using real_t = double;
  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 2, DEGREE, DEGREE>>;
  using neo_Hook_t = neo_Hook<optimizer_t, geometry_t, variable_t>;
  

    neo_Hook_t
      net(// simulation parameters
          lambda, mu, MAX_EPOCH, MIN_LOSS, DIRI_SIDES,
          // Number of neurons per layer
          {72, 72},
          // Activation functions
          {{iganet::activation::tanh},
           {iganet::activation::tanh},
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

  net.write_result();

  iganet::finalize();
  return 0;
}

// penalties for low J
// taylor approximation of ln
// loss function generell nicht für alle werte auswertbar, was macht man da?