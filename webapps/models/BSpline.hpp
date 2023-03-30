/**
   @file models/BSpline.hpp

   @brief BSpline model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <modelmanager.hpp>

namespace iganet {

  namespace model {

    /// @brief Enumerator for specifying the degree of B-splines
    enum class degree : short_t
      {
        constant  = 0, /*!<  constant B-Spline basis functions */
        linear    = 1, /*!<    linear B-Spline basis functions */
        quadratic = 2, /*!< quadratic B-Spline basis functions */
        cubic     = 3, /*!<     cubic B-Spline basis functions */
        quartic   = 4, /*!<   quartic B-Spline basis functions */
        quintic   = 5  /*!<   quintic B-Spline basis functions */
      };
  
    /// @brief B-spline model
    template<class BSpline_t>
    class BSplineModel : public Model,
                         public ModelEval<BSpline_t::geoDim()>,
                         public ModelRefine,
                         public BSpline_t {
    public:
      /// @brief Default constructor
      BSplineModel() = default;

      /// @brief Constructor for equidistant knot vectors
      BSplineModel(const std::array<int64_t, BSpline_t::parDim()> ncoeffs,
                   enum iganet::init init = iganet::init::zeros)
        : BSpline_t(ncoeffs, init)
      {}
      
      /// @brief Destructor
      ~BSplineModel() {}
      
      /// @brief Returns the model's name
      std::string getName() const override {
        if constexpr (BSpline_t::parDim() == 1)
          return "BSplineCurve";
        else if constexpr (BSpline_t::parDim() == 2)
          return "BSplineSurface";
        else if constexpr (BSpline_t::parDim() == 3)
          return "BSplineVolume";
        else if constexpr (BSpline_t::parDim() == 4)
          return "BSplineHyperVolume";
        else
          return "{ INVALID REQUEST }";
      }
      
      /// @brief Returns the model's description
      std::string getDescription() const override {
        if constexpr (BSpline_t::parDim() == 1)
          return "B-spline curve";
        else if constexpr (BSpline_t::parDim() == 2)
          return "B-spline surface";
        else if constexpr (BSpline_t::parDim() == 3)
          return "B-spline volume";
        else if constexpr (BSpline_t::parDim() == 4)
          return "B-spline hypervolume";
        else
          return "{ INVALID REQUEST }";        
      }

      /// @brief Returns the model's Options
      std::string getOptions() const override {
        if constexpr (BSpline_t::parDim() == 1)
          return "["
            "{\"name\" : \"degree\","
            " \"description\" : \"Polynomial degree of the B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"constant\", \"linear\", \"quadratic\", \"cubic\", \"quartic\", \"quintic\"],"
            " \"uiid\" : 0},"
            "{\"name\" : \"ncoeffs\","
            " \"description\" : \"Number of coefficients per parametric dimension\","
            " \"type\" : [\"int\"],"
            " \"value\" : [1],"
            " \"uiid\" : 1},"
            "{\"name\" : \"init\","
            " \"description\" : \"Initialization of the coefficients\","
            " \"type\" : \"select\","
            " \"value\" : [\"zeros\", \"ones\", \"linear\", \"random\", \"greville\"],"
            " \"uiid\" : 2},"
            "{\"name\" : \"nonuniform\","
            " \"description\" : \"Create non-uniform B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"false\", \"true\"],"
            " \"uiid\" : 3}"
            "]";
        else if constexpr (BSpline_t::parDim() == 2)
          return "["
            "{\"name\" : \"degree\","
            " \"description\" : \"Polynomial degree of the B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"constant\", \"linear\", \"quadratic\", \"cubic\", \"quartic\", \"quintic\"],"
            " \"uiid\" : 0},"
            "{\"name\" : \"ncoeffs\","
            " \"description\" : \"Number of coefficients per parametric dimension\","
            " \"type\" : [\"int\",\"int\"],"
            " \"value\" : [1,1],"
            " \"uiid\" : 1},"
            "{\"name\" : \"init\","
            " \"description\" : \"Initialization of the coefficients\","
            " \"type\" : \"select\","
            " \"value\" : [\"zeros\", \"ones\", \"linear\", \"random\", \"greville\"],"
            " \"uiid\" : 2},"
            "{\"name\" : \"nonuniform\","
            " \"description\" : \"Create non-uniform B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"false\", \"true\"],"
            " \"uiid\" : 3}"
            "]";
        else if constexpr (BSpline_t::parDim() == 3)
          return "["
            "{\"name\" : \"degree\","
            " \"description\" : \"Polynomial degree of the B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"constant\", \"linear\", \"quadratic\", \"cubic\", \"quartic\", \"quintic\"],"
            " \"uiid\" : 0},"
            "{\"name\" : \"ncoeffs\","
            " \"description\" : \"Number of coefficients per parametric dimension\","
            " \"type\" : [\"int\",\"int\",\"int\"],"
            " \"value\" : [1,1,1],"
            " \"uiid\" : 1},"
            "{\"name\" : \"init\","
            " \"description\" : \"Initialization of the coefficients\","
            " \"type\" : \"select\","
            " \"value\" : [\"zeros\", \"ones\", \"linear\", \"random\", \"greville\"],"
            " \"uiid\" : 2},"
            "{\"name\" : \"nonuniform\","
            " \"description\" : \"Create non-uniform B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"false\", \"true\"],"
            " \"uiid\" : 3}"
            "]";
        else if constexpr (BSpline_t::parDim() == 4)
          return "["
            "{\"name\" : \"degree\","
            " \"description\" : \"Polynomial degree of the B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"constant\", \"linear\", \"quadratic\", \"cubic\", \"quartic\", \"quintic\"],"
            " \"uiid\" : 0},"
            "{\"name\" : \"ncoeffs\","
            " \"description\" : \"Number of coefficients per parametric dimension\","
            " \"type\" : [int,int,int,int],"
            " \"value\" : [1,1,1,1],"
            " \"uiid\" : 1},"
            "{\"name\" : \"init\","
            " \"description\" : \"Initialization of the coefficients\","
            " \"type\" : \"select\","
            " \"value\" : [\"zeros\", \"ones\", \"linear\", \"random\", \"greville\"],"
            " \"uiid\" : 2},"
            "{\"name\" : \"nonuniform\","
            " \"description\" : \"Create non-uniform B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"false\", \"true\"],"
            " \"uiid\" : 3}"
            "]";
        else
          return "{ INVALID REQUEST }";
      }
    
      /// @brief Serializes the model to JSON
      nlohmann::json to_json(const std::string& attribute = "") const override {
        if (attribute != "") {
          nlohmann::json data;
          if (attribute == "degrees")
            data["degrees"] = this->degrees();
          else if (attribute == "geoDim")
            data["geoDim"] = this->geoDim();
          else if (attribute == "parDim")
            data["parDim"] = this->parDim();
          else if (attribute == "ncoeffs")
            data["ncoeffs"] = this->ncoeffs();
          else if (attribute == "nknots")
            data["nknots"] = this->nknots();
          else if (attribute == "knots")
            data["knots"] = this->knots_to_json();
          else if (attribute == "coeffs")
            data["coeffs"] = this->coeffs_to_json();
          return data;
        } else
          return BSpline_t::to_json();
      }

      /// @brief Updates the attrbutes of the model
      nlohmann::json updateAttribute(const std::string& attribute,
                                     const nlohmann::json& json) override {

        std::cout << "attribute: " << attribute << std::endl;
        std::cout << json.dump() << std::endl;
        
        if (attribute == "coeffs") {
          if (!json.contains("data"))
            throw InvalidModelAttributeException();
          if (!json["data"].contains("indices") ||
              !json["data"].contains("coeffs"))
            throw InvalidModelAttributeException();
          
          auto indices    = json["data"]["indices"].get<std::vector<int64_t>>();
          auto coeffs_cpu = to_tensorAccessor<typename BSpline_t::value_type,1>(BSpline_t::coeffs(), torch::kCPU);
          
          switch (BSpline_t::geoDim()) {
          case (1): {
            auto coords    = json["data"]["coeffs"].get<std::vector<std::tuple<typename BSpline_t::value_type>>>();
            auto xAccessor = std::get<1>(coeffs_cpu)[0];

            for (const auto& [index, coord] : iganet::zip(indices, coords)) {
              if (index < 0 || index >= BSpline_t::ncumcoeffs())
                throw IndexOutOfBoundsException();
              xAccessor[index] = std::get<0>(coord);
            }
            break;
          }
          case (2): {
            auto coords    = json["data"]["coeffs"].get<std::vector<std::tuple<typename BSpline_t::value_type,
                                                                               typename BSpline_t::value_type>>>();
            auto xAccessor = std::get<1>(coeffs_cpu)[0];
            auto yAccessor = std::get<1>(coeffs_cpu)[1];

            for (const auto& [index, coord] : iganet::zip(indices, coords)) {
              if (index < 0 || index >= BSpline_t::ncumcoeffs())
                throw IndexOutOfBoundsException();
              
              xAccessor[index] = std::get<0>(coord);
              yAccessor[index] = std::get<1>(coord);
            }
            break;
          }
          case (3): {
            auto coords    = json["data"]["coeffs"].get<std::vector<std::tuple<typename BSpline_t::value_type,
                                                                               typename BSpline_t::value_type,
                                                                               typename BSpline_t::value_type>>>();
            auto xAccessor = std::get<1>(coeffs_cpu)[0];
            auto yAccessor = std::get<1>(coeffs_cpu)[1];
            auto zAccessor = std::get<1>(coeffs_cpu)[2];

            for (const auto& [index, coord] : iganet::zip(indices, coords)) {
              if (index < 0 || index >= BSpline_t::ncumcoeffs())
                throw IndexOutOfBoundsException();
              
              xAccessor[index] = std::get<0>(coord);
              yAccessor[index] = std::get<1>(coord);
              zAccessor[index] = std::get<2>(coord);
            }
            break;
          }
          case (4): {
            auto coords    = json["data"]["coeffs"].get<std::vector<std::tuple<typename BSpline_t::value_type,
                                                                               typename BSpline_t::value_type,
                                                                               typename BSpline_t::value_type,
                                                                               typename BSpline_t::value_type>>>();
            auto xAccessor = std::get<1>(coeffs_cpu)[0];
            auto yAccessor = std::get<1>(coeffs_cpu)[1];
            auto zAccessor = std::get<1>(coeffs_cpu)[2];
            auto tAccessor = std::get<1>(coeffs_cpu)[3];

            for (const auto& [index, coord] : iganet::zip(indices, coords)) {
              if (index < 0 || index >= BSpline_t::ncumcoeffs())
                throw IndexOutOfBoundsException();
              
              xAccessor[index] = std::get<0>(coord);
              yAccessor[index] = std::get<1>(coord);
              zAccessor[index] = std::get<2>(coord);
              tAccessor[index] = std::get<3>(coord);
            }
            break;
          }
          default:
            throw InvalidModelAttributeException();
          }
          return to_json("coeffs");
        }
        else
          return "{ INVALID REQUEST }";
      }
      
      /// @brief Evaluates the model
      BlockTensor<torch::Tensor, 1, 1> eval(const nlohmann::json& json = NULL) const override {
        if constexpr (BSpline_t::parDim() == 1) {

          std::array<int64_t, 1> res({25});          
          if (json.contains("data"))
            if (json["data"].contains("resolution"))
              res = json["data"]["resolution"].get<std::array<int64_t,1>>();
          
          iganet::TensorArray1 xi = {torch::linspace(0, 1, res[0])};
          return BSpline_t::eval(xi);
        }
        else if constexpr (BSpline_t::parDim() == 2) {

          std::array<int64_t, 2> res({25, 25});          
          if (json.contains("data"))
            if (json["data"].contains("resolution"))
              res = json["data"]["resolution"].get<std::array<int64_t,2>>();
          
          iganet::TensorArray2 xi = {torch::linspace(0, 1, res[0]),
                                     torch::linspace(0, 1, res[1])};
          return BSpline_t::eval(xi);
        }
        else if constexpr (BSpline_t::parDim() == 3) {

          std::array<int64_t, 3> res({25, 25, 25});          
          if (json.contains("data"))
            if (json["data"].contains("resolution"))
              res = json["data"]["resolution"].get<std::array<int64_t,3>>();
          
          iganet::TensorArray3 xi = {torch::linspace(0, 1, res[0]),
                                     torch::linspace(0, 1, res[1]),
                                     torch::linspace(0, 1, res[2])};
          return BSpline_t::eval(xi);
        }
        else if constexpr (BSpline_t::parDim() == 4) {

          std::array<int64_t, 4> res({25, 25, 25, 25});          
          if (json.contains("data"))
            if (json["data"].contains("resolution"))
              res = json["data"]["resolution"].get<std::array<int64_t,4>>();
          
          iganet::TensorArray4 xi = {torch::linspace(0, 1, res[0]),
                                     torch::linspace(0, 1, res[1]),
                                     torch::linspace(0, 1, res[2]),
                                     torch::linspace(0, 1, res[3])};
          return BSpline_t::eval(xi);
        }
      }

      /// @brief Refines the model
      void refine(const nlohmann::json& json = NULL) override {
        int numRefine = 1, dim = -1;

        if (json.contains("data")) {
          if (json["data"].contains("numRefine"))
            numRefine = json["data"]["numRefine"].get<int>();
          
          if (json["data"].contains("dim"))
            dim = json["data"]["dim"].get<int>();
        }

        BSpline_t::uniform_refine(numRefine, dim);        
      }
    };    

  } // namespace model    
} // namespace iganet
