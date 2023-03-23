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
    class BSplineModel : public ModelEval<BSpline_t::geoDim(), BSpline_t::parDim()>,
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
          else if (attribute == "geoDim")
            data["nknots"] = this->nknots();
          else if (attribute == "knots")
            data["knots"] = this->knots_to_json();
          else if (attribute == "coeffs")
            data["coeffs"] = this->coeffs_to_json();
          return data;
        } else
          return BSpline_t::to_json();
      }

      /// @brief Evaluates the model
      BlockTensor<torch::Tensor, 1, 1> eval(const nlohmann::json& config = NULL) const override {
        if constexpr (BSpline_t::parDim() == 1) {
          iganet::TensorArray1 xi = {torch::linspace(0,1,100)};
          return BSpline_t::eval(xi);
        }
        else if constexpr (BSpline_t::parDim() == 2) {
          iganet::TensorArray2 xi = {torch::linspace(0,1,100),
                                     torch::linspace(0,1,100)};
          return BSpline_t::eval(xi);
        }
        else if constexpr (BSpline_t::parDim() == 3) {
          iganet::TensorArray3 xi = {torch::linspace(0,1,100),
                                     torch::linspace(0,1,100),
                                     torch::linspace(0,1,100)};
          return BSpline_t::eval(xi);
        }
        else if constexpr (BSpline_t::parDim() == 4) {
          iganet::TensorArray4 xi = {torch::linspace(0,1,100),
                                     torch::linspace(0,1,100),
                                     torch::linspace(0,1,100),
                                     torch::linspace(0,1,100)};
          return BSpline_t::eval(xi);
        }
      }
    };    

  } // namespace model    
} // namespace iganet
