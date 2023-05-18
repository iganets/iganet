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

  namespace webapp {

    /// @brief Enumerator for specifying the degree of B-splines
    enum class degree
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
                         public ModelEval,
                         public ModelRefine,
                         public ModelXML,
                         public BSpline_t {
    private:
      BSpline_t solution_;
      
    public:
      /// @brief Default constructor
      BSplineModel() = default;

      /// @brief Constructor for equidistant knot vectors
      BSplineModel(const std::array<int64_t, BSpline_t::parDim()> ncoeffs,
                   enum iganet::init init = iganet::init::zeros)
        : BSpline_t(ncoeffs, init), solution_(ncoeffs, init)
      {
        if constexpr (BSpline_t::parDim() == 1)
          solution_.transform( [](const std::array<typename BSpline_t::value_type,1> xi)
          {
            return std::array<typename BSpline_t::value_type,BSpline_t::geoDim()>{ static_cast<float>(std::sin(M_PI*xi[0])), 0.0, 0.0 };
          } );

        else if constexpr (BSpline_t::parDim() == 2)
          solution_.transform( [](const std::array<typename BSpline_t::value_type,2> xi)
          {
            return std::array<typename BSpline_t::value_type,BSpline_t::geoDim()>{ static_cast<float>(/*std::sin(M_PI*xi[0]) *
                                                                                                        std::sin(M_PI*xi[1])*/xi[0] ), 0.0, 0.0 };
          } );

        else if constexpr (BSpline_t::parDim() == 3)
          solution_.transform( [](const std::array<typename BSpline_t::value_type,3> xi)
          {
            return std::array<typename BSpline_t::value_type,BSpline_t::geoDim()>{ static_cast<float>(std::sin(M_PI*xi[0]) *
                                                                                    std::sin(M_PI*xi[1]) *
                                                                                    std::sin(M_PI*xi[2])), 0.0, 0.0 };
          } );
      }
      
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

      /// @brief Returns the model's options
      std::string getOptions() const override {
        if constexpr (BSpline_t::parDim() == 1)
          return "["
            "{\"name\" : \"degree\","
            " \"description\" : \"Polynomial degree of the B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"constant\", \"linear\", \"quadratic\", \"cubic\", \"quartic\", \"quintic\"],"
            " \"default\" : 2,"
            " \"uiid\" : 0},"
            "{\"name\" : \"ncoeffs\","
            " \"description\" : \"Number of coefficients per parametric dimension\","
            " \"type\" : [\"int\"],"
            " \"value\" : [3],"
            " \"default\" : [3],"
            " \"uiid\" : 1},"
            "{\"name\" : \"init\","
            " \"description\" : \"Initialization of the coefficients\","
            " \"type\" : \"select\","
            " \"value\" : [\"zeros\", \"ones\", \"linear\", \"random\", \"greville\"],"
            " \"default\" : 2,"
            " \"uiid\" : 2},"
            "{\"name\" : \"nonuniform\","
            " \"description\" : \"Create non-uniform B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"false\", \"true\"],"
            " \"default\" : 0,"
            " \"uiid\" : 3}"
            "]";
        else if constexpr (BSpline_t::parDim() == 2)
          return "["
            "{\"name\" : \"degree\","
            " \"description\" : \"Polynomial degree of the B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"constant\", \"linear\", \"quadratic\", \"cubic\", \"quartic\", \"quintic\"],"
            " \"default\" : 2,"
            " \"uiid\" : 0},"
            "{\"name\" : \"ncoeffs\","
            " \"description\" : \"Number of coefficients per parametric dimension\","
            " \"type\" : [\"int\",\"int\"],"
            " \"value\" : [3,3],"
            " \"default\" : [3,3],"
            " \"uiid\" : 1},"
            "{\"name\" : \"init\","
            " \"description\" : \"Initialization of the coefficients\","
            " \"type\" : \"select\","
            " \"value\" : [\"zeros\", \"ones\", \"linear\", \"random\", \"greville\"],"
            " \"default\" : 2,"
            " \"uiid\" : 2},"
            "{\"name\" : \"nonuniform\","
            " \"description\" : \"Create non-uniform B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"false\", \"true\"],"
            " \"default\" : 0,"
            " \"uiid\" : 3}"
            "]";
        else if constexpr (BSpline_t::parDim() == 3)
          return "["
            "{\"name\" : \"degree\","
            " \"description\" : \"Polynomial degree of the B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"constant\", \"linear\", \"quadratic\", \"cubic\", \"quartic\", \"quintic\"],"
            " \"default\" : 2,"
            " \"uiid\" : 0},"
            "{\"name\" : \"ncoeffs\","
            " \"description\" : \"Number of coefficients per parametric dimension\","
            " \"type\" : [\"int\",\"int\",\"int\"],"
            " \"value\" : [3,3,3],"
            " \"default\" : [3,3,3],"
            " \"uiid\" : 1},"
            "{\"name\" : \"init\","
            " \"description\" : \"Initialization of the coefficients\","
            " \"type\" : \"select\","
            " \"value\" : [\"zeros\", \"ones\", \"linear\", \"random\", \"greville\"],"
            " \"default\" : 2,"
            " \"uiid\" : 2},"
            "{\"name\" : \"nonuniform\","
            " \"description\" : \"Create non-uniform B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"false\", \"true\"],"
            " \"default\" : 0,"
            " \"uiid\" : 3}"
            "]";
        else if constexpr (BSpline_t::parDim() == 4)
          return "["
            "{\"name\" : \"degree\","
            " \"description\" : \"Polynomial degree of the B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"constant\", \"linear\", \"quadratic\", \"cubic\", \"quartic\", \"quintic\"],"
            " \"default\" : 2,"
            " \"uiid\" : 0},"
            "{\"name\" : \"ncoeffs\","
            " \"description\" : \"Number of coefficients per parametric dimension\","
            " \"type\" : [int,int,int,int],"
            " \"value\" : [3,3,3,3],"
            " \"default\" : [3,3,3,3],"
            " \"uiid\" : 1},"
            "{\"name\" : \"init\","
            " \"description\" : \"Initialization of the coefficients\","
            " \"type\" : \"select\","
            " \"value\" : [\"zeros\", \"ones\", \"linear\", \"random\", \"greville\"],"
            " \"default\" : 2,"
            " \"uiid\" : 2},"
            "{\"name\" : \"nonuniform\","
            " \"description\" : \"Create non-uniform B-spline\","
            " \"type\" : \"select\","
            " \"value\" : [\"false\", \"true\"],"
            " \"default\" : 0,"
            " \"uiid\" : 3}"
            "]";
        else
          return "{ INVALID REQUEST }";
      }

      /// @brief Returns the model's inputs
      std::string getInputs() const override {
        return "{}";
      }

      /// @brief Returns the model's outputs
      std::string getOutputs() const override {
        if constexpr (BSpline_t::geoDim() == 1)
          return "["
            "{\"name\" : \"ValueFieldMagnitude\","
            " \"description\" : \"Magnitude of the B-spline values\","
            " \"type\" : 1}"
            "]";
        else
          return "["
            "{\"name\" : \"ValueFieldMagnitude\","
            " \"description\" : \"Magnitude of the B-spline values\","
            " \"type\" : 1},"
            "{\"name\" : \"ValueField\","
            " \"description\" : \"B-spline values\","
            " \"type\" : 2}"
            "]";
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

      /// @brief Updates the attributes of the model
      nlohmann::json updateAttribute(const std::string& attribute,
                                     const nlohmann::json& json) override {
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
      nlohmann::json eval(const std::string& component,
                          const nlohmann::json& json) const override {
        
        if constexpr (BSpline_t::parDim() == 1) {

          std::array<int64_t, 1> res({25});          
          if (json.contains("data"))
            if (json["data"].contains("resolution"))
              res = json["data"]["resolution"].get<std::array<int64_t,1>>();
          
          iganet::TensorArray1 xi = {torch::linspace(0, 1, res[0])};

          if (component == "ValueFieldMagnitude") {
            return nlohmann::json::array()
              .emplace_back(::iganet::to_json<float,1>(*(solution_.eval(xi)[0])));
          }
          else if (component == "ValueField") {
            auto values = BSpline_t::eval(xi);
            auto result = nlohmann::json::array();
            for (short_t dim = 0; dim < BSpline_t::geoDim(); ++dim)
              result.emplace_back(::iganet::to_json<float,1>(*(values[dim])));
            return result;
          }
          else
            return "{ INVALID REQUEST }";          
        }
        
        else if constexpr (BSpline_t::parDim() == 2) {

          std::array<int64_t, 2> res({25, 25});          
          if (json.contains("data"))
            if (json["data"].contains("resolution"))
              res = json["data"]["resolution"].get<std::array<int64_t,2>>();
          
          iganet::TensorArray2 xi = convert<2>(torch::meshgrid({
                torch::linspace(0, 1, res[0],
                                core<typename BSpline_t::value_type>::options_),                
                torch::linspace(0, 1, res[1],
                                core<typename BSpline_t::value_type>::options_)}, "xy"));

          if (component == "ValueFieldMagnitude") {
            return nlohmann::json::array()
              .emplace_back(::iganet::to_json<float,2>(*(solution_.eval(xi)[0])));
          }
          else if (component == "ValueField") {
            auto values = BSpline_t::eval(xi);
            auto result = nlohmann::json::array();
            for (short_t dim = 0; dim < BSpline_t::geoDim(); ++dim)
              result.emplace_back(::iganet::to_json<float,2>(*(values[dim])));
            return result;
          }
          else
            return "{ INVALID REQUEST }";
        }
        
        else if constexpr (BSpline_t::parDim() == 3) {

          std::array<int64_t, 3> res({25, 25, 25});          
          if (json.contains("data"))
            if (json["data"].contains("resolution"))
              res = json["data"]["resolution"].get<std::array<int64_t,3>>();

          iganet::TensorArray3 xi = convert<3>(torch::meshgrid({
                torch::linspace(0, 1, res[0],
                                core<typename BSpline_t::value_type>::options_),                
                torch::linspace(0, 1, res[1],
                                core<typename BSpline_t::value_type>::options_),
                torch::linspace(0, 1, res[2],
                                core<typename BSpline_t::value_type>::options_)}, "xy"));
          
          if (component == "ValueFieldMagnitude") {
            return nlohmann::json::array()
              .emplace_back(::iganet::to_json<float,3>(*(solution_.eval(xi)[0])));
          }
          else if (component == "ValueField") {
            auto values = BSpline_t::eval(xi);
            auto result = nlohmann::json::array();
            for (short_t dim = 0; dim < BSpline_t::geoDim(); ++dim)
              result.emplace_back(::iganet::to_json<float,3>(*(values[dim])));
            return result;
          }
          else
            return "{ INVALID REQUEST }";
        }
        
        else if constexpr (BSpline_t::parDim() == 4) {

          std::array<int64_t, 4> res({25, 25, 25, 25});          
          if (json.contains("data"))
            if (json["data"].contains("resolution"))
              res = json["data"]["resolution"].get<std::array<int64_t,4>>();

          iganet::TensorArray4 xi = convert<4>(torch::meshgrid({
                torch::linspace(0, 1, res[0],
                                core<typename BSpline_t::value_type>::options_),                
                torch::linspace(0, 1, res[1],
                                core<typename BSpline_t::value_type>::options_),
                torch::linspace(0, 1, res[2],
                                core<typename BSpline_t::value_type>::options_),
                torch::linspace(0, 1, res[3],
                                core<typename BSpline_t::value_type>::options_)}, "xy"));
          
          if (component == "ValueFieldMagnitude") {
            return nlohmann::json::array()
              .emplace_back(::iganet::to_json<float,4>(*(solution_.eval(xi)[0])));
          }
          else if (component == "ValueField") {
            auto values = BSpline_t::eval(xi);
            auto result = nlohmann::json::array();
            for (short_t dim = 0; dim < BSpline_t::geoDim(); ++dim)
              result.emplace_back(::iganet::to_json<float,4>(*(values[dim])));
            return result;
          }
          else
            return "{ INVALID REQUEST }";
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

      /// @brief Loads the model from XML
      void loadXML(const std::string& xml,
                   const std::string& component = "") override {
      }
      
      /// @brief Saves the model to XML
      nlohmann::json saveXML(const std::string& component = "") override {
        pugi::xml_document doc;

        // add xml node
        pugi::xml_node xml = doc.append_child("xml");
        BSpline_t::to_xml(xml);

        // save to JSON
        std::ostringstream oss;
        doc.save(oss);
        
        return oss.str();
      }      
    };    

  } // namespace webapp
} // namespace iganet
