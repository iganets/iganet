/**
   @file webapps/model.hpp

   @brief Model capabilities

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <config.hpp>

namespace iganet {

/// @brief Enumerator for specifying the capabilities
enum class capability {
  /*! Model create/remove */
  create = 0, /*!< create object */
  remove = 1, /*!< remove object */

  /*!< Model evaluation and adaption */
  eval = 2,    /*!< evaluates object */
  refine = 3,  /*!< h-refines object */
  elevate = 4, /*!< p-refines object */

  /*!< Model loading/saving */
  load = 5, /*!< loads model from PyTorch file */
  save = 6, /*!< saves model to PyTorch file */

  /*!< Model import/export */
  importXML = 7, /*!< imports object from G+Smo XML file */
  exportXML = 8, /*!< exports object to G+Smo XML file */

  /*!< Error computation */
  computeL1error = 9,  /*!< computes model's L1-error */
  computeL2error = 10, /*!< computes model's L2-error */
  computeH1error = 11  /*!< computes model's H1-error */
};

/// @brief Enumerator for specifying the output type
enum class io {
  scalar = 0,               /*!< scalar value */
  scalarfield = 1,          /*!< scalar field */
  vectorfield = 2,          /*!< vector field */
  scalarfield_boundary = 3, /*!< scalar field at the boundary */
  vectorfield_boundary = 4  /*!< vector field at the boundary */
};

/// @brief IndexOutOfBounds exception
struct IndexOutOfBoundsException : public std::exception {
  const char *what() const throw() { return "Index is out of bounds"; }
};

/// @brief InvalidModel exception
struct InvalidModelException : public std::exception {
  const char *what() const throw() { return "Invalid model name"; }
};

/// @brief InvalidModelAttribute exception
struct InvalidModelAttributeException : public std::exception {
  const char *what() const throw() { return "Invalid model attribute"; }
};

/// @brief Model elevation
class ModelElevate {
public:
  /// @brief elevates model
  virtual void elevate(const nlohmann::json &json) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("elevate")};
  }
};

/// @brief Model evaluator
class ModelEval {
public:
  /// @brief Evaluates model
  virtual nlohmann::json eval(const std::string &component,
                              const nlohmann::json &json) const = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("eval")};
  }
};

/// @brief Model refinement
class ModelRefine {
public:
  /// @brief Refines model
  virtual void refine(const nlohmann::json &json) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("refine")};
  }
};

/// @brief Model serialization
class ModelSerialize {
public:
  /// @brief Loads model from LibTorch file
  virtual void load(const nlohmann::json &json) = 0;

  /// @brief Saves model to LibTorch file
  virtual nlohmann::json save() const = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("load"), std::string("save")};
  }
};

/// @brief Model XML serialization
class ModelXML {
public:
  /// @brief Imports model from XML (as JSON object)
  virtual void importXML(const nlohmann::json &json,
                         const std::string &component, std::size_t id) = 0;

  /// @brief Imports model from XML (as XML object)
  virtual void importXML(const pugi::xml_node &xml,
                         const std::string &component, std::size_t id) = 0;

  /// @brief Exports model to XML (as JSON object)
  virtual nlohmann::json exportXML(const std::string &component,
                                   std::size_t id) = 0;

  /// @brief Exports model to XML (as XML object)
  virtual pugi::xml_node &exportXML(pugi::xml_node &root,
                                    const std::string &component,
                                    std::size_t id) = 0;

  // @brief Returns model capabilities
  std::vector<std::string> getCapabilities() const {
    return std::vector{std::string("exportxml"), std::string("importxml")};
  }
};

/// @brief Model interface
class Model {
public:
  /// @brief Constructor
  Model() : transform(torch::zeros({4, 4})){};

  /// @brief Destructor
  virtual ~Model(){};

  /// @brief Returns the model's name
  virtual std::string getName() const = 0;

  /// @brief Returns the model's description
  virtual std::string getDescription() const = 0;

  /// @brief Returns the model's options
  virtual std::string getOptions() const = 0;

  /// @brief Returns the model's inputs
  virtual std::string getInputs() const = 0;

  /// @brief Returns the model's outputs
  virtual std::string getOutputs() const = 0;

  /// @brief Returns the model's JSON serialization
  virtual nlohmann::json getModel() const {
    return nlohmann::json::parse(
        std::string("{ \"name\" : \"") + getName() + "\"," +
        "\"description\" : \"" + getDescription() + "\"," + "\"options\" : " +
        getOptions() + "," + "\"capabilities\" : " + getCapabilities().dump() +
        "," + "\"inputs\" : " + getInputs() + "," +
        "\"outputs\" : " + getOutputs() + " }");
  }

  /// @brief Returns the model's capabilities
  virtual nlohmann::json getCapabilities() const {

    std::vector<std::string> capabilities;
    capabilities.push_back("create");
    capabilities.push_back("remove");

    if (auto m = dynamic_cast<const ModelElevate *>(this))
      for (auto const &capability : m->getCapabilities())
        capabilities.push_back(capability);

    if (auto m = dynamic_cast<const ModelEval *>(this))
      for (auto const &capability : m->getCapabilities())
        capabilities.push_back(capability);

    if (auto m = dynamic_cast<const ModelRefine *>(this))
      for (auto const &capability : m->getCapabilities())
        capabilities.push_back(capability);

    if (auto m = dynamic_cast<const ModelXML *>(this))
      for (auto const &capability : m->getCapabilities())
        capabilities.push_back(capability);

    auto data = nlohmann::json::array();
    for (auto const &capability : capabilities)
      data.push_back("\"" + capability + "\"");

    nlohmann::json json;
    json["capability"] = data;

    return json;
  }

  /// @brief Serializes the model to JSON
  virtual nlohmann::json to_json(const std::string &component,
                                 const std::string &attribute) const {
    if (component == "transform") {

      nlohmann::json data;
      data["matrix"] = utils::to_json<iganet::real_t, 1>(transform.flatten());

      return data;
    }

    else
      return "{ INVALID REQUEST }";
  }

  /// @brief Updates the attributes of the model
  virtual nlohmann::json updateAttribute(const std::string &component,
                                         const std::string &attribute,
                                         const nlohmann::json &json) {
    if (attribute == "transform") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("matrix"))
        throw InvalidModelAttributeException();

      auto matrix =
          json["data"]["matrix"].get<std::vector<iganet::real_t>>();

      if (matrix.size() != 16)
        throw IndexOutOfBoundsException();

      auto transform_cpu =
          utils::to_tensorAccessor<iganet::real_t, 2>(transform, torch::kCPU);
      auto transformAccessor = std::get<1>(transform_cpu);

      std::size_t index(0);
      for (const auto &entry : matrix)
        transformAccessor[index / 4][index++ % 4] = entry;

      return "{}";
    }

    else
      return "{ INVALID REQUEST }";
  }

protected:
  /// @brief Global transformation matrix
  torch::Tensor transform;
};

} // namespace iganet
