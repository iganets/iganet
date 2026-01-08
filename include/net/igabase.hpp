/**
   @file net/v1/dataset.hpp

   @brief Dataset class

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <filesystem>
#include <vector>

#include <core/core.hpp>

namespace iganet {
    
/// @brief IgA dataset base class
///
/// This class implements the specialization of the torch dataset
/// class for IgA solvers and nets
class IgADatasetBase {
protected:
  /// @brief Reads a function space from file
  template <typename T>
  inline void read_from_xml(const std::string &location, T &obj,
                            std::vector<torch::Tensor> &v) {

    std::filesystem::path path(location);

    if (std::filesystem::exists(path)) {
      if (std::filesystem::is_regular_file(path)) {
        try {
          pugi::xml_document doc;
          doc.load_file(path.c_str());
          v.emplace_back(obj.from_xml(doc).as_tensor());
        } catch (...) {
        }
      } else if (std::filesystem::is_directory(path)) {
        for (const auto &file : std::filesystem::directory_iterator(path)) {
          if (file.is_regular_file() && file.path().extension() == ".xml") {
            try {
              pugi::xml_document doc;
              doc.load_file(file.path().c_str());
              v.emplace_back(obj.from_xml(doc).as_tensor());
            } catch (...) {
            }
          }
        }
      } else
        throw std::runtime_error(
            "The path refers to neither a file nor a directory");
    } else
      throw std::runtime_error("The path does not exist");
  }
};
  
/// @brief IgA dataset class
///
/// This class implements the specialization of the torch dataset
/// class for IgA solvers and nets
/// @{
template <bool solution = false> class IgADataset;

template <>
class IgADataset<false>
    : public IgADatasetBase,
      public torch::data::Dataset<
          IgADataset<false>,
          torch::data::Example<torch::Tensor, torch::data::example::NoTarget>> {
private:
  /// @brief Vector of tensors representing the geometry maps
  std::vector<torch::Tensor> G_;

  /// @brief Vector of tensors representing the reference data
  std::vector<torch::Tensor> f_;

public:
  /// @brief Example type
  using example_type =
      torch::data::Example<torch::Tensor, torch::data::example::NoTarget>;

  /// @brief Adds a geometry map from file
  /// @{
  template <typename T> void add_geometryMap(T &obj, std::string location) {
    read_from_xml(location, obj, G_);
  }

  template <typename T> void add_geometryMap(T &&obj, std::string location) {
    read_from_xml(location, obj, G_);
  }
  /// @}

  /// @brief Adds a geometry map from XML object
  /// @{
  template <typename T>
  void add_geometryMap(T &obj, const pugi::xml_document &doc, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_geometryMap(T &&obj, const pugi::xml_document &doc, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a geometry map from XML node
  /// @{
  template <typename T>
  void add_geometryMap(T &obj, const pugi::xml_node &root, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_geometryMap(T &&obj, const pugi::xml_node &root, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a reference data set from file
  /// @{
  template <typename T> void add_referenceData(T &obj, std::string location) {
    read_from_xml(location, obj, f_);
  }

  template <typename T> void add_referenceData(T &&obj, std::string location) {
    read_from_xml(location, obj, f_);
  }
  /// @}

  /// @brief Adds a reference data set from XML object
  /// @{
  template <typename T>
  void add_referenceData(T &obj, const pugi::xml_document &doc, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_referenceData(T &&obj, const pugi::xml_document &doc, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  ///@}

  /// @brief Adds a reference data set from XML node
  /// @{
  template <typename T>
  void add_referenceData(T &obj, const pugi::xml_node &root, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_referenceData(T &&obj, const pugi::xml_node &root, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a reference data set from XML node
  /// @{
  template <typename T, typename Func>
  void add_referenceData(T &obj, Func func) {
    f_.emplace_back(obj.transform(func).as_tensor());
  }

  template <typename T, typename Func>
  void add_referenceData(T &&obj, Func func) {
    f_.emplace_back(obj.transform(func).as_tensor());
  }
  /// @}

  /// @brief Returns the data set at location index
  inline example_type get(std::size_t index) override {

    std::size_t geo_index = index / (f_.empty() ? 1 : f_.size());
    std::size_t ref_index = index - geo_index * f_.size();

    if (!G_.empty()) {
      if (!f_.empty())
        return torch::cat({G_.at(geo_index), f_.at(ref_index)});
      else
        return G_.at(geo_index);
    } else {
      if (!f_.empty())
        return f_.at(ref_index);
      else
        throw std::runtime_error("No geometry maps and reference data");
    }
  };

  // @brief Return the total size of the data set
  [[nodiscard]] inline torch::optional<std::size_t> size() const override {
    return (G_.empty() ? 1 : G_.size()) * (f_.empty() ? 1 : f_.size());
  }
};

template <>
class IgADataset<true>
    : public IgADatasetBase,
      public torch::data::Dataset<IgADataset<true>, torch::data::Example<>> {
private:
  /// @brief Vector of tensors representing the geometry maps
  std::vector<torch::Tensor> G_;

  /// @brief Vector of tensors representing the reference data
  std::vector<torch::Tensor> f_;

  /// @brief Vector of tensors representing the solution data
  std::vector<torch::Tensor> u_;

public:
  /// @brief Adds a geometry map from file
  /// @{
  template <typename T> void add_geometryMap(T &obj, std::string location) {
    read_from_xml(location, obj, G_);
  }

  template <typename T> void add_geometryMap(T &&obj, std::string location) {
    read_from_xml(location, obj, G_);
  }
  /// @}

  /// @brief Adds a geometry map from XML object
  /// @{
  template <typename T>
  void add_geometryMap(T &obj, const pugi::xml_document &doc, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_geometryMap(T &&obj, const pugi::xml_document &doc, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a geometry map from XML node
  /// @{
  template <typename T>
  void add_geometryMap(T &obj, const pugi::xml_node &root, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_geometryMap(T &&obj, const pugi::xml_node &root, int id = 0,
                       std::string label = "") {
    G_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a reference data set from file
  /// @{
  template <typename T> void add_referenceData(T &obj, std::string location) {
    read_from_xml(location, obj, f_);
  }

  template <typename T> void add_referenceData(T &&obj, std::string location) {
    read_from_xml(location, obj, f_);
  }
  /// @}

  /// @brief Adds a reference data set from XML object
  /// @{
  template <typename T>
  void add_referenceData(T &obj, const pugi::xml_document &doc, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_referenceData(T &&obj, const pugi::xml_document &doc, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  ///@}

  /// @brief Adds a reference data set from XML node
  /// @{
  template <typename T>
  void add_referenceData(T &obj, const pugi::xml_node &root, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_referenceData(T &&obj, const pugi::xml_node &root, int id = 0,
                         std::string label = "") {
    f_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a reference data set from XML node
  /// @{
  template <typename T, typename Func>
  void add_referenceData(T &obj, Func func) {
    f_.emplace_back(obj.transform(func).as_tensor());
  }

  template <typename T, typename Func>
  void add_referenceData(T &&obj, Func func) {
    f_.emplace_back(obj.transform(func).as_tensor());
  }
  /// @}

  /// @brief Adds a solution from file
  /// @{
  template <typename T> void add_solution(T &obj, std::string location) {
    read_from_xml(location, obj, u_);
  }

  template <typename T> void add_solution(T &&obj, std::string location) {
    read_from_xml(location, obj, u_);
  }
  /// @}

  /// @brief Adds a solution from XML object
  /// @{
  template <typename T>
  void add_solution(T &obj, const pugi::xml_document &doc, int id = 0,
                    std::string label = "") {
    u_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }

  template <typename T>
  void add_solution(T &&obj, const pugi::xml_document &doc, int id = 0,
                    std::string label = "") {
    u_.emplace_back(obj.from_xml(doc.child("xml"), id, label).as_tensor());
  }
  /// @}

  /// @brief Adds a solution from XML node
  /// @{
  template <typename T>
  void add_solution(T &obj, const pugi::xml_node &root, int id = 0,
                    std::string label = "") {
    u_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }

  template <typename T>
  void add_solution(T &&obj, const pugi::xml_node &root, int id = 0,
                    std::string label = "") {
    u_.emplace_back(obj.from_xml(root, id, label).as_tensor());
  }
  /// @}

  /// @brief Returns the data set at location index
  inline torch::data::Example<> get(std::size_t index) override {

    std::size_t geo_index = index / (f_.empty() ? 1 : f_.size());
    std::size_t ref_index = index - geo_index * f_.size();

    if (!G_.empty()) {
      if (!f_.empty())
        return {torch::cat({G_.at(geo_index), f_.at(ref_index)}), u_.at(index)};
      else
        return {G_.at(geo_index), u_.at(index)};
    } else {
      if (!f_.empty())
        return {f_.at(ref_index), u_.at(index)};
      else
        throw std::runtime_error("No geometry maps and reference data");
    }
  };

  // @brief Return the total size of the data set
  [[nodiscard]] inline torch::optional<std::size_t> size() const override {
    return (G_.empty() ? 1 : G_.size()) * (f_.empty() ? 1 : f_.size());
  }
};
/// @}
  
} // namespace iganet
