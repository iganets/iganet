/**
   @file splines/multipatch.hpp

   @brief Multi-patch container class

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <splines/boundary.hpp>

#include <string_view>

namespace iganet {

/// @brief Connection between two patch sides
template <typename Patch> class PatchInterface {
public:
  /// @brief Constructor
  PatchInterface(std::shared_ptr<Patch> firstPatch, enum side firstSide,
                 std::shared_ptr<Patch> secondPatch, enum side secondSide)
      : patches_{std::move(firstPatch), std::move(secondPatch)},
        sides_{firstSide, secondSide} {
    if (!patches_[0] || !patches_[1])
      throw std::invalid_argument("An interface requires two valid patches");
    if (sides_[0] == none || sides_[1] == none)
      throw std::invalid_argument("An interface requires two valid sides");
  }

  /// @brief Returns one of the two patches
  Patch &patch(std::size_t endpoint) {
    assert(endpoint < patches_.size());
    return *patches_[endpoint];
  }

  /// @brief Returns one of the two patches
  const Patch &patch(std::size_t endpoint) const {
    assert(endpoint < patches_.size());
    return *patches_[endpoint];
  }

  /// @brief Returns the shared pointer to one of the two patches
  const std::shared_ptr<Patch> &patchPtr(std::size_t endpoint) const {
    assert(endpoint < patches_.size());
    return patches_[endpoint];
  }

  /// @brief Returns the side of one of the two patches
  enum side side(std::size_t endpoint) const {
    assert(endpoint < sides_.size());
    return sides_[endpoint];
  }

  /// @brief Named endpoint accessors
  /// @{
  Patch &firstPatch() { return patch(0); }
  const Patch &firstPatch() const { return patch(0); }
  Patch &secondPatch() { return patch(1); }
  const Patch &secondPatch() const { return patch(1); }
  enum side firstSide() const { return side(0); }
  enum side secondSide() const { return side(1); }
  /// @}

private:
  std::array<std::shared_ptr<Patch>, 2> patches_;
  std::array<enum side, 2> sides_;
};

/// @brief Multi-patch container class
///
/// This class implements a container for a set of patches and their
/// topology, that is, the interface connections and outer boundary
/// faces.
template <typename Patch> class MultiPatch {

public:
  /// @brief Interface type
  using interface_type = PatchInterface<Patch>;

  /// @brief Default constructor
  MultiPatch() = default;

  /// @brief Copy constructor
  MultiPatch(const MultiPatch &other)
      : patches_(other.patches_), interfaces_(other.interfaces_) {}

  /// @brief Move constructor
  MultiPatch(MultiPatch &&other) noexcept {
    patches_.swap(other.patches_);
    interfaces_.swap(other.interfaces_);
  }

public:
  /// @brief Returns an iterator to the patches
  auto begin() { return patches_.begin(); }

  /// @brief Returns a const-iterator to the patches
  /// @{
  auto begin() const { return patches_.begin(); }
  auto cbegin() const noexcept { return patches_.cbegin(); }
  /// @}

  /// @brief Returns an iterator to the end of the patches
  auto end() { return patches_.end(); }

  /// @brief Returns a const-iterator to the end of the patches
  /// @{
  auto end() const { return patches_.end(); }
  auto cend() const noexcept { return patches_.cend(); }
  /// @}

  /// @brief Returns a reverse iterator to the patches
  auto rbegin() { return patches_.rbegin(); }

  /// @brief Returns a reverse const-iterator to the patches
  /// @{
  auto rbegin() const { return patches_.rbegin(); }
  auto crbegin() const noexcept { return patches_.crbegin(); }
  /// @}

  /// @brief Returns a reverse iterator to the end of the patches
  auto rend() { return patches_.rend(); }

  /// @brief Returns a reverse const-iterator to the end of the patches
  /// @{
  auto rend() const { return patches_.rend(); }
  auto crend() const noexcept { return patches_.crend(); }
  /// @}

public:
  /// @brief Returns the number of patches
  [[nodiscard]] std::size_t npatches() const { return patches_.size(); }

  /// @brief Returns the number of interfaces
  [[nodiscard]] std::size_t ninterfaces() const { return interfaces_.size(); }

  /// @brief Returns the number of outer boundaries
  [[nodiscard]] std::size_t nboundaries() const { return patches_.size(); }

public:
  /// @brief Adds a single patch
  /// @{
  std::size_t addPatch(std::shared_ptr<Patch> patch) {
    std::size_t index = patches_.size();
    patches_.push_back(patch);
    return index;
  }

  std::size_t addPatch(std::unique_ptr<Patch> patch) {
    std::size_t index = patches_.size();
    patches_.push_back(std::shared_ptr<Patch>(std::move(patch)));
    return index;
  }
  /// @}

  /// @brief Adds an interface between two patches identified by index
  std::size_t addInterface(std::size_t firstPatch, enum side firstSide,
                           std::size_t secondPatch, enum side secondSide) {
    assert(firstPatch < patches_.size());
    assert(secondPatch < patches_.size());
    return addInterface(patches_[firstPatch], firstSide, patches_[secondPatch],
                        secondSide);
  }

  /// @brief Adds an interface between two patches
  std::size_t addInterface(std::shared_ptr<Patch> firstPatch,
                           enum side firstSide,
                           std::shared_ptr<Patch> secondPatch,
                           enum side secondSide) {
    if (std::find(patches_.begin(), patches_.end(), firstPatch) ==
            patches_.end() ||
        std::find(patches_.begin(), patches_.end(), secondPatch) ==
            patches_.end())
      throw std::invalid_argument(
          "Interface patches must belong to the MultiPatch");

    const std::size_t index = interfaces_.size();
    interfaces_.emplace_back(std::move(firstPatch), firstSide,
                             std::move(secondPatch), secondSide);
    return index;
  }

  /// @brief Adds an interface object
  std::size_t addInterface(interface_type patchInterface) {
    return addInterface(patchInterface.patchPtr(0), patchInterface.side(0),
                        patchInterface.patchPtr(1), patchInterface.side(1));
  }

  /// @brief Removes a single interface
  void removeInterface(std::size_t index) {
    assert(index < interfaces_.size());
    interfaces_.erase(interfaces_.begin() + index);
  }

  /// @brief Removes all patches
  void clear() {
    interfaces_.clear();
    patches_.clear();
  }

  /// @brief Returns a non-constant reference to a single patch
  Patch &patch(std::size_t index) {
    assert(index < patches_.size());
    return *patches_[index];
  }

  /// @brief Returns a constant reference to a single patch
  const Patch &patch(std::size_t index) const {
    assert(index < patches_.size());
    return *patches_[index];
  }

  /// @brief Returns a reference to the vector of patches
  /// @{
  std::vector<std::shared_ptr<Patch>> &patches() { return patches_; }
  const std::vector<std::shared_ptr<Patch>> &patches() const {
    return patches_;
  }
  /// @}

  /// @brief Returns a non-constant reference to a single interface
  interface_type &interface(std::size_t index) {
    assert(index < interfaces_.size());
    return interfaces_[index];
  }

  /// @brief Returns a constant reference to a single interface
  const interface_type &interface(std::size_t index) const {
    assert(index < interfaces_.size());
    return interfaces_[index];
  }

  /// @brief Returns the interfaces for range-based iteration
  /// @{
  std::vector<interface_type> &interfaces() { return interfaces_; }
  const std::vector<interface_type> &interfaces() const { return interfaces_; }
  /// @}

  /// @brief Returns the index of a given single patch
  /// @{
  std::size_t findPatchIndex(const Patch &patch) const {
    return findPatchIndex(&patch);
  }

  std::size_t findPatchIndex(const Patch *patch) const {
    auto it = std::find_if(
        patches_.begin(), patches_.end(),
        [patch](const auto &candidate) { return candidate.get() == patch; });
    if (it == patches_.end())
      throw std::runtime_error("Did not find the patch index");

    return it - patches_.begin();
  }
  /// @}

  /// @brief Returns the index of a given patch interface
  /// @{
  std::size_t findInterfaceIndex(const interface_type &patchInterface) const {
    return findInterfaceIndex(&patchInterface);
  }

  std::size_t findInterfaceIndex(const interface_type *patchInterface) const {
    auto it = std::find_if(interfaces_.begin(), interfaces_.end(),
                           [patchInterface](const auto &candidate) {
                             return &candidate == patchInterface;
                           });
    if (it == interfaces_.end())
      throw std::runtime_error("Did not find the patch interface index");

    return it - interfaces_.begin();
  }
  /// @}

  /// @brief Returns the multi-patch object as a JSON object
  [[nodiscard]] nlohmann::json to_json() const {
    nlohmann::json json;
    json["parDim"] = Patch::parDim();
    json["patches"] = nlohmann::json::array();
    json["interfaces"] = nlohmann::json::array();
    json["boundaries"] = nlohmann::json::array();

    for (const auto &patch : patches_)
      json["patches"].push_back(patch->to_json());

    for (const auto &patchInterface : interfaces_) {
      nlohmann::json interfaceJson;
      interfaceJson["patches"] = {
          findPatchIndex(patchInterface.patchPtr(0).get()),
          findPatchIndex(patchInterface.patchPtr(1).get())};
      interfaceJson["sides"] = {static_cast<short_t>(patchInterface.side(0)),
                                static_cast<short_t>(patchInterface.side(1))};

      const short_t firstAxis =
          (static_cast<short_t>(patchInterface.side(0)) - 1) / 2;
      const short_t secondAxis =
          (static_cast<short_t>(patchInterface.side(1)) - 1) / 2;
      interfaceJson["direction"] = nlohmann::json::array();
      interfaceJson["orientation"] = nlohmann::json::array();
      for (short_t axis = 0; axis < Patch::parDim(); ++axis) {
        short_t mappedAxis = axis;
        if (axis == firstAxis)
          mappedAxis = secondAxis;
        else if (axis == secondAxis)
          mappedAxis = firstAxis;
        interfaceJson["direction"].push_back(mappedAxis);
        interfaceJson["orientation"].push_back(1);
      }
      json["interfaces"].push_back(std::move(interfaceJson));
    }

    for (std::size_t patchIndex = 0; patchIndex < patches_.size();
         ++patchIndex) {
      for (short_t patchSide = 1; patchSide <= 2 * Patch::parDim();
           ++patchSide) {
        const bool isInterface =
            std::any_of(interfaces_.begin(), interfaces_.end(),
                        [&](const auto &patchInterface) {
                          return (patchInterface.patchPtr(0).get() ==
                                      patches_[patchIndex].get() &&
                                  patchInterface.side(0) == patchSide) ||
                                 (patchInterface.patchPtr(1).get() ==
                                      patches_[patchIndex].get() &&
                                  patchInterface.side(1) == patchSide);
                        });
        if (!isInterface)
          json["boundaries"].push_back(
              {{"patch", patchIndex}, {"side", patchSide}});
      }
    }
    return json;
  }

  /// @brief Updates the multi-patch object from a JSON object
  MultiPatch &from_json(const nlohmann::json &json) {
    if (json.at("parDim").get<short_t>() != Patch::parDim())
      throw std::runtime_error(
          "MultiPatch JSON provides an incompatible parametric dimension");

    const auto &patchJson = json.at("patches");
    if (!patchJson.is_array() || patchJson.size() != patches_.size())
      throw std::runtime_error(
          "MultiPatch JSON patch count does not match the patch container");

    const auto &interfaceJson = json.at("interfaces");
    if (!interfaceJson.is_array())
      throw std::runtime_error("MultiPatch JSON interfaces must be an array");

    std::vector<interface_type> parsedInterfaces;
    parsedInterfaces.reserve(interfaceJson.size());
    for (const auto &item : interfaceJson) {
      const auto patchIndices = item.at("patches").get<std::array<size_t, 2>>();
      const auto sides = item.at("sides").get<std::array<short_t, 2>>();
      if (patchIndices[0] >= patches_.size() ||
          patchIndices[1] >= patches_.size() || sides[0] <= none ||
          sides[0] > 2 * Patch::parDim() || sides[1] <= none ||
          sides[1] > 2 * Patch::parDim())
        throw std::runtime_error("MultiPatch JSON has an invalid interface");

      if (!item.at("direction").is_array() ||
          item.at("direction").size() != Patch::parDim() ||
          !item.at("orientation").is_array() ||
          item.at("orientation").size() != Patch::parDim())
        throw std::runtime_error(
            "MultiPatch JSON has invalid interface orientation data");

      parsedInterfaces.emplace_back(
          patches_[patchIndices[0]], static_cast<enum side>(sides[0]),
          patches_[patchIndices[1]], static_cast<enum side>(sides[1]));
    }

    for (std::size_t patchIndex = 0; patchIndex < patches_.size(); ++patchIndex)
      patches_[patchIndex]->from_json(patchJson[patchIndex]);

    interfaces_ = std::move(parsedInterfaces);
    return *this;
  }

  /// @brief Returns the multi-patch object as an XML document
  [[nodiscard]] pugi::xml_document
  to_xml(int id = 0, const std::string &label = "", int index = -1) const {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("xml");
    to_xml(root, id, label, index);
    return doc;
  }

  /// @brief Appends the multi-patch object to an XML node
  pugi::xml_node &to_xml(pugi::xml_node &root, int id = 0,
                         const std::string &label = "", int index = -1) const {
    for (std::size_t patchIndex = 0; patchIndex < patches_.size(); ++patchIndex)
      patches_[patchIndex]->to_xml(root, static_cast<int>(patchIndex));

    pugi::xml_node multiPatch = root.append_child("MultiPatch");
    multiPatch.append_attribute("parDim") = Patch::parDim();
    multiPatch.append_attribute("id") = id;
    if (!label.empty())
      multiPatch.append_attribute("label") = label.c_str();
    if (index >= 0)
      multiPatch.append_attribute("index") = index;

    pugi::xml_node patches = multiPatch.append_child("patches");
    patches.append_attribute("type") = "id_range";
    if (!patches_.empty()) {
      const std::string range =
          "0 " + std::to_string(static_cast<int64_t>(patches_.size()) - 1);
      patches.append_child(pugi::node_pcdata).set_value(range.c_str());
    }

    std::stringstream interfaceData;
    for (const auto &patchInterface : interfaces_) {
      interfaceData << "\n      ";
      const auto firstPatch = findPatchIndex(patchInterface.patchPtr(0).get());
      const auto secondPatch = findPatchIndex(patchInterface.patchPtr(1).get());
      const short_t firstAxis =
          (static_cast<short_t>(patchInterface.side(0)) - 1) / 2;
      const short_t secondAxis =
          (static_cast<short_t>(patchInterface.side(1)) - 1) / 2;

      interfaceData << firstPatch << ' '
                    << static_cast<short_t>(patchInterface.side(0)) << ' '
                    << secondPatch << ' '
                    << static_cast<short_t>(patchInterface.side(1));

      for (short_t axis = 0; axis < Patch::parDim(); ++axis) {
        short_t mappedAxis = axis;
        if (axis == firstAxis)
          mappedAxis = secondAxis;
        else if (axis == secondAxis)
          mappedAxis = firstAxis;
        interfaceData << ' ' << mappedAxis;
      }
      for (short_t axis = 0; axis < Patch::parDim(); ++axis)
        interfaceData << " 1";
    }
    if (!interfaces_.empty())
      interfaceData << "\n    ";
    multiPatch.append_child("interfaces")
        .append_child(pugi::node_pcdata)
        .set_value(interfaceData.str().c_str());

    std::stringstream boundaryData;
    for (std::size_t patchIndex = 0; patchIndex < patches_.size();
         ++patchIndex) {
      for (short_t patchSide = 1; patchSide <= 2 * Patch::parDim();
           ++patchSide) {
        const bool isInterface =
            std::any_of(interfaces_.begin(), interfaces_.end(),
                        [&](const auto &patchInterface) {
                          return (patchInterface.patchPtr(0).get() ==
                                      patches_[patchIndex].get() &&
                                  patchInterface.side(0) == patchSide) ||
                                 (patchInterface.patchPtr(1).get() ==
                                      patches_[patchIndex].get() &&
                                  patchInterface.side(1) == patchSide);
                        });
        if (!isInterface)
          boundaryData << "\n      " << patchIndex << ' ' << patchSide;
      }
    }
    if (!boundaryData.str().empty())
      boundaryData << "\n    ";
    multiPatch.append_child("boundary")
        .append_child(pugi::node_pcdata)
        .set_value(boundaryData.str().c_str());

    return root;
  }

  /// @brief Updates the multi-patch object from an XML document
  MultiPatch &from_xml(const pugi::xml_document &doc, int id = 0,
                       const std::string &label = "", int index = -1) {
    return from_xml(doc.child("xml"), id, label, index);
  }

  /// @brief Updates the multi-patch object from an XML node
  MultiPatch &from_xml(const pugi::xml_node &root, int id = 0,
                       const std::string &label = "", int index = -1) {
    pugi::xml_node multiPatch;
    for (const auto &candidate : root.children("MultiPatch")) {
      if ((id >= 0 ? candidate.attribute("id").as_int() == id : true) &&
          (index >= 0 ? candidate.attribute("index").as_int() == index
                      : true) &&
          (!label.empty() ? candidate.attribute("label").value() == label
                          : true)) {
        multiPatch = candidate;
        break;
      }
    }
    if (!multiPatch)
      throw std::runtime_error("Did not find the MultiPatch XML node");

    if (multiPatch.attribute("parDim").as_int() != Patch::parDim())
      throw std::runtime_error(
          "MultiPatch XML provides an incompatible parametric dimension");

    const pugi::xml_node patchRange = multiPatch.child("patches");
    if (!patchRange || std::string_view{patchRange.attribute("type").value()} !=
                           "id_range")
      throw std::runtime_error("MultiPatch XML has no valid patch ID range");

    std::stringstream rangeData(patchRange.child_value());
    int64_t firstPatch = 0;
    int64_t lastPatch = -1;
    if (!(rangeData >> firstPatch >> lastPatch) || firstPatch != 0 ||
        lastPatch + 1 != static_cast<int64_t>(patches_.size()))
      throw std::runtime_error(
          "MultiPatch XML patch range does not match the patch container");

    std::vector<interface_type> parsedInterfaces;
    const pugi::xml_node interfaceNode = multiPatch.child("interfaces");
    if (!interfaceNode)
      throw std::runtime_error("MultiPatch XML has no interfaces node");

    std::stringstream interfaceData(interfaceNode.child_value());
    std::string line;
    while (std::getline(interfaceData, line)) {
      std::stringstream item(line);
      std::size_t firstPatchIndex;
      std::size_t secondPatchIndex;
      short_t firstSide;
      short_t secondSide;
      if (!(item >> firstPatchIndex >> firstSide >> secondPatchIndex >>
            secondSide))
        continue;

      if (firstPatchIndex >= patches_.size() ||
          secondPatchIndex >= patches_.size() || firstSide <= none ||
          firstSide > 2 * Patch::parDim() || secondSide <= none ||
          secondSide > 2 * Patch::parDim())
        throw std::runtime_error("MultiPatch XML has an invalid interface");

      short_t topologyEntry;
      for (short_t entry = 0; entry < 2 * Patch::parDim(); ++entry) {
        if (!(item >> topologyEntry))
          throw std::runtime_error(
              "MultiPatch XML has incomplete interface orientation data");
      }
      if (item >> topologyEntry)
        throw std::runtime_error(
            "MultiPatch XML has excess interface orientation data");

      parsedInterfaces.emplace_back(
          patches_[firstPatchIndex], static_cast<enum side>(firstSide),
          patches_[secondPatchIndex], static_cast<enum side>(secondSide));
    }

    for (std::size_t patchIndex = 0; patchIndex < patches_.size(); ++patchIndex)
      patches_[patchIndex]->from_xml(root, static_cast<int>(patchIndex));

    interfaces_ = std::move(parsedInterfaces);
    return *this;
  }

private:
  /// @brief Vector of single-patch objects
  std::vector<std::shared_ptr<Patch>> patches_;

  /// @brief Vector of patch-interface objects
  std::vector<interface_type> interfaces_;
};

} // namespace iganet
