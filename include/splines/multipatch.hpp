/**
   @file splines/multipatch.hpp

   @brief Multi-patch container class.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <splines/boundary.hpp>

#include <string_view>

namespace iganet {

/// @brief Connection between two patch sides.
template <typename Patch> class PatchInterface {
public:
  /// @brief Constructor.
  /// @tparam Patch Template parameter `Patch`.
  /// @param firstPatch Value of `firstPatch`.
  /// @param firstSide Value of `firstSide`.
  /// @param secondPatch Value of `secondPatch`.
  /// @param secondSide Value of `secondSide`.
  PatchInterface(std::shared_ptr<Patch> firstPatch, enum side firstSide,
                 std::shared_ptr<Patch> secondPatch, enum side secondSide)
      : patches_{std::move(firstPatch), std::move(secondPatch)},
        sides_{firstSide, secondSide} {
    if (!patches_[0] || !patches_[1])
      throw std::invalid_argument("An interface requires two valid patches");
    if (sides_[0] == none || sides_[1] == none)
      throw std::invalid_argument("An interface requires two valid sides");
  }

  /// @brief Returns one of the two patches.
  /// @param endpoint Value of `endpoint`.
  /// @return Result of the operation.
  Patch &patch(std::size_t endpoint) {
    assert(endpoint < patches_.size());
    return *patches_[endpoint];
  }

  /// @brief Returns one of the two patches.
  /// @param endpoint Value of `endpoint`.
  /// @return Result of the operation.
  const Patch &patch(std::size_t endpoint) const {
    assert(endpoint < patches_.size());
    return *patches_[endpoint];
  }

  /// @brief Returns the shared pointer to one of the two patches.
  /// @param endpoint Value of `endpoint`.
  /// @return Result of the operation.
  const std::shared_ptr<Patch> &patchPtr(std::size_t endpoint) const {
    assert(endpoint < patches_.size());
    return patches_[endpoint];
  }

  /// @brief Returns the side of one of the two patches.
  /// @param endpoint Value of `endpoint`.
  /// @return Result of the operation.
  enum side side(std::size_t endpoint) const {
    assert(endpoint < sides_.size());
    return sides_[endpoint];
  }

  /// @brief Named endpoint accessors
  /// @{
  /// @return Result of the operation.
  Patch &firstPatch() { return patch(0); }
  /// @brief Provides the `firstPatch` operation.
  /// @return Result of the operation.
  const Patch &firstPatch() const { return patch(0); }
  /// @brief Provides the `secondPatch` operation.
  /// @return Result of the operation.
  Patch &secondPatch() { return patch(1); }
  /// @brief Provides the `secondPatch` operation.
  /// @return Result of the operation.
  const Patch &secondPatch() const { return patch(1); }
  /// @brief Provides the `firstSide` operation.
  /// @return Result of the operation.
  enum side firstSide() const { return side(0); }
  /// @brief Provides the `secondSide` operation.
  /// @return Result of the operation.
  enum side secondSide() const { return side(1); }
  /// @}

private:
  std::array<std::shared_ptr<Patch>, 2> patches_;
  std::array<enum side, 2> sides_;
};

/// @brief Multi-patch container class.
///
/// This class implements a container for a set of patches and their
/// topology, that is, the interface connections and outer boundary
/// faces.
template <typename Patch> class MultiPatch {

public:
  /// @brief Interface type.
  using interface_type = PatchInterface<Patch>;

  /// @brief Default constructor.
  MultiPatch() = default;

  /// @brief Copy constructor.
  /// @param other Second input value.
  MultiPatch(const MultiPatch &other)
      : patches_(other.patches_), interfaces_(other.interfaces_) {}

  /// @brief Move constructor.
  /// @param other Second input value.
  MultiPatch(MultiPatch &&other) noexcept {
    patches_.swap(other.patches_);
    interfaces_.swap(other.interfaces_);
  }

public:
  /// @brief Returns an iterator to the patches.
  /// @return Result of the operation.
  auto begin() { return patches_.begin(); }

  /// @brief Returns a const-iterator to the patches
  /// @{
  /// @return Result of the operation.
  auto begin() const { return patches_.begin(); }
  /// @brief Provides the `cbegin` operation.
  /// @return Result of the operation.
  auto cbegin() const noexcept { return patches_.cbegin(); }
  /// @}

  /// @brief Returns an iterator to the end of the patches.
  /// @return Result of the operation.
  auto end() { return patches_.end(); }

  /// @brief Returns a const-iterator to the end of the patches
  /// @{
  /// @return Result of the operation.
  auto end() const { return patches_.end(); }
  /// @brief Provides the `cend` operation.
  /// @return Result of the operation.
  auto cend() const noexcept { return patches_.cend(); }
  /// @}

  /// @brief Returns a reverse iterator to the patches.
  /// @return Result of the operation.
  auto rbegin() { return patches_.rbegin(); }

  /// @brief Returns a reverse const-iterator to the patches
  /// @{
  /// @return Result of the operation.
  auto rbegin() const { return patches_.rbegin(); }
  /// @brief Provides the `crbegin` operation.
  /// @return Result of the operation.
  auto crbegin() const noexcept { return patches_.crbegin(); }
  /// @}

  /// @brief Returns a reverse iterator to the end of the patches.
  /// @return Result of the operation.
  auto rend() { return patches_.rend(); }

  /// @brief Returns a reverse const-iterator to the end of the patches
  /// @{
  /// @return Result of the operation.
  auto rend() const { return patches_.rend(); }
  /// @brief Provides the `crend` operation.
  /// @return Result of the operation.
  auto crend() const noexcept { return patches_.crend(); }
  /// @}

public:
  /// @brief Returns the number of patches.
  /// @return Result of the operation.
  [[nodiscard]] std::size_t npatches() const { return patches_.size(); }

  /// @brief Returns the number of interfaces.
  /// @return Result of the operation.
  [[nodiscard]] std::size_t ninterfaces() const { return interfaces_.size(); }

  /// @brief Returns the number of outer boundaries.
  /// @return Result of the operation.
  [[nodiscard]] std::size_t nboundaries() const { return patches_.size(); }

public:
  /// @brief Adds a single patch
  /// @{
  /// @param patch Patch to process.
  /// @return Result of the operation.
  std::size_t addPatch(std::shared_ptr<Patch> patch) {
    std::size_t index = patches_.size();
    patches_.push_back(patch);
    return index;
  }

  /// @brief Provides the `addPatch` operation.
  /// @param patch Patch to process.
  /// @return Result of the operation.
  std::size_t addPatch(std::unique_ptr<Patch> patch) {
    std::size_t index = patches_.size();
    patches_.push_back(std::shared_ptr<Patch>(std::move(patch)));
    return index;
  }
  /// @}

  /// @brief Adds an interface between two patches identified by index.
  /// @param firstPatch Value of `firstPatch`.
  /// @param firstSide Value of `firstSide`.
  /// @param secondPatch Value of `secondPatch`.
  /// @param secondSide Value of `secondSide`.
  /// @return Result of the operation.
  std::size_t addInterface(std::size_t firstPatch, enum side firstSide,
                           std::size_t secondPatch, enum side secondSide) {
    assert(firstPatch < patches_.size());
    assert(secondPatch < patches_.size());
    return addInterface(patches_[firstPatch], firstSide, patches_[secondPatch],
                        secondSide);
  }

  /// @brief Adds an interface between two patches.
  /// @param firstPatch Value of `firstPatch`.
  /// @param firstSide Value of `firstSide`.
  /// @param secondPatch Value of `secondPatch`.
  /// @param secondSide Value of `secondSide`.
  /// @return Result of the operation.
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

  /// @brief Adds an interface object.
  /// @param patchInterface Value of `patchInterface`.
  /// @return Result of the operation.
  std::size_t addInterface(interface_type patchInterface) {
    return addInterface(patchInterface.patchPtr(0), patchInterface.side(0),
                        patchInterface.patchPtr(1), patchInterface.side(1));
  }

  /// @brief Removes a single interface.
  /// @param index Object index.
  void removeInterface(std::size_t index) {
    assert(index < interfaces_.size());
    interfaces_.erase(interfaces_.begin() + index);
  }

  /// @brief Removes all patches.
  void clear() {
    interfaces_.clear();
    patches_.clear();
  }

  /// @brief Returns a non-constant reference to a single patch.
  /// @param index Object index.
  /// @return Result of the operation.
  Patch &patch(std::size_t index) {
    assert(index < patches_.size());
    return *patches_[index];
  }

  /// @brief Returns a constant reference to a single patch.
  /// @param index Object index.
  /// @return Result of the operation.
  const Patch &patch(std::size_t index) const {
    assert(index < patches_.size());
    return *patches_[index];
  }

  /// @brief Returns a reference to the vector of patches
  /// @{
  /// @return Result of the operation.
  std::vector<std::shared_ptr<Patch>> &patches() { return patches_; }
  /// @brief Provides the `patches` operation.
  /// @return Result of the operation.
  const std::vector<std::shared_ptr<Patch>> &patches() const {
    return patches_;
  }
  /// @}

  /// @brief Returns a non-constant reference to a single interface.
  /// @param index Object index.
  /// @return Result of the operation.
  interface_type &interface(std::size_t index) {
    assert(index < interfaces_.size());
    return interfaces_[index];
  }

  /// @brief Returns a constant reference to a single interface.
  /// @param index Object index.
  /// @return Result of the operation.
  const interface_type &interface(std::size_t index) const {
    assert(index < interfaces_.size());
    return interfaces_[index];
  }

  /// @brief Returns the interfaces for range-based iteration
  /// @{
  /// @return Result of the operation.
  std::vector<interface_type> &interfaces() { return interfaces_; }
  /// @brief Provides the `interfaces` operation.
  /// @return Result of the operation.
  const std::vector<interface_type> &interfaces() const { return interfaces_; }
  /// @}

  /// @brief Returns the index of a given single patch
  /// @{
  /// @param patch Patch to process.
  /// @return Result of the operation.
  std::size_t findPatchIndex(const Patch &patch) const {
    return findPatchIndex(&patch);
  }

  /// @brief Provides the `findPatchIndex` operation.
  /// @param patch Patch to process.
  /// @return Result of the operation.
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
  /// @param patchInterface Value of `patchInterface`.
  /// @return Result of the operation.
  std::size_t findInterfaceIndex(const interface_type &patchInterface) const {
    return findInterfaceIndex(&patchInterface);
  }

  /// @brief Provides the `findInterfaceIndex` operation.
  /// @param patchInterface Value of `patchInterface`.
  /// @return Result of the operation.
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

  /// @brief Loads the multi-patch object from a Torch archive file.
  /// @tparam P Template parameter `P`.
  /// @param filename Path of the file to process.
  /// @param key Serialization key.
  /// @param options Configuration options.
  template <typename P = Patch>
    requires requires { typename P::value_type; }
  void load(const std::string &filename, const std::string &key = "multipatch",
            Options<typename P::value_type> options =
                Options<typename P::value_type>{}) {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    read(archive, key, options);
  }

  /// @brief Reads the multi-patch object from a Torch input archive.
  /// @tparam P Template parameter `P`.
  /// @param archive Serialization archive.
  /// @param key Serialization key.
  /// @param options Configuration options.
  /// @return Result of the operation.
  template <typename P = Patch>
    requires requires { typename P::value_type; }
  torch::serialize::InputArchive &read(torch::serialize::InputArchive &archive,
                                       const std::string &key = "multipatch",
                                       Options<typename P::value_type> options =
                                           Options<typename P::value_type>{}) {
    torch::Tensor data;
    archive.read(key + ".json", data);
    data = data.to(torch::kCPU, torch::kUInt8).contiguous();
    const auto *begin =
        reinterpret_cast<const char *>(data.data_ptr<uint8_t>());
    from_json(nlohmann::json::parse(std::string(begin, begin + data.numel())),
              options);
    return archive;
  }

  /// @brief Saves the multi-patch object to a Torch archive file.
  /// @param filename Path of the file to process.
  /// @param key Serialization key.
  void save(const std::string &filename,
            const std::string &key = "multipatch") const {
    torch::serialize::OutputArchive archive;
    write(archive, key).save_to(filename);
  }

  /// @brief Writes the multi-patch object into a Torch output archive.
  /// @param archive Serialization archive.
  /// @param key Serialization key.
  /// @return Result of the operation.
  torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "multipatch") const {
    const std::string serialized = to_json().dump();
    const auto data =
        torch::from_blob(const_cast<char *>(serialized.data()),
                         {static_cast<int64_t>(serialized.size())},
                         torch::TensorOptions{}.dtype(torch::kUInt8))
            .clone();
    archive.write(key + ".json", data);
    return archive;
  }

  /// @brief Returns true if patches and topology are close up to tolerances.
  /// @tparam P Template parameter `P`.
  /// @param other Second input value.
  /// @param rtol Value of `rtol`.
  /// @param atol Value of `atol`.
  /// @return Result of the operation.
  template <typename P = Patch>
    requires requires { typename P::value_type; }
  bool
  isclose(const MultiPatch &other,
          typename P::value_type rtol = typename P::value_type{1e-5},
          typename P::value_type atol = typename P::value_type{1e-8}) const {
    return jsonIsClose(to_json(), other.to_json(), rtol, atol);
  }

  /// @brief Returns true if patches and topology are exactly equal.
  /// @param other Second input value.
  /// @return Result of the operation.
  bool operator==(const MultiPatch &other) const {
    return to_json() == other.to_json();
  }

  /// @brief Returns true if patches or topology differ.
  /// @param other Second input value.
  /// @return Result of the operation.
  bool operator!=(const MultiPatch &other) const { return !(*this == other); }

  /// @brief Prints a human-readable representation of the multi-patch object.
  /// @param os Output stream.
  void pretty_print(std::ostream &os) const noexcept {
    os << "MultiPatch(\nparDim = " << Patch::parDim()
       << ", npatches = " << npatches() << ", ninterfaces = " << ninterfaces()
       << "\n";
    for (std::size_t patchIndex = 0; patchIndex < patches_.size();
         ++patchIndex) {
      const auto json = patches_[patchIndex]->to_json();
      os << "patch[" << patchIndex << "] = {geoDim = " << json["geoDim"]
         << ", degrees = " << json["degrees"] << "}\n";
    }
    for (std::size_t interfaceIndex = 0; interfaceIndex < interfaces_.size();
         ++interfaceIndex) {
      const auto &patchInterface = interfaces_[interfaceIndex];
      os << "interface[" << interfaceIndex << "] = {patch "
         << findPatchIndex(patchInterface.patchPtr(0).get()) << ", side "
         << static_cast<short_t>(patchInterface.side(0)) << " <-> patch "
         << findPatchIndex(patchInterface.patchPtr(1).get()) << ", side "
         << static_cast<short_t>(patchInterface.side(1)) << "}\n";
    }
    os << ")";
  }

  /// @brief Returns the multi-patch object as a JSON object.
  /// @return Result of the operation.
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

  /// @brief Updates the multi-patch object from a JSON object.
  /// @tparam P Template parameter `P`.
  /// @param json JSON value to process.
  /// @param options Configuration options.
  /// @return Result of the operation.
  template <typename P = Patch>
    requires requires { typename P::value_type; }
  MultiPatch &from_json(const nlohmann::json &json,
                        Options<typename P::value_type> options =
                            Options<typename P::value_type>{}) {
    if (json.at("parDim").get<short_t>() != Patch::parDim())
      throw std::runtime_error(
          "MultiPatch JSON provides an incompatible parametric dimension");

    const auto &patchJson = json.at("patches");
    if (!patchJson.is_array() ||
        (!patches_.empty() && patchJson.size() != patches_.size()))
      throw std::runtime_error(
          "MultiPatch JSON patch count does not match the patch container");

    auto parsedPatches = patches_;
    const bool createPatches = parsedPatches.empty();
    if (createPatches) {
      parsedPatches.reserve(patchJson.size());
      for (const auto &item : patchJson) {
        try {
          parsedPatches.push_back(
              createUniformBSpline<typename Patch::value_type, Patch::geoDim(),
                                   Patch::parDim()>(item, options));
        } catch (const std::runtime_error &) {
          parsedPatches.push_back(
              createNonUniformBSpline<typename Patch::value_type,
                                      Patch::geoDim(), Patch::parDim()>(
                  item, options));
        }
      }
    }

    const auto &interfaceJson = json.at("interfaces");
    if (!interfaceJson.is_array())
      throw std::runtime_error("MultiPatch JSON interfaces must be an array");

    std::vector<interface_type> parsedInterfaces;
    parsedInterfaces.reserve(interfaceJson.size());
    for (const auto &item : interfaceJson) {
      const auto patchIndices = item.at("patches").get<std::array<size_t, 2>>();
      const auto sides = item.at("sides").get<std::array<short_t, 2>>();
      if (patchIndices[0] >= parsedPatches.size() ||
          patchIndices[1] >= parsedPatches.size() || sides[0] <= none ||
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
          parsedPatches[patchIndices[0]], static_cast<enum side>(sides[0]),
          parsedPatches[patchIndices[1]], static_cast<enum side>(sides[1]));
    }

    if (!createPatches)
      for (std::size_t patchIndex = 0; patchIndex < parsedPatches.size();
           ++patchIndex)
        parsedPatches[patchIndex]->from_json(patchJson[patchIndex]);

    patches_ = std::move(parsedPatches);
    interfaces_ = std::move(parsedInterfaces);
    return *this;
  }

  /// @brief Returns the multi-patch object as an XML document.
  /// @param id Object identifier.
  /// @param label Object label.
  /// @param index Object index.
  /// @return Result of the operation.
  [[nodiscard]] pugi::xml_document
  to_xml(int id = 0, const std::string &label = "", int index = -1) const {
    pugi::xml_document doc;
    pugi::xml_node root = doc.append_child("xml");
    to_xml(root, id, label, index);
    return doc;
  }

  /// @brief Appends the multi-patch object to an XML node.
  /// @param root Root XML node.
  /// @param id Object identifier.
  /// @param label Object label.
  /// @param index Object index.
  /// @return Result of the operation.
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

  /// @brief Updates the multi-patch object from an XML document.
  /// @tparam P Template parameter `P`.
  /// @param doc Value of `doc`.
  /// @param id Object identifier.
  /// @param label Object label.
  /// @param index Object index.
  /// @param options Configuration options.
  /// @return Result of the operation.
  template <typename P = Patch>
    requires requires { typename P::value_type; }
  MultiPatch &from_xml(const pugi::xml_document &doc, int id = 0,
                       const std::string &label = "", int index = -1,
                       Options<typename P::value_type> options =
                           Options<typename P::value_type>{}) {
    return from_xml(doc.child("xml"), id, label, index, options);
  }

  /// @brief Updates the multi-patch object from an XML node.
  /// @tparam P Template parameter `P`.
  /// @param root Root XML node.
  /// @param id Object identifier.
  /// @param label Object label.
  /// @param index Object index.
  /// @param options Configuration options.
  /// @return Result of the operation.
  template <typename P = Patch>
    requires requires { typename P::value_type; }
  MultiPatch &from_xml(const pugi::xml_node &root, int id = 0,
                       const std::string &label = "", int index = -1,
                       Options<typename P::value_type> options =
                           Options<typename P::value_type>{}) {
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
    if (!patchRange ||
        std::string_view{patchRange.attribute("type").value()} != "id_range")
      throw std::runtime_error("MultiPatch XML has no valid patch ID range");

    std::stringstream rangeData(patchRange.child_value());
    int64_t firstPatch = 0;
    int64_t lastPatch = -1;
    if (!(rangeData >> firstPatch >> lastPatch) || firstPatch != 0 ||
        (!patches_.empty() &&
         lastPatch + 1 != static_cast<int64_t>(patches_.size())))
      throw std::runtime_error(
          "MultiPatch XML patch range does not match the patch container");

    auto parsedPatches = patches_;
    const bool createPatches = parsedPatches.empty();
    if (createPatches) {
      parsedPatches.reserve(static_cast<std::size_t>(lastPatch + 1));
      for (int64_t patchIndex = 0; patchIndex <= lastPatch; ++patchIndex) {
        try {
          parsedPatches.push_back(
              createUniformBSpline<typename Patch::value_type, Patch::geoDim(),
                                   Patch::parDim()>(root, patchIndex, "", -1,
                                                    options));
        } catch (const std::runtime_error &) {
          parsedPatches.push_back(
              createNonUniformBSpline<typename Patch::value_type,
                                      Patch::geoDim(), Patch::parDim()>(
                  root, patchIndex, "", -1, options));
        }
      }
    }

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

      if (firstPatchIndex >= parsedPatches.size() ||
          secondPatchIndex >= parsedPatches.size() || firstSide <= none ||
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
          parsedPatches[firstPatchIndex], static_cast<enum side>(firstSide),
          parsedPatches[secondPatchIndex], static_cast<enum side>(secondSide));
    }

    if (!createPatches)
      for (std::size_t patchIndex = 0; patchIndex < parsedPatches.size();
           ++patchIndex)
        parsedPatches[patchIndex]->from_xml(root, static_cast<int>(patchIndex));

    patches_ = std::move(parsedPatches);
    interfaces_ = std::move(parsedInterfaces);
    return *this;
  }

private:
  template <typename real_t>
  static bool jsonIsClose(const nlohmann::json &first,
                          const nlohmann::json &second, real_t rtol,
                          real_t atol) {
    if (first.type() != second.type())
      return first.is_number() && second.is_number() &&
             std::abs(first.template get<real_t>() -
                      second.template get<real_t>()) <=
                 atol + rtol * std::abs(second.template get<real_t>());
    if (first.is_number())
      return std::abs(first.template get<real_t>() -
                      second.template get<real_t>()) <=
             atol + rtol * std::abs(second.template get<real_t>());
    if (first.is_array()) {
      if (first.size() != second.size())
        return false;
      for (std::size_t i = 0; i < first.size(); ++i)
        if (!jsonIsClose(first[i], second[i], rtol, atol))
          return false;
      return true;
    }
    if (first.is_object()) {
      if (first.size() != second.size())
        return false;
      for (const auto &[key, value] : first.items()) {
        const auto it = second.find(key);
        if (it == second.end() || !jsonIsClose(value, *it, rtol, atol))
          return false;
      }
      return true;
    }
    return first == second;
  }

  /// @brief Vector of single-patch objects.
  std::vector<std::shared_ptr<Patch>> patches_;

  /// @brief Vector of patch-interface objects.
  std::vector<interface_type> interfaces_;
};

/// @brief Writes a human-readable multi-patch representation to a stream.
/// @tparam Patch Template parameter `Patch`.
/// @param os Output stream.
/// @param obj Object to process.
/// @return Result of the operation.
template <typename Patch>
std::ostream &operator<<(std::ostream &os, const MultiPatch<Patch> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Serializes a multi-patch object.
/// @tparam Patch Template parameter `Patch`.
/// @param archive Serialization archive.
/// @param obj Object to process.
/// @return Result of the operation.
template <typename Patch>
torch::serialize::OutputArchive &
operator<<(torch::serialize::OutputArchive &archive,
           const MultiPatch<Patch> &obj) {
  return obj.write(archive);
}

/// @brief Deserializes a multi-patch object.
/// @tparam Patch Template parameter `Patch`.
/// @param archive Serialization archive.
/// @param obj Object to process.
/// @return Result of the operation.
template <typename Patch>
torch::serialize::InputArchive &
operator>>(torch::serialize::InputArchive &archive, MultiPatch<Patch> &obj) {
  return obj.read(archive);
}

} // namespace iganet
