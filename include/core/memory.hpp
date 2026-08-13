/**
   @file core/memory.hpp

   @brief Memory debugger.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core/core.hpp>

#include <chrono>
#include <utility>

namespace iganet {

/// @brief Memory debugger.
/// @tparam id Compile-time identifier used to distinguish debugger instances.
template <std::size_t id = 0> class MemoryDebugger {
private:
  /// @brief Memory object.
  struct MemoryObject {
    std::string name_;
    int64_t bytes_;

    /// @brief Constructs a registered-memory record.
    /// @param name Descriptive name of the registered object.
    /// @param bytes Number of bytes attributed to the object.
    MemoryObject(std::string name, int64_t bytes)
        : name_(std::move(name)), bytes_(bytes) {}
  };

  /// @brief Map holding the list of registered objects.
  std::map<std::chrono::high_resolution_clock::time_point, MemoryObject>
      objects_;

  /// @brief Counter holding the number of registered objects.
  int64_t counter_;

  /// @brief Counter holding the memory of registered objects in bytes.
  int64_t bytes_;

  /// @brief Reference time point.
  std::chrono::high_resolution_clock::time_point init_;

  /// @brief Converts bytes into the best human-readable unit.
  /// @param bytes Number of bytes to format.
  /// @return A value with the most appropriate binary unit suffix.
  [[nodiscard]] std::string convert_bytes(int64_t bytes) const {
    if (bytes < 1024ull)
      return std::to_string(bytes) + "b";
    else if (bytes < 1024ull * 1024ull)
      return std::to_string(bytes / static_cast<double>(1024)) + "kb";
    else if (bytes < 1024ull * 1024ull * 1024ull)
      return std::to_string(bytes / static_cast<double>(1024 * 1024)) + "mb";
    else if (bytes < 1024ull * 1024ull * 1024ull * 1024ull)
      return std::to_string(bytes / static_cast<double>(1024 * 1024 * 1024)) +
             "gb";
    else
      return std::to_string(
                 bytes / static_cast<double>(1024) / static_cast<double>(1024) /
                 static_cast<double>(1024) / static_cast<double>(1024)) +
             "tb";
  }

public:
  /// @brief Default constructor.
  MemoryDebugger()
      : counter_(0), bytes_(0),
        init_(std::chrono::high_resolution_clock::now()) {}

  /// @brief Clears the memory debugger.
  void clear() {
    counter_ = 0;
    bytes_ = 0;
    objects_.clear();
  }

  /// @brief Returns a string representation of the memory debugger.
  /// @param os Stream that receives the registered objects and total usage.
  inline void pretty_print(std::ostream &os = Log(log::info)) const {
    using namespace std::literals;

    os << "Memory debugger (ID=" << std::to_string(id) << ")\n";
    for (const auto &obj : objects_)
      os << "[" << std::right << std::setw(10) << (obj.first - init_) / 1ns
         << "ns] " << std::right << std::setw(10) << obj.second.name_ << " "
         << std::right << std::setw(10) << convert_bytes(obj.second.bytes_)
         << "\n";
    os << "[     Total  ] " << std::right << std::setw(10) << counter_ << " "
       << std::right << std::setw(10) << convert_bytes(bytes_) << "\n";
  }

  /// @brief Registers a generic type with the memory debugger.
  /// @tparam T Type of the registered object.
  /// @param name Descriptive name under which to register the object.
  /// @param obj Object whose shallow `sizeof` value is recorded.
  template <typename T>
  void add(const std::string &name, [[maybe_unused]] const T &obj) {
    counter_++;
    bytes_ += sizeof(obj);
    objects_.insert(
        std::pair<std::chrono::high_resolution_clock::time_point, MemoryObject>(
            std::chrono::high_resolution_clock::now(),
            MemoryObject(name, sizeof(obj))));
  }

  /// @brief Registers a `torch::Tensor` with the memory debugger.
  /// @param name Descriptive name under which to register the tensor.
  /// @param tensor Tensor whose element storage size is recorded.
  void add(const std::string &name, const torch::Tensor &tensor) {
    counter_++;
    bytes_ += tensor.element_size() * tensor.numel();
    objects_.insert(
        std::pair<std::chrono::high_resolution_clock::time_point, MemoryObject>(
            std::chrono::high_resolution_clock::now(),
            MemoryObject(name, tensor.element_size() * tensor.numel())));
  }

  /// @brief Registers a `std::array` with the memory debugger.
  /// @tparam T Element type of the array.
  /// @tparam N Number of array elements.
  /// @param name Base name used for the element registrations.
  /// @param array Array whose elements are registered individually.
  template <typename T, std::size_t N>
  void add(const std::string &name, const std::array<T, N> &array) {
    for (std::size_t i = 0; i < N; ++i)
      add(name + std::to_string(i), array[i]);
  }
};

/// @brief Prints a memory debugger object.
/// @tparam id Identifier of the debugger instance.
/// @param os Output stream.
/// @param obj Memory debugger to print.
/// @return `os` after the report has been written.
template <std::size_t id>
inline std::ostream &operator<<(std::ostream &os,
                                const MemoryDebugger<id> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief System-wide memory debugger.
static MemoryDebugger<std::numeric_limits<std::size_t>::max()>
    global_memory_debugger;

/// @brief Registers an object with the system-wide memory debugger.
/// @param obj Object to register; its expression is also used as its name.
#define register_memory(obj) ::iganet::global_memory_debugger.add(#obj, obj)

} // namespace iganet
