/**
   @file include/options.hpp

   @brief Options

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core.hpp>
#include <utils/fqn.hpp>
#include <utils/getenv.hpp>

#include <torch/torch.h>

namespace iganet {

struct half {};

/// @brief Concept to identify template parameters that are acceptable as DTypes
template <typename T>
concept DType =
    std::is_same_v<T, bool> || std::is_same_v<T, char> ||
    std::is_same_v<T, short> || std::is_same_v<T, int> ||
    std::is_same_v<T, long> || std::is_same_v<T, long long> ||
    std::is_same_v<T, half> || std::is_same_v<T, float> ||
    std::is_same_v<T, double> || std::is_same_v<T, std::complex<half>> ||
    std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

/// Determines the LibTorch dtype from template parameter
///
/// @tparam T C++ type
///
/// @result Torch type corresponding to the C++ type
/// @{
template <typename T>
  requires DType<T>
inline constexpr torch::Dtype dtype();

template <> inline constexpr torch::Dtype dtype<bool>() { return torch::kBool; }

template <> inline constexpr torch::Dtype dtype<char>() { return torch::kChar; }

template <> inline constexpr torch::Dtype dtype<short>() {
  return torch::kShort;
}

template <> inline constexpr torch::Dtype dtype<int>() { return torch::kInt; }

template <> inline constexpr torch::Dtype dtype<long>() { return torch::kLong; }

template <> inline constexpr torch::Dtype dtype<long long>() {
  return torch::kLong;
}

template <> inline constexpr torch::Dtype dtype<half>() { return torch::kHalf; }

template <> inline constexpr torch::Dtype dtype<float>() {
  return torch::kFloat;
}

template <> inline constexpr torch::Dtype dtype<double>() {
  return torch::kDouble;
}

template <> inline constexpr torch::Dtype dtype<std::complex<half>>() {
  return at::kComplexHalf;
}

template <> inline constexpr torch::Dtype dtype<std::complex<float>>() {
  return at::kComplexFloat;
}

template <> inline constexpr torch::Dtype dtype<std::complex<double>>() {
  return at::kComplexDouble;
}
/// @}

inline int guess_device_index() {
#ifdef IGANET_WITH_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank %
         utils::getenv("IGANET_DEVICE_COUNT", (torch::cuda::is_available()
                                                  ? torch::cuda::device_count()
                                                  : (torch::xpu::is_available() ? torch::xpu::device_count() ? 1)));
#else
  return 0;
#endif
}

/// @brief The Options class handles the automated determination of
/// dtype from the template argument and the selection of the device
///
/// @tparam real_t Type of real-valued data
template <typename real_t>
  requires DType<real_t>
class Options : private iganet::utils::FullQualifiedName {
public:
  /// Default constructor
  Options()
      : options_(
            torch::TensorOptions()
                .dtype(::iganet::dtype<real_t>())
                .device_index(utils::getenv("IGANET_DEVICE_INDEX",
                                            iganet::guess_device_index()))
                .device(
                    (utils::getenv("IGANET_DEVICE", std::string{}) == "CPU")
                        ? torch::kCPU
                    : (utils::getenv("IGANET_DEVICE", std::string{}) == "CUDA")
                        ? torch::kCUDA
                    : (utils::getenv("IGANET_DEVICE", std::string{}) == "HIP")
                        ? torch::kHIP
                    : (utils::getenv("IGANET_DEVICE", std::string{}) == "MPS")
                        ? torch::kMPS
                    : (utils::getenv("IGANET_DEVICE", std::string{}) == "XLA")
                        ? torch::kXLA
                    : (utils::getenv("IGANET_DEVICE", std::string{}) == "XPU")
                        ? torch::kXPU
                        : (torch::cuda::is_available()
                               ? torch::kCUDA
                               : (torch::xpu::is_available() ? torch::kXPU
                                                             : torch::kCPU)))) {
  }

  /// Constructor from torch::TensorOptions
  explicit Options(torch::TensorOptions &&options)
      : options_(options.dtype(::iganet::dtype<real_t>())) {}

  /// @brief Implicit conversion operator
  ///
  /// @note Do not mark this operator 'explicit' as this will prevent that objects of type Options are implicitly converted into objects of type torch::TensorOptions
  operator torch::TensorOptions() const { return options_; }

  /// @brief Returns the `device` property
  inline torch::Device device() const noexcept { return options_.device(); }

  /// @brief Returns the `device_index` property
  inline int32_t device_index() const noexcept {
    return options_.device_index();
  }

  /// @brief Returns the `dtype` property
  static inline torch::Dtype dtype() noexcept {
    return ::iganet::dtype<real_t>();
  }

  /// @brief Returns the `layout` property
  inline torch::Layout layout() const noexcept { return options_.layout(); }

  /// @brief Returns the `requires_grad` property
  inline bool requires_grad() const noexcept {
    return options_.requires_grad();
  }

  /// @brief Returns the `pinned_memory` property
  inline bool pinned_memory() const noexcept {
    return options_.pinned_memory();
  }

  /// @brief Returns if the layout is sparse
  inline bool is_sparse() const noexcept { return options_.is_sparse(); }

  /// @brief Returns a new Options object with the `device` property as given
  inline Options<real_t> device(torch::Device device) const noexcept {
    return Options(options_.device(device));
  }

  /// @brief Returns a new Options object with the `device_index` property as
  /// given
  inline Options<real_t> device_index(int16_t device_index) const noexcept {
    return Options(options_.device_index(device_index));
  }

  /// @brief Returns a new Options object with the `dtype` property as given
  template <typename other_t> inline Options<other_t> dtype() const noexcept {
    return Options<other_t>(options_.dtype(::iganet::dtype<other_t>()));
  }

  /// @brief Returns a new Options object with the `layout` property as given
  inline Options<real_t> layout(torch::Layout layout) const noexcept {
    return Options(options_.layout(layout));
  }

  /// @brief Returns a new Options object with the `requires_grad` property as
  /// given
  inline Options<real_t> requires_grad(bool requires_grad) const noexcept {
    return Options(options_.requires_grad(requires_grad));
  }

  /// @brief Returns a new Options object with the `pinned_memory` property as
  /// given
  inline Options<real_t> pinned_memory(bool pinned_memory) const noexcept {
    return Options(options_.pinned_memory(pinned_memory));
  }

  /// @brief Returns a new Options object with the `memory_format` property as
  /// given
  inline Options<real_t>
  memory_format(torch::MemoryFormat memory_format) const noexcept {
    return Options(options_.memory_format(memory_format));
  }

  /// @brief Data type
  using value_type = real_t;

  /// @brief Returns a string representation of the Options object
  inline void
  pretty_print(std::ostream &os) const noexcept override {
    os << name() << "(\noptions = " << options_ << "\n)";
  }

private:
  /// @brief Tensor options
  const torch::TensorOptions options_;
};

/// @brief Print (as string) an Options object
template <typename real_t>
inline std::ostream &operator<<(std::ostream &os, const Options<real_t> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Options dispatcher
template <typename real_t>
class Options<Options<real_t>> : public Options<real_t> {
  using Options<real_t>::Options;
};

} // namespace iganet
