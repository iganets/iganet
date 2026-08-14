/**
   @file core/options.hpp

   @brief Options.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <iganet/core/dtype.hpp>
#include <iganet/utils/fqn.hpp>
#include <iganet/utils/getenv.hpp>

namespace iganet {

/// @brief Guesses the accelerator device index for the current process.
/// @return With MPI enabled, the process rank modulo the configured or detected
/// device count; otherwise zero.
/// @note The `IGANET_DEVICE_COUNT` environment variable overrides automatic
/// device-count detection.
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
/// dtype from the template argument and the selection of the device.
///
/// @tparam real_t Type of real-valued data.
template <typename real_t>
  requires DType<real_t>
class Options : private iganet::utils::FullQualifiedName {
public:
  /// @brief Default constructor.
  Options()
      : options_(
            torch::TensorOptions()
                .dtype(::iganet::dtype_v<real_t>)
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

  /// @brief Constructor from `torch::TensorOptions`.
  /// @param options Tensor options to adopt while enforcing `real_t` as dtype.
  explicit Options(torch::TensorOptions &&options)
      : options_(options.dtype(::iganet::dtype_v<real_t>)) {}

  /// @brief Implicit conversion operator.
  ///
  /// @note Do not mark this operator 'explicit' as this will prevent that
  /// objects of type Options are implicitly converted into objects of type
  /// torch::TensorOptions.
  /// @return A copy of the underlying LibTorch tensor options.
  operator torch::TensorOptions() const { return options_; }

  /// @brief Returns the `device` property.
  /// @return Configured LibTorch device.
  inline torch::Device device() const noexcept { return options_.device(); }

  /// @brief Returns the `device_index` property.
  /// @return Configured device index.
  inline int32_t device_index() const noexcept {
    return options_.device_index();
  }

  /// @brief Returns the `dtype` property.
  /// @return LibTorch dtype corresponding to `real_t`.
  static inline torch::Dtype dtype() noexcept {
    return ::iganet::dtype_v<real_t>;
  }

  /// @brief Returns the `layout` property.
  /// @return Configured tensor layout.
  inline torch::Layout layout() const noexcept { return options_.layout(); }

  /// @brief Returns the `requires_grad` property.
  /// @return Whether tensors require gradient computation.
  inline bool requires_grad() const noexcept {
    return options_.requires_grad();
  }

  /// @brief Returns the `pinned_memory` property.
  /// @return Whether pinned memory is enabled.
  inline bool pinned_memory() const noexcept {
    return options_.pinned_memory();
  }

  /// @brief Returns whether the layout is sparse.
  /// @return `true` if the configured layout is sparse; otherwise `false`.
  inline bool is_sparse() const noexcept { return options_.is_sparse(); }

  /// @brief Returns a new Options object with the `device` property as given.
  /// @param device Device to use in the returned options.
  /// @return A copy of these options with the requested device.
  inline Options<real_t> device(torch::Device device) const noexcept {
    return Options(options_.device(device));
  }

  /// @brief Returns a new Options object with the `device_index` property as
  /// given.
  /// @param device_index Device index to use in the returned options.
  /// @return A copy of these options with the requested device index.
  inline Options<real_t> device_index(int16_t device_index) const noexcept {
    return Options(options_.device_index(device_index));
  }

  /// @brief Returns a new Options object with the `dtype` property as given.
  /// @tparam other_t C++ scalar type determining the new LibTorch dtype.
  /// @return A copy of these options represented as `Options<other_t>`.
  template <typename other_t> inline Options<other_t> dtype() const noexcept {
    return Options<other_t>(options_.dtype(::iganet::dtype_v<other_t>));
  }

  /// @brief Returns a new Options object with the `layout` property as given.
  /// @param layout Tensor layout to use in the returned options.
  /// @return A copy of these options with the requested layout.
  inline Options<real_t> layout(torch::Layout layout) const noexcept {
    return Options(options_.layout(layout));
  }

  /// @brief Returns a new Options object with the `requires_grad` property as
  /// given.
  /// @param requires_grad Whether tensors should require gradient computation.
  /// @return A copy of these options with the requested gradient setting.
  inline Options<real_t> requires_grad(bool requires_grad) const noexcept {
    return Options(options_.requires_grad(requires_grad));
  }

  /// @brief Returns a new Options object with the `pinned_memory` property as
  /// given.
  /// @param pinned_memory Whether tensors should use pinned memory.
  /// @return A copy of these options with the requested pinned-memory setting.
  inline Options<real_t> pinned_memory(bool pinned_memory) const noexcept {
    return Options(options_.pinned_memory(pinned_memory));
  }

  /// @brief Returns a new Options object with the `memory_format` property as
  /// given.
  /// @param memory_format Memory format to use in the returned options.
  /// @return A copy of these options with the requested memory format.
  inline Options<real_t>
  memory_format(torch::MemoryFormat memory_format) const noexcept {
    return Options(options_.memory_format(memory_format));
  }

  /// @brief Data type.
  using value_type = real_t;

  /// @brief Returns a string representation of the Options object.
  /// @param os Stream that receives the representation.
  inline void pretty_print(std::ostream &os) const noexcept override {
    os << name() << "(\noptions = " << options_ << "\n)";
  }

private:
  /// @brief Tensor options.
  const torch::TensorOptions options_;
};

/// @brief Prints an Options object.
/// @tparam real_t Scalar type represented by the options.
/// @param os Output stream.
/// @param obj Options object to print.
/// @return `os` after the representation has been written.
template <typename real_t>
inline std::ostream &operator<<(std::ostream &os, const Options<real_t> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Options dispatcher.
/// @tparam real_t Scalar type represented by the nested options.
template <typename real_t>
class Options<Options<real_t>> : public Options<real_t> {
  using Options<real_t>::Options;
};

} // namespace iganet
