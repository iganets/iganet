/**
   @file include/core.hpp

   @brief Core components

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <initializer_list>
#include <tuple>
#include <vector>

#ifdef WITH_GISMO
#include <gismo.h>
#endif

#undef real_t
#undef index_t
#undef short_t

#ifdef WITH_MATPLOT
#ifdef __CUDACC__
#pragma nv_diag_suppress 611
#endif
#include <matplot/matplot.h>
#ifdef __CUDACC__
#pragma nv_diag_default 611
#endif
#endif

#include <pugixml.hpp>
#include <nlohmann/json.hpp>

#include <torch/torch.h>
#include <torch/csrc/api/include/torch/types.h>

namespace iganet {

  using short_t = unsigned short int;

  namespace literals {
    inline int64_t operator""_i64(unsigned long long value) { return value; };
  }
  
  using TensorArray1 = std::array<torch::Tensor,1>;
  using TensorArray2 = std::array<torch::Tensor,2>;
  using TensorArray3 = std::array<torch::Tensor,3>;
  using TensorArray4 = std::array<torch::Tensor,4>;

  /// Determines the LibTorch dtype from template parameter
  ///
  /// @tparam T C++ type
  ///
  /// @result Torch type corresponding to the C++ type
  /// @{
  template<typename real_t>
  inline constexpr auto dtype() { return torch::kByte; }

  template<>
  inline constexpr auto dtype<double>() { return torch::kFloat64; }

  template<>
  inline constexpr auto dtype<float>() { return torch::kFloat32; }

  template<>
  inline constexpr auto dtype<long int>() { return torch::kLong; }

  template<>
  inline constexpr auto dtype<int>() { return torch::kInt; };

  template<>
  inline constexpr auto  dtype<short>() { return torch::kShort; }

  template<>
  inline constexpr auto dtype<char>() { return torch::kChar; };
  /// @}

  /// Stream manipulator
  /// @{
  inline int get_iomanip()
  {
    static int i = std::ios_base::xalloc();
    return i;
  }

  inline std::ostream& verbose(std::ostream& os) { os.iword(get_iomanip()) = 1; return os; }
  inline std::ostream& regular(std::ostream& os) { os.iword(get_iomanip()) = 0; return os; }

  inline bool is_verbose(std::ostream& os) { return os.iword(get_iomanip()) != 0; }
  /// @}

  /// @brief Full qualified name descriptor
  class fqn {
  public:
    /// @brief Returns the full qualified name of the object
    ///
    /// @result Full qualified name of the object as string
    inline const virtual std::string& name() const noexcept
    {
      // If the name optional is empty at this point, we grab the name of the
      // dynamic type via RTTI. Note that we cannot do this in the constructor,
      // because in the constructor of a base class `this` always refers to the base
      // type. Inheritance effectively does not work in constructors. Also this note
      // from http://en.cppreference.com/w/cpp/language/typeid:
      // If typeid is used on an object under construction or destruction (in a
      // destructor or in a constructor, including constructor's initializer list
      // or default member initializers), then the std::type_info object referred
      // to by this typeid represents the class that is being constructed or
      // destroyed even if it is not the most-derived class.
      if (!name_.has_value()) {
        name_ = c10::demangle(typeid(*this).name());
#if defined(_WIN32)
        // Windows adds "struct" or "class" as a prefix.
        if (name_->find("struct ") == 0) {
          name_->erase(name_->begin(), name_->begin() + 7);
        } else if (name_->find("class ") == 0) {
          name_->erase(name_->begin(), name_->begin() + 6);
        }
#endif // defined(_WIN32)
        }
      return *name_;
    }
  protected:
    /// @brief String storing the full qualified name of the object
    mutable at::optional<std::string> name_;
  };

  /// @brief LibTorch core object handles the automated determination
  /// of dtype from the template argument and the selection of the
  /// device
  ///
  /// @tparam real_t Type of real-valued data
  template<typename real_t, bool memory_optimized = false>
  class core : public fqn {
  public:
    /// Default constructor
    core()
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
                 .requires_grad(true))
    {}

    /// Constructor with user-defined device type
    core(c10::DeviceType deviceType)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .device(deviceType)
                 .requires_grad(true))
    {}

    /// Constructor with user-defined device
    core(c10::Device device)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .device(device)
                 .requires_grad(true))
    {}

    /// Constructor with user-defined layout
    core(torch::Layout layout)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .layout(layout)
                 .requires_grad(true))
    {}

    /// Constructor with user-defined memory format
    core(torch::MemoryFormat memoryFormat)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .memoryFormat(memoryFormat)
                 .requires_grad(true))
    {}

    /// Constructor with user-defined gradient calculation
    core(bool requiresGrad)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
                 .requires_grad(requiresGrad))
    {}

    /// Constructor with user-defined device type and gradient calculation
    core(c10::DeviceType deviceType, bool requiresGrad)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .device(deviceType)
                 .requires_grad(requiresGrad))
    {}

    /// Constructor with user-defined device and gradient calculation
    core(c10::Device device, bool requiresGrad)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .device(device)
                 .requires_grad(requiresGrad))
    {}

    /// Constructor with user-defined device and all other parameters
    core(c10::DeviceType deviceType, torch::Layout layout, torch::MemoryFormat memoryFormat,
         bool requiresGrad, bool pinnedMemory)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .device(deviceType)
                 .layout(layout)
                 .memoryFormat(memoryFormat)
                 .requires_grad(requiresGrad)
                 .pinnedMemory(pinnedMemory))
    {}

    /// Constructor with user-defined device and all other parameters
    core(c10::Device device, torch::Layout layout, torch::MemoryFormat memoryFormat,
         bool requiresGrad, bool pinnedMemory)
      : options_(torch::TensorOptions()
                 .dtype(::iganet::dtype<real_t>())
                 .device(device)
                 .layout(layout)
                 .memoryFormat(memoryFormat)
                 .requires_grad(requiresGrad)
                 .pinnedMemory(pinnedMemory))
    {}    

    /// Destructor
    virtual ~core() {}

    /// @brief Returns the `device` property
    torch::Device device() const noexcept {
      return options_.device();
    }

    /// @brief Returns the `device_index` property
    int32_t device_index() const noexcept {
      return options_.device_index();
    }

    /// @brief Returns the `dtype` property
    caffe2::TypeMeta dtype() const noexcept {
      return options_.dtype();
    }

    /// @brief Returns the `layout` property
    torch::Layout layout() const noexcept {
      return options_.layout();
    }
    
    /// @brief Returns the `requires_grad` property
    bool requires_grad() const noexcept {
      return options_.requires_grad();
    }

    /// @brief Returns the `pinned_memory` property
    bool pinned_memory() const noexcept {
      return options_.pinned_memory();
    }

    /// @brief Returns if the layout is sparse
    bool is_sparse() const noexcept {
      return options_.is_sparse();
    }

    /// Sets the `device` property
    virtual core<real_t>& device(torch::Device device) noexcept {
      options_ = options_.device(device);
      return *this;
    }

    /// Sets the `device_index` property
    virtual core<real_t>& device_index(int16_t device_index) noexcept {
      options_ = options_.device_index(device_index);
      return *this;
    }
    
    /// Sets the `dtype` property
    virtual core<real_t>& dtype(caffe2::TypeMeta dtype) noexcept {
      options_ = options_.dtype(dtype);
      return *this;
    }

    /// Sets the `dtype` property
    virtual core<real_t>& dtype(torch::ScalarType dtype) noexcept {
      options_ = options_.dtype(dtype);
      return *this;
    }

    /// Sets the `layout` property
    virtual core<real_t>& layout(torch::Layout layout) noexcept {
      options_ = options_.layout(layout);
      return *this;
    }
    
    /// Sets the `requires_grad` property
    virtual core<real_t>& requires_grad(bool requires_grad) noexcept {
      options_ = options_.requires_grad(requires_grad);
      return *this;
    }

    /// Sets the `pinned_memory` property
    virtual core<real_t>& pinned_memory(bool pinned_memory) noexcept {
      options_ = options_.pinned_memory(pinned_memory);
      return *this;
    }

    /// Sets the `memory_format` property
    virtual core<real_t>& memory_format(torch::MemoryFormat memory_format) noexcept {
      options_ = options_.memory_format(memory_format);
      return *this;
    }
    
    /// @brief Returns constant reference to options
    const torch::TensorOptions& options() const
    {
      return options_;
    }
    
    /// @brief Serialization to JSON
    virtual nlohmann::json to_json() const
    {
      return "not implemented";
    }

    /// @brief Data type
    using value_type = real_t;

    /// @brief Returns a string representation of the core object
    inline virtual void pretty_print(std::ostream& os = std::cout) const
    {
      os << name()
         << "(\nreal_t = " << typeid(real_t).name()
         << ", memory_optimized = " << memory_optimized_
         << ", options = " << options_
         << "\n)";
    }
    
  protected:
    /// @brief Optimize for memory usage
    static constexpr bool memory_optimized_ = memory_optimized;

    /// @brief Tensor options
    torch::TensorOptions options_;
  };

  /// @brief Print (as string) a core object
  template<typename real_t>
  inline std::ostream& operator<<(std::ostream& os,
                                  const core<real_t>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

  /// @brief Dispatcher
  template<typename real_t, bool memory_optimized, bool memory_optimized_>
  class core<core<real_t, memory_optimized>, memory_optimized_> : public core<real_t, memory_optimized>
  {
    using core<real_t, memory_optimized>::core;
  };

  /// @brief Initializes the library
  inline void init(std::ostream& os = std::clog)
  {
    os << "LibTorch version: "
       << TORCH_VERSION_MAJOR << "."
       << TORCH_VERSION_MINOR << "."
       << TORCH_VERSION_PATCH << "\n";
    torch::manual_seed(1);
  }

} // namespace iganet

namespace std {

  /// Print (as string) an std::array of torch::Tensor objects
  template<std::size_t N>
  inline std::ostream& operator<<(std::ostream& os,
                                  const std::array<torch::Tensor, N>& obj)
  {
    at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
    // Windows adds "struct" or "class" as a prefix.
    if (name_->find("struct ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 7);
    } else if (name_->find("class ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 6);
    }
#endif // defined(_WIN32)

    os << *name_ << "(\n";
    for (const auto& i : obj)
      if (!i.numel())
        os << "{}\n";
      else
        os << ((i.sizes().size() == 1) ? i.view({1,i.size(0)}) : i) << "\n";
    os << ")";

    return os;
  }

  /// Print (as string) an std::array of generic objects
  template<typename T, std::size_t N>
  inline std::ostream& operator<<(std::ostream& os,
                                  const std::array<T, N>& obj)
  {
    at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
    // Windows adds "struct" or "class" as a prefix.
    if (name_->find("struct ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 7);
    } else if (name_->find("class ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 6);
    }
#endif // defined(_WIN32)

    os << *name_ << "(";
    for (const auto& i : obj)
      os << i << (&i==&(*obj.rbegin()) ? "" : ",");
    os << ")";

    return os;
  }

  namespace detail {
    template<typename... Ts, std::size_t... Is>
    inline std::ostream& output_tuple(std::ostream& os,
                                      const std::tuple<Ts...>& obj,
                                      std::index_sequence<Is...>)
    {
      (..., (os << std::get<Is>(obj) << "\n"));
      return os;
    }

  } // namespace detail

  /// Print (as string) an std::tuple of generic objects
  template<typename... Ts>
  inline std::ostream& operator<<(std::ostream& os,
                                  const std::tuple<Ts...>& obj)
  {
    at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
    // Windows adds "struct" or "class" as a prefix.
    if (name_->find("struct ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 7);
    } else if (name_->find("class ") == 0) {
      name_->erase(name_->begin(), name_->begin() + 6);
    }
#endif // defined(_WIN32)

    os << *name_ << "(\n";
    detail::output_tuple(os, obj, std::make_index_sequence<sizeof...(Ts)>());
    os << "\n)";

    return os;
  }

} // namespace std
