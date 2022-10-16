/**
   @file include/core.hpp

   @brief Core components

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <array>
#include <initializer_list>
#include <tuple>
#include <vector>

#include <matplot/matplot.h>
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/types.h>

#pragma once

namespace iganet {

#define short_t unsigned short int

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
  
  std::ostream& verbose(std::ostream& os) { os.iword(get_iomanip()) = 1; return os; } 
  std::ostream& regular(std::ostream& os) { os.iword(get_iomanip()) = 0; return os; }
  
  bool is_verbose(std::ostream& os) { return os.iword(get_iomanip()) != 0; }
  /// @}

  /// LibTorch core object handles the automated determination of dtype
  /// from the template argument and the selection of the device
  ///
  /// @tparam real_t Type of real-valued data
  template<typename real_t>
  class core {
  public:
    /// Default constructor
    core()
      : options_(torch::TensorOptions()
                 .dtype(dtype<real_t>())
                 .device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
                 .requires_grad(true))
    {}

    /// Constructor with user-defined device type
    core(c10::DeviceType deviceType)
      : options_(torch::TensorOptions()
                 .dtype(dtype<real_t>())
                 .device(deviceType)
                 .requires_grad(true))
    {}

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

    /// @brief Returns constant reference to options
    const torch::TensorOptions options() const
    {
      return options_;
    }
    
  protected:
    /// @brief Tensor options
    const torch::TensorOptions options_;

    /// @brief String storing the full qualified name of the object
    mutable at::optional<std::string> name_;
  };

  /// @brief Initializes the library
  inline void init(std::ostream& os = std::cout)
  {
    os << "LibTorch version: "
       << TORCH_VERSION_MAJOR << "."
       << TORCH_VERSION_MINOR << "."
       << TORCH_VERSION_PATCH << std::endl;
    torch::manual_seed(1);
  }

} // namespace iganet

namespace std {
  
  /// Print (as string) an array of torch::Tensor objects
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
    for (auto i : obj)
      os << ((i.sizes().size() == 1) ? i.view({1,i.size(0)}) : i) << std::endl;
    os << ")";
    
    return os;
  }

} // namespace std
  

