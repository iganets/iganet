/**
   @file include/blocktensor.hpp

   @brief Compile-time block-tensor

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <array>
#include <exception>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include <core.hpp>

#pragma once

namespace iganet {

  template<typename T>
  struct is_shared_ptr : std::false_type {};
  
  template<typename T>
  struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

  template<typename T>
  inline auto make_shared(T arg)
  {
    if constexpr (is_shared_ptr<typename std::decay<T>::type>::value)
      return std::forward<T>(arg);
    else
      return std::make_shared<T>(arg);
  }

  /// @brief Compile-time block-tensor  
  template<typename T, short_t Rows, short_t Cols>
  class BlockTensor : public iganet::core<T> {
    
  private:
    /// @brief Array storing the data
    std::array<std::shared_ptr<T>, Rows*Cols> data_;

  public:
    /// @brief Data constructor
    template<typename... Ts>
    BlockTensor(Ts&&... data)
      : data_({make_shared<Ts>(data)...})
    {}    
    
    /// @brief Returns the number of rows
    inline static constexpr short_t rows()
    {
      return Rows;
    }

    /// @brief Returns the number of columns
    inline static constexpr short_t cols()
    {
      return Cols;
    }

    /// @brief Returns the number of entries
    inline static constexpr short_t entries()
    {
      return Rows*Cols;
    }

    inline std::array<std::shared_ptr<T>, Rows*Cols> data()
    {
      return data_;      
    }

    /// @brief Returns a constant shared pointer to entry (idx)
    inline const std::shared_ptr<T>& operator[](short_t idx) const
    {
      return data_[idx];
    }

    /// @brief Returns a non-constant shared pointer to entry (idx)
    inline std::shared_ptr<T>& operator[](short_t idx)
    {
      return data_[idx];
    }
    
    /// @brief Returns a constant reference to entry (idx)
    inline const T& operator()(short_t idx) const
    {
      return *data_[idx];
    }

    /// @brief Returns a non-constant reference to entry (idx)
    inline T& operator()(short_t idx)
    {
      return *data_[idx];
    }
    
    /// @brief Returns a constant reference to entry (row, col)
    inline const T& operator()(short_t row, short_t col) const
    {
      return *data_[Cols*row+col];
    }

    /// @brief Returns a non-constant reference to entry (row, col)
    inline T& operator()(short_t row, short_t col)
    {
      return *data_[Cols*row+col];
    }

    /// @brief Returns the transpose of the block-tensor
    inline auto tr() const
    {
      BlockTensor<T, Cols, Rows> result;
      for (short_t row = 0; row<Rows; ++row)
        for (short_t col = 0; col<Cols; ++col)
          result[Cols*row+col] = data_[Rows*col+row];
      return result;
    }

    /// @brief Returns the inverse of the block-tensor
    inline auto inv() const
    {      
      if constexpr (Rows == 1 && Cols == 1) {
        BlockTensor<T, Rows, Cols> result;
        result[0] = std::make_shared<T>(torch::reciprocal(*data_[0]));
        return result;
      }
      else if constexpr (Rows == 2 && Cols == 2) {        
        auto det = torch::mul(*data_[0], *data_[3])
                 - torch::mul(*data_[1], *data_[2]);
        
        BlockTensor<T, Rows, Cols> result;
        result[0] = std::make_shared<T>(torch::div(*data_[3], det));
        result[1] = std::make_shared<T>(torch::div(*data_[2],-det));
        result[2] = std::make_shared<T>(torch::div(*data_[1],-det));
        result[3] = std::make_shared<T>(torch::div(*data_[0], det));
        return result;
      }
      else if constexpr (Rows == 3 && Cols == 3) {
        // DET  =   a11(a33a22-a32a23)
        //        - a21(a33a12-a32a13)
        //        + a31(a23a12-a22a13)
        auto det = torch::mul(*data_[0],
                              torch::mul(*data_[8], *data_[4]) -
                              torch::mul(*data_[7], *data_[5]))
                 - torch::mul(*data_[3],
                              torch::mul(*data_[8], *data_[1]) -
                              torch::mul(*data_[7], *data_[2]))
                 - torch::mul(*data_[6],
                              torch::mul(*data_[5], *data_[1]) -
                              torch::mul(*data_[4], *data_[2]));

        // |  a33a22-a32a23  -(a33a12-a32a13)   a23a12-a22a13 |
        // |-(a33a21-a31a23)   a33a11-a31a13  -(a23a11-a21a13)|
        // |  a32a21-a31a22  -(a32a11-a31a12)   a22a11-a21a12 |
             
        auto a11 = torch::mul(*data_[8], *data_[4]) - torch::mul(*data_[7], *data_[5]);
        auto a12 = torch::mul(*data_[7], *data_[2]) - torch::mul(*data_[8], *data_[1]);
        auto a13 = torch::mul(*data_[5], *data_[1]) - torch::mul(*data_[4], *data_[2]);
        auto a21 = torch::mul(*data_[6], *data_[5]) - torch::mul(*data_[8], *data_[3]);
        auto a22 = torch::mul(*data_[8], *data_[0]) - torch::mul(*data_[6], *data_[2]);
        auto a23 = torch::mul(*data_[3], *data_[2]) - torch::mul(*data_[5], *data_[0]);
        auto a31 = torch::mul(*data_[7], *data_[3]) - torch::mul(*data_[6], *data_[4]);
        auto a32 = torch::mul(*data_[6], *data_[1]) - torch::mul(*data_[7], *data_[0]);
        auto a33 = torch::mul(*data_[4], *data_[0]) - torch::mul(*data_[3], *data_[1]);
        
        BlockTensor<T, Rows, Cols> result;
        result[0] = std::make_shared<T>(torch::div(a11, det));
        result[1] = std::make_shared<T>(torch::div(a21, det));
        result[2] = std::make_shared<T>(torch::div(a31, det));
        result[3] = std::make_shared<T>(torch::div(a12, det));
        result[4] = std::make_shared<T>(torch::div(a22, det));
        result[5] = std::make_shared<T>(torch::div(a32, det));
        result[6] = std::make_shared<T>(torch::div(a13, det));
        result[7] = std::make_shared<T>(torch::div(a23, det));
        result[8] = std::make_shared<T>(torch::div(a33, det));
        return result;
      }
      else
        throw std::runtime_error("Unsupported block-tensor dimension");
    }

        /// @brief Returns the inverse transpose of the block-tensor
    inline auto invtr() const
    {      
      if constexpr (Rows == 1 && Cols == 1) {
        BlockTensor<T, Rows, Cols> result;
        result[0] = std::make_shared<T>(torch::reciprocal(*data_[0]));
        return result;
      }
      else if constexpr (Rows == 2 && Cols == 2) {        
        auto det = torch::mul(*data_[0], *data_[3])
                 - torch::mul(*data_[1], *data_[2]);
        
        BlockTensor<T, Rows, Cols> result;
        result[0] = std::make_shared<T>(torch::div(*data_[3], det));
        result[1] = std::make_shared<T>(torch::div(*data_[1],-det));
        result[2] = std::make_shared<T>(torch::div(*data_[2],-det));
        result[3] = std::make_shared<T>(torch::div(*data_[0], det));
        return result;
      }
      else if constexpr (Rows == 3 && Cols == 3) {
        // DET  =   a11(a33a22-a32a23)
        //        - a21(a33a12-a32a13)
        //        + a31(a23a12-a22a13)
        auto det = torch::mul(*data_[0],
                              torch::mul(*data_[8], *data_[4]) -
                              torch::mul(*data_[7], *data_[5]))
                 - torch::mul(*data_[3],
                              torch::mul(*data_[8], *data_[1]) -
                              torch::mul(*data_[7], *data_[2]))
                 - torch::mul(*data_[6],
                              torch::mul(*data_[5], *data_[1]) -
                              torch::mul(*data_[4], *data_[2]));

        // |  a33a22-a32a23  -(a33a12-a32a13)   a23a12-a22a13 |
        // |-(a33a21-a31a23)   a33a11-a31a13  -(a23a11-a21a13)|
        // |  a32a21-a31a22  -(a32a11-a31a12)   a22a11-a21a12 |
             
        auto a11 = torch::mul(*data_[8], *data_[4]) - torch::mul(*data_[7], *data_[5]);
        auto a12 = torch::mul(*data_[7], *data_[2]) - torch::mul(*data_[8], *data_[1]);
        auto a13 = torch::mul(*data_[5], *data_[1]) - torch::mul(*data_[4], *data_[2]);
        auto a21 = torch::mul(*data_[6], *data_[5]) - torch::mul(*data_[8], *data_[3]);
        auto a22 = torch::mul(*data_[8], *data_[0]) - torch::mul(*data_[6], *data_[2]);
        auto a23 = torch::mul(*data_[3], *data_[2]) - torch::mul(*data_[5], *data_[0]);
        auto a31 = torch::mul(*data_[7], *data_[3]) - torch::mul(*data_[6], *data_[4]);
        auto a32 = torch::mul(*data_[6], *data_[1]) - torch::mul(*data_[7], *data_[0]);
        auto a33 = torch::mul(*data_[4], *data_[0]) - torch::mul(*data_[3], *data_[1]);
        
        BlockTensor<T, Rows, Cols> result;
        result[0] = std::make_shared<T>(torch::div(a11, det));
        result[1] = std::make_shared<T>(torch::div(a12, det));
        result[2] = std::make_shared<T>(torch::div(a13, det));
        result[3] = std::make_shared<T>(torch::div(a21, det));
        result[4] = std::make_shared<T>(torch::div(a22, det));
        result[5] = std::make_shared<T>(torch::div(a23, det));
        result[6] = std::make_shared<T>(torch::div(a31, det));
        result[7] = std::make_shared<T>(torch::div(a32, det));
        result[8] = std::make_shared<T>(torch::div(a33, det));
        return result;
      }
      else
        throw std::runtime_error("Unsupported block-tensor dimension");
    }

    /// Returns a string representation of the BSplineCommon object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << iganet::core<T>::name() << "\n";
      for (short_t row = 0; row<Rows; ++row)
        for (short_t col = 0; col<Cols; ++col)
          os << "[" << row << "," << col << "] = \n"
             << *data_[Cols*row+col] << "\n";
    }
  };

  /// Print (as string) a compile-time block-tensor object
  template<typename T, short_t Rows, short_t Cols>
  inline std::ostream& operator<<(std::ostream& os,
                                  const BlockTensor<T, Rows, Cols>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

#define unary_op(name)                                                  \
  template<typename T, short_t Rows, short_t Cols>                      \
  inline auto name(const BlockTensor<T, Rows, Cols>& input)             \
  {                                                                     \
    BlockTensor<T, Rows, Cols> result;                                  \
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         \
      result[idx] = std::make_shared<T>(torch::name(*input[idx]));      \
    return result;                                                      \
  }

#define unary_special_op(name)                                          \
  template<typename T, short_t Rows, short_t Cols>                      \
  inline auto name(const BlockTensor<T, Rows, Cols>& input)             \
  {                                                                     \
    BlockTensor<T, Rows, Cols> result;                                  \
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         \
      result[idx] = std::make_shared<T>(torch::special::name(*input[idx])); \
    return result;                                                      \
  }
  
#define binary_op(name)                                                 \
  template<typename T, typename U, short_t Rows, short_t Cols>          \
  inline auto name(const BlockTensor<T, Rows, Cols>& input,             \
                   const BlockTensor<U, Rows, Cols>& other)             \
  {                                                                     \
    BlockTensor<typename std::common_type<T,U>::type, Rows, Cols> result; \
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         \
      result[idx] = std::make_shared<T>(torch::name(*input[idx], *other[idx])); \
    return result;                                                      \
  }

#define binary_special_op(name)                                         \
  template<typename T, typename U, short_t Rows, short_t Cols>          \
  inline auto name(const BlockTensor<T, Rows, Cols>& input,             \
                   const BlockTensor<U, Rows, Cols>& other)             \
  {                                                                     \
    BlockTensor<typename std::common_type<T,U>::type, Rows, Cols> result; \
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         \
      result[idx] = std::make_shared<T>(torch::special::name(*input[idx], *other[idx])); \
    return result;                                                      \
  }
    
  /// @brief Returns a new block-tensor with the absolute value of the
  /// elements of \a input
  unary_op(abs);

  /// @brief Alias for \a abs()
  unary_op(absolute);

  /// @brief Returns a new block-tensor with the inverse cosine of the
  /// elements of \a input
  unary_op(acos);

  /// @brief Alias for \a acos()
  unary_op(arccos);

  /// @brief Returns a new block-tensor with the inverse hyperbolic
  /// cosine of the elements of \a input
  unary_op(acosh);

  /// @brief Alias for \a acosh()
  unary_op(arccosh);

  /// @brief Returns a new block-tensor with the elements of \a other,
  /// scaled by \a alpha, added to the elements of \a input
  template<typename T, typename U, typename V, short_t Rows, short_t Cols>          
  inline auto add(const BlockTensor<T, Rows, Cols>& input,                  
                  const BlockTensor<U, Rows, Cols>& other,
                  V alpha = 1.0)                   
  {                                                                     
    BlockTensor<typename std::common_type<T,U>::type, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::add(*input[idx],
                                                   *other[idx],
                                                   alpha)); 
    return result;                                                      
  }

  /// @brief Returns a new block-tensor with the elements of \a other,
  /// scaled by \a alpha, added to the elements of \a input
  template<typename T, typename U, typename V, short_t Rows, short_t Cols>          
  inline auto add(const BlockTensor<T, Rows, Cols>& input,                  
                  U other,
                  V alpha = 1.0)                   
  {                                                                     
    BlockTensor<T, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::add(*input[idx],
                                                   other,
                                                   alpha)); 
    return result;                                                      
  }

  /// @brief Returns a new block-tensor with the elements of \a other,
  /// scaled by \a alpha, added to the elements of \a input
  template<typename T, typename U, typename V, short_t Rows, short_t Cols>          
  inline auto add(T input,                  
                  const BlockTensor<U, Rows, Cols>& other,
                  V alpha = 1.0)                   
  {                                                                     
    BlockTensor<U, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::add(input,
                                                   *other[idx],
                                                   alpha)); 
    return result;                                                      
  }

  /// @brief Returns a new block-tensor with the elements of \a
  /// tensor1 divided by the elements of \a tensor2, with the result
  /// multiplied by the scalar \a value and added to the elements of
  /// \a input
  template<typename T, typename U, typename V, typename W, short_t Rows, short_t Cols>          
  inline auto addcdiv(const BlockTensor<T, Rows, Cols>& input,                  
                      const BlockTensor<U, Rows, Cols>& tensor1,
                      const BlockTensor<V, Rows, Cols>& tensor2,
                      W value = 1.0)                   
  {                                                                     
    BlockTensor<typename std::common_type<T,U,V>::type, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::addcdiv(*input[idx],
                                                       *tensor1[idx],
                                                       *tensor2[idx],
                                                       value)); 
    return result;                                                      
  }

  /// @brief Returns a new block-tensor with the elements of \a
  /// tensor1 multiplied by the elements of \a tensor2, with the
  /// result multiplied by the scalar \a value and added to the
  /// elements of \a input
  template<typename T, typename U, typename V, typename W, short_t Rows, short_t Cols>          
  inline auto addcmul(const BlockTensor<T, Rows, Cols>& input,                  
                      const BlockTensor<U, Rows, Cols>& tensor1,
                      const BlockTensor<V, Rows, Cols>& tensor2,
                      W value = 1.0)                   
  {                                                                     
    BlockTensor<typename std::common_type<T,U,V>::type, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::addcmul(*input[idx],
                                                       *tensor1[idx],
                                                       *tensor2[idx],
                                                       value)); 
    return result;                                                      
  }
  
  /// @brief Returns a new block-tensor with the angle (in radians) of
  /// the elements of \a input
  unary_op(angle);

  /// @brief Returns a new block-tensor with the arcsine of the
  /// elements of \a input
  unary_op(asin);

  /// @brief Alias for asin()
  unary_op(arcsin);

  /// @brief Returns a new block-tensor with the inverse hyperbolic
  /// sine of the elements of \a input
  unary_op(asinh);

  /// @brief Alias for asinh()
  unary_op(arcsinh);

  /// @brief Returns a new block-tensor with the arctangent of the
  /// elements of \a input
  unary_op(atan);

  /// @brief Alias for atan()
  unary_op(arctan);

  /// @brief Returns a new block-tensor with the inverse hyperbolic
  /// tangent of the elements of \a input
  unary_op(atanh)

  /// @brief Alias for atanh()
  unary_op(arctanh);

  /// @brief Returns a new block-tensor with the arctangent of the
  /// elements in \a input and \a other with consideration of the
  /// quadrant
  binary_op(atan2);

  /// @brief Alias for atan2()
  binary_op(arctan2);

  /// @brief Returns a new block-tensor with the bitwise NOT of the
  /// elements of \a input
  unary_op(bitwise_not);

  /// @brief Returns a new block-tensor with the bitwise AND of the
  /// elements of \a input and \a other
  binary_op(bitwise_and);

  /// @brief Returns a new block-tensor with the bitwise OR of the
  /// elements of \a input and \a other
  binary_op(bitwise_or);

  /// @brief Returns a new block-tensor with the bitwise XOR of the
  /// elements of \a input and \a other
  binary_op(bitwise_xor);

  /// @brief Returns a new block-tensor with the left arithmetic shift
  /// of the elements of \a input by \a other bits
  binary_op(bitwise_left_shift);

  /// @brief Returns a new block-tensor with the right arithmetic
  /// shift of the element of \a input by \a other bits
  binary_op(bitwise_right_shift);

  /// @brief Returns a new block-tensor with the ceil of the elements of
  /// input, the smallest integer greater than or equal to each
  /// element
  unary_op(ceil);

  /// @brief Returns a new block-tensor with the elements of \a input
  /// clamped into the range [ min, max ]
  template<typename T, typename U, short_t Rows, short_t Cols>          
  inline auto clamp(const BlockTensor<T, Rows, Cols>& input,                  
                    U min, U max)                   
  {                                                                     
    BlockTensor<T, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::clamp(*input[idx], min, max)); 
    return result;                                                      
  }

  /// @brief Alias for clamp()
  template<typename T, typename U, short_t Rows, short_t Cols>          
  inline auto clip(const BlockTensor<T, Rows, Cols>& input,                  
                   U min, U max)                   
  {                                                                     
    BlockTensor<T, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::clip(*input[idx], min, max)); 
    return result;                                                      
  }

  /// @brief Returns a new block-tensor with the conjugate of the
  /// elements of \a input tensor
  unary_op(conj_physical);

  /// @brief Returns a new block-tensor with the magnitude of the
  /// elements of \a input and the sign of the elements of \a other
  binary_op(copysign);

  /// @brief Returns a new block-tensor with the cosine of the
  /// elements of \a input
  unary_op(cos);

  /// @brief Returns a new block-tensor with the hyperbolic cosine of
  /// the elements of \a input
  unary_op(cosh);

  /// @brief Returns a new block-tensor with the elements of \a input
  /// converted from angles in degrees to radians
  unary_op(deg2rad)

  /// @brief Returns a new block-tensor with the elements of \a input
  /// divided by the elements of \a other
  binary_op(div);

  /// @brief Alias for div()
  binary_op(divide);

  /// @brief Returns a new block-tensor with the logarithmic
  /// derivative of the gamma function of the elements of \a input
  unary_op(digamma);

  /// @brief Returns a new block-tensor with the error function of the
  /// elements of \a input
  unary_op(erf);

  /// @brief Returns a new block-tensor with the complementary error
  /// function of the elements of \a input
  unary_op(erfc);

  /// @brief Returns a new block-tensor with the inverse error
  /// function of the elements of \a input
  unary_op(erfinv);

  /// @brief Returns a new block-tensor with the exponential of the
  /// elements of \a input
  unary_op(exp);

  /// @brief Returns a new block-tensor with the base-2 exponential of
  /// the elements of \a input
  unary_op(exp2);

  /// @brief Returns a new block-tensor with the exponential minus 1
  /// of the elements of \a input
  unary_op(expm1);

  /// @brief Alias for trunc()
  unary_op(fix);

  /// @brief Returns a new block-tensor with the elements of \a input
  /// raised to the power of \a exponent, elementwise, in double
  /// precision
  binary_op(float_power);

  /// @brief Returns a new block-tensor with the floor of the elements
  /// of \a input, the largest integer less than or equal to each element
  unary_op(floor);

  /// @brief Returns a new block-tensor with the fmod of the elements
  /// of \a input and \a other
  binary_op(fmod);

  /// @brief Returns a new block-tensor with the fractional portion of
  /// the elements of \a input
  unary_op(frac);

  /// @brief Returns a new block-tensor with the decomposition of the
  /// elements of \a input into mantissae and exponents
  unary_op(frexp);

  /// @brief Returns a new block-tensor with the imaginary values of
  /// the elements of \a input
  unary_op(imag);

  /// @brief Returns a new block-tensor with the elements of \a input
  /// multiplied by 2**other
  binary_op(ldexp);

  /// @brief Returns a new block-tensor with the natural logarithm of
  /// the absolute value of the gamma function of the elements of \a
  /// input
  unary_op(lgamma);

  /// @brief Returns a new block-tensor with the natural logarithm of
  /// the elements of \a input
  unary_op(log);

  /// @brief Returns a new block-tensor with the logarithm to the
  /// base-10 of the elements of \a input
  unary_op(log10);

  /// @brief Returns a new block-tensor with the natural logarithm of
  /// (1 + the elements of \a input)
  unary_op(log1p);

  /// @brief Returns a new block-tensor with the logarithm to the
  /// base-2 of the elements of \a input
  unary_op(log2);

  /// @brief Returns a new block-vector with the logarithm of the sum
  /// of exponentiations of the elements of \a input
  binary_op(logaddexp);

  /// @brief Returns a new block-vector with the logarithm of the sum
  /// of exponentiations of the elements of \a input in base-2
  binary_op(logaddexp2);

  /// @brief Returns a new block-tensor with the element-wise logical
  /// AND of the elements of \a input and \a other
  binary_op(logical_and)

  /// @brief Returns a new block-tensor with the element-wise logical
  /// NOT of the elements of \a input
  unary_op(logical_not)

  /// @brief Returns a new block-tensor with the element-wise logical
  /// OR of the elements of \a input and \a other
  binary_op(logical_or)

  /// @brief Returns a new block-tensor with the element-wise logical
  /// XOR of the elements of \a input and \a other
  binary_op(logical_xor);

  /// logit

  /// @brief Given the legs of a right triangle, return its hypotenuse
  binary_op(hypot);

  /// @brief Returns a new block-tensor with the element-wise zeroth
  /// order modified Bessel function of the first kind for each
  /// element of \a input
  unary_op(i0);

  /// @brief Returns a new block-tensor with the regularized lower
  /// incomplete gamma function of each element of \a input
  binary_special_op(gammainc);

  /// @brief Alias for gammainc()
  binary_op(igamma);

  /// @brief Returns a new block-tensor with the regularized upper
  /// incomplete gamma function of each element of \a input
  binary_special_op(gammaincc);
  
  /// @brief Alias for gammainc()
  binary_op(igammac);

  /// @brief Returns a new block-tensor with the product of each
  /// element of \a input and \a other
  binary_op(mul);

  /// @brief Alias for mul()
  binary_op(multiply);

  /// @brief Returns a new block-tensor with the negative of the
  /// elements of \a input
  unary_op(neg);

  /// @brief Alias for neg()
  unary_op(negative);

  /// @brief Return a new block-tensor with the next elementwise
  /// floating-point value after \a input towards \a other
  binary_op(nextafter);
  
  /// @brief Returns a new block-tensor with the \a input
  unary_op(positive);

  /// @brief Returns a new block-tensor with the power of each element
  /// in \a input with exponent \other
  binary_op(pow);

  /// @brief Returns a new block-tensor with each of the elements of
  /// \a input converted from angles in radians to degrees
  unary_op(rad2deg);

  /// @brief Returns a new block-tensor with the real values of the
  /// elements of \a input
  unary_op(real);

  /// @brief Returns a new block-tensor with the reciprocal of the
  /// elements of \a input
  unary_op(reciprocal);

  /// @brief Returns a new block-tensor with the modulus of the
  /// elements of \a input
  binary_op(remainder);

  /// @brief Returns a new block-tensor with the elements of \a input
  /// rounded to the nearest integer
  unary_op(round);

  /// @brief Returns a new block-tensor with the reciprocal of the
  /// square-root of the elements of \a input
  unary_op(rsqrt);

  /// @brief Returns a new block-tensor with the expit (also known as
  /// the logistic sigmoid function) of the elements of \a input
  unary_special_op(expit);

  /// @brief Alias for expit()
  unary_op(sigmoid);

  /// @brief Returns a new block-tensor with the signs of the elements
  /// of \a input
  unary_op(sign);

  /// @brief Returns a new block-tensor with the signs of the elements
  /// of \a input, extension to complex value
  unary_op(sgn);

  /// @brief Tests if each element of \a input has its sign bit set
  /// (is less than zero) or not
  unary_op(signbit);

  /// @brief Returns a new block-tensor with the sine of the elements
  /// of \a input
  unary_op(sin);

  /// @brief Returns a new block-tensor with the normalized sinc of
  /// the elements of \a input
  unary_op(sinc);

  /// @brief Returns a new block-tensor with the hyperbolic sine of
  /// the elements of \a input
  unary_op(sinh);

  /// @brief Returns a new block-tensor with the square-root of the
  /// elements of \a input
  unary_op(sqrt);

  /// @brief Returns a new block-tensor with the square of the
  /// elements of \a input
  unary_op(square);

  /// @brief Subtracts other, scaled by alpha, from input
  template<typename T, typename U, typename V, short_t Rows, short_t Cols>          
  inline auto sub(const BlockTensor<T, Rows, Cols>& input,                  
                  const BlockTensor<U, Rows, Cols>& other,
                  V alpha = 1.0)                   
  {                                                                     
    BlockTensor<typename std::common_type<T,U>::type, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::sub(*input[idx],
                                                   *other[idx],
                                                   alpha)); 
    return result;                                                      
  }

  /// @brief Alias for sub()
  template<typename T, typename U, typename V, short_t Rows, short_t Cols>          
  inline auto subtract(const BlockTensor<T, Rows, Cols>& input,                  
                       const BlockTensor<U, Rows, Cols>& other,
                       V alpha = 1.0)                   
  {                                                                     
    BlockTensor<typename std::common_type<T,U>::type, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::sub(*input[idx],
                                                   *other[idx],
                                                   alpha)); 
    return result;                                                      
  }
  
  /// @brief Returns a new tensor with the tangent of the elements of
  /// input
  unary_op(tan);

  /// @brief Returns a new tensor with the hyperbolic tangent of the
  /// elements of input
  unary_op(tanh);

  /// @brief Returns a new tensor with the truncated integer values of
  /// the elements of input
  unary_op(trunc)

  /// @brief Computes input * log(other)
  binary_op(xlogy);
  
  /// @brief Adds one compile-time block-tensor to another
  template<typename T, typename U, short_t Rows, short_t Cols>
  inline auto operator+(const BlockTensor<T, Rows, Cols>& lhs, const BlockTensor<U, Rows, Cols>& rhs)
  {
    BlockTensor<typename std::common_type<T, U>::type, Rows, Cols> result;
    for (short_t idx = 0; idx<Rows*Cols; ++idx)
      result[idx] = std::make_shared<T>(*lhs[idx] + *rhs[idx]);
    return result;
  }

  /// @brief Adds a compile-time block-tensor to a scalar
  template<typename T, typename U, short_t Rows, short_t Cols>
  inline auto operator+(const BlockTensor<T, Rows, Cols>& lhs, U rhs)
  {
    BlockTensor<T, Rows, Cols> result;
    for (short_t idx = 0; idx<Rows*Cols; ++idx)
      result[idx] = std::make_shared<T>(*lhs[idx] + rhs);
    return result;
  }

    /// @brief Adds a compile-time block-tensor to a scalar
  template<typename T, typename U, short_t Rows, short_t Cols>
  inline auto operator+(T lhs, const BlockTensor<U, Rows, Cols>& rhs)
  {
    BlockTensor<U, Rows, Cols> result;
    for (short_t idx = 0; idx<Rows*Cols; ++idx)
      result[idx] = std::make_shared<U>(lhs + *rhs[idx]);
    return result;
  }

  /// @brief Subtracts one compile-time block-tensor from another
  template<typename T, typename U, short_t Rows, short_t Cols>
  inline auto operator-(const BlockTensor<T, Rows, Cols>& lhs, const BlockTensor<U, Rows, Cols>& rhs)
  {
    BlockTensor<typename std::common_type<T, U>::type, Rows, Cols> result;
    for (short_t idx = 0; idx<Rows*Cols; ++idx)
      result[idx] = std::make_shared<T>(*lhs[idx] - *rhs[idx]);
    return result;
  }

  /// @brief Multiplies one compile-time block-tensor with another
  template<typename T, typename U, short_t Rows, short_t Common, short_t Cols>
  inline auto operator*(const BlockTensor<T, Rows, Common>& lhs, const BlockTensor<U, Common, Cols>& rhs)
  {
    BlockTensor<typename std::common_type<T, U>::type, Rows, Cols> result;
    for (short_t row = 0; row<Rows; ++row)
      for (short_t col = 0; col<Cols; ++col) {
        T tmp = torch::mul(*lhs[Common*row], *rhs[col]);
          for (short_t idx = 1; idx<Common; ++idx)
            tmp += torch::mul(*lhs[Common*row+idx], *rhs[Cols*idx+col]);
        result[Cols*row+col] = std::make_shared<T>(tmp);
      }
    return result;
  }

  /// @brief Returns true if both compile-time block-tensors are equal
  template<typename T, typename U, short_t RowsT, short_t ColsT, short_t RowsU, short_t ColsU>
  inline bool operator==(const BlockTensor<T, RowsT, ColsT>& lhs, const BlockTensor<U, RowsU, ColsU>& rhs)
  {
    if constexpr ((RowsT != RowsU) || (ColsT != ColsU))
      return false;
    
    bool result = true;
    for (short_t idx = 0; idx<RowsT*ColsT; ++idx)
      result = result && torch::equal(*lhs[idx], *rhs[idx]);

    return result;
  }

  /// @brief Returns true if both compile-time block-tensors are not equal
  template<typename T, typename U, short_t RowsT, short_t ColsT, short_t RowsU, short_t ColsU>
  inline bool operator!=(const BlockTensor<T, RowsT, ColsT>& lhs, const BlockTensor<U, RowsU, ColsU>& rhs)
  {
    return !(lhs == rhs);
  }
  
} // namespace iganet
