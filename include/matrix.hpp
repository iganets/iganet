/**
   @file include/matrix.hpp

   @brief Compile-time matrix

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

  /// @brief Compile-time matrix  
  template<typename T, short_t Rows, short_t Cols>
  class Matrix : public core<T> {
    
  private:
    /// @brief Array storing the data
    std::array<std::shared_ptr<T>, Rows*Cols> data_;

  public:
    /// @brief Data constructor
    template<typename... Ts>
    Matrix(Ts&&... data)
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

    /// @brief Returns the transpose of the matrix
    inline auto tr() const
    {
      Matrix<T, Cols, Rows> result;
      for (short_t row = 0; row<Rows; ++row)
        for (short_t col = 0; col<Cols; ++col)
          result[Cols*row+col] = data_[Rows*col+row];
      return result;
    }

    /// @brief Returns the inverse of the matrix
    inline auto inv() const
    {      
      if constexpr (Rows == 1 && Cols == 1) {
        Matrix<T, Rows, Cols> result;
        result[0] = std::make_shared<T>(torch::reciprocal(*data_[0]));
        return result;
      }
      else if constexpr (Rows == 2 && Cols == 2) {        
        auto det = torch::mul(*data_[0], *data_[3])
                 - torch::mul(*data_[1], *data_[2]);
        
        Matrix<T, Rows, Cols> result;
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
        
        Matrix<T, Rows, Cols> result;
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
        throw std::runtime_error("Unsupported matrix dimension");
    }

    /// Returns a string representation of the BSplineCommon object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<T>::name() << "\n";
      for (short_t row = 0; row<Rows; ++row)
        for (short_t col = 0; col<Cols; ++col)
          os << "[" << row << "," << col << "] = \n"
             << *data_[Cols*row+col] << "\n";
    }
  };

  /// Print (as string) a compile-time matrix object
  template<typename T, short_t Rows, short_t Cols>
  inline std::ostream& operator<<(std::ostream& os,
                                  const Matrix<T, Rows, Cols>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

#define unary_op(name)                                              \
  template<typename T, short_t Rows, short_t Cols>                  \
  inline auto name(const Matrix<T, Rows, Cols>& input)              \
  {                                                                 \
    Matrix<T, Rows, Cols> result;                                   \
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                     \
      result[idx] = std::make_shared<T>(torch::name(*input[idx]));  \
    return result;                                                  \
  }
  
#define binary_op(name)                                                 \
  template<typename T, typename U, short_t Rows, short_t Cols>          \
  inline auto name(const Matrix<T, Rows, Cols>& input,                  \
                   const Matrix<T, Rows, Cols>& other)                  \
  {                                                                     \
    Matrix<typename std::common_type<T,U>::type, Rows, Cols> result;    \
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         \
      result[idx] = std::make_shared<T>(torch::name(*input[idx], *other[idx])); \
    return result;                                                      \
  }

#define unary_op_alias(name, alias)                 \
  template<typename T, short_t Rows, short_t Cols>  \
  auto alias = name<T, Rows, Cols>;
  
#define binary_op_alias(name, alias)                             \
  template<typename T, typename U, short_t Rows, short_t Cols>   \
  auto alias = name<T, U, Rows, Cols>;
  
  /// @brief Computes the absolute value of each element in input
  unary_op(abs);

  /// @brief Alias for abs
  unary_op_alias(abs, absolute);

  /// @brief Computes the inverse cosine of each element in input
  unary_op(acos);

  /// @brief Alias for acos
  unary_op_alias(acos, arccos);

  /// @brief Returns a new tensor with the inverse hyperbolic cosine of the elements of input
  unary_op(acosh);

  /// @brief Aloas for acosh
  unary_op_alias(acosh, arccosh);

  /// @brief Adds other, scaled by alpha, to input
  template<typename T, typename U, typename V, short_t Rows, short_t Cols>          
  inline auto add(const Matrix<T, Rows, Cols>& input,                  
                  const Matrix<U, Rows, Cols>& other,
                  V alpha = (double)1.0)                   
  {                                                                     
    Matrix<typename std::common_type<T,U>::type, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::add(*input[idx],
                                                   *other[idx],
                                                   alpha)); 
    return result;                                                      
  }

  /// @brief Performs the element-wise division of tensor1 by tensor2,
  /// multiply the result by the scalar value and add it to input
  template<typename T, typename U, typename V, typename W, short_t Rows, short_t Cols>          
  inline auto addcdiv(const Matrix<T, Rows, Cols>& input,                  
                      const Matrix<U, Rows, Cols>& tensor1,
                      const Matrix<V, Rows, Cols>& tensor2,
                      W value = (double)1.0)                   
  {                                                                     
    Matrix<typename std::common_type<T,U,V,W>::type, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::addcdiv(*input[idx],
                                                       *tensor1[idx],
                                                       *tensor2[idx],
                                                       value)); 
    return result;                                                      
  }

  /// @brief Performs the element-wise multiplication of tensor1 by
  /// tensor2, multiply the result by the scalar value and add it to
  /// input
  template<typename T, typename U, typename V, typename W, short_t Rows, short_t Cols>          
  inline auto addcmul(const Matrix<T, Rows, Cols>& input,                  
                      const Matrix<U, Rows, Cols>& tensor1,
                      const Matrix<V, Rows, Cols>& tensor2,
                      W value = (double)1.0)                   
  {                                                                     
    Matrix<typename std::common_type<T,U,V,W>::type, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::addcmul(*input[idx],
                                                       *tensor1[idx],
                                                       *tensor2[idx],
                                                       value)); 
    return result;                                                      
  }
  
  /// @brief Computes the element-wise angle (in radians) of the given input tensor
  unary_op(angle);

  /// @brief Returns a new tensor with the arcsine of the elements of input
  unary_op(asin);

  /// @brief Alias for ason
  unary_op_alias(asin, arcsin);

  /// @brief Returns a new tensor with the inverse hyperbolic sine of the elements of input
  unary_op(asinh);

  /// @brief Alias for asinh
  unary_op_alias(asinh, arcsinh);

  /// @brief Returns a new tensor with the arctangent of the elements of input
  unary_op(atan);

  /// @brief Alias for atan
  unary_op_alias(atan, arctan);

  /// @brief Returns a new tensor with the inverse hyperbolic tangent of the elements of input
  unary_op(atanh)

  /// @brief Alias for atanh
  unary_op_alias(atanh, arctanh);

  /// @brief Element-wise arctangent of input/other with consideration of the quadrant
  binary_op(atan2);

  /// @brief Alias for atan2
  binary_op_alias(atan2, arctan2);

  /// @brief Computes the bitwise NOT of the given input tensor
  unary_op(bitwise_not);

  /// @brief Computes the bitwise AND of input and other
  binary_op(bitwise_and);

  /// @brief Computes the bitwise OR of input and other
  binary_op(bitwise_or);

  /// @brief Computes the bitwise XOR of input and other
  binary_op(bitwise_xor);

  /// @brief Computes the left arithmetic shift of input by other bits
  binary_op(bitwise_left_shift);

  /// @brief Computes the right arithmetic shift of input by other bits
  binary_op(bitwise_right_shift);

  /// @brief Returns a new tensor with the ceil of the elements of
  /// input, the smallest integer greater than or equal to each
  /// element
  unary_op(ceil);

  /// @brief Clamps all elements in input into the range [ min, max ]
  template<typename T, typename U, typename V, short_t Rows, short_t Cols>          
  inline auto clamp(const Matrix<T, Rows, Cols>& input,                  
                    U min, U max)                   
  {                                                                     
    Matrix<T, Rows, Cols> result;    
    for (short_t idx = 0; idx<Rows*Cols; ++idx)                         
      result[idx] = std::make_shared<T>(torch::clamp(*input[idx], min, max)); 
    return result;                                                      
  }

  /// @brief Alias for clamp()
  template<typename T, typename U, short_t Rows, short_t Cols>  \
  auto clip = clamp<T, Rows, Cols>;     

  /// ... add more
  
  /// @brief Adds one compile-time matrix to another
  template<typename T, typename U, short_t Rows, short_t Cols>
  inline auto operator+(const Matrix<T, Rows, Cols>& lhs, const Matrix<U, Rows, Cols>& rhs)
  {
    Matrix<typename std::common_type<T, U>::type, Rows, Cols> result;
    for (short_t idx = 0; idx<Rows*Cols; ++idx)
      result[idx] = std::make_shared<T>(*lhs[idx] + *rhs[idx]);
    return result;
  }

  /// @brief Subtracts one compile-time matrix from another
  template<typename T, typename U, short_t Rows, short_t Cols>
  inline auto operator-(const Matrix<T, Rows, Cols>& lhs, const Matrix<U, Rows, Cols>& rhs)
  {
    Matrix<typename std::common_type<T, U>::type, Rows, Cols> result;
    for (short_t idx = 0; idx<Rows*Cols; ++idx)
      result[idx] = std::make_shared<T>(*lhs[idx] - *rhs[idx]);
    return result;
  }

  /// @brief Multiplies one compile-time matrix with another
  template<typename T, typename U, short_t Rows, short_t Common, short_t Cols>
  inline auto operator*(const Matrix<T, Rows, Common>& lhs, const Matrix<U, Common, Cols>& rhs)
  {
    Matrix<typename std::common_type<T, U>::type, Rows, Cols> result;
    for (short_t row = 0; row<Rows; ++row)
      for (short_t col = 0; col<Cols; ++col) {
        T tmp = torch::mul(*lhs[Common*row], *rhs[col]);
          for (short_t idx = 1; idx<Common; ++idx)
            tmp += torch::mul(*lhs[Common*row+idx], *rhs[Cols*idx+col]);
        result[Cols*row+col] = std::make_shared<T>(tmp);
      }
    return result;
  }

  /// @brief Returns true if both compile-time matrices are equal
  template<typename T, typename U, short_t RowsT, short_t ColsT, short_t RowsU, short_t ColsU>
  inline bool operator==(const Matrix<T, RowsT, ColsT>& lhs, const Matrix<U, RowsU, ColsU>& rhs)
  {
    if constexpr ((RowsT != RowsU) || (ColsT != ColsU))
      return false;
    
    bool result = true;
    for (short_t idx = 0; idx<RowsT*ColsT; ++idx)
      result = result && torch::equal(*lhs[idx], *rhs[idx]);

    return result;
  }

  /// @brief Returns true if both compile-time matrices are not equal
  template<typename T, typename U, short_t RowsT, short_t ColsT, short_t RowsU, short_t ColsU>
  inline bool operator!=(const Matrix<T, RowsT, ColsT>& lhs, const Matrix<U, RowsU, ColsU>& rhs)
  {
    return !(lhs == rhs);
  }
  
} // namespace iganet
