/**
   @file include/creator.hpp

   @brief Geometry creator

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <bspline.hpp>

#pragma once

namespace iganet {

  /// @brrief Abstract creator class
  template<typename T>
  class CreatorCore : public fqn
  {
  public:
    /// Returns a string representation of the CreatorCore object
    virtual void pretty_print(std::ostream& os = std::cout) const = 0;
  };

  /// Print (as string) a CreatorCore object
  template<typename T>
  inline std::ostream& operator<<(std::ostream& os,
                                  const CreatorCore<T>& obj)
  {
    obj.pretty_print(os);
    return os;
  }
  
  /// @brief Interval creator class
  ///
  /// This geometry creator generates a sequence of intervals in the
  /// specified bounds [xmin, xmax]
  template<typename T>
  class IntervalCreator : public CreatorCore<T>
  {
  public:
    /// Default constructor
    IntervalCreator()
      : x0min_(0.0), x0max_(0.1),
        x1min_(0.9), x1max_(1.0)
    {
      //std::seed();
    }

    /// Bounds constructor
    IntervalCreator(const T& x0min, const T& x0max,
                    const T& x1min, const T& x1max)
    : x0min_(x0min), x0max_(x0max),
      x1min_(x1min), x1max_(x1max)
    {
      //std::seed();
    }
    
    template<typename bspline_t>
    auto next(bspline_t& obj)
    {
      static_assert(bspline_t::parDim() == 1 &&
                    bspline_t::geoDim() == 1,
                    "Interval creator requires parDim=1 and geoDim=1");

      T x0 = x0min_ + (x0max_-x0min_) * T(std::rand()) / T(RAND_MAX);
      T x1 = x1min_ + (x1max_-x1min_) * T(std::rand()) / T(RAND_MAX);

      return x1;
    }

    /// Returns a string representation of the IntervalCreator object
    virtual void pretty_print(std::ostream& os = std::cout) const
    {
      os << CreatorCore<T>::name() << "\n"
         << "(x0min = " << x0min_ << ", x0max = " << x0max_
         << "; x1min = " << x1min_ << ", x1max = " << x1max_ << ")";
    }
    
  private:
    T x0min_, x0max_, x1min_, x1max_;
  };


  
} // namespace iganet
