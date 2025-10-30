/**
   @file include/creator.hpp

   @brief Geometry creator

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <cstdlib>
#include <ctime>

#include <bspline.hpp>
#include <utils/fqn.hpp>

namespace iganet {

/// @brief Abstract creator class
template <typename T>
class CreatorCore : protected iganet::utils::FullQualifiedName {
public:
  /// Returns a string representation of the CreatorCore object
  void pretty_print(std::ostream &os) const noexcept override = 0;
};

/// Print (as string) a CreatorCore object
template <typename T>
inline std::ostream &operator<<(std::ostream &os, const CreatorCore<T> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief Interval creator class
///
/// This geometry creator generates a sequence of intervals in the
/// specified bounds [xmin, xmax]
template <typename T> class IntervalCreator : public CreatorCore<T> {
public:
  /// Default constructor
  IntervalCreator() : x0min_(0.0), x0max_(0.1), x1min_(0.9), x1max_(1.0) {
    std::srand(std::time(0));
  }

  /// Bounds constructor
  IntervalCreator(const T &x0min, const T &x0max, const T &x1min,
                  const T &x1max)
      : x0min_(x0min), x0max_(x0max), x1min_(x1min), x1max_(x1max) {
    std::srand(std::time(0));
  }

  template <typename Spline>
    requires SplineType<Spline>
  auto &next(Spline &obj) const {
    static_assert(Spline::parDim() == 1 && Spline::geoDim() == 1,
                  "Interval creator requires parDim=1 and geoDim=1");

    T x0 = x0min_ + (x0max_ - x0min_) * T(std::rand()) / T(RAND_MAX);
    T x1 = x1min_ + (x1max_ - x1min_) * T(std::rand()) / T(RAND_MAX);
    T xmin = std::min(x0, x1);
    T xmax = std::max(x0, x1);

    obj.transform([xmin, xmax](const std::array<T, 1> X) {
      return std::array<T, 1>{
          static_cast<T>(xmax * X[0] + xmin * (X[0] - T(1)))};
    });

    return obj;
  }

  /// Returns a string representation of the IntervalCreator object
  void pretty_print(std::ostream &os) const noexcept override {
    os << CreatorCore<T>::name() << "\n"
       << "(x0min = " << x0min_ << ", x0max = " << x0max_
       << "; x1min = " << x1min_ << ", x1max = " << x1max_ << ")";
  }

private:
  T x0min_, x0max_, x1min_, x1max_;
};

/// @brief Rectangle creator class
///
/// This geometry creator generates a sequence of rectangles in the
/// specified bounds [xmin, xmax] x [ymin, ymax]
template <typename T> class RectangleCreator : public CreatorCore<T> {
public:
  /// Default constructor
  RectangleCreator()
      : x0min_(0.0), x0max_(0.1), x1min_(0.9), x1max_(1.0), y0min_(0.0),
        y0max_(0.1), y1min_(0.9), y1max_(1.0) {
    std::srand(std::time(0));
  }

  /// Bounds constructor
  RectangleCreator(const T &x0min, const T &x0max, const T &x1min,
                   const T &x1max, const T &y0min, const T &y0max,
                   const T &y1min, const T &y1max)
      : x0min_(x0min), x0max_(x0max), x1min_(x1min), x1max_(x1max),
        y0min_(y0min), y0max_(y0max), y1min_(y1min), y1max_(y1max) {
    std::srand(std::time(0));
  }

  template <typename Spline>
    requires SplineType<Spline>
  auto &next(Spline &obj) const {
    static_assert(Spline::parDim() == 2 && Spline::geoDim() == 2,
                  "Rectangle creator requires parDim=2 and geoDim=2");

    T x0 = x0min_ + (x0max_ - x0min_) * T(std::rand()) / T(RAND_MAX);
    T x1 = x1min_ + (x1max_ - x1min_) * T(std::rand()) / T(RAND_MAX);
    T xmin = std::min(x0, x1);
    T xmax = std::max(x0, x1);

    T y0 = y0min_ + (y0max_ - y0min_) * T(std::rand()) / T(RAND_MAX);
    T y1 = y1min_ + (y1max_ - y1min_) * T(std::rand()) / T(RAND_MAX);
    T ymin = std::min(y0, y1);
    T ymax = std::max(y0, y1);

    obj.transform([xmin, xmax, ymin, ymax](const std::array<T, 2> X) {
      return std::array<T, 2>{
          static_cast<T>(xmax * X[0] + xmin * (X[0] - T(1))),
          static_cast<T>(ymax * X[1] + ymin * (X[1] - T(1)))};
    });

    return obj;
  }

  /// Returns a string representation of the RectangleCreator object
  void pretty_print(std::ostream &os) const noexcept override {
    os << CreatorCore<T>::name() << "\n"
       << "(x0min = " << x0min_ << ", x0max = " << x0max_
       << "; x1min = " << x1min_ << ", x1max = " << x1max_
       << "; y0min = " << y0min_ << ", y0max = " << y0max_
       << "; y1min = " << y1min_ << ", y1max = " << y1max_ << ")";
  }

private:
  T x0min_, x0max_, x1min_, x1max_, y0min_, y0max_, y1min_, y1max_;
};

/// @brief Cuboid creator class
///
/// This geometry creator generates a sequence of cuboids in the
/// specified bounds [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
template <typename T> class CuboidCreator : public CreatorCore<T> {
public:
  /// Default constructor
  CuboidCreator()
      : x0min_(0.0), x0max_(0.1), x1min_(0.9), x1max_(1.0), y0min_(0.0),
        y0max_(0.1), y1min_(0.9), y1max_(1.0), z0min_(0.0), z0max_(0.1),
        z1min_(0.9), z1max_(1.0) {
    std::srand(std::time(0));
  }

  /// Bounds constructor
  CuboidCreator(const T &x0min, const T &x0max, const T &x1min, const T &x1max,
                const T &y0min, const T &y0max, const T &y1min, const T &y1max,
                const T &z0min, const T &z0max, const T &z1min, const T &z1max)
      : x0min_(x0min), x0max_(x0max), x1min_(x1min), x1max_(x1max),
        y0min_(y0min), y0max_(y0max), y1min_(y1min), y1max_(y1max),
        z0min_(z0min), z0max_(z0max), z1min_(z1min), z1max_(z1max) {
    std::srand(std::time(0));
  }

  template <typename Spline>
    requires SplineType<Spline>
  auto &next(Spline &obj) const {
    static_assert(Spline::parDim() == 3 && Spline::geoDim() == 3,
                  "Cuboid creator requires parDim=3 and geoDim=3");

    T x0 = x0min_ + (x0max_ - x0min_) * T(std::rand()) / T(RAND_MAX);
    T x1 = x1min_ + (x1max_ - x1min_) * T(std::rand()) / T(RAND_MAX);
    T xmin = std::min(x0, x1);
    T xmax = std::max(x0, x1);

    T y0 = y0min_ + (y0max_ - y0min_) * T(std::rand()) / T(RAND_MAX);
    T y1 = y1min_ + (y1max_ - y1min_) * T(std::rand()) / T(RAND_MAX);
    T ymin = std::min(y0, y1);
    T ymax = std::max(y0, y1);

    T z0 = z0min_ + (z0max_ - z0min_) * T(std::rand()) / T(RAND_MAX);
    T z1 = z1min_ + (z1max_ - z1min_) * T(std::rand()) / T(RAND_MAX);
    T zmin = std::min(z0, z1);
    T zmax = std::max(z0, z1);

    obj.transform(
        [xmin, xmax, ymin, ymax, zmin, zmax](const std::array<T, 3> X) {
          return std::array<T, 3>{
              static_cast<T>(xmax * X[0] + xmin * (X[0] - T(1))),
              static_cast<T>(ymax * X[1] + ymin * (X[1] - T(1))),
              static_cast<T>(zmax * X[2] + zmin * (X[2] - T(1)))};
        });

    return obj;
  }

  /// Returns a string representation of the RectangleCreator object
  void pretty_print(std::ostream &os) const noexcept override {
    os << CreatorCore<T>::name() << "\n"
       << "(x0min = " << x0min_ << ", x0max = " << x0max_
       << "; x1min = " << x1min_ << ", x1max = " << x1max_
       << "; y0min = " << y0min_ << ", y0max = " << y0max_
       << "; y1min = " << y1min_ << ", y1max = " << y1max_
       << "; z0min = " << z0min_ << ", z0max = " << z0max_
       << "; z1min = " << z1min_ << ", z1max = " << z1max_ << ")";
  }

private:
  T x0min_, x0max_, x1min_, x1max_, y0min_, y0max_, y1min_, y1max_, z0min_,
      z0max_, z1min_, z1max_;
};

} // namespace iganet
