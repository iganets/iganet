#include <exception>
#include <functional>

#include <core.hpp>

#pragma once

namespace iganet {

  /// Enumerator for specifying the initialization of B-Spline coefficients
  enum class BSplineInit : short_t
    {
      zeros    = 0,
      ones     = 1,
      linear   = 2,
      random   = 3,
      greville = 4
    };

  /// Enumerator for specifying the derivative of B-Spline evaluation
  enum class BSplineDeriv : short_t
    {
      func   =    0,
      dx     =    1,
      dx2    =    2,
      dx3    =    3,
      dx4    =    4,
      dy     =   10,
      dy2    =   20,
      dy3    =   30,
      dy4    =   40,
      dz     =  100,
      dz2    =  200,
      dz3    =  300,
      dz4    =  400,
      dt     = 1000,
      dt2    = 2000,
      dt3    = 3000,
      dt4    = 4000
    };

  /// Tensor-product uniform B-Spline (core functionality)
  ///
  /// This class implements the core functionality of all B-Spline
  /// classes and serves as base class for non-uniform B-Splines
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  class UniformBSplineCore : public core<real_t>
  {
  public:
    // Dimension of the parametric space
    static constexpr const short_t parDim_ = sizeof...(Degrees);

    // Dimension of the geometric space
    static constexpr const short_t geoDim_ = GeoDim;

    // Array storing the degrees per dimension
    static constexpr const std::array<short_t, parDim_> degrees_ = { Degrees... };

  protected:
    // Array storing the knot vectors
    std::array<torch::Tensor, parDim_> knots_;

    // Array storing the sizes of the knot vectors
    std::array<int64_t, parDim_> nknots_;

    // Array storing the coefficients of the control net
    std::array<torch::Tensor, geoDim_> coeffs_;

    // Array storing the sizes of the coefficients of the control net
    std::array<int64_t, parDim_> ncoeffs_;

    // String storing the full qualified name of the object
    mutable at::optional<std::string> name_;

    // LibTorch constants
    const torch::Tensor one_, zero_;

  public:
    // Constructor: equidistant knot vectors
    UniformBSplineCore(std::array<int64_t, parDim_> ncoeffs,
                       BSplineInit init = BSplineInit::zeros)
      : core<real_t>(),
        ncoeffs_(ncoeffs),
        one_(torch::ones(1, core<real_t>::options_)),
        zero_(torch::zeros(1, core<real_t>::options_))
    {
      for (short_t i=0; i<parDim_; ++i) {

        // Create open uniform knot vectors
        std::vector<real_t> kv;

        for (int64_t j=0; j<degrees_[i]; ++j)
          kv.push_back(static_cast<real_t>(0));

        for (int64_t j=0; j<ncoeffs[i]-degrees_[i]+1; ++j)
          kv.push_back(static_cast<real_t>(j/real_t(ncoeffs[i]-degrees_[i])));

        for (int64_t j=0; j<degrees_[i]; ++j)
          kv.push_back(static_cast<real_t>(1));

        knots_[i] = torch::from_blob(static_cast<real_t*>(kv.data()),
                                     kv.size(), core<real_t>::options_).clone();

        // Store the size of the knot vector
        nknots_[i] = knots_[i].size(0);
      }

      // Initialize coefficients
      init_coeffs(init);
    }

    // Returns a constant reference to the array of degrees
    inline static constexpr const std::array<short_t, parDim_>& degrees()
    {
      return degrees_;
    }

    // Returns a constant reference to the degree in the i-th dimension
    inline static constexpr const short_t& degree(short_t i)
    {
      assert(i>=0 && i<parDim_);
      return degrees_[i];
    }

    // Returns a constant reference to the array of knot vectors
    inline const std::array<torch::Tensor, parDim_>& knots() const
    {
      return knots_;
    }

    // Returns a constant reference to the knot vector in the i-th dimension
    inline const torch::Tensor& knots(short_t i) const
    {
      assert(i>=0 && i<parDim_);
      return knots_[i];
    }

    // Returns a non-constant reference to the array of knot vectors
    inline std::array<torch::Tensor, parDim_>& knots()
    {
      return knots_;
    }

    // Returns a non-constant reference to the knot vector in the i-th dimension
    inline torch::Tensor& knots(short_t i)
    {
      assert(i>=0 && i<parDim_);
      return knots_[i];
    }

    // Returns a constant reference to the array of knot vector dimensions
    inline const std::array<int64_t, parDim_>& nknots() const
    {
      return nknots_;
    }

    // Returns the dimension of the knot vector in the i-th dimension
    inline int64_t nknots(short_t i) const
    {
      assert(i>=0 && i<parDim_);
      return nknots_[i];
    }

    // Returns a constant reference to the array of coefficients. If
    // flatten=false, this function returns an std::array of
    // torch::Tensor objects with coefficients reshaped according to
    // the dimensions of the knot vectors (return by value in this
    // case)
    template<bool flatten=true>
    inline auto coeffs() const
      -> typename std::conditional<flatten,
                                   const std::array<torch::Tensor, geoDim_>&,
                                   std::array<torch::Tensor, geoDim_>>::type
    {
      if constexpr (flatten)
                     return coeffs_;
      else {
        std::array<torch::Tensor, geoDim_> result;
        for (short_t i=0; i< geoDim_; ++i)
          result[i] = coeffs_[i].view(ncoeffs_);
        return result;
      }
    }

    // Returns a constant reference to the coefficients in the i-th
    // dimension. If flatten=false, this function returns a
    // torch::Tensor with coefficients in the i-th dimension reshaped
    // according to the dimensions of the knot vectors (return by
    // value in this case)
    template<bool flatten=true>
    inline auto coeffs(short_t i) const
      -> typename std::conditional<flatten,
                                   const torch::Tensor&,
                                   torch::Tensor>::type
    {
      assert(i>=0 && i<geoDim_);
      if constexpr (flatten)
                     return coeffs_[i];
      else
        return coeffs_[i].view(ncoeffs_);
    }

    // Returns a non-constant reference to the array of coefficients
    inline std::array<torch::Tensor, geoDim_>& coeffs()
    {
      return coeffs_;
    }

    // Returns a non-constant reference to the coefficients in the i-th dimension
    inline torch::Tensor& coeffs(short_t i)
    {
      assert(i>=0 && i<geoDim_);
      return coeffs_[i];
    }

    // Returns the total number of coefficients
    inline int64_t ncoeffs() const
    {
      int64_t s=1;
      for (short_t i=0; i<parDim_; ++i)
        s *= ncoeffs(i);
      return s;
    }

    // Returns the total number of coefficients in the i-th direction
    inline int64_t ncoeffs(short_t i) const
    {
      assert(i>=0 && i<parDim_);
      return ncoeffs_[i];
    }

    // Returns the parametric dimension
    inline short_t parDim() const
    {
      return parDim_;
    }

    // Returns the geometric dimension
    inline short_t geoDim() const
    {
      return geoDim_;
    }

    // Returns the value of the B-spline object in the points \f$ \xi \f$.
    //
    // To this end, the function first determines the interval
    // \f$
    //   [knot[i], knot[i+1])
    // \f$
    // that contains the point \f$ \xi \f$ and evaluates the vector of
    // basis functions (or their derivatives)
    // \f$
    //   \left[ D^r B_{i-d,d}, \dots, D^r B_{i,d} \right]
    // \f$,
    // where
    // \f$
    //   d
    // \f$
    // is the degree of the B-spline and
    // \f$
    //   r
    // \f$
    // denotes the requested derivative. Next, the function multiplies
    // the above row vector by the column vector of control points
    // \f$
    //   \left[ c_{i-d}, \dots, c_{i} \right]^\top
    // \f$.
    //
    // This functions applies the above procedure to the tensor
    // product of B-splines in all spatial dimensions.
    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi) const
    {
      static_assert(parDim_ <= 4, "Unsupported parametric dimension");

      // 1D
      if constexpr (parDim_ == 1) {
        int64_t i = int64_t(xi[0].item<real_t>()*(nknots_[0]-2*degrees_[0]-1)+degrees_[0]);
        return eval_1d(xi, i);
      }

      // 2D
      else if constexpr (parDim_ == 2) {
        int64_t i = int64_t(xi[0].item<real_t>()*(nknots_[0]-2*degrees_[0]-1)+degrees_[0]);
        int64_t j = int64_t(xi[1].item<real_t>()*(nknots_[1]-2*degrees_[1]-1)+degrees_[1]);
        return eval_2d(xi, i, j);
      }

      // 3D
      else if constexpr (parDim_ == 3) {
        int64_t i = int64_t(xi[0].item<real_t>()*(nknots_[0]-2*degrees_[0]-1)+degrees_[0]);
        int64_t j = int64_t(xi[1].item<real_t>()*(nknots_[1]-2*degrees_[1]-1)+degrees_[1]);
        int64_t k = int64_t(xi[2].item<real_t>()*(nknots_[2]-2*degrees_[2]-1)+degrees_[2]);
        return eval_3d(xi, i, j, k);
      }

      // 4D
      else if constexpr (parDim_ == 4) {
        int64_t i = int64_t(xi[0].item<real_t>()*(nknots_[0]-2*degrees_[0]-1)+degrees_[0]);
        int64_t j = int64_t(xi[1].item<real_t>()*(nknots_[1]-2*degrees_[1]-1)+degrees_[1]);
        int64_t k = int64_t(xi[2].item<real_t>()*(nknots_[2]-2*degrees_[2]-1)+degrees_[2]);
        int64_t l = int64_t(xi[3].item<real_t>()*(nknots_[3]-2*degrees_[3]-1)+degrees_[3]);
        return eval_4d(xi, i, j, k, l);
      }
    }

    // Transforms the coefficients based on the given mapping
    inline UniformBSplineCore& transform(const std::function<std::array<real_t, geoDim_>(const std::array<real_t, parDim_>&)> transformation)
    {
      static_assert(parDim_ <= 4, "Unsupported parametric dimension");

      // 1D
      if constexpr (parDim_ == 1) {
#pragma omp parallel for simd
        for (int64_t i=0; i<ncoeffs_[0]; ++i) {
          auto c = transformation( std::array<real_t,1>{i/real_t(ncoeffs_[0]-1)} );
          for (short_t d=0; d<geoDim_; ++d)
            coeffs_[d].detach()[i] = c[d];
        }
      }

      // 2D
      else if constexpr (parDim_ == 2) {
#pragma omp parallel for simd collapse(2)
        for (int64_t i=0; i<ncoeffs_[0]; ++i) {
          for (int64_t j=0; j<ncoeffs_[1]; ++j) {
            auto c = transformation( std::array<real_t,2>{i/real_t(ncoeffs_[0]-1), j/real_t(ncoeffs_[1]-1)} );
            for (short_t d=0; d<geoDim_; ++d)
              coeffs_[d].detach()[i*ncoeffs_[1]+j] = c[d];
          }
        }
      }

      // 3D
      else if constexpr (parDim_ == 3) {
#pragma omp parallel for simd collapse(3)
        for (int64_t i=0; i<ncoeffs_[0]; ++i) {
          for (int64_t j=0; j<ncoeffs_[1]; ++j) {
            for (int64_t k=0; k<ncoeffs_[2]; ++k) {
              auto c = transformation( std::array<real_t,3>{i/real_t(ncoeffs_[0]-1), j/real_t(ncoeffs_[1]-1), k/real_t(ncoeffs_[2]-1)} );
              for (short_t d=0; d<geoDim_; ++d)
                coeffs_[d].detach()[i*ncoeffs_[1]*ncoeffs_[2]+j*ncoeffs_[2]+k] = c[d];
            }
          }
        }
      }

      // 4D
      else if constexpr (parDim_ == 4) {
#pragma omp parallel for simd collapse(4)
        for (int64_t i=0; i<ncoeffs_[0]; ++i) {
          for (int64_t j=0; j<ncoeffs_[1]; ++j) {
            for (int64_t k=0; k<ncoeffs_[2]; ++k) {
              for (int64_t l=0; l<ncoeffs_[3]; ++l) {
                auto c = transformation( std::array<real_t,4>{i/real_t(ncoeffs_[0]-1), j/real_t(ncoeffs_[1]-1), k/real_t(ncoeffs_[2]-1), l/real_t(ncoeffs_[3]-1)} );
                for (short_t d=0; d<geoDim_; ++d)
                  coeffs_[d].detach()[i*ncoeffs_[1]*ncoeffs_[2]*ncoeffs_[3]+j*ncoeffs_[2]*ncoeffs_[3]+k*ncoeffs_[3]+l] = c[d];
              }
            }
          }
        }
      }

      return *this;
    }

    // Returns a string representation of the UniformBSpline object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << name()
         << "(\n  parDim=" << parDim_
         << ", geoDim=" << geoDim_

         << ", degrees=";
      for (short_t i=0; i<parDim_-1; ++i)
        os << degree(i) << "x";
      os << degree(parDim_-1)

         << ", knots=";
      for (short_t i=0; i<parDim_-1; ++i)
        os << nknots(i) << "x";
      os << nknots(parDim_-1)

         << ", coeffs=";
      for (short_t i=0; i<parDim_-1; ++i)
        os << ncoeffs(i) << "x";
      os << ncoeffs(parDim_-1)
         << "\n)";
    }

    inline const std::string& name() const noexcept {
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
    // Initialize coefficients
    inline void init_coeffs(BSplineInit init)
    {
      switch (init) {

      case (BSplineInit::zeros): {

        // Fill coefficients with zeros
        for (short_t i=0; i<geoDim_; ++i) {

          int64_t size = 1;
          for (short_t j=0; j<parDim_; ++j)
            size *= ncoeffs_[j];

          coeffs_[i] = torch::zeros(size, core<real_t>::options_);
        }
        break;
      }

      case (BSplineInit::ones): {

        // Fill coefficients with ones
        for (short_t i=0; i<geoDim_; ++i) {

          int64_t size = 1;
          for (short_t j=0; j<parDim_; ++j)
            size *= ncoeffs_[j];

          coeffs_[i] = torch::ones(size, core<real_t>::options_);
        }
        break;
      }

      case (BSplineInit::linear): {

        // Fill coefficients with the tensor-product of linearly
        // increasing values between 0 and 1 per univariate dimension
        for (short_t i=0; i<geoDim_; ++i) {
          coeffs_[i] = torch::ones(1, core<real_t>::options_);

          for (short_t j=0; j<parDim_; ++j)
            {
              if (i==j)
                coeffs_[i] = coeffs_[i].kron(torch::linspace(static_cast<real_t>(0),
                                                             static_cast<real_t>(1),
                                                             ncoeffs_[j],
                                                             core<real_t>::options_));
              else
                coeffs_[i] = coeffs_[i].kron(torch::ones(ncoeffs_[j],
                                                         core<real_t>::options_));
            }
        }
        break;
      }

      case (BSplineInit::random): {

        // Fill coefficients with random values
        for (short_t i=0; i<geoDim_; ++i) {

          int64_t size = 1;
          for (short_t j=0; j<parDim_; ++j)
            size *= ncoeffs_[j];

          coeffs_[i] = torch::rand(size, core<real_t>::options_);
        }
        break;
      }

      case (BSplineInit::greville): {

        // Fill coefficients with the tensor-product of Greville
        // abscissae values per univariate dimension
        for (short_t i=0; i<geoDim_; ++i) {
          coeffs_[i] = torch::ones(1, core<real_t>::options_);

          for (short_t j=0; j<parDim_; ++j)
            {
              if (i==j) {
                auto greville_ = torch::zeros(ncoeffs_[j], core<real_t>::options_);
                auto greville = greville_.template accessor<real_t,1>();
                auto knots = knots_[j].template accessor<real_t,1>();
                for (int64_t k=0; k<ncoeffs_[j]; ++k) {
                  for (short_t l=1; l<=degrees_[j]; ++l)
                    greville[k] += knots[k+l];
                  greville[k] /= degrees_[j];
                }
                coeffs_[i] = coeffs_[i].kron(greville_);
              } else
                coeffs_[i] = coeffs_[i].kron(torch::ones(ncoeffs_[j],
                                                         core<real_t>::options_));
            }
        }
        break;
      }

      default:
        throw std::runtime_error("Unsupported BSplineInit option");
      }
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_1d(const torch::Tensor& xi, int64_t i) const
    {
      if constexpr (geoDim_ > 1) {
        auto basfunc = eval_impl<degrees_[0],0,(short_t)deriv%10>(i, xi[0]);
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t k=0; k<geoDim_; ++k)
          result[k] = torch::matmul(basfunc,
                                    coeffs<false>(k).index(
                                                           {
                                                             torch::indexing::Slice(i-degrees_[0], i+1, 1)
                                                           }
                                                           ).flatten());
        return result;
      } else
        return
          torch::matmul(eval_impl<degrees_[0],0,(short_t)deriv%10>(i, xi[0]),
                        coeffs<false>(0).index(
                                               {
                                                 torch::indexing::Slice(i-degrees_[0], i+1, 1)
                                               }
                                               ).flatten());
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_2d(const torch::Tensor& xi, int64_t i, int64_t j) const
    {
      if constexpr (geoDim_ > 1) {
        auto basfunc = torch::kron(
                                   eval_impl<degrees_[0],0, (short_t)deriv    %10>(i, xi[0]),
                                   eval_impl<degrees_[1],1,((short_t)deriv/10)%10>(j, xi[1])
                                   );
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t k=0; k<geoDim_; ++k)
          result[k] = torch::matmul(basfunc,
                                    coeffs<false>(k).index(
                                                           {
                                                             torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                             torch::indexing::Slice(j-degrees_[1], j+1, 1)
                                                           }
                                                           ).flatten());
        return result;
      } else
        return
          torch::matmul(torch::kron(
                                    eval_impl<degrees_[0],0, (short_t)deriv    %10>(i, xi[0]),
                                    eval_impl<degrees_[1],1,((short_t)deriv/10)%10>(j, xi[1])
                                    ),
                        coeffs<false>(0).index(
                                               {
                                                 torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                 torch::indexing::Slice(j-degrees_[1], j+1, 1)
                                               }
                                               ).flatten());
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_3d(const torch::Tensor& xi, int64_t i, int64_t j, int64_t k) const
    {
      if constexpr (geoDim_ > 1) {
        auto basfunc = torch::kron(
                                   torch::kron(
                                               eval_impl<degrees_[0],0, (short_t)deriv    %10>(i, xi[0]),
                                               eval_impl<degrees_[1],1,((short_t)deriv/10)%10>(j, xi[1])
                                               ),
                                   eval_impl<degrees_[2],2,((short_t)deriv/100)%10>(k, xi[2])
                                   );
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t k=0; k<geoDim_; ++k)
          result[k] = torch::matmul(basfunc,
                                    coeffs<false>(k).index(
                                                           {
                                                             torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                             torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                                             torch::indexing::Slice(k-degrees_[2], k+1, 1)
                                                           }
                                                           ).flatten());
        return result;
      } else
        return
          torch::matmul(torch::kron(
                                    torch::kron(
                                                eval_impl<degrees_[0],0, (short_t)deriv    %10>(i, xi[0]),
                                                eval_impl<degrees_[1],1,((short_t)deriv/10)%10>(j, xi[1])
                                                ),
                                    eval_impl<degrees_[2],2,((short_t)deriv/100)%10>(k, xi[2])
                                    ),
                        coeffs<false>(0).index(
                                               {
                                                 torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                 torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                                 torch::indexing::Slice(k-degrees_[2], k+1, 1)
                                               }
                                               ).flatten());
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval_4d(const torch::Tensor& xi, int64_t i, int64_t j, int64_t k, int64_t l) const
    {
      if constexpr (geoDim_ > 1) {
        auto basfunc = torch::kron(
                                   torch::kron(
                                               eval_impl<degrees_[0],0, (short_t)deriv      %10>(i, xi[0]),
                                               eval_impl<degrees_[1],1,((short_t)deriv/  10)%10>(j, xi[1])
                                               ),
                                   torch::kron(
                                               eval_impl<degrees_[2],2,((short_t)deriv/ 100)%10>(k, xi[2]),
                                               eval_impl<degrees_[3],3,((short_t)deriv/1000)%10>(l, xi[3])
                                               )
                                   );
        std::array<torch::Tensor, geoDim_> result;
        for (std::size_t k=0; k<geoDim_; ++k)
          result[k] = torch::matmul(basfunc,
                                    coeffs<false>(k).index(
                                                           {
                                                             torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                             torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                                             torch::indexing::Slice(k-degrees_[2], k+1, 1),
                                                             torch::indexing::Slice(l-degrees_[3], l+1, 1)
                                                           }
                                                           ).flatten());
        return result;
      } else
        return
          torch::matmul(torch::kron(
                                    torch::kron(
                                                eval_impl<degrees_[0],0, (short_t)deriv      %10>(i, xi[0]),
                                                eval_impl<degrees_[1],1,((short_t)deriv/  10)%10>(j, xi[1])
                                                ),
                                    torch::kron(
                                                eval_impl<degrees_[2],2,((short_t)deriv/ 100)%10>(k, xi[2]),
                                                eval_impl<degrees_[3],3,((short_t)deriv/1000)%10>(l, xi[3])
                                                )
                                    ),
                        coeffs<false>(0).index(
                                               {
                                                 torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                 torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                                 torch::indexing::Slice(k-degrees_[2], k+1, 1),
                                                 torch::indexing::Slice(l-degrees_[3], l+1, 1)
                                               }
                                               ).flatten());
    }

    // Returns the values of the vector of B-spline basis function (or
    // their derivatives) evaluated in the point \f$ \xi \f$
    // \f$
    //   \left[ D^r B_{i-d,d}, \dots, D^r B_{i,d} \right]
    // \f$,
    // where
    // \f$
    //   d
    // \f$
    // is the degree of the B-spline and
    // \f$
    //   r
    // \f$
    // denotes the requested derivative.
    //
    // For a detailed descriptions see Section 2.3 in
    // https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT5340/v05/undervisningsmateriale/kap2-new.pdf
    // and Section 3.2 in
    // https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT5340/v05/undervisningsmateriale/kap3-new.pdf
    template<short_t degree, short_t dim, short_t deriv>
    inline auto eval_impl(int64_t i, const torch::Tensor& xi) const
    {
      static_assert(degree <= 5, "Degrees higher than 5 are not implemented");

      // linear B-Splines
      if constexpr (degree == 1) {

        // zero-th derivative
        if constexpr (deriv == 0) {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::stack(
                         {
                           ( knots_[dim][i+1]-xi             ) / ( knots_[dim][i+1]-knots_[dim][i] ),
                           ( xi              -knots_[dim][i] ) / ( knots_[dim][i+1]-knots_[dim][i] )
                         }
                         ).view({1,2})
            :
            torch::stack(
                         {
                           one_[0]
                         }
                         )
            ;
        }

        // first derivative
        else if constexpr (deriv == 1) {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::stack(
                         {
                           -one_[0] / ( knots_[dim][i+1]-knots_[dim][i] ),
                           one_[0] / ( knots_[dim][i+1]-knots_[dim][i] )
                         }
                         ).view({1,2})
            :
            torch::stack(
                         {
                           one_[0]
                         }
                         )
            ;
        }

        // second or higher-order derivatives
        else {
          return torch::stack(
                              {
                                zero_[0],
                                zero_[0]
                              }
                              ).view({1,2});
        }
      }

      // quadratic B-splines
      else if constexpr (degree == 2) {

        // zero-th derivative
        if constexpr (deriv == 0) {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::matmul(eval_impl<1,dim,0>(i,xi),
                          torch::stack(
                                       {
                                         ( knots_[dim][i+1]-xi               ) / ( knots_[dim][i+1]-knots_[dim][i-1] ),
                                         ( xi              -knots_[dim][i-1] ) / ( knots_[dim][i+1]-knots_[dim][i-1] ),
                                         zero_[0],

                                         zero_[0],
                                         ( knots_[dim][i+2]-xi             ) / ( knots_[dim][i+2]-knots_[dim][i] ),
                                         ( xi              -knots_[dim][i] ) / ( knots_[dim][i+2]-knots_[dim][i] )
                                       }
                                       ).view({2,3}))
            :
            torch::stack(
                         {
                           zero_[0], one_[0]
                         }
                         ).view({1,2})
            ;
        }

        // first or higher-order derivatives
        else {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::matmul(eval_impl<1,dim,deriv-1>(i,xi),
                          torch::stack(
                                       {
                                         -one_[0] / ( knots_[dim][i+1]-knots_[dim][i-1] ),
                                         one_[0] / ( knots_[dim][i+1]-knots_[dim][i-1] ),
                                         zero_[0],

                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+2]-knots_[dim][i] ),
                                         one_[0] / ( knots_[dim][i+2]-knots_[dim][i] )
                                       }
                                       ).view({2,3}))
            :
            torch::stack(
                         {
                           zero_[0], one_[0]
                         }
                         ).view({1,2})
            ;
        }
      }

      // cubic B-splines
      else if constexpr (degree == 3) {

        // zero-th derivative
        if constexpr (deriv == 0) {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::matmul(eval_impl<2,dim,0>(i,xi),
                          torch::stack(
                                       {
                                         ( knots_[dim][i+1]-xi               ) / ( knots_[dim][i+1]-knots_[dim][i-2] ),
                                         ( xi              -knots_[dim][i-2] ) / ( knots_[dim][i+1]-knots_[dim][i-2] ),
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         ( knots_[dim][i+2]-xi               ) / ( knots_[dim][i+2]-knots_[dim][i-1] ),
                                         ( xi              -knots_[dim][i-1] ) / ( knots_[dim][i+2]-knots_[dim][i-1] ),
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         ( knots_[dim][i+3]-xi             ) / ( knots_[dim][i+3]-knots_[dim][i] ),
                                         ( xi              -knots_[dim][i] ) / ( knots_[dim][i+3]-knots_[dim][i] )
                                       }
                                       ).view({3,4}))
            :
            torch::stack(
                         {
                           zero_[0], zero_[0], one_[0]
                         }
                         ).view({1,3})
            ;
        }

        // first or higher-order derivatives
        else {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::matmul(eval_impl<2,dim,deriv-1>(i,xi),
                          torch::stack(
                                       {
                                         -one_[0] / ( knots_[dim][i+1]-knots_[dim][i-2] ),
                                         one_[0] / ( knots_[dim][i+1]-knots_[dim][i-2] ),
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+2]-knots_[dim][i-1] ),
                                         one_[0] / ( knots_[dim][i+2]-knots_[dim][i-1] ),
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+3]-knots_[dim][i] ),
                                         one_[0] / ( knots_[dim][i+3]-knots_[dim][i] )
                                       }
                                       ).view({3,4}))
            :
            torch::stack(
                         {
                           zero_[0], zero_[0], one_[0]
                         }
                         ).view({1,3})
            ;
        }
      }

      // quartic B-splines
      else if constexpr (degree == 4) {

        // zero-th derivative
        if constexpr (deriv == 0) {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::matmul(eval_impl<3,dim,0>(i,xi),
                          torch::stack(
                                       {
                                         ( knots_[dim][i+1]-xi               ) / ( knots_[dim][i+1]-knots_[dim][i-3] ),
                                         ( xi              -knots_[dim][i-3] ) / ( knots_[dim][i+1]-knots_[dim][i-3] ),
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         ( knots_[dim][i+2]-xi               ) / ( knots_[dim][i+2]-knots_[dim][i-2] ),
                                         ( xi              -knots_[dim][i-2] ) / ( knots_[dim][i+2]-knots_[dim][i-2] ),
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         ( knots_[dim][i+3]-xi               ) / ( knots_[dim][i+3]-knots_[dim][i-1] ),
                                         ( xi              -knots_[dim][i-1] ) / ( knots_[dim][i+3]-knots_[dim][i-1] ),
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         ( knots_[dim][i+4]-xi             ) / ( knots_[dim][i+4]-knots_[dim][i] ),
                                         ( xi              -knots_[dim][i] ) / ( knots_[dim][i+4]-knots_[dim][i] )
                                       }
                                       ).view({4,5}))
            :
            torch::stack(
                         {
                           zero_[0], zero_[0], zero_[0], one_[0]
                         }
                         ).view({1,4})
            ;
        }

        // first or higher-order derivatives
        else {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::matmul(eval_impl<3,dim,deriv-1>(i,xi),
                          torch::stack(
                                       {
                                         -one_[0] / ( knots_[dim][i+1]-knots_[dim][i-3] ),
                                         one_[0] / ( knots_[dim][i+1]-knots_[dim][i-3] ),
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+2]-knots_[dim][i-2] ),
                                         one_[0] / ( knots_[dim][i+2]-knots_[dim][i-2] ),
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+3]-knots_[dim][i-1] ),
                                         one_[0] / ( knots_[dim][i+3]-knots_[dim][i-1] ),
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+4]-knots_[dim][i] ),
                                         one_[0] / ( knots_[dim][i+4]-knots_[dim][i] )
                                       }
                                       ).view({4,5}))
            :
            torch::stack(
                         {
                           zero_[0], zero_[0], zero_[0], one_[0]
                         }
                         ).view({1,4})
            ;
        }
      }

      // quintic B-splines
      else if constexpr (degree == 5) {

        // zero-th derivative
        if constexpr (deriv == 0) {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::matmul(eval_impl<4,dim,0>(i,xi),
                          torch::stack(
                                       {
                                         ( knots_[dim][i+1]-xi               ) / ( knots_[dim][i+1]-knots_[dim][i-4] ),
                                         ( xi              -knots_[dim][i-4] ) / ( knots_[dim][i+1]-knots_[dim][i-4] ),
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         ( knots_[dim][i+2]-xi               ) / ( knots_[dim][i+2]-knots_[dim][i-3] ),
                                         ( xi              -knots_[dim][i-3] ) / ( knots_[dim][i+2]-knots_[dim][i-3] ),
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         ( knots_[dim][i+3]-xi               ) / ( knots_[dim][i+3]-knots_[dim][i-2] ),
                                         ( xi              -knots_[dim][i-2] ) / ( knots_[dim][i+3]-knots_[dim][i-2] ),
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         ( knots_[dim][i+4]-xi               ) / ( knots_[dim][i+4]-knots_[dim][i-1] ),
                                         ( xi              -knots_[dim][i-1] ) / ( knots_[dim][i+4]-knots_[dim][i-1] ),
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         ( knots_[dim][i+5]-xi             ) / ( knots_[dim][i+5]-knots_[dim][i] ),
                                         ( xi              -knots_[dim][i] ) / ( knots_[dim][i+5]-knots_[dim][i] )
                                       }
                                       ).view({5,6}))
            :
            torch::stack(
                         {
                           zero_[0], zero_[0], zero_[0], zero_[0], one_[0]
                         }
                         ).view({1,5})
            ;
        }

        // first or higher-order derivatives
        else {
          return
            (xi < one_[0]).template item<bool>()
            ?
            torch::matmul(eval_impl<4,dim,deriv-1>(i,xi),
                          torch::stack(
                                       {
                                         -one_[0] / ( knots_[dim][i+1]-knots_[dim][i-4] ),
                                         one_[0] / ( knots_[dim][i+1]-knots_[dim][i-4] ),
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+2]-knots_[dim][i-3] ),
                                         one_[0] / ( knots_[dim][i+2]-knots_[dim][i-3] ),
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+3]-knots_[dim][i-2] ),
                                         one_[0] / ( knots_[dim][i+3]-knots_[dim][i-2] ),
                                         zero_[0],
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+4]-knots_[dim][i-1] ),
                                         one_[0] / ( knots_[dim][i+4]-knots_[dim][i-1] ),
                                         zero_[0],

                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         zero_[0],
                                         -one_[0] / ( knots_[dim][i+5]-knots_[dim][i] ),
                                         one_[0] / ( knots_[dim][i+5]-knots_[dim][i] )
                                       }
                                       ).view({5,6}))
            :
            torch::stack(
                         {
                           zero_[0], zero_[0], zero_[0], zero_[0], one_[0]
                         }
                         ).view({1,5})
            ;
        }
      }
    }
  };



  /// Tensor-product non-uniform B-Spline (core functionality)
  ///
  /// This class extends the base class UniformBSplineCore to
  /// non-uniform B-Splines. Like its base class it only implements
  /// the core functionality of non-uniform B-Splines
  template<typename real_t, short_t GeoDim, short_t... Degrees>
  class NonUniformBSplineCore : public UniformBSplineCore<real_t, GeoDim, Degrees...>
  {
  private:
    using Base = UniformBSplineCore<real_t, GeoDim, Degrees...>;

  public:
    // Constructor: equidistant knot vectors
    using UniformBSplineCore<real_t, GeoDim, Degrees...>::UniformBSplineCore;

    // Constructor: non-equidistant knot vectors
    NonUniformBSplineCore(std::array<std::vector<real_t>, Base::parDim_> kv,
                          BSplineInit init = BSplineInit::zeros)
      : Base(std::array<int64_t, Base::parDim_>{0}, init)
    {
      for (short_t i=0; i<Base::parDim_; ++i) {
        Base::knots_[i] = torch::from_blob(static_cast<real_t*>(kv[i].data()),
                                           kv[i].size(), core<real_t>::options_).clone();

        // Store the size of the knot vector
        Base::nknots_[i] = Base::knots_[i].size(0);

        // Store the size of the coefficient vector
        Base::ncoeffs_[i] = Base::nknots_[i]-Base::degrees_[i]-1;
      }

      // Initialize coefficients
      Base::init_coeffs(init);
    }

    template<BSplineDeriv deriv = BSplineDeriv::func>
    inline auto eval(const torch::Tensor& xi) const
    {
      static_assert(Base::parDim_ <= 4, "Unsupported parametric dimension");

      // 1D
      if constexpr (Base::parDim_ == 1) {
        int64_t i;
        auto knots = Base::knots_[0].template accessor<real_t,1>();
        for (i=Base::degrees_[0]; i<Base::nknots_[0]-Base::degrees_[0]-1; ++i)
          if (knots[i+1] > xi[0].item<real_t>())
            break;

        return Base::eval_1d(xi, i);
      }

      // 2D
      else if constexpr (Base::parDim_ == 2) {
        int64_t i;
        auto knots = Base::knots_[0].template accessor<real_t,1>();
        for (i=Base::degrees_[0]; i<Base::nknots_[0]-Base::degrees_[0]-1; ++i)
          if (knots[i+1] > xi[0].item<real_t>())
            break;

        int64_t j;
        knots = Base::knots_[1].template accessor<real_t,1>();
        for (j=Base::degrees_[1]; j<Base::nknots_[1]-Base::degrees_[1]-1; ++j)
          if (knots[j+1] > xi[1].item<real_t>())
            break;

        return Base::eval_2d(xi, i, j);
      }

      // 3D
      else if constexpr (Base::parDim_ == 3) {
        int64_t i;
        auto knots = Base::knots_[0].template accessor<real_t,1>();
        for (i=Base::degrees_[0]; i<Base::nknots_[0]-Base::degrees_[0]-1; ++i)
          if (knots[i+1] > xi[0].item<real_t>())
            break;

        int64_t j;
        knots = Base::knots_[1].template accessor<real_t,1>();
        for (j=Base::degrees_[1]; j<Base::nknots_[1]-Base::degrees_[1]-1; ++j)
          if (knots[j+1] > xi[1].item<real_t>())
            break;

        int64_t k;
        knots = Base::knots_[2].template accessor<real_t,1>();
        for (k=Base::degrees_[2]; k<Base::nknots_[2]-Base::degrees_[2]-1; ++k)
          if (knots[k+1] > xi[2].item<real_t>())
            break;

        return Base::eval_3d(xi, i, j, k);
      }

      // 4D
      else if constexpr (Base::parDim_ == 4) {
        int64_t i;
        auto knots = Base::knots_[0].template accessor<real_t,1>();
        for (i=Base::degrees_[0]; i<Base::nknots_[0]-Base::degrees_[0]-1; ++i)
          if (knots[i+1] > xi[0].item<real_t>())
            break;

        int64_t j;
        knots = Base::knots_[1].template accessor<real_t,1>();
        for (j=Base::degrees_[1]; j<Base::nknots_[1]-Base::degrees_[1]-1; ++j)
          if (knots[j+1] > xi[1].item<real_t>())
            break;

        int64_t k;
        knots = Base::knots_[2].template accessor<real_t,1>();
        for (k=Base::degrees_[2]; k<Base::nknots_[2]-Base::degrees_[2]-1; ++k)
          if (knots[k+1] > xi[2].item<real_t>())
            break;

        int64_t l;
        knots = Base::knots_[3].template accessor<real_t,1>();
        for (l=Base::degrees_[3]; l<Base::nknots_[3]-Base::degrees_[3]-1; ++l)
          if (knots[l+1] > xi[3].item<real_t>())
            break;

        return Base::eval_4d(xi, i, j, k, l);
      }
    }
  };

  /// B-Spline (common high-level functionality)
  ///
  /// This class implements some high-level common functionality of
  /// all B-spline classes, e.g., plotting which rely on low-level
  /// functionality that is implemented differently for uniform and
  /// non-uniform B-spline. C++ suggests to use virtual methods for
  /// this purpose and implement the common functionality in a base
  /// class. However, this is not performance for low-level
  /// functionality, e.g., point-wise function evaluation which is
  /// called repeatedly. Moreover, virtual methods do not work with
  /// templated functions, which is why we implement high-level common
  /// functionality here and 'inject' the core functionality by
  /// deriving from a particular base class.
  template<typename real_t, typename BSplineCore>
  class BSplineCommon : public BSplineCore
  {
  public:
    using BSplineCore::BSplineCore;

    // Plots the B-Spline object using matplotlibcpp
    inline auto plot(int64_t xres=10, int64_t yres=10, int64_t zres=10) const
    {
      return plot(*this, xres, yres, zres);
    }

    // Plots the B-Spline object using matplotlibcpp
    template<typename BSplineCore_t>
    inline auto plot(const BSplineCommon<real_t, BSplineCore_t>& color,
                     int64_t xres=10, int64_t yres=10, int64_t zres=10) const
    {
      static_assert(BSplineCore::parDim_ == BSplineCore_t::parDim_,
                    "Parametric dimensions must match");

      if ((void*)this != (void*)&color && BSplineCore_t::geoDim_>1)
        throw std::runtime_error("BSpline for coloring must have geoDim=1");

      if constexpr(BSplineCore::parDim_==1 && BSplineCore::geoDim_==1) {

        //
        // mapping: [0,1] -> R^1
        //

        matplot::vector_1d Xfine(xres, 0.0);
        matplot::vector_1d Yfine(xres, 0.0);

#pragma omp parallel for simd
        for (int64_t i=0; i<xres; ++i)
          Xfine[i] = eval(torch::stack(
                                       {
                                         torch::full({1}, i/real_t(xres-1))
                                       }
                                       ).view({1})
                          ).template item<real_t>();

        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
#pragma omp parallel for simd
            for (int64_t i=0; i<xres; ++i)
              Yfine[i] = color.eval(torch::stack(
                                                 {
                                                   torch::full({1}, i/real_t(xres-1)),
                                                 }
                                                 ).view({1})
                                    ).template item<real_t>();
          }
        }

        // Plotting ...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
            matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          }
        } else {
          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);

          auto x = BSplineCore::coeffs<false>(0);

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = x[i].template item<real_t>();
          }

          matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot(X, Y, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R");
        matplot::xlabel("x");
        matplot::ylabel("y");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim_==1 && BSplineCore::geoDim_==2) {

        //
        // mapping: [0,1] -> R^2
        //

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
            matplot::vector_2d Xfine(1, matplot::vector_1d(xres, 0.0));
            matplot::vector_2d Yfine(1, matplot::vector_1d(xres, 0.0));
            matplot::vector_2d Zfine(1, matplot::vector_1d(xres, 0.0));

#pragma omp parallel for simd
            for (int64_t i=0; i<xres; ++i) {
              auto coords = eval(torch::stack(
                                              {
                                                torch::full({1}, i/real_t(xres-1))
                                              }
                                              ).view({1})
                                 );
              Xfine[0][i] = coords[0].template item<real_t>();
              Yfine[0][i] = coords[1].template item<real_t>();
              Zfine[0][i] = color.eval(torch::stack(
                                                    {
                                                      torch::full({1}, i/real_t(xres-1))
                                                    }
                                                    ).view({1})
                                       ).template item<real_t>();
            }
            matplot::view(2);
            matplot::mesh(Xfine, Yfine, Zfine);
          }
        } else {
          matplot::vector_1d Xfine(xres, 0.0);
          matplot::vector_1d Yfine(xres, 0.0);

#pragma omp parallel for simd
          for (int64_t i=0; i<xres; ++i) {
            auto coords = eval(torch::stack(
                                            {
                                              torch::full({1}, i/real_t(xres-1))
                                            }
                                            ).view({1})
                               );
            Xfine[i] = coords[0].template item<real_t>();
            Yfine[i] = coords[1].template item<real_t>();
          }

          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);

          auto x = BSplineCore::coeffs<false>(0);
          auto y = BSplineCore::coeffs<false>(1);

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = x[i].template item<real_t>();
            Y[i] = y[i].template item<real_t>();
          }

          matplot::plot(Xfine, Yfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot(X, Y, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R^2");
        matplot::xlabel("x");
        matplot::ylabel("y");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim_==1 && BSplineCore::geoDim_==3) {

        //
        // mapping: [0,1] -> R^3
        //

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
            matplot::vector_2d Xfine(1, matplot::vector_1d(xres, 0.0));
            matplot::vector_2d Yfine(1, matplot::vector_1d(xres, 0.0));
            matplot::vector_2d Zfine(1, matplot::vector_1d(xres, 0.0));
            matplot::vector_2d Cfine(1, matplot::vector_1d(xres, 0.0));

#pragma omp parallel for simd
            for (int64_t i=0; i<xres; ++i) {
              auto coords = eval(torch::stack(
                                              {
                                                torch::full({1}, i/real_t(xres-1))
                                              }
                                              ).view({1})
                                 );
              Xfine[0][i] = coords[0].template item<real_t>();
              Yfine[0][i] = coords[1].template item<real_t>();
              Zfine[0][i] = coords[2].template item<real_t>();

              Cfine[0][i] = color.eval(torch::stack(
                                                    {
                                                      torch::full({1}, i/real_t(xres-1))
                                                    }
                                                    ).view({1})
                                       ).template item<real_t>();
            }

            matplot::mesh(Xfine, Yfine, Zfine, Cfine);
          }
        } else {
          matplot::vector_1d Xfine(xres, 0.0);
          matplot::vector_1d Yfine(xres, 0.0);
          matplot::vector_1d Zfine(xres, 0.0);

#pragma omp parallel for simd
          for (int64_t i=0; i<xres; ++i) {
            auto coords = eval(torch::stack(
                                            {
                                              torch::full({1}, i/real_t(xres-1))
                                            }
                                            ).view({1})
                               );
            Xfine[i] = coords[0].template item<real_t>();
            Yfine[i] = coords[1].template item<real_t>();
            Zfine[i] = coords[2].template item<real_t>();
          }

          matplot::vector_1d X(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Y(BSplineCore::ncoeffs(0), 0.0);
          matplot::vector_1d Z(BSplineCore::ncoeffs(0), 0.0);

          auto x = BSplineCore::coeffs<false>(0);
          auto y = BSplineCore::coeffs<false>(1);
          auto z = BSplineCore::coeffs<false>(2);

#pragma omp parallel for simd
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i) {
            X[i] = x[i].template item<real_t>();
            Y[i] = y[i].template item<real_t>();
            Z[i] = z[i].template item<real_t>();
          }

          matplot::plot3(Xfine, Yfine, Zfine, "b-")->line_width(2);
          matplot::hold(matplot::on);
          matplot::plot3(X, Y, Z, ".k-")->line_width(1);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1] -> R^3");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim_==2 && BSplineCore::geoDim_==2) {

        //
        // mapping: [0,1]^2 -> R^2
        //

        matplot::vector_2d Xfine(yres, matplot::vector_1d(xres, 0.0));
        matplot::vector_2d Yfine(yres, matplot::vector_1d(xres, 0.0));
        matplot::vector_2d Zfine(yres, matplot::vector_1d(xres, 0.0));

#pragma omp parallel for simd collapse(2)
        for (int64_t i=0; i<xres; ++i)
          for (int64_t j=0; j<yres; ++j) {
            auto coords = BSplineCore::eval(torch::stack(
                                                         {
                                                           torch::full({1}, i/real_t(xres-1)),
                                                           torch::full({1}, j/real_t(yres-1))
                                                         }
                                                         ).view({2})
                                            );
            Xfine[j][i] = coords[0].template item<real_t>();
            Yfine[j][i] = coords[1].template item<real_t>();
          }

        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
#pragma omp parallel for simd collapse(2)
            for (int64_t i=0; i<xres; ++i)
              for (int64_t j=0; j<yres; ++j) {
                Zfine[j][i] = color.eval(torch::stack(
                                                      {
                                                        torch::full({1}, i/real_t(xres-1)),
                                                        torch::full({1}, j/real_t(yres-1))
                                                      }
                                                      ).view({2})
                                         ).template item<real_t>();
              }
          }
        }

        // Plotting...
        if ((void*)this != (void*)&color && BSplineCore_t::geoDim_==1) {
          matplot::view(2);
          matplot::colormap(matplot::palette::hsv());
          matplot::mesh(Xfine, Yfine, Zfine)->palette_map_at_surface(true).face_alpha(0.7);
        } else {
          matplot::view(2);
          matplot::vector_2d X(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Y(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Z(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));

          auto x = BSplineCore::template coeffs<false>(0);
          auto y = BSplineCore::template coeffs<false>(1);

#pragma omp parallel for simd collapse(2)
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i)
            for (int64_t j=0; j<BSplineCore::ncoeffs(1); ++j) {
              X[j][i] = x[i][j].template item<real_t>();
              Y[j][i] = y[i][j].template item<real_t>();
            }

          matplot::colormap(matplot::palette::winter());
          matplot::mesh(Xfine, Yfine, Zfine);
          matplot::hold(matplot::on);
          matplot::surf(X, Y, Z)->palette_map_at_surface(true).face_alpha(0);
          matplot::hold(matplot::off);
        }
        matplot::title("BSpline: [0,1]^2 -> R^2");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else if constexpr(BSplineCore::parDim_==2 && BSplineCore::geoDim_==3) {

        ///
        // mapping: [0,1]^2 -> R^3
        ///

        matplot::vector_2d Xfine(yres, matplot::vector_1d(xres, 0.0));
        matplot::vector_2d Yfine(yres, matplot::vector_1d(xres, 0.0));
        matplot::vector_2d Zfine(yres, matplot::vector_1d(xres, 0.0));

#pragma omp parallel for simd collapse(2)
        for (int64_t i=0; i<xres; ++i)
          for (int64_t j=0; j<yres; ++j) {
            auto coords = BSplineCore::eval(torch::stack(
                                                         {
                                                           torch::full({1}, i/real_t(xres-1)),
                                                           torch::full({1}, j/real_t(yres-1))
                                                         }
                                                         ).view({2})
                                            );
            Xfine[j][i] = coords[0].template item<real_t>();
            Yfine[j][i] = coords[1].template item<real_t>();
            Zfine[j][i] = coords[2].template item<real_t>();
          }

        // Plotting...
        if ((void*)this != (void*)&color) {
          if constexpr (BSplineCore_t::geoDim_==1) {
            matplot::vector_2d Cfine(yres, matplot::vector_1d(xres, 0.0));

#pragma omp parallel for simd collapse(2)
            for (int64_t i=0; i<xres; ++i)
              for (int64_t j=0; j<yres; ++j) {
                Cfine[j][i] = color.eval(torch::stack(
                                                      {
                                                        torch::full({1}, i/real_t(xres-1)),
                                                        torch::full({1}, j/real_t(yres-1))
                                                      }
                                                      ).view({2})
                                         ).template item<real_t>();
              }
            matplot::colormap(matplot::palette::hsv());
            matplot::mesh(Xfine, Yfine, Zfine, Cfine);
          }
        }
        else {
          matplot::vector_2d X(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Y(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));
          matplot::vector_2d Z(BSplineCore::ncoeffs(1), matplot::vector_1d(BSplineCore::ncoeffs(0), 0.0));

          auto x = BSplineCore::coeffs<false>(0);
          auto y = BSplineCore::coeffs<false>(1);
          auto z = BSplineCore::coeffs<false>(2);

#pragma omp parallel for simd collapse(2)
          for (int64_t i=0; i<BSplineCore::ncoeffs(0); ++i)
            for (int64_t j=0; j<BSplineCore::ncoeffs(1); ++j) {
              X[j][i] = x[i][j].template item<real_t>();
              Y[j][i] = y[i][j].template item<real_t>();
              Z[j][i] = z[i][j].template item<real_t>();
            }

          matplot::colormap(matplot::palette::winter());
          matplot::mesh(Xfine, Yfine, Zfine);

          matplot::hold(matplot::on);
          matplot::surf(X, Y, Z)->palette_map_at_surface(true).face_alpha(0);
          matplot::hold(matplot::off);
        }

        matplot::title("BSpline: [0,1]^2 -> R^3");
        matplot::xlabel("x");
        matplot::ylabel("y");
        matplot::zlabel("z");
        return matplot::show();
      }

      else
        throw std::runtime_error("Unsupported combination of parametric/geometric dimensions");
    }
  };

  /// Tensor-product uniform B-spline
  template<typename real_t, short_t geoDim, short_t... Degrees>
  using UniformBSpline = BSplineCommon<real_t, UniformBSplineCore<real_t, geoDim, Degrees...>>;

  /// Print (as string) a UniformBSpline object
  template<typename real_t, short_t geoDim, short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const UniformBSpline<real_t, geoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

  /// Tensor-product non-uniform B-spline
  template<typename real_t, short_t geoDim, short_t... Degrees>
  using NonUniformBSpline = BSplineCommon<real_t, NonUniformBSplineCore<real_t, geoDim, Degrees...>>;
  
  /// Print (as string) a UniformBSpline object
  template<typename real_t, short_t geoDim, short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const NonUniformBSpline<real_t, geoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

} // namespace iganet
