#include <array>
#include <exception>
#include <functional>
#include <vector>

#include <torch/torch.h>

#include <core.hpp>

#pragma once

namespace iganet {

  enum class BSplineInit
    {
      zeros,
      ones,
      linear,
      random
    };

  template<typename real_t,
           short_t GeoDim, short_t... Degrees>
  class BSpline : public core<real_t>
  {
  private:
    // Dimension of the parametric space
    static constexpr const short_t parDim_ = sizeof...(Degrees);

    // Dimension of the geometric space
    static constexpr const short_t geoDim_ = GeoDim;

    // Array storing the degrees per dimension
    static constexpr const std::array<short_t,parDim_> degrees_ = { Degrees... };

    // Array storing the knot vectors
    std::array<torch::Tensor,parDim_> knots_;

    // Array storing the dimensions of the knot vectors
    std::array<int64_t,parDim_> nknots_;

    // Array storing the coefficients of the control net
    std::array<torch::Tensor,geoDim_> coeffs_;

    // Array storing the dimensions of the coefficients of the control net
    std::array<int64_t,parDim_> ncoeffs_;

    // String storing the full qualified name of the object
    mutable at::optional<std::string> name_;

    // LibTorch constants
    const torch::Tensor one_, zero_;

  public:
    // Constructor: number of knots
    BSpline(std::array<int64_t,parDim_> nknots, BSplineInit init = BSplineInit::zeros)
      : core<real_t>(),
        one_(torch::ones(1, core<real_t>::options_)),
        zero_(torch::zeros(1, core<real_t>::options_))
    {
      // Create open uniform knot vector
      for (short_t i=0; i<parDim_; ++i)
        {
          std::vector<real_t> kv;

          for (int64_t j=0; j<degrees_[i]; ++j)
            kv.push_back(static_cast<real_t>(0));

          for (int64_t j=0; j<nknots[i]; ++j)
            kv.push_back(static_cast<real_t>(j/real_t(nknots[i]-1)));

          for (int64_t j=0; j<degrees_[i]; ++j)
            kv.push_back(static_cast<real_t>(1));

          knots_[i] = torch::from_blob(static_cast<real_t*>(kv.data()),
                                       kv.size(), core<real_t>::options_).clone();

          // Store the dimension of the knot vector
          nknots_[i] = knots_[i].size(0);

          // Store the dimension of the coefficient vector
          ncoeffs_[i] = nknots_[i]-degrees_[i];
        }

      // Create coefficies of the control net
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

        // Fill coefficients with linearly increasing values in a single direction
        for (short_t i=0; i<geoDim_; ++i)
          {
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

      default:
        throw std::runtime_error("Unsupported BSplineInit option");
      }
    }

    // Returns a constant reference to the array of degrees
    inline static constexpr const std::array<short_t,parDim_>& degrees()
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
    inline const std::array<torch::Tensor,parDim_>& knots() const
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
    inline std::array<torch::Tensor,parDim_>& knots()
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
    inline const std::array<int64_t,parDim_>& nknots() const
    {
      return nknots_;
    }

    // Returns the dimension of the knot vector in the i-th dimension
    inline int64_t nknots(short_t i) const
    {
      assert(i>=0 && i<parDim_);
      return nknots_[i];
    }

    // Returns a constant reference to the array of coefficients
    inline const std::array<torch::Tensor,geoDim_>& coeffs() const
    {
      return coeffs_;
    }

    // Returns a constant reference to the coefficients in the i-th
    // dimension. If flatten=false, this function returns a
    // torch::Tensor with coefficients in the i-th dimension reshaped
    // according to the dimensions of the knot vectors
    template<bool flatten=true>
    inline auto coeffs(short_t i) const
      -> typename std::conditional<flatten, const torch::Tensor&, torch::Tensor>::type
    {
      assert(i>=0 && i<geoDim_);
      if constexpr (flatten)
        return coeffs_[i];
      else
        return coeffs_[i].view(ncoeffs_);
    }

    // Returns a non-constant reference to the array of coefficients
    inline std::array<torch::Tensor,geoDim_>& coeffs()
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

    // Returns all B-spline basis functions that are not zero in xi
    inline torch::Tensor eval(const torch::Tensor& xi) const
    {
      if constexpr (parDim_==1) {
        int64_t i = int64_t(xi[0].item<real_t>()*(nknots_[0]-2*degrees_[0]-1)+degrees_[0]);
        return
          torch::matmul(eval_<degrees_[0],0>(i, xi[0]),
                        coeffs<false>(0).index(
                                               {
                                                 torch::indexing::Slice(i-degrees_[0], i+1, 1)
                                               }
                                               ).flatten().view({degrees_[0]+1,1}));
      }

      else if constexpr (parDim_==2) {
        int64_t i = int64_t(xi[0].item<real_t>()*(nknots_[0]-2*degrees_[0]-1)+degrees_[0]);
        int64_t j = int64_t(xi[1].item<real_t>()*(nknots_[1]-2*degrees_[1]-1)+degrees_[1]);
        return
          torch::matmul(torch::kron(
                                    eval_<degrees_[0],0>(i, xi[0]),
                                    eval_<degrees_[1],1>(j, xi[1])
                                    ),
                        coeffs<false>(0).index(
                                               {
                                                 torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                 torch::indexing::Slice(j-degrees_[1], j+1, 1)
                                               }
                                               ).flatten().view({(degrees_[0]+1)*(degrees_[1]+1),1}));
      }

      else if constexpr (parDim_==3) {
        int64_t i = int64_t(xi[0].item<real_t>()*(nknots_[0]-2*degrees_[0]-1)+degrees_[0]);
        int64_t j = int64_t(xi[1].item<real_t>()*(nknots_[1]-2*degrees_[1]-1)+degrees_[1]);
        int64_t k = int64_t(xi[2].item<real_t>()*(nknots_[2]-2*degrees_[2]-2)+degrees_[2]);
        return
          torch::matmul(torch::kron(
                                    torch::kron(
                                                eval_<degrees_[0],0>(i, xi[0]),
                                                eval_<degrees_[1],1>(j, xi[1])
                                                ),
                                    eval_<degrees_[2],2>(k, xi[2])
                                    ),
                        coeffs<false>(0).index(
                                               {
                                                 torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                 torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                                 torch::indexing::Slice(k-degrees_[2], k+1, 1)
                                               }
                                               ).flatten().view({(degrees_[0]+1)*(degrees_[1]+1)*(degrees_[2]+1),1}));
      }

      else if constexpr (parDim_==4) {
        int64_t i = int64_t(xi[0].item<real_t>()*(nknots_[0]-2*degrees_[0]-1)+degrees_[0]);
        int64_t j = int64_t(xi[1].item<real_t>()*(nknots_[1]-2*degrees_[1]-1)+degrees_[1]);
        int64_t k = int64_t(xi[2].item<real_t>()*(nknots_[2]-2*degrees_[2]-1)+degrees_[2]);
        int64_t l = int64_t(xi[3].item<real_t>()*(nknots_[3]-2*degrees_[3]-1)+degrees_[3]);
        return
          torch::matmul(torch::kron(
                                    torch::kron(
                                                eval_<degrees_[0],0>(i, xi[0]),
                                                eval_<degrees_[1],1>(j, xi[1])
                                                ),
                                    torch::kron(
                                                eval_<degrees_[2],2>(k, xi[2]),
                                                eval_<degrees_[3],3>(l, xi[3])
                                                )
                                    ),
                        coeffs<false>(0).index(
                                               {
                                                 torch::indexing::Slice(i-degrees_[0], i+1, 1),
                                                 torch::indexing::Slice(j-degrees_[1], j+1, 1),
                                                 torch::indexing::Slice(k-degrees_[2], k+1, 1),
                                                 torch::indexing::Slice(l-degrees_[3], l+1, 1)
                                               }
                                               ).flatten().view({(degrees_[0]+1)*(degrees_[1]+1)*(degrees_[2]+1)*(degrees_[3]+1),1}));
      }

      else {
        throw std::runtime_error("Unsupported parametric dimension");
      }
    }

    // Transforms the coefficients based on the given mapping
    inline BSpline& transform(const std::function< std::array<real_t,geoDim_> (const std::array<real_t,parDim_>& )> transformation)
    {
      if constexpr (parDim_==1) {
        for (int64_t i=0; i<ncoeffs_[0]; ++i) {
          auto c = transformation( std::array<real_t,1>{i/real_t(ncoeffs_[0]-1)} );
          for (short_t d=0; d<geoDim_; ++d)
            coeffs_[d].detach()[i] = c[d];
        }
      }

      else if constexpr (parDim_==2) {
        for (int64_t i=0; i<ncoeffs_[0]; ++i) {
          for (int64_t j=0; j<ncoeffs_[1]; ++j) {
            auto c = transformation( std::array<real_t,2>{i/real_t(ncoeffs_[0]-1), j/real_t(ncoeffs_[1]-1)} );
            for (short_t d=0; d<geoDim_; ++d)
              coeffs_[d].detach()[i*ncoeffs_[1]+j] = c[d];
          }
        }
      }

      else if constexpr (parDim_==3) {
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

      else if constexpr (parDim_==4) {
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

      else
        throw std::runtime_error("Unsupported parametric dimension");

      return *this;
    }

    // Returns a string representation of the BSpline object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << name()
         << "(\nparDim=" << parDim_
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

  private:

    template<short_t degree, short_t dim>
    inline torch::Tensor eval_(int64_t i, const torch::Tensor& xi) const
    {
      if constexpr (degree==1) {
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
                         zero_[0], one_[0]
                       }
                       ).view({1,2})
          ;
      }

      else if constexpr (degree==2) {
        return
          (xi < one_[0]).template item<bool>()
          ?
          torch::matmul(eval_<1,dim>(i,xi),
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
                         zero_[0], zero_[0], one_[0]
                       }
                       ).view({1,3})
          ;
      }

      else if constexpr (degree==3) {
        return
          (xi < one_[0]).template item<bool>()
          ?
          torch::matmul(eval_<2,dim>(i,xi),
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
                         zero_[0], zero_[0], zero_[0], one_[0]
                       }
                       ).view({1,4})
          ;
      }

      else if constexpr (degree==4) {
        return
          (xi < one_[0]).template item<bool>()
          ?
          torch::matmul(eval_<3,dim>(i,xi),
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
                         zero_[0], zero_[0], zero_[0], zero_[0], one_[0]
                       }
                       ).view({1,5})
          ;
      }

      else if constexpr (degree==5) {
        return
          (xi < one_[0]).template item<bool>()
          ?
          torch::matmul(eval_<4,dim>(i,xi),
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
                         zero_[0], zero_[0], zero_[0], zero_[0], zero_[0], one_[0]
                       }
                       ).view({1,6})
          ;
      } 
      
      else {
        throw std::runtime_error("Degrees higher than 5 are not implemented");
      }
      
    }
  };

  /// Print (as string) a BSpline object
  template<typename real_t, short_t geoDim, short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const BSpline<real_t, geoDim, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

} // namespace iganet
