#include <array>
#include <exception>
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
  class BSpline
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

    torch::Tensor one_, zero_;
    
  public:
    // Constructor: number of knots
    BSpline(std::array<int64_t,parDim_> nknots, BSplineInit init = BSplineInit::zeros)
      : one_(torch::ones(1)), zero_(torch::zeros(1))
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
                                       kv.size(),
                                       torch::TensorOptions()
                                       .dtype(dtype<real_t>())).clone();

          // Store the dimension of the knot vector
          nknots_[i] = knots_[i].size(0);
          
          // Store the dimension of the coefficient vector
          ncoeffs_[i] = nknots_[i]-degrees_[i]-1;
        }
      
      // Create coefficies of the control net
      switch (init) {
        
      case (BSplineInit::zeros): {
        
        // Fill coefficients with zeros
        for (short_t i=0; i<geoDim_; ++i) {
          
          int64_t size = 1;
          for (short_t j=0; j<parDim_; ++j)
            size *= ncoeffs_[j];
          
          coeffs_[i] = torch::zeros(size,
                                    torch::TensorOptions()
                                    .dtype(dtype<real_t>()));
        }
        break;
      }
        
      case (BSplineInit::ones): {
        
        // Fill coefficients with ones
        for (short_t i=0; i<geoDim_; ++i) {
          
          int64_t size = 1;
          for (short_t j=0; j<parDim_; ++j)
            size *= ncoeffs_[j];
          
          coeffs_[i] = torch::ones(size,
                                   torch::TensorOptions()
                                   .dtype(dtype<real_t>()));
        }
        break;
      }
        
      case (BSplineInit::linear): {
        
        // Fill coefficients with linearly increasing values in a single direction
        for (short_t i=0; i<geoDim_; ++i)
          {
            coeffs_[i] = torch::ones(1,
                                     torch::TensorOptions()
                                     .dtype(dtype<real_t>()));
            
            for (short_t j=0; j<parDim_; ++j)
              {
                if (i==j)
                  coeffs_[i] = coeffs_[i].kron(torch::linspace(static_cast<real_t>(0),
                                                               static_cast<real_t>(1),
                                                               ncoeffs_[j],
                                                               torch::TensorOptions()
                                                               .dtype(dtype<real_t>())));
                else
                  coeffs_[i] = coeffs_[i].kron(torch::ones(ncoeffs_[j],
                                                           torch::TensorOptions()
                                                           .dtype(dtype<real_t>())));
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
          
          coeffs_[i] = torch::rand(size,
                                   torch::TensorOptions()
                                   .dtype(dtype<real_t>()));
        }
        break;
      }
        
      default:
        throw std::runtime_error("Unsupported BSplineInit option");
      }
    }
    
    // Returns a constant reference to the array of degrees
    static constexpr const std::array<short_t,parDim_>& degrees()
    {
      return degrees_;
    }

    // Returns a constant reference to the degree in the i-th dimension
    static constexpr const short_t& degree(short_t i)
    {
      assert(i>=0 && i<parDim_);
      return degrees_[i];
    }

    // Returns a constant reference to the array of knot vectors
    const std::array<torch::Tensor,parDim_>& knots() const
    {
      return knots_;
    }

    // Returns a constant reference to the knot vector in the i-th dimension
    const torch::Tensor& knots(short_t i) const
    {
      assert(i>=0 && i<parDim_);
      return knots_[i];
    }

    // Returns a non-constant reference to the array of knot vectors
    std::array<torch::Tensor,parDim_>& knots()
    {
      return knots_;
    }

    // Returns a non-constant reference to the knot vector in the i-th dimension
    torch::Tensor& knots(short_t i)
    {
      assert(i>=0 && i<parDim_);
      return knots_[i];
    }

    // Returns a constant reference to the array of knot vector dimensions
    const std::array<int64_t,parDim_>& nknots() const
    {
      return nknots_;
    }

    // Returns the dimension of the knot vector in the i-th dimension
    int64_t nknots(short_t i) const
    {
      assert(i>=0 && i<parDim_);
      return nknots_[i];
    }
    
    // Returns a constant reference to the array of coefficients
    const std::array<torch::Tensor,geoDim_>& coeffs() const
    {
      return coeffs_;
    }

    // Returns a constant reference to the coefficients in the i-th
    // dimension. If flatten=false, this function returns a
    // torch::Tensor with coefficients in the i-th dimension reshaped
    // according to the dimensions of the knot vectors
    template<bool flatten=true>
    auto coeffs(short_t i) const
      -> typename std::conditional<flatten, const torch::Tensor&, torch::Tensor>::type
    {
      assert(i>=0 && i<geoDim_);
      if constexpr (flatten)
        return coeffs_[i];
      else
        return coeffs_[i].reshape(torch::IntArrayRef(ncoeffs_));
    }

    // Returns a non-constant reference to the array of coefficients
    std::array<torch::Tensor,geoDim_>& coeffs()
    {
      return coeffs_;
    }

    // Returns a non-constant reference to the coefficients in the i-th dimension
    torch::Tensor& coeffs(short_t i)
    {
      assert(i>=0 && i<geoDim_);
      return coeffs_[i];
    }

    // Returns the total number of coefficients
    int64_t ncoeffs() const
    {
      int64_t s=0;
      for (short_t i=0; i<parDim_; ++i)
        s += ncoeffs(i);
      return s;
    }

    // Returns the total number of coefficients in the i-th direction
    int64_t ncoeffs(short_t i) const
    {
      assert(i>=0 && i<parDim_);
      return ncoeffs_[i];
    }

    // Returns the parametric dimension
    short_t parDim() const
    {
      return parDim_;
    }

    // Returns the geometric dimension
    short_t geoDim() const
    {
      return geoDim_;
    }

    // Returns all B-spline basis functions that are not zero in xi
    torch::Tensor eval(const torch::Tensor& xi) const
    {
      if constexpr (parDim_==1) {
        return
          eval_<degrees_[0],0>(int64_t(xi[0].item<real_t>()*(ncoeffs_[0]-2*degrees_[0]-1)+degrees_[0]), xi[0]);
      } else if constexpr (parDim_==2) {
        return
          torch::kron(
                      eval_<degrees_[0],0>(int64_t(xi[0].item<real_t>()*(ncoeffs_[0]-2*degrees_[0]-1)+degrees_[0]), xi[0]),
                      eval_<degrees_[1],1>(int64_t(xi[1].item<real_t>()*(ncoeffs_[1]-2*degrees_[1]-1)+degrees_[1]), xi[1])
                      );
      } else if constexpr (parDim_==3) {
        return
          torch::kron(
                      torch::kron(
                                  eval_<degrees_[0],0>(int64_t(xi[0].item<real_t>()*(ncoeffs_[0]-2*degrees_[0]-1)+degrees_[0]), xi[0]),
                                  eval_<degrees_[1],1>(int64_t(xi[1].item<real_t>()*(ncoeffs_[1]-2*degrees_[1]-1)+degrees_[1]), xi[1])
                                  ),
                      eval_<degrees_[2],2>(int64_t(xi[2].item<real_t>()*(ncoeffs_[2]-2*degrees_[2]-2)+degrees_[2]), xi[2])
                      );
      } else if constexpr (parDim_==3) {
        return
          torch::kron(
                      torch::kron(
                                  eval_<degrees_[0],0>(int64_t(xi[0].item<real_t>()*(ncoeffs_[0]-2*degrees_[0]-1)+degrees_[0]), xi[0]),
                                  eval_<degrees_[1],1>(int64_t(xi[1].item<real_t>()*(ncoeffs_[1]-2*degrees_[1]-1)+degrees_[1]), xi[1])
                                  ),
                      torch::kron(
                                  eval_<degrees_[2],2>(int64_t(xi[2].item<real_t>()*(ncoeffs_[2]-2*degrees_[2]-2)+degrees_[2]), xi[2]),
                                  eval_<degrees_[3],3>(int64_t(xi[3].item<real_t>()*(ncoeffs_[3]-3*degrees_[3]-3)+degrees_[3]), xi[3])
                                  )
                      );
      } else {
        throw "Invalid parametric dimension";
      }
    }
    
    // Returns a string representation of the BSpline object
    void pretty_print(std::ostream& os = std::cout) const
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

    const std::string& name() const noexcept {
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
    torch::Tensor eval_(int64_t i, const torch::Tensor& xi) const
    {
      if constexpr (degree==1) {
        return
          torch::reshape(torch::stack(
                                      {
                                        (knots_[dim][i+1]-xi) / (knots_[dim][i+1]-knots_[dim][i]), ( xi-knots_[dim][i] ) / (knots_[dim][i+1]-knots_[dim][i])
                                      }
                                      ), {1,2});
      } else if constexpr (degree==2) {
          return
            torch::matmul(eval_<1,dim>(i,xi),
            torch::reshape(torch::stack(
                                        {
                                          (knots_[dim][i+1]-xi) / (knots_[dim][i+1]-knots_[dim][i-1]), ( xi-knots_[dim][i-1] ) / (knots_[dim][i+1]-knots_[dim][i-1]), zero_[0],
                                          zero_[0], (knots_[dim][i+2]-xi) / (knots_[dim][i+2]-knots_[dim][i]), ( xi-knots_[dim][i] ) / (knots_[dim][i+2]-knots_[dim][i])
                                        }
                                        ), {2,3}));                          
      } else if constexpr (degree==3) {
          return
            torch::matmul(eval_<2,dim>(i,xi),
            torch::reshape(torch::stack(
                                        {
                                          (knots_[dim][i+1]-xi) / (knots_[dim][i+1]-knots_[dim][i-2]), ( xi-knots_[dim][i-2] ) / (knots_[dim][i+1]-knots_[dim][i-2]), zero_[0], zero_[0],
                                          zero_[0], (knots_[dim][i+2]-xi) / (knots_[dim][i+2]-knots_[dim][i-1]), ( xi-knots_[dim][i-1] ) / (knots_[dim][i+2]-knots_[dim][i-1]), zero_[0],
                                          zero_[0], zero_[0], (knots_[dim][i+3]-xi) / (knots_[dim][i+3]-knots_[dim][i]), ( xi-knots_[dim][i] ) / (knots_[dim][i+3]-knots_[dim][i])
                                        }
                                        ), {3,4}));                          
      } else {
        throw "Degrees higher than 3 are not implemented.";
        return torch::eye(1);
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
