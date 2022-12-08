#include <iganet.hpp>
#include <iostream>

int main()
{
  std::cout << iganet::verbose;
  using real_t = double;
  iganet::init();

  std::array<torch::Tensor, 1> coeffs = {5*torch::ones({6})};
  
  // Univariate uniform B-spline of degree 2 with 6 control points in R^1
  iganet::UniformBSpline<real_t,1,2> a({6}), b(std::move(a), std::array<torch::Tensor, 1>{5*torch::ones({6})} );

  // Print information
  std::cout << a << std::endl;
  std::cout << b << std::endl;
  
  // Map control points to phyiscal coordinates
  a.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,1>{ xi[0]*xi[0] }; } );
  b.transform( [](const std::array<real_t,1> xi){ return std::array<real_t,1>{ xi[0]*10 }; } );
  
  // Print information
  std::cout << a << std::endl;
  std::cout << b << std::endl;  
  
  return 0;
}
