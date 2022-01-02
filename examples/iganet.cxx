#include <iganet.hpp>
#include <iostream>

int main()
{
  torch::manual_seed(1);
  iganet::core<double> core;
  
  iganet::IgANet<double,5,5,5,5> net( {50,30,70}, // Number of neurons per layers
                                      {6,6,6,6}   // Number of B-spline coefficients
                                  );

  std::cout << net << std::endl;
  
  //net.sol().transform( [](const std::array<double,1> X){ return std::array<double,1>{ X[0] /*sin(M_PI*X[0])*/ }; } );
  //net.sol().transform( [](const std::array<double,2> X){ return std::array<double,1>{ sin(M_PI*X[0])*sin(M_PI*X[1]) }; } );
  //net.sol().transform( [](const std::array<double,3> X){ return std::array<double,1>{ sin(M_PI*X[0])*sin(M_PI*X[1])*sin(M_PI*X[2]) }; } );
  net.sol().transform( [](const std::array<double,4> X){ return std::array<double,1>{ sin(M_PI*X[0])*sin(M_PI*X[1])*sin(M_PI*X[2])*sin(M_PI*X[3]) }; } );

  std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5), torch::full({1}, 1.0), torch::full({1}, 1.0), torch::full({1}, 1.0)}).flatten() ) << std::endl;
  
  //net.plot();
}
