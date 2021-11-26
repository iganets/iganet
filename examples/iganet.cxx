#include <iganet.hpp>
#include <iostream>

int main()
{
  torch::manual_seed(1);
  
  iganet::IgANet<double, 2,2> net( {50,30,70}, // Number of neurons per layers
                                   {6,6}       // B-spline knots
                                   );
  
  //net.sol().transform( [](const std::array<double,1> X){ return std::array<double,1>{ sin(3.141*X[0]) }; } );
  net.sol().transform( [](const std::array<double,2> X){ return std::array<double,1>{ sin(3.141*X[0])*sin(3.141*X[1]) }; } );
  //net.sol().transform( [](const std::array<double,3> X){ return std::array<double,1>{ sin(3.141*X[0])*sin(3.141*X[1])*sin(3.141*X[2]) }; } );
  //net.sol().transform( [](const std::array<double,4> X){ return std::array<double,1>{ sin(3.141*X[0])*sin(3.141*X[1])*sin(3.141*X[2])*sin(3.141*X[3]) }; } );
  
  std::cout << net.sol().eval( torch::full({2}, 0.5) ) << std::endl;

  //net.plot();
}
