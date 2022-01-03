#include <iganet.hpp>
#include <iostream>

int main()
{
  using real_t = double;
  torch::manual_seed(1);

  {
    iganet::IgANet<real_t,5> net( {50,30,70}, // Number of neurons per layers
                                  {6}         // Number of B-spline coefficients
                                  );
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,1> X){ return std::array<real_t,1>{ X[0] /*sin(M_PI*X[0])*/ }; } );
    std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5)}).flatten() ) << std::endl;
  }

  {
    iganet::IgANet<real_t,5,5> net( {50,30,70}, // Number of neurons per layers
                                    {6,6}       // Number of B-spline coefficients
                                    );
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,2> X){ return std::array<real_t,1>{ sin(M_PI*X[0])*sin(M_PI*X[1]) }; } );
    std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5), torch::full({1}, 0.5)}).flatten() ) << std::endl;
  }

  {
    iganet::IgANet<real_t,5,5,5> net( {50,30,70}, // Number of neurons per layers
                                      {6,6,6}     // Number of B-spline coefficients
                                      );
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,3> X){ return std::array<real_t,1>{ sin(M_PI*X[0])*sin(M_PI*X[1])*sin(M_PI*X[2]) }; } );
    std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5), torch::full({1}, 0.5), torch::full({1}, 0.5)}).flatten() ) << std::endl;
  }

  {
    iganet::IgANet<real_t,5,5,5,5> net( {50,30,70}, // Number of neurons per layers
                                        {6,6,6,6}   // Number of B-spline coefficients
                                        );
    std::cout << net << std::endl;
    net.sol().transform( [](const std::array<real_t,4> X){ return std::array<real_t,1>{ sin(M_PI*X[0])*sin(M_PI*X[1])*sin(M_PI*X[2])*sin(M_PI*X[3]) }; } );
    std::cout << net.sol().eval( torch::stack({torch::full({1}, 0.5), torch::full({1}, 0.5), torch::full({1}, 0.5), torch::full({1}, 0.5)}).flatten() ) << std::endl;
  }
  
  //net.plot();
}
