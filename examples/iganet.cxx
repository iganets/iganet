#include <iganet.hpp>
#include <iostream>

int main()
{
  torch::manual_seed(1);
  
  iganet::IgANet<double, 2, 2> net( {50,30,70}, // Number of neurons per layers
                                    {5,7}       // B-spline knots
                                    );
  net.plot();
}
