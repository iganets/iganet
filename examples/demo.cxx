#include <iganet.hpp>
#include <iostream>

template<typename real_t,
         typename optimizer_t,
         short_t GeoDim, short_t PdeDim,
         template<typename, short_t, short_t...> class bspline_t,
         short_t... Degrees>
class poisson : public iganet::IgANet<real_t, optimizer_t, GeoDim, PdeDim, bspline_t, Degrees...>
{
public:
  using iganet::IgANet<real_t, optimizer_t, GeoDim, PdeDim, bspline_t, Degrees...>::IgANet;

  virtual iganet::IgaNetDataStatus get_epoch(int64_t epoch) const override
  {
    std::cout << "Epoch " << std::to_string(epoch) << ": ";
    return iganet::IgaNetDataStatus(0);
  }
};

int main()
{
  using real_t      = float;
  using optimizer_t = torch::optim::Adam;

  torch::autograd::AnomalyMode::set_enabled(true);
  
  iganet::init();
  iganet::verbose(std::cout);
  
  poisson<real_t, optimizer_t, 1, 1, iganet::UniformBSpline,
          2> net({100,100}, // Number of neurons per layers
                 {
                   {iganet::activation::relu},
                   {iganet::activation::relu},
                   {iganet::activation::none}
                 },         // Activation functions
                 {5});    // Number of B-spline coefficients
  
  std::cout << net.get_samples() << std::endl;
  exit(0);
  
  // Set rhs to x
  //  net.rhs().transform( [](const std::array<real_t,1> X){ return std::array<real_t,1>{ static_cast<real_t>( X[0] ) }; } );
  
  // Set left boundary value to 0
  //  net.bdr().coeffs()[0].accessor<real_t,1>()[0] = 0;
  //  net.bdr().coeffs()[1].accessor<real_t,1>()[0] = 1;
  
  net.options().max_epoch(1000);
  net.options().min_loss(1e-8);
  
  net.train();
  
  // iganet::UniformBSpline<real_t,2,1,1> u({6,5});
  
  // iganet::RectangleCreator<real_t> creator(-1.5, -1.2, 0.8, 2.5,
  //                                         -0.1,  0.1, 0.9, 1.1);

  // std::cout << creator << std::endl;

  // for (int i=0; i<20; ++i)
  //   std::cout << creator.next(u) << std::endl;
  
  return 0;

}
