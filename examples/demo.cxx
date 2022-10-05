#include <iganet.hpp>
#include <iostream>

int main()
{
  auto a = torch::linspace(0,19,20).reshape({4,5});

  std::cout
    << "a=\n"
    << a << std::endl;
  
  std::cout
    << "a.index=\n"
    << a.index({
        torch::indexing::Slice(1,3,1),
        torch::indexing::Slice(1,4,1)
      }) << std::endl;

  torch::Tensor idx0 = torch::linspace(1,2,2).to(torch::kInt64);
  torch::Tensor idx1 = torch::linspace(1,3,3).to(torch::kInt64);

  std::cout
    << "idx0=\n"
    << idx0 << std::endl;

  std::cout
    << "idx1=\n"
    << idx1 << std::endl;
  
  std::cout
    << "a.index_select=\n"
    << a.index_select(0, idx0).index_select(1, idx1)
    << std::endl;

  torch::Tensor idx = torch::full({4}, 2).to(torch::kInt64);
  std::cout
    << "VSlice=\n"
    << iganet::VSlice(idx, -1, 1) << std::endl;
  
  return 0;
}
