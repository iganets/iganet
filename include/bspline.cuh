/**
   @file include/bspline.cuh

   @brief CUDA kernels for multivariate B-splines

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#ifdef __CUDACC__
namespace iganet {
  namespace cuda {
    
    /**
       @brief Compute Greville abscissae
    */
    template<typename real_t>
    __global__ void greville_cuda_kernel(torch::PackedTensorAccessor64<real_t, 1>       greville,  
                                         const torch::PackedTensorAccessor64<real_t, 1> knots,
                                         int64_t ncoeffs, short_t degree)
    {
      for (int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
           k < ncoeffs;
           k += blockDim.x * gridDim.x) {
        for (short_t l = 1; l <= degree; ++l)
          greville[k] += knots[k + l];
        greville[k] /= real_t(degree);
      }      
    }
    
  } // namespace cuda
} // namespace iganet
#endif
