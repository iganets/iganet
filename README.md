# IgANet: Physics-machine learning embedded into Isogeometric Analysis

## Compilation instructions

IgANet requires a C++17 compiler, CMake and LibTorch (the C++ API of PyTorch). If CUDA is installed on your system, IgANet will be compiled with CUDA support. Optionally, you can configure IgANet to compile the Python module pyiganet which requires a Python version to be installed.

### macOS

1. Install prerequisites (CMake and LibTorch)
   ```
   brew install cmake pytorch
   ```

2. Configure
    ```
    cmake .. -DTorch_DIR=/opt/homebrew/Cellar/pytorch/2.0.1/share/cmake/Torch -DCMAKE_PREFIX_PATH=/opt/homebrew/Cellar/protobuf@21/21.12/
    ```

    Note that the specific version of PyTorch and/or protobuf might be different on your system.

3. Compile
    ```
    make
    ```

4. _Optionally_: Compile pyiganet
   ```
   Torch_DIR=/opt/homebrew/Cellar/pytorch/2.0.1/share/cmake/Torch CMAKE_PREFIX_PATH=/opt/homebrew/Cellar/protobuf@21/21.12/ python setup.py develop python setup.py develop
   ```

   Again, the specific version of PyTorch and/or protobuf might be different on your system.
