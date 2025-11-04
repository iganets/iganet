# IGAnets: Isogeometric analysis networks

[![GitlabSync](https://github.com/iganets/iganet/actions/workflows/gitlab-sync.yml/badge.svg)](https://github.com/iganets/iganet/actions/workflows/gitlab-sync.yml)
[![CI](https://github.com/iganets/iganet/actions/workflows/ci-push-pr.yml/badge.svg)](https://github.com/iganets/iganet/actions/workflows/ci-push-pr.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://iganets.github.io/iganet/)

[![GitHub Releases](https://img.shields.io/github/release/iganets/iganet.svg)](https://github.com/iganets/iganet/releases)
[![GitHub Downloads](https://img.shields.io/github/downloads/iganets/iganet/total)](https://github.com/iganets/iganet/releases)
[![GitHub Issues](https://img.shields.io/github/issues/iganets/iganet.svg)](https://github.com/iganets/iganet/issues)

IGAnets is a novel approach to combine the concept of deep operator learning with the mathematical framework of isogeometric analysis.

## Installation instructions

IGAnets require a C++20 compiler, CMake and LibTorch (the C++ API of PyTorch).

**Supported CMake flags**:

- `-DIGANET_BUILD_CPUONLY=ON` builds IGAnets in CPU mode even if CUDA, ROCm, etc. is found (default `OFF`).

- `-DIGANET_BUILD_DOCS=ON` builds the documentation (default `OFF`). To build the documentation you need [Doxygen](https://www.doxygen.nl) and [Sphinx](https://www.sphinx-doc.org/en/master/) installed on you system.

- `-DIGANET_BUILD_PCH=ON` builds IGAnets with precompiled headers (default `ON`).

- `-DIGANET_OPTIONAL="module1[branch];module2[branch];..."` builds optional modules (default `NONE`)

  Optional modules are downloaded into the directory `optional`. If the current IGAnets checkout is a git repository (i.e. if CMake finds the directory `.git`) optional modules are also checked out as git repositories. Otherwise, CMake downloads the ZIP archive of the optional module.

  The following optional modules are available:
  - [Examples](https://github.com/iganets/iganet-examples) `examples[main]`
  - [Unit tests](https://github.com/iganets/iganet-unittests) `unittests[main]`
  - [Performance tests](https://github.com/iganets/iganet-perftests) `perftests[main]`
  - [Python bindings](https://github.com/iganets/iganet-python) `python[main]`
  - [MATLAB bindings](https://github.com/iganets/iganet-matlab) `matlab[main]`

  If `[branch]` is not given then `[main]` is assumed by default. There might exist further optional modules that are not visible publicly.

- `-DIGANET_WITH_GISMO=ON` compiles IGAnets with support for the open-source Geometry plus Simulation Modules library [G+Smo](https://github.com/gismo/gismo) enabled (default `OFF`).

- `-DIGANET_WITH_MATPLOT=ON` compiles IGAnets with support for the open-source library [Matplot-cpp](https://github.com/lava/matplotlib-cpp) enabled (default `OFF`). _Note that this option can cause compilation errors with GCC._

- `-DIGANET_WITH_MPI=ON` compiles IGAnets with MPI support enabled (default `OFF`).

- `-DIGANET_WITH_OPENMP=ON` compiles IGAnets with OpenMP support enabled (default `ON`). _Note that this option can cause compilation errors with Clang._

### Linux

1.  Install prerequisites (CMake and LibTorch)

    #### Ubuntu
      ```shell
      apt-get install build-essential cmake unzip wget
      ```

    #### RedHat
      ```shell
      yum install make cmake gcc gcc-c++ unzip wget
      ```

    #### Install LibTorch

    Pre-compiled versions of LibTorch are available at [PyTorch.org](https://pytorch.org/get-started/locally/). Depending on your compiler toolchain you need to choose between the pre-cxx11 and the cxx11 ABI, i.e.

    ```shell
    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.7.1%2Bcpu.zip -O libtorch.zip
    unzip libtorch.zip -d $HOME/
    rm -f libtorch
    ```
    or
    ```shell
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip -O libtorch.zip
    unzip libtorch.zip -d $HOME/
    rm -f libtorch
    ```

    Note that there might be a newer LibTorch version available than indicated in the above code snippet.

2.  Configure
    ```shell
    cmake .. -DTorch_DIR=${HOME}/libtorch/share/cmake/Torch
    ```

3.  Compile
    ```shell
    make -j 8
    ```

    Depending on the number of cores of your CPU you may want to change 8 to a different number.

### macOS

1.  Install prerequisites (CMake and LibTorch)
    ```shell
    brew install cmake pytorch
    ```

    Note that since version 2.2.0, official builds of the LibTorch library for ARM64 and X86_64 can be downloaded from PyTorch.org:

    - https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-2.7.1.zip
    - https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip

    If you decide to use these version download and unzip them as shown for the Linux installation. It is, however, recommended to install LibTorch through `brew` as described above since this method is tested regularly by the IGAnets authors.

    Note that there might be a newer LibTorch version available than indicated in the above code snippet.

3.  Configure
    ```shell
    cmake .. -DTorch_DIR=/opt/homebrew/Cellar/pytorch/2.7.1/share/cmake/Torch
    ```

    Note that the specific version of PyTorch and/or protobuf might be different on your system.

4.  Compile
    ```shell
    make -j 8
    ```

    Depending on the number of cores of your CPU you may want to change 8 to a different number.

## Compilation with CUDA support (only Linux)

1.  Install the CUDA-enabled version of LibTorch

    - https://download.pytorch.org/libtorch/cu121/libtorch-shared-with-deps-2.7.1%2Bcu128.zip
    - https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu128.zip

    Note that the version must be compatible with the CUDA version installed on your system.

2. Configure and compile

   All further steps are the same as described above (Linux)

## Compilation with ROCm support (only Linux)

1.  Install the ROCm-enabled version of LibTorch

    - https://download.pytorch.org/libtorch/rocm6.3/libtorch-shared-with-deps-2.7.1%2Brocm6.3.zip
    - https://download.pytorch.org/libtorch/rocm6.3/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Brocm6.3.zip

    Note that the version must be compatible with the ROCm version installed on your system.

2.  Configure and compiled

    All further steps are the same as described above (Linux)

## Compilation with Intel GPU support (only Linux)

1. Install the Intel GPU drivers and PyTorch version as decribed
   [here](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html). If
   you do not own an Intel GPU, you can create a free account at the
   [Intel Tiber AI Cloud](https://console.cloud.intel.com), which
   provides a free access to Intel datacenter GPU for testing
   purposes.

2. Install the XPU-enabled version of PyTorch in a virtual python environment

   ```shell
   python3 -m venv $HOME/.venv/torch-xpu
   source $HOME/.venv/torch-xpu/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/xpu
   ```

3. Configure
    ```shell
    cmake .. -DTorch_DIR=$HOME/.venv/torch-xpu/lib/python3.11/site-packages/torch/share/cmake/Torch/
    ```

    Note that on the latest Intel Tiber AI Cloud installation, ZLib is
    not found by default. This can be corrected by calling CMake with
    the additional parameters
    ```shell
    -DZLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/libz.so.1 -DZLIB_INCLUDE_DIR=/usr/include
    ```

## Compilation with Intel Extensions for PyTorch support (only Linux)

1.  Install the Intel Extensions for PyTorch as described [here](https://github.com/intel/intel-extension-for-pytorch?tab=readme-ov-file).

2.  Add the CMake option `-DIPEX_DIR=<path/to/IPEX/installation>`
