.. _LibKet Installation Guide:

Installation Guide
==================

LibKet is designed as header-only C++ library with minimal external
dependencies. All you need to get started is a C++14 (or better)
compiler and, optionally, Python 3.x to execute quantum algorithms
directly from within LibKet. Instructions on how to :ref:`install
prerequisites<LibKet Installing prerequisites>`, :ref:`download LibKet<LibKet
Downloading LibKet>`, and :ref:`configure and
build<LibKet Configuring and building LibKet>` it can be found
below. 

.. _LibKet Installing prerequisites:

Installing prerequisites
------------------------

LibKet uses standard C++14 code and has minimal requirements, which are as follows:

- C/C++compiler that supports the C++14 standard (or better);
- `CMake <http://www.cmake.org>`_ configuration tools version 3.x (or better);
- `Python <https://www.python.org>`_ version 3.x (or better) header and library files (optional);
- `Doxygen <http://www.doxygen.org>`_ documentation tool (optional);
- `Sphinx <https://www.sphinx-doc.org>`_ documentation tool (optional)

These prerequisites can be installed as follows:

Linux RedHat/CentOS 7.x (or better)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, install the basic development tools

.. code-block:: bash

    sudo yum update
    sudo yum group install "Development Tools"

Next, install CMake3 and Doxygen (optional) and Python 3.x
(optional). If you have RedHat/CentOS 7.7 or better, you can simply
run

.. code-block:: bash
                
    sudo yum install cmake3 doxygen git python3 python3-devel python3-libs

Releases prior to 7.7 do not provide Python 3.x and require to install
it from a third-party repository such as [IUS](https://ius.io/setup)
or [EPEL](https://fedoraproject.org/wiki/EPEL).

If you are running **RedHat/CentOS 8.x** or better you are done
here. The GCC version that is shipped with **RedHat/CentOS 7.x** is
too old to compile **LibKet** and needs to be updated from the
`Software Collections <https://www.softwarecollections.org/en/>`_

.. code-block:: bash
                
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7

From now on, GCC v7.x can be used by running

.. code-block:: bash
                
    scl enable devtoolset-7 bash

Linux Ubuntu 18.x (or better)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All prerequisites can be installed by running the following oneliner

.. code-block:: bash
                
    sudo apt-get update
    sudo apt install build-essential cmake doxygen git python3-dev python3-pip --fix-missing

macOS
~~~~~

The easiest way to get started under macOS is to install the XCode
Command Line Tools by opening a Terminal in /Applications/Utilities/
and running the one-liner

.. code-block:: bash

   xcode-select --install

Afterwards, install the package manager `homebrew <https://brew.sh>`_
by running

.. code-block:: bash

   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Once that is done you can install the prerequisites as follows

.. code-block:: bash
                
    brew update
    brew install cmake doxygen gcc python 

Note that the `QX` simulator does not run under Apple Silicon
(M1). Moreover, some Python packages might not install correctly. We
are working on a one-liner installation procedure.

Windows 10
~~~~~~~~~~

The easiest way to get started under Windows 10 is to install the
`Windows 10 Linux subsystem
<https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_ and a
Linux distribution of your choice and compile and run LibKet
inside the Linux subsystem. In our experience, `Ubuntu Linux
<https://tutorials.ubuntu.com/tutorial/tutorial-ubuntu-on-windows>`_
works fine.

Note that LibKet does not compile under Microsoft Visual Studio 2017
or 2019. We are working on a native port.

.. _LibKet Downloading LibKet:

Downloading LibKet
------------------

.. _LibKet Stable version:

Stable version
~~~~~~~~~~~~~~

The latest **stable version** of the LibKet code can be obtained
from `GitLab <https://gitlab.com/libket/LibKet>`_ as:

-  https://gitlab.com/libket/LibKet/-/archive/master/LibKet-master.zip
-  https://gitlab.com/libket/LibKet/-/archive/master/LibKet-master.tar.gz
-  https://gitlab.com/libket/LibKet/-/archive/master/LibKet-master.tar.bz2
-  https://gitlab.com/libket/LibKet/-/archive/master/LibKet-master.tar

If you are interested in a specific version you can download zip files
of `specific releases <https://gitlab.com/libket/LibKet/releases>`_.

A better way to obtain the latest revision from `GitLab
<https://gitlab.com/libket/LibKet>`_ and additionally have the
convenience to receive updates of the code is to use Git.

On **Linux**/**macOS**, you may checkout the latest revision using
        
.. code-block:: bash
                
    git clone https://gitlab.com/libket/LibKet.git

or

.. code-block:: bash
                
    git clone git@gitlab.com:libket/LibKet.git


On **Windows**, you can use `GitHub Windows client
<https://windows.github.com>`_ or any other Git client.


.. _LibKet Developer version:

Developer version
~~~~~~~~~~~~~~~~~

If you are interested in trying out the **development version** of the
LibKet code switch to the `develop` branch once you the initial
cloning of the Git repository succeeded

.. code-block:: bash
                
    cd LibKet
    git checkout --track origin/develop

    
.. _LibKet Configuring and building LibKet:

Configuring and building LibKet
-------------------------------

Assuming that LibKet has been downloaded to the source folder 
``LibKet`` the following sequence of commands will compile all
examples with the `common Quantum Assembly Language (CQASM) v1.0
<https://arxiv.org/abs/1805.09607>`_ (cQASMv1) backend enabled and
execute the program ``tutorial01_simple``.

.. code-block:: bash
                
   cd LibKet
   mkdir build
   cd build
   cmake .. -DLIBKET_WITH_EXAMPLES=ON -DLIBKET_WITH_CQASM=ON
   make
   ...
   [100%] Built
   ./examples/tutorial01_simple

Please note that the so-compiled tutorials try to establish a connection with the 
remote QI-Simulator. Details on how to configure LibKet for the QI-backend can be 
found in :ref:`LibKet Activating additional quantum backends<LibKet Activating additional quantum backends>`

The following configuration options can be used with the :code:`cmake` :code:`-D` flag:

+--------------------------+---------------------------------------------------------------------------------+
| Configuration            | Description                                                                     |
| Command                  |                                                                                 |
+==========================+=================================================================================+
| LIBKET_BUILD_COVERAGE    | Build LibKet with code coverage                                                 |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILD_DOCS        | Enable generation of Doxyen and Sphinx Docs                                     |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILD_C_API       | Build the C API library                                                         |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILD_PYTHON_API  | Build the Python API library                                                    |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILD_EXAMPLES    | Build the example and tutorial programs                                         |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILD_UNITTESTS   | Build unit tests                                                                |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILTIN_OPENQL    | Use built-in OpenQL Simulator                                                   |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILTIN_QUEST     | Use built-in QuEST Simulator                                                    |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILTIN_QX        | Use built-in QX Simulator                                                       |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_BUILTIN_UNITTESTS | Use built-in UnitTests++                                                        |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_L2R_EVALUATION    | Enable left-to-right evaluation                                                 |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_GEN_PROFILING     | Enable generation of profiling data                                             |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_OPTIMIZE_GATES    | Enable optimization of gates, e.g. H(H(q0)) = I(q0)                             |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_PROF_COMPILE      | Enable profiling of compilation                                                 |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_USE_PCH           | Enable use of precompiled headers                                               |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_AQASM        | Enable support for Atos QASM                                                    |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_CIRQ         | Enable support for Cirq used by Google                                          |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_CQASM        | Enable support for Common QASM used by QuTech's QX simulator                    |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_MPI          | Enable support for MPI                                                          |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_OPENMP       | Enable support for OpenMP                                                       |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_OPENQASM     | Enable support for OpenQASM used by Qiskit as well as IBMQ and IonQ devices     |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_OPENQL       | Enable support for OpenQL used by QuTech's OpenQL simulator                     |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_QASM         | Enable support for Qasm2circ LaTeX export                                       |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_QUEST        | Enable support for Quantum exact simulation toolkit by University of Oxford, UK |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_QUIL         | Enable support for Quantum instruction set architecture used by Rigetti         |
+--------------------------+---------------------------------------------------------------------------------+
| LIBKET_WITH_QX           | Enable support for QuTech's QX simulator                                        |
+--------------------------+---------------------------------------------------------------------------------+

.. _LibKet Activating additional quantum backends:

Activating additional quantum backends
--------------------------------------

LibKet supports the following quantum computing backends

+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| backend name                     | description                                                                                        | note |
+==================================+====================================================================================================+======+
| ``LibKet::QBackend::AQASM``      | `Atos Quantum Assembly Language (AQASM) <https://atos.net/en/solutions/quantum-learning-machine>`_ | 1    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| ``LibKet::QBackend::Cirq``       | `Cirq <https://github.com/quantumlib/Cirq>`_                                                       | 2    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| ``LibKet::QBackend::cQASMv1``    | `Common Quantum Assembly Language (cQASM) v1.0 <https://arxiv.org/abs/1805.09607>`_                | 3    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| ``LibKet::QBackend::openQASMv2`` | `Open Quantum Assembly Language (openQASM) v2.0 <https://arxiv.org/abs/1707.03429>`_               | 4    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| ``LibKet::QBackend::OpenQL``     | `QuTech's OpenQL framework <https://github.com/QE-Lab/OpenQL>`_                                    | 6    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| ``LibKet::QBackend::QASM``       | `QASM for the quantum circuit viewer qasm2circ <https://www.media.mit.edu/quanta/qasm2circ>`_      | 6    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| ``LibKet::QBackend::Quil``       | `Rigetti's Quantum Instruction Language <https://arxiv.org/abs/1608.03355>`_                       | 5    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| ``LibKet::QBackend::QuEST``      | `Quantum Exact Simulation Toolkit (QuEST) <https://quest.qtechtheory.org>`_                        | 6    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+
| ``LibKet::QBackend::QX``         | `QuTech's QX simulator <https://github.com/QE-Lab/qx-simulator>`_                                  | 6    |
+----------------------------------+----------------------------------------------------------------------------------------------------+------+

1. For using the full functionality of the ``AQASM`` backend you need
   to have access to a Quantum Learning Maching (QLM). This is
   proprietary software. Further information will come soon.

2. For using the full functionality of the ``Cirq`` backend you need
   to have the ``cirq`` Python package installed. This can be done by
   running either of the following commands:

   .. code-block:: bash

      pip3 install cirq      # installs Cirq in the global environment 
      make install-cirq-venv # installs Cirq in a virtual environment
      make install-cirq      # installs Cirq in the global environment

   When using the CMake approach (``make``) the default location of
   the Python virtual environment is
   ``$PYTHON_VENV_DIR/venv/cirq-$CIRQ_VERSION``, where
   ``$PYTHON_VENV_DIR`` and ``CIRQ_VERSION`` are environment
   variables. If not given then the virtual environment is installed
   in the CMake project binary directory and/or without version
   number.

3. For using the full functionality of the ``cQASMv1`` backend you
   need to have the ``quantuminspire`` Python package installed (see above for an explanation of the following commands).

   .. code-block:: bash

      pip3 install quantuminspire
      make install-quantuminspire-venv
      make install-quantuminspire

   In order to execute the quantum kernels on QuTech's `Quantum Inspire (QI)
   <https://quantuminspire.com>`_ cloud platform, you are required to
   have a user account, which can be created free-of-charge `here
   <https://www.quantum-inspire.com/account/create>`_. Once you created a
   free user account it suffices to set the following environment
   variables:   
   
   **Bash**

   .. code-block:: bash

    export QI_USERNAME=<your username>
    export QI_PASSWORD=<your password>

   **Csh/Tcsh**

   .. code-block:: bash

    setenv QI_USERNAME <your username>
    setenv QI_PASSWORD <your password>  

   LibKet will use this information to establish the connection
   with the remote QI simulator.
 
4. For using the full functionality of the ``openQASMv2`` backend you need to have the ``qiskit`` Python package installed (see above for an explanation of the following commands).

   .. code-block:: bash

        pip3 install qiskit
        make install-qiskit-venv
        make install-qiskit

    

   For additional use of the IonQ simulator, an additional qiskit package needs to be installed

   .. code-block:: bash

    pip3 install qiskit-ionq
    make install-qiskit-ionq-venv
    make install-qiskit-ionq

   IBMQ and IonQ's remote devices can be accessed by creating an account for their services and obtaining an API tokens. Keys can be exported using terminal command:

   .. code-block:: bash 

    export IBMQ_API_TOKEN="<Your IBMQ token>"
    export IONQ_API_TOKEN="<Your IonQ token>"

5. For using the full functionality of the ``Quil`` backend you need
   to have the ``pyquil`` Python package installed (see above for an
   explanation of the following commands).

   .. code-block:: bash

      pip3 install pyquil
      make install-pyquil-venv
      make install-pyquil

   In addition, you need to have the `Forest SDK
   <https://qcs.rigetti.com/sdk-downloads>`_ installed which includes
   the `Rigetti quil compiler <https://github.com/rigetti/quilc>`_ and
   the `Rigetti quantum virtual machine
   <https://github.com/rigetti/qvm>`_. The CMake targets only point
   you to the website but do not install the Forest SDK.

6. Prerequisites for these backends are bundled with LibKet as Git
   submodules and do not have to be installed separately. It is,
   however, still possible to install them externally, e.g.,
   system-wide and request LibKet to use them by passing the following
   arguments to CMake, e.g.

   .. code-block:: bash

      cmake .. -DLIBKET_BUILTIN_OPENQL=OFF -DOPENQL_INCLUDE_PATH=<path to OpenQL include files>
      cmake .. -DLIBKET_BUILTIN_QUEST=OFF  -DQUEST_INCLUDE_PATH=<path to QuEST include files>
      cmake .. -DLIBKET_BUILTIN_QX=OFF     -DQX_INCLUDE_PATH=<path to QX include files>

   LibKet makes use of the `UnitTest++
   <https://unittest-cpp.github.io>`_ framework for unit testing. Like
   the above, it is bundled with LibKet as Git submodule but can be
   overwritten as follows

   .. code-block:: bash

      cmake .. -DLIBKET_BUILTIN_UNITTESTS=OFF -DUNITTESTPP_INCLUDE_PATH=<path to UnitTest++ include files>

.. _LibKet Docker images:

Docker images
-------------

The quickest way to explore LibKet without going through all
installation steps is by trying one of the pre-build `images
<https://hub.docker.com/repository/docker/mmoelle1/libket>`_ for
`Docker <https://www.docker.com/get-started>`_ or its daemonless
counterpart `Podman <https://podman.io>`_.

Once you have installed one of these tools, getting started with
LibKet is as easy as running the following one-liner in your terminal

.. code-block:: bash

   docker run --rm -ti mmoelle1/libket:qx

or

.. code-block:: bash

   podman run --rm -ti mmoelle1/libket:qx

Please check the full online `documentation
<https://hub.docker.com/repository/docker/mmoelle1/libket>`_ for
additional configuration options.

.. _LibKet Generating the LibKet documentation:

Generating the LibKet documentation
-----------------------------------
Libket supports the generation of project documentations with `Doxygen <http://www.doxygen.org/>`_ 
and `Sphinx <https://www.sphinx-doc.org/>`_. Make sure to set the :code:`-DLIBKET_BUILD_DOCS=ON` flag 
when configuring :code:`cmake`.


If `Doxygen <http://www.doxygen.org/>`_ is available on your system,
you can generate and open the Doxygen HTML pages by executing

.. code-block:: bash
                
    cd build
    make Doxygen
    ...
    Built target Doxygen
    firefox doc/doxygen/html/index.html


If `Sphinx <https://www.sphinx-doc.org/>`_ is available on your
system, you can generate and open the Sphinx HTML pages by executing

.. code-block:: bash
                
    cd build
    make Sphinx
    ...
    Built target Sphinx
    firefox doc/sphinx/index.html

If you want to generate both documentations simply type

.. code-block:: bash

    cd build
    make docs

Next read :ref:`Components<LibKet Basics>`.
