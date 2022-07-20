.. _LibKet Library:

Library
=======

.. _LibKet Library Filters:

Filters
-------

.. doxygengroup:: filters

.. _LibKet Library Gates:

Gates
-----

Initialisation
""""""""""""""

.. doxygengroup:: init

.. doxygengroup:: prep_x

.. doxygengroup:: prep_y

.. doxygengroup:: prep_z

.. doxygengroup:: reset

Measurements
""""""""""""

.. doxygengroup:: measure

.. doxygengroup:: measure_x

.. doxygengroup:: measure_y

.. doxygengroup:: measure_z

Single Qubit Gates
""""""""""""""""""

.. doxygengroup:: pauli_x  

.. doxygengroup:: pauli_y  

.. doxygengroup:: pauli_z

.. doxygengroup:: identity

.. doxygengroup:: hadamard

.. doxygengroup:: s

.. doxygengroup:: sdag

.. doxygengroup:: rotate_x90

.. doxygengroup:: rotate_mx90

.. doxygengroup:: rotate_y90

.. doxygengroup:: rotate_my90

.. doxygengroup:: sqrtnot

.. doxygengroup:: t

.. doxygengroup:: tdag

.. doxygengroup:: barrier

.. doxygengroup:: u2

.. doxygengroup:: u2dag


Parameterized Single Qubit Gates
""""""""""""""""""""""""""""""""

.. doxygengroup:: rotate_x

.. doxygengroup:: rotate_xdag

.. doxygengroup:: rotate_y

.. doxygengroup:: rotate_ydag

.. doxygengroup:: rotate_z

.. doxygengroup:: rotate_zdag

.. doxygengroup:: phase

.. doxygengroup:: phasedag


Two Qubit Gates
"""""""""""""""

.. doxygengroup:: cnot

.. doxygengroup:: cy

.. doxygengroup:: cz

.. doxygengroup:: swap

.. doxygengroup:: sqrtswap

.. doxygengroup:: cu2

.. doxygengroup:: cu2dag


Parameterized Two Qubit Gates
"""""""""""""""""""""""""""""

.. doxygengroup:: cphase

.. doxygengroup:: cphasedag

.. doxygengroup:: cphasek

.. doxygengroup:: cphasekdag

.. doxygengroup:: crx

.. doxygengroup:: cry

.. doxygengroup:: crz

.. doxygengroup:: rxx

.. doxygengroup:: ryy

.. doxygengroup:: rzz


Three Qubit Gates
"""""""""""""""""

.. doxygengroup:: ccnot

.. doxygengroup:: qor


.. _LibKet Library Circuits:

Circuits
--------

Quantum Fourier Transform
"""""""""""""""""""""""""

The code below can be used to apply the Quantum Fourier Transform on qubits 0 to n. 

.. code-block:: cpp

  auto expr = qft(range<0,n>(init()));

This generates the following circuit for :math:`n = 5`:

.. tikz:: This image shows the circuit created with the above line of code

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \swap{4} & \qw      & \gate{H} & \gate{S} & \qw      & \gate{T} & \qw      & \gate{Z^{1/8}}& \qw      & \gate{Z^{1/16}}& \qw      & \qw \\
        \lstick{$q_1$} & \qw      & \swap{2} & \qw      & \ctrl{-1}& \gate{H} & \gate{S} & \qw      & \gate{T}      & \qw      & \gate{Z^{1/8}} & \qw      & \qw \\
        \lstick{$q_2$} & \qw      & \qw      & \qw      & \qw      & \qw      & \ctrl{-2}& \gate{H} & \gate{S}      & \qw      & \gate{T}       & \qw      & \qw \\
        \lstick{$q_3$} & \qw      & \targX{} & \qw      & \qw      & \qw      & \qw      & \qw      & \ctrl{-3}     & \gate{H} & \gate{S}       & \qw      & \qw \\
        \lstick{$q_4$} & \targX{} & \qw      & \qw      & \qw      & \qw      & \qw      & \qw      & \qw           & \qw      & \ctrl{-4}      & \gate{H} & \qw     
    \end{quantikz}};
    :libs: quantikz  

Inverse QFT is called using function :code:`qftdag()`.

.. _LibKet Circuits AllSwap:

AllSwap
"""""""

The LibKet AllSwap circuit swaps all qubits in a given selection. The LibKet AllSwap circuit can be applied to the first n qubits of your register as follows: 

.. code-block:: cpp

  auto expr = allswap(range<0,n>(init()));


This creates the following circuit for n = 5:

.. tikz:: This image shows the circuit created with the above line of code

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \swap{4} & \qw      & \qw \\
        \lstick{$q_1$} & \qw      & \swap{2} & \qw \\
        \lstick{$q_2$} & \qw      & \qw      & \qw \\
        \lstick{$q_3$} & \qw      & \targX{} & \qw \\
        \lstick{$q_4$} & \targX{} & \qw      & \qw     
    \end{quantikz}};
    :libs: quantikz 

Arbitrary Control
"""""""""""""""""

The Arbitrary control circuit allows controlled unitary qubit gates (e.g. cx, cy, cz, cphase, etc) to be controlled by multiple qubits. For every :math:`N` control qubits, :math:`N-1` ancilla qubits are needed. The following code snippet constructs a cnot gate controlled by qubits 0 to 3.

.. code-block:: cpp

      auto expr = arb_ctrl<>(cx(),              //Control gate
                             sel<0,1,2,3>(),    //Control qubits
                             sel<7>(),          //Target qubits
                             sel<4,5,6>(init()) //Ancilla qubits
                            );

This generates the following circuit:

.. tikz:: Arbitrary Control circuit for cnot gate

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \ctrl{1} & \qw      & \qw      & \qw      & \qw      & \qw      & \ctrl{1} & \qw \\
        \lstick{$q_1$} & \ctrl{3} & \qw      & \qw      & \qw      & \qw      & \qw      & \ctrl{3} & \qw \\
        \lstick{$q_2$} & \qw      & \ctrl{2} & \qw      & \qw      & \qw      & \ctrl{2} & \qw      & \qw \\
        \lstick{$q_3$} & \qw      & \qw      & \ctrl{2} & \qw      & \ctrl{2} & \qw      & \qw      & \qw \\
        \lstick{$q_4$} & \targ{}  & \ctrl{1} & \qw      & \qw      & \qw      & \ctrl{1} & \targ{}  & \qw \\ 
        \lstick{$q_5$} & \qw      & \targ{}  & \ctrl{1} & \qw      & \ctrl{1} & \targ{}  & \qw      & \qw \\
        \lstick{$q_6$} & \qw      & \qw      & \targ{}  & \ctrl{1} & \targ{}  & \qw      & \qw      & \qw \\
        \lstick{$q_7$} & \qw      & \qw      & \qw      & \targ{}  & \qw      & \qw      & \qw      & \qw    
    \end{quantikz}};
    :libs: quantikz 

Quantum Phase Estimation
""""""""""""""""""""""""

Oracle
""""""

.. _LibKet Library Devices:

Devices
-------

Atos QLM
""""""""

   This class executes quantum circuits on the Atos Quantum Learning
   Machine (QLM) simulator. It adopts Atos' AQASM quantum assembly
   language: `Atos Website <https://atos.net/en/solutions/quantum-learning-machine>`_

Available QDevices in LibKet:

.. code-block:: cpp

  atos_qlm_feynman_simulator  /**< Atos QLM Feynman integral path simulator          */
  atos_qlm_linalg_simulator   /**< Atos QLM Linear algebra-based  simulator          */
  atos_qlm_stabs_simulator    /**< Atos QLM Stabilizer-based simulator               */
  atos_qlm_mps_simulator      /**< Atos QLM Matrix product state-based simulator     */

Cirq
""""
   This class executes quantum circuits locally on the Cirq simulator, a Python software library for writing, manipulating, and optimizing quantum circuits, and then running them on quantum computers and quantum simulators. It adopts the Cirq quantum assembly language. Cirq provides useful features such as dealing with today’s noisy intermediate-scale quantum computers: `Cirq Website <https://quantumai.google/cirq>`_

Available QDevices in LibKet:

.. code-block:: cpp

  cirq_simulator             /**< Cirq simulator                                    */
  cirq_simulator_simulator   /**< Cirq simulator (name demangling)                  */
  cirq_bristlecone_simulator /**< Cirq Bristlecone simulator                        */
  cirq_foxtail_simulator     /**< Cirq Foxtail simulator                            */
  cirq_sycamore_simulator    /**< Cirq Sycamore simulator                           */
  cirq_sycamore23_simulator  /**< Cirq Sycamore23 simulator                         */

Qiskit
""""""

Qiskit is another python basesd open-source SDK for working with quantum computers and simulators at the level of pulses, circuits and application modules: `Qiskit Website <https://qiskit.org/>`_

Available QDevices in LibKet:

.. code-block:: cpp

  qiskit_almaden_simulator      /**< Qiskit  20-qubit local simulator                  */
  qiskit_armonk_simulator       /**< Qiskit   1-qubit local simulator                  */
  qiskit_athens_simulator       /**< Qiskit   5-qubit local simulator                  */
  qiskit_belem_simulator        /**< Qiskit   5-qubit local simulator                  */
  qiskit_boeblingen_simulator   /**< Qiskit  20-qubit local simulator                  */
  qiskit_bogota_simulator       /**< Qiskit   5-qubit local simulator                  */
  qiskit_brooklyn_simulator     /**< Qiskit  65-qubit local simulator                  */
  qiskit_burlington_simulator   /**< Qiskit   5-qubit local simulator                  */
  qiskit_cairo_simulator        /**< Qiskit  27-qubit local simulator                  */
  qiskit_cambridge_simulator    /**< Qiskit  28-qubit local simulator                  */
  qiskit_casablanca_simulator   /**< Qiskit   7-qubit local simulator                  */
  qiskit_dublin_simulator       /**< Qiskit  27-qubit local simulator                  */
  qiskit_essex_simulator        /**< Qiskit   5-qubit local simulator                  */
  qiskit_guadalupe_simulator    /**< Qiskit  16-qubit local simulator                  */
  qiskit_hanoi_simulator        /**< Qiskit  27-qubit local simulator                  */
  qiskit_jakarta_simulator      /**< Qiskit   7-qubit local simulator                  */
  qiskit_johannesburg_simulator /**< Qiskit  20-qubit local simulator                  */
  qiskit_kolkata_simulator      /**< Qiskit  27-qubit local simulator                  */
  qiskit_lagos_simulator        /**< Qiskit   7-qubit local simulator                  */
  qiskit_lima_simulator         /**< Qiskit   5-qubit local simulator                  */
  qiskit_london_simulator       /**< Qiskit   5-qubit local simulator                  */
  qiskit_manhattan_simulator    /**< Qiskit  65-qubit local simulator                  */
  qiskit_manila_simulator       /**< Qiskit   5-qubit local simulator                  */
  qiskit_melbourne_simulator    /**< Qiskit  15-qubit local simulator                  */    
  qiskit_montreal_simulator     /**< Qiskit  27-qubit local simulator                  */
  qiskit_mumbai_simulator       /**< Qiskit  27-qubit local simulator                  */
  qiskit_nairobi_simulator      /**< Qiskit   7-qubit local simulator                  */
  qiskit_ourense_simulator      /**< Qiskit   5-qubit local simulator                  */
  qiskit_paris_simulator        /**< Qiskit  27-qubit local simulator                  */
  qiskit_peekskill_simulator    /**< Qiskit  27-qubit local simulator                  */
  qiskit_poughkeepsie_simulator /**< Qiskit  20-qubit local simulator                  */
  qiskit_quito_simulator        /**< Qiskit   5-qubit local simulator                  */
  qiskit_rochester_simulator    /**< Qiskit  53-qubit local simulator                  */
  qiskit_rome_simulator         /**< Qiskit   5-qubit local simulator                  */
  qiskit_rueschlikon_simulator  /**< Qiskit  16-qubit local simulator                  */
  qiskit_santiago_simulator     /**< Qiskit   5-qubit local simulator                  */
  qiskit_singapore_simulator    /**< Qiskit  20-qubit local simulator                  */
  qiskit_sydney_simulator       /**< Qiskit  27-qubit local simulator                  */
  qiskit_tenerife_simulator     /**< Qiskit   5-qubit local simulator                  */
  qiskit_tokyo_simulator        /**< Qiskit  20-qubit local simulator                  */
  qiskit_toronto_simulator      /**< Qiskit  27-qubit local simulator                  */
  qiskit_valencia_simulator     /**< Qiskit   5-qubit local simulator                  */
  qiskit_vigo_simulator         /**< Qiskit   5-qubit local simulator                  */
  qiskit_yorktown_simulator     /**< Qiskit   5-qubit local simulator                  */
  qiskit_washington_simulator   /**< Qiskit 127-qubit local simulator                  */
  qiskit_perth_simulator        /**< Qiskit   7-qubit local simulator                  */

  qiskit_pulse_simulator       /**< Qiskit pulse local simulator                      */
  qiskit_qasm_simulator        /**< Qiskit universal local simulator                  */
  qiskit_statevector_simulator /**< Qiskit statevector local simulator                */
  qiskit_unitary_simulator     /**< Qiskit density matrix local simulator             */

  qiskit_aer_density_matrix_simulator       /**< Qiskit Aer density matrix local simulator         */
  qiskit_aer_extended_stabilizer_simulator  /**< Qiskit Aer extended stabilizer local simulator    */
  qiskit_aer_matrix_product_state_simulator /**< Qiskit Aer matrix product state local simulator   */
  qiskit_aer_simulator                      /**< Qiskit Aer local simulator                        */
  qiskit_aer_stabilizer_simulator           /**< Qiskit Aer stabilizer local simulator             */
  qiskit_aer_statevector_simulator          /**< Qiskit Aer statevector local simulator            */
  qiskit_aer_superop_simulator              /**< Qiskit Aer superop local simulator                */
  qiskit_aer_unitary_simulator              /**< Qiskit Aer unitary local simulator                */

IBMQ
""""
This class executes quantum circuits remotely on physical quantum devices made accessible through IBM's Quantum Experience cloud services. It adopts the OpenQASM v2.0 quantum assembly language: `IBMQ Website <https://quantum-computing.ibm.com/>`_

Available QDevices in LibKet:

.. code-block:: cpp

  // IBM-Q Experience
  ibmq_almaden_simulator      /**< IBM-Q  20-qubit remote simulator                  */
  ibmq_armonk_simulator       /**< IBM-Q   1-qubit remote simulator                  */
  ibmq_athens_simulator       /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_belem_simulator        /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_boeblingen_simulator   /**< IBM-Q  20-qubit remote simulator                  */
  ibmq_bogota_simulator       /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_brooklyn_simulator     /**< IBM-Q  65-qubit remote simulator                  */
  ibmq_burlington_simulator   /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_cairo_simulator        /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_cambridge_simulator    /**< IBM-Q  28-qubit remote simulator                  */
  ibmq_casablanca_simulator   /**< IBM-Q   7-qubit remote simulator                  */
  ibmq_dublin_simulator       /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_essex_simulator        /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_guadalupe_simulator    /**< IBM-Q  16-qubit remote simulator                  */
  ibmq_hanoi_simulator        /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_jakarta_simulator      /**< IBM-Q   7-qubit remote simulator                  */
  ibmq_johannesburg_simulator /**< IBM-Q  20-qubit remote simulator                  */
  ibmq_kolkata_simulator      /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_lagos_simulator        /**< IBM-Q   7-qubit remote simulator                  */
  ibmq_lima_simulator         /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_london_simulator       /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_manhattan_simulator    /**< IBM-Q  65-qubit remote simulator                  */
  ibmq_manila_simulator       /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_melbourne_simulator    /**< IBM-Q  15-qubit remote simulator                  */    
  ibmq_montreal_simulator     /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_mumbai_simulator       /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_nairobi_simulator      /**< IBM-Q   7-qubit remote simulator                  */
  ibmq_ourense_simulator      /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_paris_simulator        /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_peekskill_simulator    /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_poughkeepsie_simulator /**< IBM-Q  20-qubit remote simulator                  */
  ibmq_quito_simulator        /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_rochester_simulator    /**< IBM-Q  53-qubit remote simulator                  */
  ibmq_rome_simulator         /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_rueschlikon_simulator  /**< IBM-Q  16-qubit remote simulator                  */
  ibmq_santiago_simulator     /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_singapore_simulator    /**< IBM-Q  20-qubit remote simulator                  */
  ibmq_sydney_simulator       /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_tenerife_simulator     /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_tokyo_simulator        /**< IBM-Q  20-qubit remote simulator                  */
  ibmq_toronto_simulator      /**< IBM-Q  27-qubit remote simulator                  */
  ibmq_valencia_simulator     /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_vigo_simulator         /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_yorktown_simulator     /**< IBM-Q   5-qubit remote simulator                  */
  ibmq_washington_simulator   /**< IBM-Q 127-qubit remote simulator                  */
  ibmq_perth_simulator        /**< IBM-Q   7-qubit remote simulator                  */
  
  ibmq_qasm_simulator         /**< IBM-Q universal remote simulator                  */

  ibmq_almaden                /**< IBM-Q  20-qubit processor                         */
  ibmq_armonk                 /**< IBM-Q   1-qubit processor                         */
  ibmq_athens                 /**< IBM-Q   5-qubit processor                         */
  ibmq_belem                  /**< IBM-Q   5-qubit processor                         */
  ibmq_boeblingen             /**< IBM-Q  20-qubit processor                         */
  ibmq_bogota                 /**< IBM-Q   5-qubit processor                         */
  ibmq_brooklyn               /**< IBM-Q  65-qubit processor                         */
  ibmq_cairo                  /**< IBM-Q  27-qubit processor                         */
  ibmq_burlington             /**< IBM-Q   5-qubit processor                         */
  ibmq_cambridge              /**< IBM-Q  28-qubit processor                         */
  ibmq_casablanca             /**< IBM-Q   7-qubit processor                         */
  ibmq_dublin                 /**< IBM-Q  27-qubit processor                         */
  ibmq_essex                  /**< IBM-Q   5-qubit processor                         */
  ibmq_guadalupe              /**< IBM-Q  16-qubit processor                         */
  ibmq_hanoi                  /**< IBM-Q  27-qubit processor                         */
  ibmq_jakarta                /**< IBM-Q   7-qubit processor                         */
  ibmq_johannesburg           /**< IBM-Q  20-qubit processor                         */
  ibmq_kolkata                /**< IBM-Q  27-qubit processor                         */
  ibmq_lagos                  /**< IBM-Q   7-qubit processor                         */
  ibmq_lima                   /**< IBM-Q   5-qubit processor                         */
  ibmq_london                 /**< IBM-Q   5-qubit processor                         */
  ibmq_manhattan              /**< IBM-Q  65-qubit processor                         */
  ibmq_manila                 /**< IBM-Q   5-qubit processor                         */
  ibmq_melbourne              /**< IBM-Q  15-qubit processor                         */    
  ibmq_montreal               /**< IBM-Q  27-qubit processor                         */
  ibmq_mumbai                 /**< IBM-Q  27-qubit processor                         */
  ibmq_nairobi                /**< IBM-Q   7-qubit processor                         */
  ibmq_ourense                /**< IBM-Q   5-qubit processor                         */
  ibmq_paris                  /**< IBM-Q  27-qubit processor                         */
  ibmq_peekskill              /**< IBM-Q  27-qubit processor                         */
  ibmq_poughkeepsie           /**< IBM-Q  20-qubit processor                         */
  ibmq_quito                  /**< IBM-Q   5-qubit processor                         */
  ibmq_rochester              /**< IBM-Q  53-qubit processor                         */
  ibmq_rome                   /**< IBM-Q   5-qubit processor                         */
  ibmq_rueschlikon            /**< IBM-Q  16-qubit processor                         */
  ibmq_santiago               /**< IBM-Q   5-qubit processor                         */
  ibmq_singapore              /**< IBM-Q  20-qubit processor                         */
  ibmq_sydney                 /**< IBM-Q  27-qubit processor                         */
  ibmq_tenerife               /**< IBM-Q   5-qubit processor                         */
  ibmq_tokyo                  /**< IBM-Q  20-qubit processor                         */
  ibmq_toronto                /**< IBM-Q  27-qubit processor                         */
  ibmq_valencia               /**< IBM-Q   5-qubit processor                         */
  ibmq_vigo                   /**< IBM-Q   5-qubit processor                         */
  ibmq_yorktown               /**< IBM-Q   5-qubit processor                         */
  ibmq_washington             /**< IBM-Q 127-qubit processor                         */
  ibmq_perth                  /**< IBM-Q   7-qubit processor                         */

Quantum Inspire
"""""""""""""""

This class executes quantum circuits remotely on the Quantum-Inspire simulator made accessible through QuTech's Quantum-Inspire cloud services. It adopts the commonQASM v1.0  quantum assembly language. The goal of Quantum Inspire is to provide users access to various technologies to perform quantum computations and insights in principles of quantum computing and access to the community: `Quantum Inpsire Website <https://www.quantum-inspire.com/>`_



.. code-block:: cpp

  qi_26_simulator  /**< Quantum Inspire 26-qubit simulator                */
  qi_34_simulator  /**< Quantum Inspire 34-qubit simulator                */
  qi_spin2         /**< Quantum Inspire spin-2 processor (2 qubits)       */
  qi_starmon5      /**< Quantum Inspire starmon-5 processor (5 qubits)    */

Rigetti
"""""""

This class executes quantum circuits remotely on physical quantum devices made accessible through Rigetti's Quantum Cloud Service (QCS). It adopts Rigetti's Quantum Instruction Language. Rigetti builds quantum computers and the superconducting quantum processors that power them: `Rigetti Website <https://www.rigetti.com/about-rigetti-computing>`_ 

Available QDevices in LibKet:

.. code-block:: cpp

  rigetti_aspen_8_simulator   /**< Rigetti Aspen-8 simulator                         */
  rigetti_aspen_9_simulator   /**< Rigetti Aspen-9 simulator                         */
  rigetti_aspen_10_simulator  /**< Rigetti Aspen-10 simulator                        */
  rigetti_9q_square_simulator /**< Rigetti 9Q-square simulator                       */
  rigetti_aspen_8             /**< Rigetti Aspen-8 processor                         */
  rigetti_aspen_9             /**< Rigetti Aspen-9 processor                         */
  rigetti_aspen_10            /**< Rigetti Aspen-10 processor                        */

IonQ
""""

The IonQ backend provides acces to an 11-qubit trapped ion quantum computer and simulator. One of the main benifits of this quantum computer is its complete connectivity, which means that qubits can interact with eachother without swaps: `IonQ Website <https://ionq.com/>`_

Available QDevices in LibKet:

.. code-block:: cpp

  ionq_simulator    /**< IonQ Simulator           */
  ionq_qpu          /**< IonQ Trapped-Ion QPU     */
  

QuEST
"""""

The Quantum Exact Simulation Toolkit is a high performance simulator of quantum circuits, state-vectors and density matrices. QuEST implements multiple useful feathuers such as multithreading, GPU acceleration and distribution: `QuEST Website <https://quest.qtechtheory.org/>`_ 

Available QDevices in LibKet:

.. code-block:: cpp
    
    quest   /**< QuEST simulator                                   */

QX
"""

The QX Simulator is a universal quantum computer simulator developped at QuTech by Nader Khammassi. The QX allows quantum algorithm designers to simulate the execution of their quantum circuits on a quantum computer. It adopts a low-level quantum assembly language Quantum Code: `QX Website <http://www.quantum-studio.net/>`_

Available QDevices in LibKet:

.. code-block:: cpp

    qx    /**< QX simulator                                      */

OpenQL
""""""

This class compiles the quantum circuit using the OpenQL backend. It adopts the OpenQL quantum assembly language. OpenQL is a framework for high-level quantum programming in C++/Python. The framework provides a compiler for compiling and optimizing quantum code. The compiler produces the intermediate quantum assembly language and the compiled micro-code for various target platforms: `OpenQL website <https://github.com/QE-Lab/OpenQL>`_

Available QDevices in LibKet:

.. code-block:: cpp

  openql_cc_light_compiler     /**< OpenQL compiler for CC-Light                      */
  openql_cc_light17_compiler   /**< OpenQL compiler for CC-Light17                    */
  openql_qx_compiler           /**< OpenQL compiler for QX simulator                  */