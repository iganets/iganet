.. _LibKet Tutorials:

Tutorials
=========

These set of tutorials provide more insight in the workings of LibKet for different commonly used quantum circuits, backends and advanced operations. All tutorial code can be found in the LibKet examples folder. If not mentioned otherwise, all expressions are evaluated for 1024 shots on all active backends. This section is complemantary material to the example code and as such the tutorials are best followed with code and documenation side to side.

**Tutorial 1: Bell state**

This tutorial shows how to create a simple LibKet expression, the Bell circuit.

.. tikz:: Bell state circuit

	\node at (0,0) []{
    \tikzset{
    phase label/.append style={above right,xshift=0.1cm}
    }
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm, transparent]
        \lstick{$q_0$} & \gate{H}   & \ctrl{1}  & \meter{} \\
        \lstick{$q_1$} & \qw 		& \targ{}   & \meter{} 
    \end{quantikz}};	
    :libs:quantikz

**Tutorial 2: Quantum Teleportation**

This tutorial implements the quantum teleportation circuit, which teleports the state of the first qubit to the third qubit by using measurements taken from the first two qubits. 

.. tikz:: Quantum Teleportation circuit

    \node at (0,0) []{
    \tikzset{
    phase label/.append style={above right,xshift=0.1cm}
    }
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm, transparent]
        \lstick{$\ket{\psi}$} & \qw      & \qw      & \ctrl{1} & \gate{H} & \meter{} & \cw        & \cwbend{2} \\
        \lstick{$\ket{0}$}    & \gate{H} & \ctrl{1} & \targ{}  & \qw      & \meter{} & \cwbend{1} \\
        \lstick{$\ket{0}$}    & \qw      & \targ{}  & \qw      & \qw      & \qw      & \gate{X}   & \gate{Z} & \qw \rstick{$\ket{\psi}$}
    \end{quantikz}};    
    :libs:quantikz

**Tutorial 3: Advanced**

This tutorial shows the use of more advanced filters and gates to create this arbitrary circuit. The circuit itselfs serves no real purpose, but is rather an example of how LibKet handles filters and gate functors.

.. tikz:: Advanced circuit

    \node at (0,0) []{
    \tikzset{
    phase label/.append style={above right,xshift=0.1cm}
    }
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm, transparent]
        \lstick{$q_0$} & \gate{R_y(1)}  & \ctrl{1}            & \ctrl{1}                      & \qw      & \meter{} \\
        \lstick{$q_1$} & \gate{H}       &\phase{U_1(\pi)} \qw &\phase{U_1(\frac{\pi}{4})} \qw & \gate{H} & \meter{}
     
    \end{quantikz}};    
    :libs:quantikz

**Tutorial 4: Static_for with graph**

This tutorial shows how to implement the :code:`static_for` function in combination with the LibKet graph class. Here the :code:`static_for` loop is used to apply a cnot gate between every connected node in the graph. The ring graph with five nodes is represented as a list of edges. From this, it follows that the static_for loop should iterate over all edges in this list and apply a cnot function between two nodes that are connected, where the first node is the control and the second node the target qubit.

.. tikz:: Static_for circuit

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm, transparent]
        \lstick{$q_0$} & \ctrl{1} & \qw      & \qw      & \qw      & \targ{}  & \meter{} \\
        \lstick{$q_1$} & \targ{}  & \ctrl{1} & \qw      & \qw      & \qw      & \meter{} \\
        \lstick{$q_2$} & \qw      & \targ{}  & \ctrl{1} & \qw      & \qw      & \meter{} \\
        \lstick{$q_3$} & \qw      & \qw      & \targ{}  & \ctrl{1} & \qw      & \meter{} \\
        \lstick{$q_4$} & \qw      & \qw      & \qw      & \targ{}  & \ctrl{-4}& \meter{}
     
    \end{quantikz}};    
    :libs:quantikz

**Tutorial 5: QPU execution**

In this tutorial, three methods of retrieving QPU results are shown:
    
- Asynchronous execution
- Synchronous execution
- Evaluation

Both asynchronous and synchrounous execution return a pointer to the quantum job. The asynchronous option does not interrupt the code exection and the QPU execution will run in the background, so you can run code while the quantum expression is being evaluated. The synchronous option waits until the QPU has finished evaluating the quantum expression before contuing with the main code. For both methods, results can be retreived using the :code:`job->get()` function.

The evaluation method is similar to the sychronous execution, but directly returns results in JSON format instead of the :code:`QObj` pointer. The circuit belows shows the simple quantum expression used in this tutorial:    

.. tikz:: Simple quantum expression 

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm, transparent]
        \lstick{$q_0$} & \gate{X} & \gate{Y} & \meter{} \\
        \lstick{$q_1$} & \gate{X} & \qw      & \meter{} \\
        \lstick{$q_2$} & \gate{X} & \gate{Y} & \meter{} \\
        \lstick{$q_3$} & \gate{X} & \qw      & \meter{} 
     
    \end{quantikz}};    
    :libs:quantikz

**Tutorial 6: Quantum Fourier Transform**

Here, the LibKet circuit QFT is used to construct a QFT circuit with allswap at the end. The Quantum Fourier Tranform. More information on the QFT can be found `here <https://en.wikipedia.org/wiki/Quantum_Fourier_transform>`_. The inverse QFT can be applied by using the :code:`QFTdag()` circuit.

.. tikz:: QFT circuit for 6 qubits

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \gate{H} & \gate{S} & \qw      & \gate{T} & \qw      & \gate{Z^{1/8}}& \qw      & \gate{Z^{1/16}}& \qw      & \gate{Z^{1/32}}& \qw      & \qw      & \qw      & \swap{5} & \meter{}  \\
        \lstick{$q_1$} & \qw      & \ctrl{-1}& \gate{H} & \gate{S} & \qw      & \gate{T}      & \qw      & \gate{Z^{1/8}} & \qw      & \gate{Z^{1/16}}& \qw      & \qw      & \swap{3} & \qw      & \meter{}  \\
        \lstick{$q_2$} & \qw      & \qw      & \qw      & \ctrl{-2}& \gate{H} & \gate{S}      & \qw      & \gate{T}       & \qw      & \gate{Z^{1/8}} & \qw      & \swap{1} & \qw      & \qw      & \meter{}  \\
        \lstick{$q_3$} & \qw      & \qw      & \qw      & \qw      & \qw      & \ctrl{-3}     & \gate{H} & \gate{S}       & \qw      & \gate{T}       & \qw      & \targX{} & \qw      & \qw      & \meter{}  \\
        \lstick{$q_4$} & \qw      & \qw      & \qw      & \qw      & \qw      & \qw           & \qw      & \ctrl{-4}      & \gate{H} & \gate{S}       & \qw      & \qw      & \targX{} & \qw      & \meter{}  \\
        \lstick{$q_5$} & \qw      & \qw      & \qw      & \qw      & \qw      & \qw           & \qw      & \qw            & \qw      & \ctrl{-5}      & \gate{H} & \qw      & \qw      & \targX{} & \meter{}  
    \end{quantikz}};
    :libs: quantikz 

**Tutorial 7: Arbitrary Control Circuit**

The following circuit implements the arbitrary control circuit. It takes four parameters: A controlled binary gate, a filter for the control qubits, a filter for the ancilla qubits and a filter for the target qubit. This example implements a 4-qubit controlled X-gate. The first four qubits are used a control for the target qubit :math:`q_4`. For every :math:`N` control qubits, :math:`N-1` ancilla qubits are needed, in this case the last three qubits.

.. tikz:: Arbitrary Control circuit for cnot gate

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \ctrl{1} & \qw      & \qw      & \qw      & \qw      & \qw      & \ctrl{1} & \meter{} \\
        \lstick{$q_1$} & \ctrl{4} & \qw      & \qw      & \qw      & \qw      & \qw      & \ctrl{4} & \meter{} \\
        \lstick{$q_2$} & \qw      & \ctrl{3} & \qw      & \qw      & \qw      & \ctrl{3} & \qw      & \meter{} \\
        \lstick{$q_3$} & \qw      & \qw      & \ctrl{3} & \qw      & \ctrl{3} & \qw      & \qw      & \meter{} \\
        \lstick{$q_4$} & \qw      & \qw      & \qw      & \gate{X} & \qw      & \qw      & \qw      & \meter{} \\ 
        \lstick{$q_5$} & \targ{}  & \ctrl{1} & \qw      & \qw      & \qw      & \ctrl{1} & \targ{}  & \meter{} \\
        \lstick{$q_6$} & \qw      & \targ{}  & \ctrl{1} & \qw      & \ctrl{1} & \targ{}  & \qw      & \meter{} \\
        \lstick{$q_7$} & \qw      & \qw      & \targ{}  & \ctrl{-3}& \targ{}  & \qw      & \qw      & \meter{}    
    \end{quantikz}};
    :libs: quantikz 

**Tutorial 8: Allswap**

The Allswap circuit creates a quantum expression where the qubit order in a given selection is flipped.

.. tikz:: Allswap circuit

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \qw      & \qw      & \swap{5} & \qw      & \meter{} \\
        \lstick{$q_1$} & \qw      & \swap{3} & \qw      & \qw      & \meter{} \\
        \lstick{$q_2$} & \swap{1} & \qw      & \qw      & \qw      & \meter{} \\
        \lstick{$q_3$} & \targX{} & \qw      & \qw      & \qw      & \meter{} \\ 
        \lstick{$q_4$} & \qw      & \targX{} & \qw      & \qw      & \meter{} \\
        \lstick{$q_5$} & \qw      & \qw      & \targX{} & \qw      & \meter{}  
    \end{quantikz}};
    :libs: quantikz 

**Tutorial 9: QAOA**

This tutorial shows the implementation of a QAOA circuit for the Maximum Cut problem on arbitrary graph. More information on the QAOA can be found `here <https://arxiv.org/abs/1411.4028>`_. This toturial shows some more advance use of the :code:`static_for()` function. The circuit is constructed for a singe QAOA iteration (:math:`p=1`). This tutorial only shows how to create the QAOA circuit. For the QAOA to function, an classical optimizer is needed to optimize parameters :math:`\beta` and :math:`\gamma`.

.. tikz:: QAOA circuit for MaxCut

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \gate{H} & \ctrl{1} & \qw                & \ctrl{1} & \ctrl{4} & \qw               & \ctrl{4} & \qw      & \qw               & \qw      & \qw      & \qw               & \qw      & \qw      & \qw               & \qw      & \gate{R_x(\beta)} & \meter{} \\
        \lstick{$q_1$} & \gate{H} & \targ{}  & \gate{R_z(\gamma)} & \targ{1} & \qw      & \qw               & \qw      & \ctrl{1} & \qw               & \ctrl{1} & \ctrl{3} & \qw               & \ctrl{3} & \qw      & \qw               & \qw      & \gate{R_x(\beta)} & \meter{} \\
        \lstick{$q_2$} & \gate{H} & \qw      & \qw                & \qw      & \qw      & \qw               & \qw      & \targ{}  & \gate{R_z(\gamma)}& \targ{}  & \qw      & \qw               & \qw      & \ctrl{1} & \qw               & \ctrl{1} & \gate{R_x(\beta)} & \meter{} \\
        \lstick{$q_3$} & \gate{H} & \qw      & \qw                & \qw      & \qw      & \qw               & \qw      & \qw      & \qw               & \qw      & \qw      & \qw               & \qw      & \targ{}  & \gate{R_z(\gamma)}& \targ{}  & \gate{R_x(\beta)} & \meter{} \\ 
        \lstick{$q_4$} & \gate{H} & \qw      & \qw                & \qw      & \targ{}  & \gate{R_z(\gamma)}& \targ{}  & \qw      & \qw               & \qw      & \targ{}  & \gate{R_z(\gamma)}& \targ{}  & \qw      & \qw               & \qw      & \gate{R_x(\beta)} & \meter{} \\
        \lstick{$q_5$} & \gate{H} & \qw      & \qw                & \qw      & \qw      & \qw               & \qw      & \qw      & \qw               & \qw      & \qw      & \qw               & \qw      & \qw      & \qw               & \qw      & \gate{R_x(\beta)} & \meter{}  
    \end{quantikz}};
    :libs: quantikz 

**Tutorial 10: Hook**

This tutorial illustrates the usage of the hook gate, which is able to reference another LibKet expression or create an expression from a cQASM string.

.. tikz:: Circuit created with the Hook function

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \gate{H} & \meter{} \\
        \lstick{$q_1$} & \gate{H} & \meter{} \\
        \lstick{$q_2$} & \gate{H} & \meter{} \\
        \lstick{$q_3$} & \gate{H} & \meter{} \\
        \lstick{$q_4$} & \gate{H} & \meter{} \\
        \lstick{$q_5$} & \gate{H} & \meter{} 
    \end{quantikz}};
    :libs: quantikz 

**Tutorial 11: Just-In-Time Compilation**

Using Just-In-Time Compilation, LibKet is able to use command line inputs for compile-time expresions. In this tutorial, a LibKet expression can be entered via the command line and will be processed later on in program.

**Tutorial 12: Execution Scripts**

he optional :code:`ftor_init`, :code:`ftor_before`, and :code:`ftor_after` make it possible to inject user-defined code at three different locations of the execution process. In this tutorial, a simple statement after the execution collects the histogram data of the experiment using Qiskit's :code:`get_count()` function, generates a histogram plot and saves it to a file named 'histogram.png' in the build folder.

The init script imports the necessary packages for the qiksit visualization. After execution, the after script gets the counts and plots the histogram. It should be noted that the code injections are idented automatically and must not have trailing :code:`\t`'s. Each line must end with :code:`\n`.


.. tikz:: Simple quantum expression for scripts tutorial

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm, transparent]
        \lstick{$q_0$} & \gate{H} & \meter{} \\
        \lstick{$q_1$} & \qw      & \meter{} \\
        \lstick{$q_2$} & \gate{H} & \meter{} \\
        \lstick{$q_3$} & \qw      & \meter{} 
     
    \end{quantikz}};    
    :libs:quantikz

**Tutorial 13: Unitary decomposition**

This tutorial illustrates the basic usage of the built-in decomposition of a controlled 2x2 unitary gate into native gates. The unitary gate accepts an arbitrary 2x2
unitary matrix :math:`U` as input and performs the ZYZ decomposition of :math:`U`. For example the following unitary matrix is used:

.. math::
    
   U = \frac{1}{\sqrt{2}}
    \begin{pmatrix}
    1 & -1\\
    1 & 1
    \end{pmatrix}    

The decomposition created the following quantum circuit:

.. tikz:: ZYZ decomposition of the unitary matrix

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \gate{R_z(0)} & \gate{R_y(\frac{\pi}{2})} & \gate{R_z(0)} & \meter{} \\
        \lstick{$q_1$} & \gate{R_z(0)} & \gate{R_y(\frac{\pi}{2})} & \gate{R_z(0)} & \meter{} \\
        \lstick{$q_2$} & \gate{R_z(0)} & \gate{R_y(\frac{\pi}{2})} & \gate{R_z(0)} & \meter{} \\
        \lstick{$q_3$} & \gate{R_z(0)} & \gate{R_y(\frac{\pi}{2})} & \gate{R_z(0)} & \meter{}
    \end{quantikz}};
    :libs: quantikz 



**Tutorial 14: Controlled unitary decomposition**

This tutorial implementes the controlled unitary decomposition, which is similar the the previous tutorial on the decomposed unitary. In this case, a control qubit is incluced to form a binary gate which controls the rotations of the unitary. For example the following unitary matrix is used:

.. math::
    
   U = \frac{1}{\sqrt{2}}
    \begin{pmatrix}
    1 & -1\\
    1 & 1
    \end{pmatrix}    

The controlled decomposition created the following quantum circuit:

.. tikz:: ZYZ decomposition of the unitary matrix

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \ctrl{1}      & \ctrl{1}                  & \ctrl{1}      & \meter{} \\
        \lstick{$q_1$} & \gate{R_z(0)} & \gate{R_y(\frac{\pi}{2})} & \gate{R_z(0)} & \meter{} 

    \end{quantikz}};
    :libs: quantikz 

**Tutorial 15: Quantum Program**

The quantum program allows for a linear approach to constructing a quantum expression, as often seen in QASM languages. Here, qubit gates and operations are added sequantially and are translated to a quantum expression. 


.. tikz:: Circuit created by using the QProgram

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$q_0$} & \gate{R_x(3.141)} & \gate{H}          & \qw      & \qw      & \meter{} \\
        \lstick{$q_1$} & \gate{R_x(3.141)} & \gate{H}          & \qw      & \qw      & \meter{} \\
        \lstick{$q_2$} & \gate{R_x(3.141)} & \gate{H}          & \qw      & \qw      & \meter{} \\
        \lstick{$q_3$} & \gate{H}          & \gate{R_x(3.141)} & \qw      & \ctrl{3} & \meter{} \\
        \lstick{$q_4$} & \gate{R_x(3.141)} & \ctrl{3}          & \qw      & \qw      & \meter{} \\
        \lstick{$q_5$} & \gate{R_x(3.141)} & \qw               & \ctrl{3} & \qw      & \meter{} \\
        \lstick{$q_6$} & \qw               & \qw               & \qw      & \targ{}  & \meter{} \\
        \lstick{$q_7$} & \qw               & \targ{}           & \qw      & \qw      & \meter{} \\
        \lstick{$q_8$} & \qw               & \qw               & \targ{}  & \qw      & \meter{} 
    \end{quantikz}};
    :libs: quantikz 

**Tutorial 16: HHL Algorithm**

This tutorial shows the implementation of the Harrow-Hassidim-Lloyd (HHL) algorithm (see `link <https://arxiv.org/pdf/2108.09004.pdf>`_). This algorithm is used to solve Linear systems of the form:

.. math::

   A \vec{x} = \vec{b}

where A is an :math:`N_{b}×N_{b}` Hermitian matrix and :math:`\vec{x}` and :math:`\vec{b}` are :math:`N_b`-dimensional vectors. In this example, :math:`A` and :math:`\vec{b}` are set to:   

.. math::
    
   A =
    \begin{pmatrix}
    1 & -\frac{1}{3}\\
    -\frac{1}{3} & 1
    \end{pmatrix}\\

  \vec{b} = \begin{pmatrix} 0 \\ 1\end{pmatrix}   

The controlled unitary evolution is computed to be:

.. math::
    
   U = e^{iAt} = e^{iA\frac{3\pi}{4}} = \frac{1}{2}
    \begin{pmatrix}
    -1+i & 1+i\\
     1+1 & -1+i
    \end{pmatrix}\\ 

Which results in the following circuit example:

.. tikz:: HHL Algorithm for a 2x2 matrix A and 2x1 vector b

    \node at (0,0) []{
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm]
        \lstick{$ancilla_0$} & \qw      & \qw      & \qw      & \qw        & \qw      & \qw      & \qw      & \qw      & \gate{R_Y (\pi)} & \gate{R_Y (\pi/3)} & \meter{} & \qw      & \qw      & \qw       & \qw       & \qw        & \qw      & \qw      & \qw \\
        \lstick{$clock_0$}   & \qw      & \gate{H} & \ctrl{2} & \qw        & \qw      & \ctrl{1} & \gate{H} & \swap{1} & \ctrl{-1}        & \qw                & \qw      & \swap{1} & \gate{H} & \gate{S}  & \qw       & \qw        & \ctrl{2} & \gate{H} & \qw \\
        \lstick{$clock_1$}   & \qw      & \gate{H} & \qw      & \ctrl{1}   & \gate{H} & \gate{S} & \qw      & \targX{} & \qw              & \ctrl{-2}          & \qw      & \targX{} & \qw      & \ctrl{-1} & \gate{H}  & \ctrl{1}   & \qw      & \gate{H} & \qw \\
        \lstick{$b_0$}       & \gate{X} & \qw      & \gate{U} & \gate{U^2} & \qw      & \qw      & \qw      & \qw      & \qw              & \qw                & \qw      & \qw      & \qw      & \qw       & \qw       & \gate{U^2} & \gate{U} & \qw      & \meter{} \\

    \end{quantikz}};
    :libs: quantikz     


The output result indeed confirms the expected ratio found in the HHL paper, which should be around :math:`prob(b_0)` : :math:`prob(b_1)` =  1 : 9.
