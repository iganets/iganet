.. _LibKet Advanced:

Advanced
========


.. _LibKet Static For Loop:

Static For Loop
---------------

Since quantum expressions are evaluated at compile time, it is not possible to generate an
expression with a runtime for-loop. However, in order to build larger quantum circuits, using
a for loop will become a necessaty. Looping over quantum expressions can thus be done with the
:code:`constexpr_for()` function. The generic interface of the :code:`constexpr_for()` function reads:

.. code-block:: cpp

	template<index_t for_start,
         	 index_t for_end,
         	 index_t for_step,
         	 template<index_t start, index_t end, index_t step, index_t index> class functor,
         	 typename functor_return_type,
         	 typename... functor_types>
	inline auto
	constexpr_for(functor_return_type&& functor_return_arg,
           	   functor_types&&... functor_args)

Below is a small example on how to use the :code:`constexpr_for()` function. First, a **functor**
needs to be created, which represents the loop's body:

.. code-block:: cpp

	template<index_t start, index_t end, index_t step, index_t index>
	struct ftor
	{
	    template<typename Expr>
	    inline constexpr auto operator()(Expr&& expr) noexcept
	    {
	        // Returns the controlled phase shift gate with angle
	        // theta = pi/2^(index+1) between qubits index and index+1
	        return crk<index+1>(sel<index>  (gototag<0>()),
	                            sel<index+1>(gototag<0>(expr))
	                           );
	    }
	};

To loop through this functor (at compile time) we call the :code:`utils::constexpr_for<start, end, step, body>(...)` function as follows.
Note the usefulness of the tag/gototag mechanism to restore the original filter settings easily.

.. code-block:: cpp

	auto expr = utils::constexpr_for<0,4,1,ftor>(tag<0>(init()));

This then generates the following 6 qubit circuit with only one line of code! Notice that the :code:`end` index is set to 4 and is thus also included in the loop, similar to the regular :code:`for(int i=start; i<=end; i+=step){}`

.. tikz:: Generated circuit by the constexpr_for loop

	\node at (0,0) []{
    \tikzset{
    phase label/.append style={above right,xshift=0.1cm}
    }
    \begin{quantikz}[row sep={0.75cm,between origins}, column sep=0.2cm, transparent]
        \lstick{$q_0$} & \ctrl{1}             & \qw & \qw                            & \qw & \qw                            & \qw & \qw                            & \qw & \qw                             & \qw \\
        \lstick{$q_1$} & \phase{U_1(\pi)} \qw & \qw & \ctrl{1}                       & \qw & \qw                            & \qw & \qw                            & \qw & \qw                             & \qw \\
        \lstick{$q_2$} & \qw                  & \qw & \phase{U_2(\frac{\pi}{2})} \qw & \qw & \ctrl{1}                       & \qw & \qw                            & \qw & \qw                             & \qw \\
        \lstick{$q_3$} & \qw                  & \qw & \qw                            & \qw & \phase{U_3(\frac{\pi}{4})} \qw & \qw & \ctrl{1}                       & \qw & \qw                             & \qw \\
        \lstick{$q_4$} & \qw                  & \qw & \qw                            & \qw & \qw                            & \qw & \phase{U_4(\frac{\pi}{8})} \qw & \qw & \ctrl{1}                        & \qw \\
        \lstick{$q_5$} & \qw                  & \qw & \qw                            & \qw & \qw                            & \qw & \qw                            & \qw & \phase{U_5(\frac{\pi}{16})} \qw & \qw
    \end{quantikz}};
    :libs:quantikz

.. _LibKet computational offloading:

Computational offloading
------------------------

LibKet's computation offloading model is very similar to that of CUDA to ease the transition from GPU- to quantum-accelerated computing. The :code:`device.eval(...)` is just one of three ways to run a quantum expression, which we will refer to as quantum kernel, on a quantum device.

:code:`LibKet::utils::json device.eval(...)` :
	This function offloads the quantum computation to the quantum device and returns the evaluated result as JSON object once the quantum computation has completed. Exceptions are the QuEST and QX simulators where a reference to the internal state vector is returned.

:code:`LibKet::QJob* device.execute(...)`
	Offloads the quantum computation to the quantum device and returns a QJob pointer once the quantum computation has completed.

:code:`LibKet::QJob* device.execute_async(...)`
	Offloads the quantum computation to the quantum device and returns a QJob pointer immediately.

The :code:`execute()` and :code:`execute_async()` have a similar interface as the :code:`eval()` function:

.. code-block:: cpp

	QJob<QJobType::CXX>* execute(std::size_t shots       			     = [default from ctor],
				     std::function<void(QDevice_QuEST*)> ftor_init   = NULL,
				     std::function<void(QDevice_QuEST*)> ftor_before = NULL,
				     std::function<void(QDevice_QuEST*)> ftor_after  = NULL,
				     QStream<QJobType::CXX>*             stream      = NULL)

Notice that different functors are optional parameters, which will be elaborated on in the next section. The QJob objects supports the following functionality

- :code:`QObj* wait()`: waits for the job to complete (blocking)
- :code:`bool query()`: returns true if the job completed and false otherwise (non-blocking)
- :code:`utils::json get()`: returns the result as JSON object after completion (blocking)

Let's conclude these exection options with an example. Here an expression is executed asychronously for 20 shots on the Qiskit QASM simulator. The results are retreived with the :code:`get()` function, which waits for the qpu to finish execution and return results.

.. code-block:: cpp

	QDevice<QDeviceType::qiskit_qasm_simulator, 2> qpu;
	qpu(expr);
	auto job = qpu.execute_async(20);
	result = job->get();
	std::cout << result.dump(2) << std::endl;


.. _LibKet Execution Scripts:

Execution Scripts
-----------------

The optional hooks :code:`ftor_init`, :code:`ftor_before`, and :code:`ftor_after` make it possible to inject user-defined code at three different locations of the execution process:

:code:`script_init`
	This functor is performed before any other code of the execution process. It can be used for importing additional Python modules.

:code:`script_before`
	This functor is performed just before sending the instructions to the quantum device. It can be used to pre-process the quantum circuit, e.g., to perform user-specific optimizations on the raw quantum circuit, before it runs through the backend-specific pipeline

:code:`script_after`
	Performed just after receiving the result from the quantum device. It can be used to post-process the raw results received from the quantum device, e.g., to generate histograms or other types of visualizations

Let's inject a simple statement after the execution that collects the histogram data of the experiment using Qiskit's :code:`get_count()` function.

.. code-block:: cpp

	auto job = qiskit.execute_async(20,
                                	/* init_script   */
	                                "",
	                                /* before_script */
	                                "",
	                                /* after_script */
	                                "counts = result.get_counts(qc)\n"
	                                "return json.dumps(counts)\n"
                               		);
	std::cout << job->get().dump(2) << std::endl;

It should be noted that the code injections are idented automatically and must not have trailing :code:`\t`'s. Each line must end with :code:`\n`.

.. _LibKet Parameterized circuits:

Parameterized circuits [WIP]
----------------------------

The creating of parameteterized circuits is still under development. When finished,
LibKet will be able to support platforms that use parameterised circuits, such
as `Qiskit Runtime <https://quantumcomputing.com/strangeworks/qiskit-runtime>`_.
