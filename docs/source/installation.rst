Installation
=====================

`CMake <https://cmake.org>`_ is our primary build system. To install LMN-Hydro, follow these steps:

.. code-block:: bash
    mkdir build
    cd build
    cmake ..
    make -j

**That's it!**


.. Under construction:
.. You can also customize the build using various CMake options. For example, to enable KOKKOS support, you can run:

.. .. code-block:: bash
..     cmake -DENABLE_KOKKOS=ON ..
