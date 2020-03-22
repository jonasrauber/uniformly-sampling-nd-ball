=========================================================
Efficiently sampling vectors from the n-sphere and n-ball
=========================================================

This repository provides an implementation of the algorithm proposed by `Voelker et al., 2017 <http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf>`_ for efficient uniform sampling from the n-dimensional ball (section 3.1) and compares its performance with the baseline algorithm (section 2.2).

The implementation works with

* NumPy
* PyTorch
* TensorFlow
* JAX

and is based on `EagerPy <https://github.com/jonasrauber/eagerpy>`_.

Code
====

The implementation can be found in `sample.py <sample.py>`_.

Results
=======

All experiments were run on a 64 core Intel Xeon processor and an Nvidia Tesla V100 GPU.

NumPy
-----

.. literalinclude:: results/numpy.txt

PyTorch
-------

.. literalinclude:: results/pytorch.txt

PyTorch (GPU)
-------------

.. literalinclude:: results/pytorch-gpu.txt

TensorFlow (GPU)
----------------

.. literalinclude:: results/tensorflow.txt

JAX (GPU)
---------

.. literalinclude:: results/jax.txt
