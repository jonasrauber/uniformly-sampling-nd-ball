============================================================
Efficiently sampling vectors from the n-sphere and n-ball 🏀
============================================================

This repository provides an implementation of the algorithm proposed by `Voelker et al., 2017 <http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf>`_ for efficient uniform sampling from the n-dimensional ball (section 3.1) and compares its performance to the baseline (section 2.2).

Code 🔥
=======

The implementation can be found in `sample.py <sample.py>`_. It is based on `EagerPy <https://github.com/jonasrauber/eagerpy>`_ and was tested with NumPy 1.18.1, PyTorch 1.4.0, TensorFlow 2.0.0 and JAX 0.1.59.

Results 🎉
==========

All experiments were run on a 64 core Intel Xeon processor and an Nvidia Tesla V100 GPU.

NumPy
-----

.. code-block::

        1 @ 1000 repeats: 5.60e-02 vs. 4.07e-02 -> alternative takes 0.7x as long
        2 @ 1000 repeats: 5.66e-02 vs. 3.99e-02 -> alternative takes 0.7x as long
        4 @ 1000 repeats: 5.80e-02 vs. 4.00e-02 -> alternative takes 0.7x as long
        8 @ 1000 repeats: 5.83e-02 vs. 4.03e-02 -> alternative takes 0.7x as long
       16 @ 1000 repeats: 5.90e-02 vs. 4.08e-02 -> alternative takes 0.7x as long
       32 @ 1000 repeats: 5.98e-02 vs. 4.21e-02 -> alternative takes 0.7x as long
       64 @ 1000 repeats: 6.17e-02 vs. 4.34e-02 -> alternative takes 0.7x as long
      128 @ 1000 repeats: 6.67e-02 vs. 4.16e-02 -> alternative takes 0.6x as long
      256 @ 1000 repeats: 6.79e-02 vs. 5.49e-02 -> alternative takes 0.8x as long
      512 @ 1000 repeats: 8.78e-02 vs. 6.85e-02 -> alternative takes 0.8x as long
     1024 @ 1000 repeats: 1.14e-01 vs. 9.30e-02 -> alternative takes 0.8x as long
     2048 @ 1000 repeats: 1.65e-01 vs. 1.22e-01 -> alternative takes 0.7x as long
     4096 @ 1000 repeats: 2.31e-01 vs. 2.36e-01 -> alternative takes 1.0x as long
     8192 @ 1000 repeats: 4.36e-01 vs. 3.82e-01 -> alternative takes 0.9x as long
    16384 @ 1000 repeats: 7.74e-01 vs. 7.22e-01 -> alternative takes 0.9x as long
    32768 @ 1000 repeats: 1.56e+00 vs. 1.41e+00 -> alternative takes 0.9x as long
    65536 @ 1000 repeats: 3.00e+00 vs. 2.86e+00 -> alternative takes 1.0x as long
   131072 @ 1000 repeats: 6.23e+00 vs. 5.61e+00 -> alternative takes 0.9x as long
   262144 @ 1000 repeats: 1.32e+01 vs. 1.12e+01 -> alternative takes 0.8x as long
   524288 @ 1000 repeats: 2.52e+01 vs. 2.22e+01 -> alternative takes 0.9x as long

PyTorch
-------

.. code-block::

        1 @ 1000 repeats: 1.20e-01 vs. 8.31e-02 -> alternative takes 0.7x as long
        2 @ 1000 repeats: 1.18e-01 vs. 8.20e-02 -> alternative takes 0.7x as long
        4 @ 1000 repeats: 1.19e-01 vs. 1.02e-01 -> alternative takes 0.9x as long
        8 @ 1000 repeats: 1.20e-01 vs. 8.30e-02 -> alternative takes 0.7x as long
       16 @ 1000 repeats: 1.17e-01 vs. 8.00e-02 -> alternative takes 0.7x as long
       32 @ 1000 repeats: 1.18e-01 vs. 8.05e-02 -> alternative takes 0.7x as long
       64 @ 1000 repeats: 1.20e-01 vs. 8.12e-02 -> alternative takes 0.7x as long
      128 @ 1000 repeats: 1.19e-01 vs. 8.19e-02 -> alternative takes 0.7x as long
      256 @ 1000 repeats: 1.22e-01 vs. 8.39e-02 -> alternative takes 0.7x as long
      512 @ 1000 repeats: 1.25e-01 vs. 8.60e-02 -> alternative takes 0.7x as long
     1024 @ 1000 repeats: 1.29e-01 vs. 9.04e-02 -> alternative takes 0.7x as long
     2048 @ 1000 repeats: 1.39e-01 vs. 9.95e-02 -> alternative takes 0.7x as long
     4096 @ 1000 repeats: 1.58e-01 vs. 1.17e-01 -> alternative takes 0.7x as long
     8192 @ 1000 repeats: 2.00e-01 vs. 1.52e-01 -> alternative takes 0.8x as long
    16384 @ 1000 repeats: 2.75e-01 vs. 2.27e-01 -> alternative takes 0.8x as long
    32768 @ 1000 repeats: 5.16e-01 vs. 4.46e-01 -> alternative takes 0.9x as long
    65536 @ 1000 repeats: 7.96e-01 vs. 7.47e-01 -> alternative takes 0.9x as long
   131072 @ 1000 repeats: 1.33e+00 vs. 1.26e+00 -> alternative takes 0.9x as long
   262144 @ 1000 repeats: 2.31e+00 vs. 2.20e+00 -> alternative takes 1.0x as long
   524288 @ 1000 repeats: 4.16e+00 vs. 4.07e+00 -> alternative takes 1.0x as long

PyTorch (GPU)
-------------

.. code-block::

        1 @ 1000 repeats: 4.47e+00 vs. 1.46e-01 -> alternative takes 0.0x as long
        2 @ 1000 repeats: 2.28e-01 vs. 1.45e-01 -> alternative takes 0.6x as long
        4 @ 1000 repeats: 2.25e-01 vs. 1.67e-01 -> alternative takes 0.7x as long
        8 @ 1000 repeats: 2.26e-01 vs. 1.45e-01 -> alternative takes 0.6x as long
       16 @ 1000 repeats: 2.27e-01 vs. 1.45e-01 -> alternative takes 0.6x as long
       32 @ 1000 repeats: 2.24e-01 vs. 1.55e-01 -> alternative takes 0.7x as long
       64 @ 1000 repeats: 2.32e-01 vs. 1.82e-01 -> alternative takes 0.8x as long
      128 @ 1000 repeats: 2.91e-01 vs. 1.87e-01 -> alternative takes 0.6x as long
      256 @ 1000 repeats: 2.91e-01 vs. 1.87e-01 -> alternative takes 0.6x as long
      512 @ 1000 repeats: 2.94e-01 vs. 1.88e-01 -> alternative takes 0.6x as long
     1024 @ 1000 repeats: 2.74e-01 vs. 1.97e-01 -> alternative takes 0.7x as long
     2048 @ 1000 repeats: 2.97e-01 vs. 1.88e-01 -> alternative takes 0.6x as long
     4096 @ 1000 repeats: 2.93e-01 vs. 1.88e-01 -> alternative takes 0.6x as long
     8192 @ 1000 repeats: 2.94e-01 vs. 1.89e-01 -> alternative takes 0.6x as long
    16384 @ 1000 repeats: 2.94e-01 vs. 1.87e-01 -> alternative takes 0.6x as long
    32768 @ 1000 repeats: 2.95e-01 vs. 1.91e-01 -> alternative takes 0.6x as long
    65536 @ 1000 repeats: 2.99e-01 vs. 1.91e-01 -> alternative takes 0.6x as long
   131072 @ 1000 repeats: 3.12e-01 vs. 2.02e-01 -> alternative takes 0.6x as long
   262144 @ 1000 repeats: 3.75e-01 vs. 1.98e-01 -> alternative takes 0.5x as long
   524288 @ 1000 repeats: 3.03e-01 vs. 1.95e-01 -> alternative takes 0.6x as long

TensorFlow (GPU)
----------------

.. code-block::

        1 @ 1000 repeats: 6.74e-01 vs. 4.15e-01 -> alternative takes 0.6x as long
        2 @ 1000 repeats: 6.78e-01 vs. 4.15e-01 -> alternative takes 0.6x as long
        4 @ 1000 repeats: 6.77e-01 vs. 4.14e-01 -> alternative takes 0.6x as long
        8 @ 1000 repeats: 6.96e-01 vs. 5.02e-01 -> alternative takes 0.7x as long
       16 @ 1000 repeats: 7.92e-01 vs. 4.79e-01 -> alternative takes 0.6x as long
       32 @ 1000 repeats: 7.90e-01 vs. 4.79e-01 -> alternative takes 0.6x as long
       64 @ 1000 repeats: 7.89e-01 vs. 4.80e-01 -> alternative takes 0.6x as long
      128 @ 1000 repeats: 7.93e-01 vs. 4.73e-01 -> alternative takes 0.6x as long
      256 @ 1000 repeats: 7.93e-01 vs. 4.96e-01 -> alternative takes 0.6x as long
      512 @ 1000 repeats: 7.96e-01 vs. 4.81e-01 -> alternative takes 0.6x as long
     1024 @ 1000 repeats: 7.95e-01 vs. 4.81e-01 -> alternative takes 0.6x as long
     2048 @ 1000 repeats: 7.92e-01 vs. 4.80e-01 -> alternative takes 0.6x as long
     4096 @ 1000 repeats: 7.92e-01 vs. 4.91e-01 -> alternative takes 0.6x as long
     8192 @ 1000 repeats: 8.04e-01 vs. 4.90e-01 -> alternative takes 0.6x as long
    16384 @ 1000 repeats: 8.02e-01 vs. 4.94e-01 -> alternative takes 0.6x as long
    32768 @ 1000 repeats: 8.05e-01 vs. 4.98e-01 -> alternative takes 0.6x as long
    65536 @ 1000 repeats: 8.02e-01 vs. 4.90e-01 -> alternative takes 0.6x as long
   131072 @ 1000 repeats: 8.04e-01 vs. 4.90e-01 -> alternative takes 0.6x as long
   262144 @ 1000 repeats: 8.06e-01 vs. 4.98e-01 -> alternative takes 0.6x as long
   524288 @ 1000 repeats: 8.14e-01 vs. 5.01e-01 -> alternative takes 0.6x as long

JAX (GPU)
---------

.. code-block::

        1 @ 1000 repeats: 3.79e+00 vs. 3.35e+00 -> alternative takes 0.9x as long
        2 @ 1000 repeats: 3.10e+00 vs. 3.39e+00 -> alternative takes 1.1x as long
        4 @ 1000 repeats: 2.50e+00 vs. 3.34e+00 -> alternative takes 1.3x as long
        8 @ 1000 repeats: 3.13e+00 vs. 3.43e+00 -> alternative takes 1.1x as long
       16 @ 1000 repeats: 3.12e+00 vs. 3.37e+00 -> alternative takes 1.1x as long
       32 @ 1000 repeats: 3.12e+00 vs. 3.40e+00 -> alternative takes 1.1x as long
       64 @ 1000 repeats: 3.13e+00 vs. 3.36e+00 -> alternative takes 1.1x as long
      128 @ 1000 repeats: 3.12e+00 vs. 3.37e+00 -> alternative takes 1.1x as long
      256 @ 1000 repeats: 3.14e+00 vs. 3.37e+00 -> alternative takes 1.1x as long
      512 @ 1000 repeats: 3.11e+00 vs. 3.35e+00 -> alternative takes 1.1x as long
     1024 @ 1000 repeats: 3.18e+00 vs. 3.35e+00 -> alternative takes 1.1x as long
     2048 @ 1000 repeats: 3.02e+00 vs. 3.40e+00 -> alternative takes 1.1x as long
     4096 @ 1000 repeats: 2.62e+00 vs. 3.29e+00 -> alternative takes 1.3x as long
     8192 @ 1000 repeats: 2.66e+00 vs. 3.28e+00 -> alternative takes 1.2x as long
    16384 @ 1000 repeats: 2.58e+00 vs. 3.35e+00 -> alternative takes 1.3x as long
    32768 @ 1000 repeats: 2.64e+00 vs. 3.28e+00 -> alternative takes 1.2x as long
    65536 @ 1000 repeats: 2.62e+00 vs. 3.27e+00 -> alternative takes 1.2x as long
   131072 @ 1000 repeats: 2.66e+00 vs. 3.34e+00 -> alternative takes 1.3x as long
   262144 @ 1000 repeats: 2.64e+00 vs. 3.38e+00 -> alternative takes 1.3x as long
   524288 @ 1000 repeats: 2.71e+00 vs. 3.37e+00 -> alternative takes 1.2x as long
