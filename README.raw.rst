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
