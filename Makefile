.PHONY: run
run:
	./sample.py numpy | tee results/numpy.txt
	./sample.py pytorch | tee results/pytorch.txt
	./sample.py pytorch-gpu | tee results/pytorch-gpu.txt
	./sample.py tensorflow | tee results/tensorflow.txt
	./sample.py jax | tee results/jax.txt

.PHONY: readme
readme:
	-rm README.rst
	cp README.raw.rst README.rst
	echo "\nNumPy\n-----\n\n.. code-block:: python\n" >> README.rst
	cat results/numpy.txt >> README.rst
	echo "\nPyTorch\n-------\n\n.. code-block:: python\n" >> README.rst
	cat results/pytorch.txt >> README.rst
	echo "\nPyTorch (GPU)\n-------------\n\n.. code-block:: python\n" >> README.rst
	cat results/pytorch-gpu.txt >> README.rst
	echo "\nTensorFlow (GPU)\n----------------\n\n.. code-block:: python\n" >> README.rst
	cat results/tensorflow.txt >> README.rst
	echo "\nJAX (GPU)\n---------\n\n.. code-block:: python\n" >> README.rst
	cat results/jax.txt >> README.rst
