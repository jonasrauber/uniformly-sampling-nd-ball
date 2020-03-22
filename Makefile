run:
	./sample.py numpy | tee results/numpy.txt
	./sample.py pytorch | tee results/pytorch.txt
	./sample.py pytorch-gpu | tee results/pytorch-gpu.txt
	./sample.py tensorflow | tee results/tensorflow.txt
	./sample.py jax | tee results/jax.txt
