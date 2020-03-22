#!/usr/bin/env python3
"""Sampling from the n-sphere and n-ball

Implementations of the algorithms in [1]_.

References
----------

.. [1]: Voelker et al., 2017, Efficiently sampling vectors and coordinates
        from the n-sphere and n-ball
        http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
"""

import eagerpy as ep
import argparse
import time
import sys


def uniform_n_sphere(dummy: ep.Tensor, n: int) -> ep.Tensor:
    x = ep.normal(dummy, n + 1)
    r = x.norms.l2()
    s = x / r
    return s


def uniform_n_ball(dummy: ep.Tensor, n: int) -> ep.Tensor:
    s = uniform_n_sphere(dummy, n - 1)
    c = ep.uniform(dummy, 1)
    b = c.pow(1 / n) * s
    return b


def uniform_n_ball_alternative(dummy: ep.Tensor, n: int) -> ep.Tensor:
    s = uniform_n_sphere(dummy, n + 1)
    b = s[:n]
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "backend", choices=["numpy", "pytorch", "pytorch-gpu", "tensorflow", "jax"]
    )
    args = parser.parse_args()
    dummy = ep.utils.get_dummy(args.backend)

    for k in range(20):
        n = 2 ** k
        repeats = 1000

        t1 = time.time()
        for _ in range(repeats):
            uniform_n_ball(dummy, n)
        t1 = time.time() - t1

        t2 = time.time()
        for _ in range(repeats):
            uniform_n_ball_alternative(dummy, n)
        t2 = time.time() - t2

        print(f"{n:9} @ {repeats:3} repeats: {t1:.2e} vs. {t2:.2e} -> alternative takes {t2 / t1:.1f}x as long")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
