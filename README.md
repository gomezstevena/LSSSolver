(Least-Squares Sensitvity Solver) LSSSolver:
============================================

Python Code for solving least-squares sensitivity problems for Ordinary Differential Equations (ODE's) both sequentially and in parallel. Used to compute sensitvities to Long-Time-Average quantities for periodic and chaotic ODE's.

A simple example is given in examples/testode.py

Dependencies:
-------------

* Python 2.7 (2.6?)
* NumPy
* SciPy
* mpi4py
* Optional: If using isotropic turbulence code install [anfft](https://code.google.com/p/anfft/) to use a faster FFT implementation (based on FFTW)
* Optional: matplotlib required for example code

Copyright:
==============
Copyright (C) 2013 Steven Gomez and Qiqi Wang


License:
=======

LSSSolver is licensed under the GPLv3 [license ](http://www.gnu.org/licenses/gpl-3.0.txt)