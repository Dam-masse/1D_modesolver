# 1D_modesolver
This repository contains a simple simple study of alghoritms and tools to solve 1D waveguide mode.
It should be intended as a pure learning exercise and no guarantee on the correctness or accuracy of the results should be made.

There are 3 files: 
- 1dMode_solver_matching.py 
- 1DMode_solver_matrixmethod.py
- 1DMode_solver_taichi.py

### 1dMode_solver_matching.py 
This method uses the bisection method to find the 0 difference between different equations assumed in the different part of the media.

The mode solver can be used changing the lines from 27 to 32 

disclaimer: this code was developed in my early python days(12/2019) so not all the best coding practices have been followed. 

#### Requires:
matplotlib
numpy

### 1dMode_solver_matrixmethod.py

This mode solver calculate the eingenmode of the representative sparse matrix using the inverse method. it is more efficient than the previous method but it will not find all the solution. 

#### Requires
matplotlib
numpy
scipy

### 1dMode_solver_taichi.py

The solution alghoritm is implemented as in the previous file. With the difference that in this one a library called Taichi is used for matrix multiplication, this allows C-style precompilation and execution of the program, leading to more than 10x faster computation. 

#### Requires
matplotlib
numpy
scipy
taichi
