# Parallel SGD

A simple implementation of "Parallelized Stochastic Gradient Descent" (Zinkevich et al., 2010) with C++ and OpenMP. 

An example of Linear Regression is demonstrated by encoding the closed form of objective function (least squres) and gradient.

### How to use?

First download the source code from the repo

```
cd src

make

./lin_reg
```

To clean up the `.o` and executable file, use

``` 
make clean 
```
