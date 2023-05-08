# DiNoMat

**Di**fferentiable **No**n-Local **Mat**erial Homogenization

This repository contains code for the paper "On the Physical Significance of Non-Local Material Parameters in Optical Metamaterials".

## Installation

DiNoMat's only real dependency is [JAX](https://github.com/google/jax) and can be installed into an existing Python environment via:
```
pip install "dinomat @ git+https://github.com/tfp-photonics/dinomat"
```
The examples (currently only parameter retrieval) have additional dependencies. If you want to include those, you can install DiNoMat via:
```
pip install "dinomat[examples] @ git+https://github.com/tfp-photonics/dinomat"
```
