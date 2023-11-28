# DiNoMat

**Di**fferentiable **No**n-Local **Mat**erial Homogenization

This repository contains code for the paper "On the Physical Significance of Non-Local Material Parameters in Optical Metamaterials" ([NJP](https://iopscience.iop.org/article/10.1088/1367-2630/ad1010)]).

## Installation

DiNoMat's only real dependency is [JAX](https://github.com/google/jax) and can be installed into an existing Python environment via:
```
pip install "dinomat @ git+https://github.com/tfp-photonics/dinomat"
```
The examples (currently only parameter retrieval) have additional dependencies. If you want to include those, you can install DiNoMat via:
```
pip install "dinomat[examples] @ git+https://github.com/tfp-photonics/dinomat"
```

## Citing

If you use this code or associated data for your research, please cite:

```bibtex
@article{venkitakrishnan2023physical,
  title = {On the Physical Significance of Non-Local Material Parameters in Optical Metamaterials},
  author = {Venkitakrishnan, Ramakrishna and Augenstein, Yannick and Zerulla, Benedikt and Goffi, Fatima and Plum, Michael and Rockstuhl, Carsten},
  year = 2023,
  journal = {New Journal of Physics},
  doi = {10.1088/1367-2630/ad1010}
}
```
