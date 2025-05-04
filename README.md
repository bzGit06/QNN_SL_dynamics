# QNN training dynamics in supervised learning
This repository contains the official Python implementation of [*Quantum-data-driven dynamical transition in quantum learning*](https://arxiv.org/abs/2410.01955), an article by [Bingzhi Zhang](https://sites.google.com/view/bingzhi-zhang/home), [Junyu Liu](https://sites.google.com/view/junyuliu/main), [Liang Jiang](https://pme.uchicago.edu/group/jiang-group), and [Quntao Zhuang](https://sites.usc.edu/zhuang).

## Citation
```
@misc{zhang2024quantum,
      title={Quantum-data-driven dynamical transition in quantum learning}, 
      author={Zhang, Bingzhi and Liu, Junyu and Jiang, Liang and Zhuang, Quntao},
      year={2024},
      eprint={2410.01955},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## Prerequisite
The simulation of quantum circuits is performed via the [TensorCircuit](https://tensorcircuit.readthedocs.io/en/latest/#) package with [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) backend. Use of GPU is not required, but highly recommended. 

Additionally, the packages [`opt_einsum`](https://optimized-einsum.readthedocs.io/en/stable/) is used for speeding up certain evaluation, and [Pennylane](https://docs.pennylane.ai/en/stable) is needed for experiments on IBM Quantum device.

The package [`RTNI`](https://github.com/MotohisaFukuda/RTNI) is used to derive analytical unitary ensemble average result.

## File Structure
The file `RPA_jax_orth.ipynb` contains all numerical simulation with random Pauli ansatz. The file `restrict_haar.ipynb` contains simulation codes for restricted Haar ensemble. The file `restrict_haar_theory.py` includes functions to calculate theoretical results to calculate restricted Haar ensemble. The folder `restrictHaar` includes mathematica codes to derive restricted Haar ensemble result for QNTK and dQNTK.
