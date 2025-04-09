# Bounds in Wasserstein Distance for Locally Stationary Functional Time Series
This repository provides the Python codes used for the numerical experiments described in the paper

> *Bounds in Wasserstein Distance for Locally Stationary Functional Time Series*
> 
> by Jan N. Tinio, Mokhtar Z. Alaya and Salim Bouzebda
> arXiv link: 
> 
## Introduction
A brief introduction about the folders and files:
* `data/`: locally stationary functional real-world datasets.

* `src/`: methods and implementations.
    * `kernels.py`: 
    * `utils.py`: standard kernels in torch-mode calls.

* `models/`: python files containing all the used illustrated models.

* `notebooks/`: notebooks for implementing experiments on synthetic and real-world data.

## Requirements
Python: > 3.10
Pytorch
Sckit-Learn

## Reproducibility
* You can run the provided notebooks to reproduce the results.

## Citation
If you use this toolbox in your research and find it useful, please cite:
```
@article{tinioetal2025,
  title={Bounds in Wasserstein Distance for Locally Stationary Functional Time Series},
  author={J.N. Tinio, M. Z. Alaya and S. Bouzebda},
  journal={arXiv preprint },
  year={2025}
}
```
