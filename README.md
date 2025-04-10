# Bounds in Wasserstein Distance for Locally Stationary Functional Time Series
This repository provides the Python codes used for the numerical experiments described in the paper

> *Bounds in Wasserstein Distance for Locally Stationary Functional Time Series*
> 
> by Jan N. Tinio, Mokhtar Z. Alaya and Salim Bouzebda
> arXiv link: https://arxiv.org/abs/2504.06453
> 
## Introduction
A brief introduction about the folders and files:
* `data/`: locally stationary functional real-world datasets.

* `src/`: methods and implementations.
    * `kernels.py`: 
    * `utils.py`: standard kernels in torch-mode calls.

* `models/`: python files containing all the used illustrated models.

* `notebooks/`: notebooks for implementing synthetic and real-world data experiments.

## Requirements
Python: > 3.10
Pytorch
Sckit-Learn

## Reproducibility
* You can just run the notebooks provided to reproduce the results.

## Citation
If you use this toolbox in your research and find it useful, please cite:
```
@article{tinioetal2025,
  title={Bounds in Wasserstein Distance for Locally Stationary Functional Time Series},
  author={J.N. Tinio, M. Z. Alaya and S. Bouzebda},
  journal={arXiv preprint },
  doi = {https://doi.org/10.48550/arXiv.2504.06453},
  year={2025}
}
```
