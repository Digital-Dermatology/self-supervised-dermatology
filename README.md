# Towards Small-Scale Foundation Models for Digital Dermatology (ML4H'24)
This repository contains the code to reproduce all evaluations in the paper "Towards Foundation Models for Digital Dermatology".

## Usage
Run `make` for a list of possible targets.

## Installation
Run this command for installation
`make install`

## Reproducibility of the Paper
To reproduce our experiments, we list the detailed comments needed for replicating each experiment below.
Note that our experiments were run on a DGX Workstation 1.
If less computational power is available, this would require adaptations of the configuration file.

### Experiment: Frozen Evaluation

kNN evaluation (i.e. Figure 1 and Table 2):
> python -m src.knn_figs --knn_performance

Linear evaluation (i.e. Table 2):
> python -m src.knn_figs --linear_evaluation

kNN / linear few-shot evaluation (i.e. Figure 2):
> python -m src.knn_figs

## Code and test conventions
- `black` for code style
- `isort` for import sorting
- docstring style: `sphinx`
- `pytest` for running tests

### Development installation and configurations
To set up your dev environment run:
```bash
pip install -r requirements.txt
# Install pre-commit hook
pre-commit install
```
