# Optimal Algorithms for Resolvent Splitting (oars)

This package provides tools for building custom proximal splitting algorithms.

## Using and citing the package

This code comes jointly with the following [`reference`](https://arxiv.org/pdf/2407.16159.pdf):

    Bassett, R. L., & Barkley, P. (2024). Optimal Design of Resolvent Splitting Algorithms. 

When using the package in a project, please use this Bibtex entry:

```
@article{bassett2024optimaldesignresolventsplitting,
      title={Optimal Design of Resolvent Splitting Algorithms}, 
      author={Robert L Bassett and Peter Barkley},
      journal={arXiv preprint arXiv:2407.16159},
      year={2024},
      url={https://arxiv.org/abs/2407.16159}, 
}
```

## Installation

After cloning the repository, the package can be installed with
`pip install .`

## Quick Start

See the [quick start guide]([docs/source/quickstart.rst](https://oars.readthedocs.io/en/latest/quickstart.html) in the documentation.

## Documentation

You can find the [documentation](https://oars.readthedocs.io/) for this package on readthedocs.

## Reproducibility

If you would like to run the experiments from [our paper](https://arxiv.org/pdf/2407.16159.pdf), use `pip install .[all]` and run the `paper.ipynb` notebook in the examples directory.
