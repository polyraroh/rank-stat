# Rank-stat
A library for computation of R-estimators and visualizing the nice geometry of it.

Currently, there is little to no documentation. Some is available as docstrings in the files.

## Usage
The main file is `rs.py`. It provides class `rankStat`, which provides all the functionalities. The following code snippet creates a random 2D instance, evaluates estimator using WoA algorithm (an exact LP-based algorithm) and draws the problem.
```
instance = rs.rankStat(10,2)
instance.WoA()
instance.visualize()
```

## Requirements
Currently, `gurobipy` module and `gurobi` itself is required for LP solving. TeX distribution with TikZ is necessary for visualizations.

Aside from these, there are no uncommon packages that would be hard to obtain.
