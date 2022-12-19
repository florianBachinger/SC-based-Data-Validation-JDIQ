# SC-DataValidation-Datasets
This project was used to create the experiment data for our publication. The scripts of this repository can be used to repeat and check our experiments or to further analyze the results.

*Summary:*
```
We investigate the shape properties of selected equations from the Feynman lectures. Each shape property can be formulated as a constraint for shape-constrained regression. One example for such a shape constraint from the Feynman2 equation is, that it is monotonically decreasing over sigma. Subsequently, we add random normal noise to the target and we generate different errors that deliberatly violate the known constraints. 

In the subsequent data validation phase we train prediction models using shape-constrained regression algorithms. As all errors are designed to violate the constraints, the models are restricted from fitting to these erroneous shapes. Our data validation approach utilizes the training error of prediction models to distinguish between erroneous and unmodified datasets. 
```
The Feynman Equations: https://space.mit.edu/home/tegmark/aifeynman.html.

Our analysis of the shape properties of each Feynman equation: https://github.com/florianBachinger/FeynmanEquations-Python


## Replication of analysis
Our results can be further investigated without repeating the grid search and model training, which both require a Mosek license and installation. Simply extract the data from `data/[uni/multi]variate/*.zip` and skip the scripts `/[uni/multi]variate/1-generate-multivariate-gridsearch.py`, `/[uni/multi]variate/2-execute-gridsearch.cmd`, `/[uni/multi]variate/4-generate_multivariate_inkl_error`, `/[uni/multi]variate/5-execute-validation-modeling.cmd`.

The scripts `/[uni/multi]variate/99-*` provide analysis and figures for all or individual datasets for debugging or in-depth analysis. 


## Replication of experiments
All scripts in this project are executed from the context of this root folder (the location of this README.md). 
All relative paths are defined from this location. All python scripts are executed from this context.

### Prerequisites
1. obtain a Mosek solver license (https://www.mosek.com/) 
1. install Mosek 9.2.49 (alternatively, upgrade the NuGet dependency in `/shared_packages/SCPRRunner/`)
1. install the python dependencies as specified by pip freeze in `pip-requirements.txt` (preferably a local environment with virtualenv) 
1. build a release version of the runner application in the default `/shared_packages/SCPRRunner/bin/Release/net6.0/`

### Repeat univariate experiments
1. execute the python and cmd scripts of the folder `/univariate_scripts/` in order


### Repeat multivariate experiments
1. execute the python and cmd scripts of the folder `/multivariate_scripts/` in order

