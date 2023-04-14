# DSTAR : **D**ft & STructure free Active motif based Representation

This repository contains codes and notebooks used to create results in our paper.

For more details, check out this paper (https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00726).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Usage](#usage)

## Prerequisites
* Generalized Adsorption Simulator for Python ([GASpy](https://github.com/ulissigroup/GASpy))

* Additional packages required for gaspy enviroment:
- [scikit-learn](http://scikit-learn.org/stable/) (0.24.2)
- [pymatgen](http://pymatgen.org) (2021.3.3)

## Usage
### Machine Learning Model
See DSTAR_Guide_.pdf to use ML model.

### CO2RR Application
To reproduce DSTAR application for CO2RR, please refer to three ipynb files in `application/CO2RR/`. Each ipynb file will do the following: 

`01_Scaler.ipynb` will generates scaler to normalize the productivity.

`02_Heatmap.ipynb` will visualize productivty heatmap.

`03_Selectivity_Plot.ipynb` will plot the productivity for each product corresponding to applied potential, composition and coordination number.

More details are in each ipynb files.

### Data for CO2RR Application
All predicted CO* / H* / OH* binding energies and coordination number of prototype surface used for application can be found in `application/CO2RR/data/energy` and `application/CO2RR/data/CN_dict.pkl`, and the boundary conditions of selectivity map are in `application/CO2RR/script/condition.py`.

