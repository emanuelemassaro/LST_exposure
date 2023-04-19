# Spatially-optimized urban greening for reduction of population exposure to land surface temperature extremes
### E. Massaro, R. Schifanella, L. Caporaso, M. Piccardo, H. Taubenböck, A. Cescatti and G. Duveiller
#### contact: ema.massaro@gmail.com

This repository contains the codes and the data used for producing the results of the paper.

For each city we collected information of land surface properties from the [Google Earth Engine](https://earthengine.google.com/) and the population from the [EU JRC Global Human Settlement Layer](https://ghsl.jrc.ec.europa.eu/). 

In this repository we can find python notebooks and python scripts to run the main analysis for our results and the response to the reviwers.

- [crossValidation.ipynb](notebooks/crossValidation.ipynb): this notebook generates the list of the cities for the cross validation
- [data.ipynb](notebooks/data.ipynb): this notebook allows the user to download the data from the Google Earth Engine
- [Figure1.ipynb](notebooks/Figure1.ipynb): contains the link to the data and the codes to generate the Figure 1 of the main text
- [Figure2.ipynb](notebooks/Figure2.ipynb): contains the link to the data and the codes to generate the Figure 2 of the main text
- [Figure3.ipynb](notebooks/Figure3.ipynb): contains the link to the data and the codes to generate the Figure 3 of the main text
- [Figure4.ipynb](notebooks/Figure4.ipynb): contains the link to the data and the codes to generate the Figure 4 of the main text
- [Figure5.ipynb](notebooks/Figure5.ipynb): contains the link to the data and the codes to generate the Figure 5 of the main text
- [SI_figures.ipynb](notebooks/SI_figures.ipynb): contains the link to data and the codes to generate the figures for the Supporting Information.
- [SUHI.ipynb](notebooks/SUHI.ipynb): this notebook estimates the SUHI from the Google Earth Engine
- [testFinal.ipynb](notebooks/testFinal.ipynb): this notebook run the test phases of the regression models
- [trainingValidation.ipynb](notebooks/trainingValidation.ipynb): this notebook run the training validation phases of the regression models
- [tresholds.ipynb](notebooks/tresholds.ipynb): this notebook estimates the LST tresholds from the Google Earth Engine

* All the collected data and the generated data are the folder [data_revision](data_revision). 
* All the python fuctions and codes are in the folder [codes](codes). 
* All the Figures are in the folder [figures_revision](figures_revision). 
