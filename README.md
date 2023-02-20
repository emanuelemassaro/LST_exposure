# Reducing extreme heat exposure in cities through spatial analysis of urban greening
### E. Massaro, G. Duveiller, R. Schifanella, L. Caporaso, M. Piccardo, H. Taubenböck and A. Cescatti
#### contact: ema.massaro@gmail.com

This repository contains the codes and the data used for producing the results of the paper. The codes are in form of of jupyter notebooks.

For each city we collected information of land surface properties from the [Google Eart Engine](https://earthengine.google.com/) and the population from the [EU JRC Global Human Settlement Layer](https://ghsl.jrc.ec.europa.eu/). 

In this repository we can find python notebooks and python scripts to run the main analysis for our results. 

- Notebooks: in this folder we can find the following
  - hotDays.ipynb: this notebook allow to download from the [Google Eart Engine](https://earthengine.google.com/) the days and nights over the thresholds for all the cities
  - NDBI_GEE_Tiff.ipynb: this notebook allow to download from the [Google Eart Engine](https://earthengine.google.com/) the NDBI values for all the cities
  - NDVI_GEE_Tiff.ipynb: this notebook allow to download from the [Google Eart Engine](https://earthengine.google.com/) the NDVI values for all the cities
- Codes: in this folder we can find the following
  - modelTrainingValidation.py: it allow us to run the regression models in the training validation phase
