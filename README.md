# SUTD 50.039 Theory and Practice of Deep Learning Big Project

## Traffic Congestion Prediction using Spatio-Temporal Graph Convolutional Networks

### Project Summary
Our project aims to conduct traffic congestion prediction using a Spatial-Temporal Graph Convolutional Network (STGCN) trained on Singapore traffic speed dataset by Land Transport Authority (LTA). Using the STGCN, we further conducted various analyses on its effectiveness across time and across geographical regions (different roads). 


### Github Directory Guide
`data/processed:` Contains the traffic dataset used in this project

`interactive-app:` Code for deploying the model on the web-app

`saved-models:` Contains the weights for the models

`Analysis.ipynb:` Notebook to generate the analysis used in the ***Results and Analysis*** section of the report

`Arima.ipynb:` Notebook to train the alternative model used in the ***Comparison with Other Models*** section of the report

`Data Visualisation.ipynb:` Notebook to generate the visualisation images used in the ***Data Visualisations*** section of the report

`Hyperparameter Tuning.ipynb:` Notebook used to test the different hyperparameter settings for the model. The results were used and elaborated further in the ***Hyperparameter Tuning*** section of the report

`STGCN Timestep Comparison.ipynb:` Notebook used to test the effects of using different input and output timesteps for the model, as described in the ***Impact of Input and Output Timesteps*** section of the report

`STGCN Traffic.ipynb:` Main Notebook used to train the STGCN model

`analysis.py:` Supporting code that was used in the `Analysis.ipynb` notebook

`model.py:` Code for our STGCN model

`model_utils.py:` Supporting code that is used to train the model

`preprocessing_utils.py:` Code for preprocessing the traffic data

`visualisation.py:` Supporting code that was used in the `Data Visualisation.ipynb` notebook