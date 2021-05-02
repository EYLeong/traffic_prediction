import streamlit as st
from os.path import dirname, abspath
import os

def app():
    st.title('Model')

    st.write('### Choice of Model')
    st.write('We used a **Spatio-Temporal Graph Convolutional Networks (STGCN)** framework to tackle this time series '
             'prediction problem, as road traffic congestion prediction is a complex problem that has both spatial '
             'and temporal qualities. For example, if a particular road faces a congestion (for various reasons such '
             'as peak hour, or car crash), the surrounding connected nodes (roads) will face a similar increment in '
             'congestion levels that cascades with time (dependent on various factors such as hop-distance and other '
             'potential node features). From our literature reviews, STGCNs have shown good performance against '
             'existing models on various prediction problems including traffic forecasting, and have proven to '
             'capture spatio-temporal correlations well with very fast training speeds. Hence, our group decided to '
             'experiment and utilize STGCN for our deep learning project.')

    st.write('### Model Architecture')
    st.write('We heavily adopted the STGCN architecture from the novel STGCN deep learning framework proposed by the '
             'paper [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic '
             'Forecasting](https://arxiv.org/abs/1709.04875)')
    # image_path = os.path.join(dirname(dirname(abspath(__file__))), 'utils', 'model_architecture.jpg')
    image_path = os.path.join('utils', 'model_architecture.jpg')
    st.image(image_path)
    st.write("""
    In the figure above, the overall architecture contains 2 ST-Conv blocks followed by an output block, which itself is composed of a temporal block and a last fully connected layer.
    The ST-Conv block combines the spatial (graph) and temporal convolutions to capture the spatio-temporal behaviors. (View our full report for further details on the implementation of the model).
    
    For the final Fully Connected layer, the size of out-features will be the number of output steps in our parameters, which follows our regression problem design.
    """)

    st.write('### Model Training')
    st.write('The Mean Squared Error (MSE or L2 Loss) is used as our loss function in training our model')

    st.write('### Hyperparameter Tuning')
    st.write('In order to fine-tune and improve the performance of our model, we conducted an automatic hyperparameter tuning using Optuna framework. In brief, through Optuna we used the Tree Parzen Estimator which uses Bayesian reasoning to construct the surrogate model and select the next hyperparameters using Expected Improvement. The detailed code can be viewed in our [Jupyter Notebook](https://github.com/EYLeong/traffic_prediction/blob/master/Hyperparameter%20Tuning.ipynb)')

    st.write('_______')
    st.write('View our full report (to be disclosed at a later date) for in-depth explanation of our model and implementations.')




