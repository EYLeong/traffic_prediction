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
    # st.write('### Model Architecture')
    # image_path = os.path.join(dirname(dirname(abspath(__file__))), 'utils', 'model_architecture.jpg')
    # st.image(image=image_path, width=250)




