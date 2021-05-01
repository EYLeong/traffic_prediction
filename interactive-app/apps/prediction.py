import streamlit as st
from utils.prediction import calculate, predict
from components.buttons import download_local_button
import datetime
import pandas as pd
import numpy as np
import base64
import os
from zipfile import ZipFile
import torch
from os.path import dirname, abspath
import sys


import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import contextily as ctx
import matplotlib.pyplot as plt

def app():
    st.beta_set_page_config(page_title='your_title', page_icon="hi")
    st.title('Traffic Speed Prediction')
    st.write("> Let's try predicting some traffic speed!")
    current_dir = dirname(dirname(abspath(__file__)))
    d = dirname(dirname(abspath(__file__)))
    print(d)
    dd = dirname(dirname(dirname(abspath(__file__))))
    print(dd)
    ddd = os.path.join(dd, 'truncated_data')
    print(ddd)

    # Sidebar --------------------------------------------------------------------------------
    st.sidebar.title("Model Options")
    st.sidebar.write("Select your prediction options below")
    st.sidebar.write(
        "> There are a few components to this model to indicate how much past data you want to input, and how far in the future you want to predict until.")

    # Input Timesteps
    st.sidebar.subheader("Select Input Timesteps")
    st.sidebar.write("How much past data to input into the model for prediction")
    # input_timestep_options = {1: "1 (5 minutes)", 2: "2 (10 minutes)", 3: "3 (15 minutes) - optimal"}
    input_timestep_options = {7: "7 (35 minutes) - default"}
    num_input_timesteps = st.sidebar.selectbox("Number of Input Timesteps", options=list(input_timestep_options.keys()),
                                               format_func=lambda x: input_timestep_options[x])

    # Output Timesteps
    st.sidebar.subheader("Select Output Timesteps")
    st.sidebar.write("How far do you want to predict the traffic speeds")
    output_timestep_options = {1: "1 (5 minutes)", 2: "2 (10 minutes)", 3: "3 (15 minutes)",
                               4: "4 (20 minutes) - default"}
    num_output_timesteps = st.sidebar.selectbox("Number of Output Timesteps",
                                                options=list(output_timestep_options.keys()),
                                                format_func=lambda x: output_timestep_options[x], index=3)
    # Slider
    # @st.cache()
    # slider_val = st.sidebar.slider("Maximum number of iterations", 1, 5)
    # --------------------------------

    # another_val = st.slider("Maximum number of iterationsss", 1, 5)
    # field_1 = st.text_input('Your Name')
    # field_2 = st.text_area("Your address")
    # start_date = datetime.date(1990, 7, 6)
    # date = st.date_input('Your birthday', start_date)
    # ------------------------------------------------------------------------------------------------

    sample_zip_path = os.path.join(current_dir, 'data', 'sample', 'input.zip')
    # st.write(sample_zip_path)
    # st.write(os.getcwd())
    # st.write(d)
    # st.write(dd)
    st.write("### 1. Download Sample Input Files")
    st.write("Here's a sample input file with the format that is required for the model prediction. You can download this, change the data and upload the zip file below.")
    download_local_button(sample_zip_path, 'input.zip', 'Download files')
    st.write("___________________________")
    st.write("### 2. Upload Input Files")
    st.write("Please upload the zip file with the correct format below")
    zip_file = st.file_uploader("Upload file", type="zip")
    if zip_file is not None:
        file_details = {'file_name': zip_file.name, 'file_type': zip_file.type}
        # st.write(file_details)

        # Saving File
        saved_zip_path = os.path.join(current_dir, 'data', zip_file.name)
        with open(saved_zip_path, 'wb') as f:
            f.write(zip_file.getbuffer())

        with ZipFile(saved_zip_path, 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()
            # extracting all the files
            print('Extracting all the files now...')
            unzip_path = os.path.join(current_dir, 'data', 'raw')
            zip.extractall(path=unzip_path)
            print('Done!')

        st.success('File Uploaded! You can now predict traffic speeds')

        # Predict Traffic Speeds here
        if st.button("Predict Traffic Speeds", key='predict'):
            with st.spinner("Please wait for prediction results...."):
                st.write('## Results')
                results, A, X, metadata = predict(num_timesteps_input=7, num_timesteps_output=4)

                # Display Metadata
                st.write('#### Metadata')
                metadata_expander = st.beta_expander("Click to expand", expanded=False)
                with metadata_expander:
                    st.write("Here's the metadata of the input data you have uploaded")
                    df = pd.DataFrame(metadata).transpose()
                    st.write(df)

                # Display Results
                st.write('#### Predictions')
                predictions_expander = st.beta_expander("Click to expand", expanded=False)
                with predictions_expander:
                    def loc_to_linestring(loc):
                        coordArr = loc.split()
                        coordArr = [float(coord) for coord in coordArr]
                        return LineString([coordArr[1::-1], coordArr[3:1:-1]])

                    def plotGeoPerformance(metadata, speedbands):
                        df = pd.DataFrame(metadata).transpose()
                        df["speedbands"] = speedbands
                        loc = df["start_pos"] + " " + df["end_pos"]
                        linestrings = loc.apply(loc_to_linestring)
                        gdf = gpd.GeoDataFrame(df, geometry=linestrings, crs="EPSG:4326")
                        gdf = gdf.to_crs('EPSG:3857')
                        fig, ax = plt.subplots(figsize=(10, 10))
                        gdf.plot(ax=ax, column="speedbands", legend=True, cmap="OrRd",
                                 legend_kwds={'label': 'speedbands'})
                        ax.set_xlabel("Longitude")
                        ax.set_ylabel("Latitude")
                        ctx.add_basemap(ax)

                    timestep_speedbands = results.reshape(predicted_denorm.shape[2], predicted_denorm.shape[1])
                    st.write(plotGeoPerformance(metadata, timestep_speedbands[0]))
                    st.write("Here are the prediction results")
                    results = results[:, :, :num_output_timesteps]
                    st.write(results)




                # st.success('File uploaded succesful!!')
                # st.error('Unable to Load the selected file.Please choose another!!')








