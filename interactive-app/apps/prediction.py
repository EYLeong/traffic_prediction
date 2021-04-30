import streamlit as st
from utils.prediction import calculate
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

def app():
    st.title('Traffic Speed Prediction')
    st.write("Let's try predicting some traffic speed!")
    d = dirname(dirname(abspath(__file__)))
    print(d)
    dd = dirname(dirname(dirname(abspath(__file__))))
    print(dd)
    ddd = os.path.join(dd, 'truncated_data')
    print(ddd)

    # raw_trunc_dir = "./data/raw/trunc/"
    # process_dir = "./data/processed/"
    #
    # # overwrite = False means that the processing function will only run if the process data files do not exist
    # # overwrite = True => functions will run regardless
    # preprocessing_utils.processed(raw_trunc_dir, process_dir, overwrite=False)
    # A, X, metadata, cat2index, means, stds = preprocessing_utils.load(process_dir)
    filename = 'README.md'
    with open(filename, 'rb') as f:
        bytes = f.read()
    b64 = base64.b64encode(bytes).decode()
    download_href = f'<a href="data:file/zip;base64,{b64}" download=\'README.md\'>\
Click to download\
        </a>'
    st.markdown(download_href, unsafe_allow_html=True)
    sample_zip_path = os.path.join('data', 'sample', 'pred_data.zip')
    download_local_button(sample_zip_path, 'pred_data.zip', 'Download sample files')

    zip_file = st.file_uploader("Upload file", type="zip")
    if zip_file is not None:
        file_details = {'file_name': zip_file.name, 'file_type': zip_file.type}
        st.write(file_details)

        # Saving File
        saved_zip_path = os.path.join('data', zip_file.name)
        with open(saved_zip_path, 'wb') as f:
            f.write(zip_file.getbuffer())
            st.write(saved_zip_path)
        with ZipFile(saved_zip_path, 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()
            # extracting all the files
            print('Extracting all the files now...')
            unzip_path = os.path.join('data', 'temp')
            zip.extractall(path=unzip_path)
            print('Done!')

        st.success('File Uploaded! You can now predict traffic speeds')

    # ------------ Sidebar -----------
    st.sidebar.title("Model Options")
    st.sidebar.write("Select your prediction options below")
    st.sidebar.write("> There are a few components to this model to indicate how much past data you want to input, and how far in the future you want to predict until.")

    # Input Timesteps
    st.sidebar.subheader("Select Input Timesteps")
    st.sidebar.write("How much past data to input into the model for prediction")
    input_timestep_options = {1: "1 (5 minutes)", 2: "2 (10 minutes)", 3: "3 (15 minutes) - optimal"}
    num_input_timesteps = st.sidebar.selectbox("Number of Input Timesteps", options=list(input_timestep_options.keys()), format_func=lambda x: input_timestep_options[x])

    # Output Timesteps
    st.sidebar.subheader("Select Output Timesteps")
    st.sidebar.write("How far do you want to predict the traffic")
    output_timestep_options = {1: "1 (5 minutes) - optimal", 2: "2 (10 minutes)", 3: "3 (15 minutes)"}
    num_output_timesteps = st.sidebar.selectbox("Number of Output Timesteps", options=list(output_timestep_options.keys()),format_func=lambda x: output_timestep_options[x])

    # Slider
    #@st.cache()
    slider_val = st.sidebar.slider("Maximum number of iterations", 1, 5)
    # --------------------------------
    st.write(f"You selected this Input Timestep {num_input_timesteps}")
    st.write(f"You selected this Output Timestep {num_output_timesteps}")
    st.write(f"You selected this Slider Val {slider_val}")


    another_val = st.slider("Maximum number of iterationsss", 1, 5)
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
    field_1 = st.text_input('Your Name')
    field_2 = st.text_area("Your address")
    start_date = datetime.date(1990, 7, 6)
    date = st.date_input('Your birthday', start_date)

    if st.button("Classify", key='classify'):
        with st.spinner("Please wait for prediction results...."):
            st.subheader("Support Vector Machine (SVM) Results: ")
            st.write("Another val {}".format(calculate(another_val, another_val)))
            st.write("Summed value {}".format(calculate(num_input_timesteps, num_output_timesteps)))
            # st.success('File uploaded succesful!!')
            # st.error('Unable to Load the selected file.Please choose another!!')

    # --------------
    # Randomly fill a dataframe and cache it
    @st.cache(allow_output_mutation=True)
    def get_dataframe():
        return pd.DataFrame(
            np.random.randn(50, 20),
            columns=('col %d' % i for i in range(20)))

    df = get_dataframe()

    # Create row, column, and value inputs
    row = st.number_input('row', max_value=df.shape[0])
    col = st.number_input('column', max_value=df.shape[1])
    value = st.number_input('value')

    # Change the entry at (row, col) to the given value
    df.values[row][col] = value

    # And display the result!
    st.dataframe(df)
    # --------------

    my_expander = st.beta_expander("Expand", expanded=True)
    with my_expander:
        st.write("Hello! This is hidden content")


