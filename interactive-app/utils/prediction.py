import streamlit as st
import pandas as pd
import numpy as np
import time


@st.cache(persist=True)
def load_data(data):
    return data


@st.cache(persist=True)
def split_data(data):
    return data

#@st.cache(suppress_st_warning=True)
def calculate(num_input, num_output):
    time.sleep(2)
    return num_input + num_output
