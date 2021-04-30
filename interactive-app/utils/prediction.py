import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from os.path import dirname, abspath
import sys
sys.path.append('..')
from preprocessing_utils import processed


@st.cache(persist=True)
def load_data(data):
    return data


@st.cache(persist=True)
def split_data(data):
    return data

#@st.cache(suppress_st_warning=True)
def calculate(num_input, num_output):
    time.sleep(2)
    dd = dirname(dirname(dirname(abspath(__file__))))
    ddd = os.path.join(dd, 'truncated_data')
    print("here",dd, ddd)
    print("syspath", sys.path[0])
    return num_input + num_output
