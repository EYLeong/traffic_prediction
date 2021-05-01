import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, prediction

st.set_page_config(page_title="Traffic Prediction ⦾ ⦿ ⦾ ⦿ ⦾ ⦿ ⦾ ⦿ ⦾ ⦿", page_icon="🚦")

TRAFFICLIGHT_EMOJI_URL = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/271/vertical-traffic-light_1f6a6.png"
CAR_EMOJI_URL = "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/271/automobile_1f697.png"

col1, col2, _ = st.beta_columns([1, 1, 6])
col1.image(TRAFFICLIGHT_EMOJI_URL, width=80)
col2.image(CAR_EMOJI_URL, width=80)
st.markdown("""
# Traffic Prediction
Deep Learning project to predict traffic speeds using *Spatio-Temporal Graph Convolutional Network (STGCN)*  
""")

# Add all your application here
app = MultiApp()
app.add_app("Home", home.app)
app.add_app("Data", data.app)
app.add_app("Model", model.app)
app.add_app("Prediction", prediction.app)
# The main app
app.run()
