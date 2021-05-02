import streamlit as st

def app():
    st.title('Home')

    # About Project
    st.write('### About our project')
    st.write('Our project aims to conduct traffic congestion prediction using a Spatial-Temporal Graph Convolutional Network (STGCN) trained on Singapore traffic speed dataset by Land Transport Authority (LTA). Using the STGCN, we further conducted various analyses on its effectiveness across time and across geographical regions (different roads). ')
    st.write('View our [Github Repository](https://github.com/EYLeong/traffic_prediction) for further implementation details')

    st.write('### Problem & Scope of Project')
    st.write('We aim to tackle the problem of traffic congestion prediction in Singapore. This will help improve the '
             'overall accuracy for trip planners for urban transportation in Singapore by allowing algorithms to take '
             'into account the traffic congestions during the planned trip. Ideally, this can lead to improved road '
             'utilisation and commuter satisfaction.')
    st.write('Our project can hence be designed as a regression problem, where given past data of speed-bands per '
             'road per given timesteps (further explained in the ‘Data’ section), our STGCN will predict the '
             'speed-band for the predetermined output timesteps. ')
    st.write('As a special note, we designed this to be a regression problem rather than a classification problem ('
             'which will involve correctly classifying these speed bins), as traffic speed is inherently continuous '
             'and we wanted our model to capture this quantification of impact (of how much the information from the '
             'spatio-temporal graph embedding has resulted in contribution to the model’s decision). We recognise '
             'that this is a limitation of our project (that we are not using continuous speed data), and will be '
             'addressed again in the ‘Limitations’ section.')
    st.write('________')

    # Team Members
    st.write('### Team Members')
    st.write("""
            - Soo Han Son
            - Cornelius Yap
            - Leong Enyi
            - Mario Josephan Kosasih
            - Daryll Wong
            """)



