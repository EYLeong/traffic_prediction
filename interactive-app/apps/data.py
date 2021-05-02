import streamlit as st
import numpy as np
import pandas as pd

def app():
    st.title('Data')

    st.write('### Raw data collection')
    st.write("We used the Singapore Land Transport Authority (LTA) Traffic Speed Band dataset from [DataMall]("
             "https://datamall.lta.gov.sg/content/datamall/en.html). This is a dynamic dataset that is updated every "
             "5 minutes and is available for query through an API. To construct the dataset used to train our model, "
             "the API was queried every 5 minutes over the 27 day period from Mar 8 2021 to Apr 3 2021")
    st.write('Each query of the dynamic dataset gives a list of road links in Singapore together with other details '
             'such as their start and end coordinates and minimum and maximum speeds. An example of the information '
             'given for each road link is presented in the table below.')
    sample_data = {'LinkID': '103000000', 'Location': '1.3170142376560023 103.85298052044503 1.3166840028663076 '
                                                       '103.85259882242372',
                    'MaximumSpeed': '39',
                    'MinimumSpeed': '30',
                    'RoadCategory': 'E',
                    'RoadName': "KENT ROAD",
                    'SpeedBand': 4
                    }
    st.json(sample_data)

    st.write('### Refinement of Target Roads')
    st.write('The dataset collected from LTA consists of all the roads in Singapore. However, to keep the scope of '
             'this project feasible, we selected a few inter-connected roads in the Central Business District (CBD) '
             'area of Singapore. These roads were also chosen because they were found to have '
             'moderate to high levels of congestion, according to Google Maps.')

    st.write('### Description of Relevant Road Features')
    st.write('The main feature that we will be using for our regression problem would be the traffic speedbands. '
             'These speedbands describe the range of vehicle speeds (table below) that were observed on that particular '
             'stretch of road over a 5 minute interval. The goal would be to use past speedband data to predict the '
             'speedbands of roads in the network in a future timestep.')
    speedband_data = np.array([[1, '0 - 9'], [2, '10 - 19'], [3, '20 - 29'], [4, '30 - 39'], [5, '40 - 49'], [6, '50 - 59'], [7, '60 - 69'], [8, '>70']])
    df2 = pd.DataFrame(speedband_data, columns=['SpeedBand', 'Traffic Speed (km/hr)'])
    st.write(df2)

    st.write('Apart from the traffic speed bands themselves, we selected two other road features from the available '
             'data to consider when training our model. The first is the length of the road, which is determined by '
             'taking the distance between the start and end coordinates of each road. The second is the road '
             'category, which roughly describes the capacity and size of a road through a classification, as seen in '
             'the table below')
    roadcategory_data = np.array(
        [['A', 'Expressway'], ['B', 'Major Arterial Road'], ['C', 'Arterial Road'], ['D', 'Minor Arterial Road'], ['E', 'Small Road'], ['F', 'Slip Road'], ['G', 'Unknown']])
    df3 = pd.DataFrame(roadcategory_data, columns=['RoadCategory', 'Road Classification'])
    st.write(df3)
