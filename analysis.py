import torch.nn as nn

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import contextily as ctx
import matplotlib.pyplot as plt

def rmse_per_link(predicted, actual):
    '''
    Calculates the RMSE of the speedbands for each road separately
    -----------------------------
    :params:
        list (3-dimensions of samples, roads, output timesteps) predicted: The predicted speedbands
        list (3-dimensions of samples, roads, output timesteps) actual: The actual speedbands
    -----------------------------
    :returns:
        list: rmse for each road
    '''
    rmses = []
    for i in range(predicted.shape[1]):
        linkPreds = predicted[:,i,:]
        linkActs = actual[:,i,:]
        rmse = nn.MSELoss()(linkPreds, linkActs).sqrt()
        rmses.append(rmse.item())
    return rmses

def loc_to_linestring(loc):
    '''
    Utility function to create shapely linestrings from road location data
    -----------------------------
    :params:
        string loc: Location data of format (start_lat start_lon end_lat end_lon)
    -----------------------------
    :returns:
        LineString: Corresponding LineString representing road
    '''
    coordArr = loc.split()
    coordArr = [float(coord) for coord in coordArr]
    return LineString([coordArr[1::-1], coordArr[3:1:-1]])

def plot_geo_performance(metadata, rmses):
    '''
    Generates a geographical map of the roads color coded with their corresponding RMSE
    -----------------------------
    :params:
        dict metadata: Metadata linking road index to other road information
        list rmses: RMSE of each road
    -----------------------------
    :returns:
        None
    '''
    df = pd.DataFrame(metadata).transpose()
    df["RMSE"] = rmses
    loc = df["start_pos"] + " " + df["end_pos"]
    linestrings = loc.apply(loc_to_linestring)
    gdf = gpd.GeoDataFrame(df, geometry=linestrings, crs="EPSG:4326")
    gdf = gdf.to_crs('EPSG:3857')
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, column="RMSE", legend=True, cmap="OrRd", legend_kwds={'label': 'RMSE'})
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ctx.add_basemap(ax)
    plt.show()
    
def plot_pred_actual(predicted, actual, idx, ts):
    '''
    Generates a plot of the predicted vs actual speedbands across all test samples for a specific road
    -----------------------------
    :params:
        list (3-dimensions of samples, roads, output timesteps) predicted: The predicted speedbands
        list (3-dimensions of samples, roads, output timesteps) actual: The actual speedbands
        int idx: The index of the road that should be plotted
        int ts: The index of the timestep that should be plotted (0 is 5 min, 1 in 10 min, 2 is 15 min, etc)
    -----------------------------
    :returns:
        None
    '''
    fig, ax = plt.subplots()
    ax.plot(actual[:,idx,ts], label="Actual")
    ax.plot(predicted[:,idx,ts], label="Predicted")
    ax.set_ylabel("Speedband")
    ax.set_xlabel("Timestep")
    ax.legend()
    ax.set_title("{} minutes".format((ts+1) * 5))
    plt.show()
    
def rmse_per_time(predicted, actual, timestamps, timeidx = 0):
    '''
    Calculates the RMSE of the speedbands for each time period
    -----------------------------
    :params:
        list (3-dimensions of samples, roads, output timesteps) predicted: The predicted speedbands
        list (3-dimensions of samples, roads, output timesteps) actual: The actual speedbands
        dict timestamps: Metadata linking index of timestamp to date string
        int timeidx: Index of the period of time to be analysed. Date strings are of format DAYOFWEEK_MTH_DAY_YEAR_H_M_S, hence a timeidx of 0 means splitting by day, 4 means splitting by hour, etc.
    -----------------------------
    :returns:
        dict: Dictionary of time period to RMSE
        dict: Dictionary of time period to how many times it is represented in the test set
    '''
    end = len(timestamps) - predicted.shape[2]
    start = end - len(predicted) + 1
    timecounts = {}
    time_total_se = {}
    for i in range(start, end + 1):
        idx = i - start
        for j in range(predicted.shape[2]):
            date_time = timestamps[str(i+j)]
            time = date_time.split("_")[timeidx]
            if time not in timecounts:
                timecounts[time] = 0
                time_total_se[time] = 0
            timecounts[time] += 1
            time_total_se[time] += nn.MSELoss()(predicted[idx,:,j], actual[idx,:,j])
    for k,v in timecounts.items():
        time_total_se[k] = (time_total_se[k]/v).sqrt().item()
    return time_total_se, timecounts

def plot_rmse_time(rmses, xlabel="Time Period"):
    '''
    Generates a plot of the RMSE of all roads across different time periods
    -----------------------------
    :params:
        dict rmses: Dictionary of time period to RMSE
        string xlabel: Label of the time period e.g. Day, Hour, etc
    -----------------------------
    :returns:
        None
    '''
    fig, ax = plt.subplots()
    x = []
    y = []
    for k, v in rmses.items():
        y.append(v)
        x.append(k)
    ax.plot(x, y)
    ax.set_ylabel("RMSE")
    ax.set_xlabel(xlabel)
    ax.set_title("RMSE against "+xlabel)
    plt.show()