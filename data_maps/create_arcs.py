# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:48:02 2024

@author: P304937
"""

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gb
from gurobipy import *
import pandas as pd
from itertools import product
import math


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    This is the haversine formula from wikipedia
    Returns: distance between two points on earth

    """
    R = 6373.0  # earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dist_lat = lat2 - lat1
    dist_lon = lon2 - lon1
    a = np.sin(dist_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dist_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


file_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\data\all_countries.xlsx"
data_emitters = pd.read_excel(file_path, "emitters")
data_utilizers = pd.read_excel(file_path, "utilization_sites")
data_storage = pd.read_excel(file_path, "storage_sites")
# Add 'Capacity (Million ton)' to data_emitters with default value
data_emitters["SourceFile"] = "emitters"
data_utilizers["SourceFile"] = "utilizers"
data_storage["SourceFile"] = "storage"
data_storage["cost_2025"]=data_storage
data_emitters["Capacity (Million ton)"] = 0  # or None

# Ensure all DataFrames have the same columns
data_emitters = data_emitters[["ID", "LAT", "LON", "SourceFile", "Capacity (Million ton)"]]
data_utilizers = data_utilizers[["ID", "LAT", "LON", "SourceFile", "Capacity (Million ton)"]]
data_storage = data_storage[["ID", "LAT", "LON", "SourceFile", "Capacity (Million ton)"]]
# data_emitters = data_emitters[["ID", "LAT", "LON"]].copy()
# data_utilizers = data_utilizers[["ID", "LAT", "LON","Capacity (Million ton)"]].copy()
# data_storage = data_storage[["ID", "LAT", "LON","Capacity (Million ton)"]].copy()
#data_emitters["Capacity (Million ton)"] = 0


print(data_emitters.columns)
print(data_utilizers.columns)
print(data_storage.columns)

# print(data_storage.columns)
# print(data_storage)
# print(data_utilizers)
combined_df = pd.concat([data_emitters, data_utilizers, data_storage], ignore_index=True)
#print(combined_df)
#combined_df = pd.concat([data_emitters, data_utilizers, data_storage], ignore_index=True)
# print(combined_df)
combined_df.to_excel("combined_data.xlsx", index=False)

# Generate all pairs of nodes for distance calculation
node_pairs = pd.DataFrame(list(product(combined_df.index, repeat=2)), columns=["Index1", "Index2"])

# Merge coordinates for each pair
node_pairs = node_pairs.merge(
    combined_df[["ID", "LAT", "LON", "SourceFile"]].rename(columns={
        "ID": "Node1", "LAT": "LAT1", "LON": "LON1", "SourceFile": "SourceFile1"
    }),
    left_on="Index1",
    right_index=True
)
node_pairs = node_pairs.merge(
    combined_df[["ID", "LAT", "LON", "SourceFile"]].rename(columns={
        "ID": "Node2", "LAT": "LAT2", "LON": "LON2", "SourceFile": "SourceFile2"
    }),
    left_on="Index2",
    right_index=True
)

# Calculate distance for each pair
node_pairs["Distance (km)"] = node_pairs.apply(
    lambda row: calculate_distance(row["LAT1"], row["LON1"], row["LAT2"], row["LON2"]),
    axis=1
)

# Filter out self-distances (optional)
node_pairs = node_pairs[node_pairs["Index1"] != node_pairs["Index2"]]

# Select relevant columns
result_df = node_pairs[["Node1", "SourceFile1", "Node2", "SourceFile2", "Distance (km)"]]

# Save the results to an Excel file
result_df.to_excel("distances_between_all_points.xlsx", index=False)
print(result_df)
print("Distances saved to distances_between_all_points.xlsx")

# print(result_df)
print("Distances saved to distances_between_all_points.xlsx")

print(data_emitters.duplicated().sum(), "duplicate rows in emitters")
print(data_utilizers.duplicated().sum(), "duplicate rows in utilizers")
print(data_storage.duplicated().sum(), "duplicate rows in storage")

# Create arcs based on rules
arcs = []

for _, row in node_pairs.iterrows():
    if row["SourceFile1"] == "emitters":  # Source node
        if row["SourceFile2"] == "emitters" and row["Node1"] != row["Node2"]:
            # Source to Source
            arcs.append((row["Node1"], row["Node2"], row["SourceFile2"],row["Distance (km)"]))
        elif row["SourceFile2"] in ["utilizers", "storage"]:
            # Source to Sink
            arcs.append((row["Node1"], row["Node2"],row["SourceFile2"] ,row["Distance (km)"]))

    # No connections from sinks to sources or sinks
    # (This part is excluded by default as we skip these combinations)

# Convert arcs to a DataFrame for easier manipulation and counting
arcs_df = pd.DataFrame(arcs, columns=["From", "To", "destination","Distance (km)"])
print(arcs_df.groupby(["From", "To"]).size())
# arcs_df["capacity"] = 0
#
# # Map capacities based on the 'SourceFile' column
# arcs_df.loc[arcs_df["destination"] == "utilizers", "capacity"] = arcs_df["To"].map(
#     data_utilizers.set_index("ID")["Capacity (Million ton)"]
# )
# arcs_df.loc[arcs_df["destination"] == "storage", "capacity"] = arcs_df["To"].map(
#     data_storage.set_index("ID")["Capacity (Million ton)"]
# )
#
# # Fill missing capacities with 0 (optional)
# arcs_df["capacity"] = arcs_df["capacity"].fillna(0)
# Function to fetch capacity based on node type
def get_capacity(node_id, node_type):
    if node_type == "emitters":
        return 0  # Emitters don't have capacity
    elif node_type == "utilizers":
        return data_utilizers.set_index("ID")["Capacity (Million ton)"].get(node_id, 0)
    elif node_type == "storage":
        return data_storage.set_index("ID")["Capacity (Million ton)"].get(node_id, 0)
    else:
        return 0  # Default if type is unknown

# Apply the function to arcs_df
arcs_df["capacity"] = arcs_df.apply(
    lambda row: get_capacity(row["To"], row["destination"]),
    axis=1
)

def enviromental_impact(node_id, node_type):
    """

    Args:
        node_id:  string
        node_type: string

    Returns: float on the capacity of each node

    """
    if node_type == "emitters":
        return 0  # we include the cost in transportation(can be set 0.1)
    elif node_type == "utilizers":
        return -0.1#  since we minimize, we favour the utilization nodes
    elif node_type == "storage":
        return -0.1
    else:
        return 0  # Default if type is unknown
arcs_df["enviromental_impact"]=arcs_df.apply(
    lambda row: enviromental_impact(row["To"], row["destination"]), axis=1
)
# add transportation costs to the sheet
tanker_cost_per_km = 0.05
arcs_df["Tanker_Cost"] = arcs_df["Distance (km)"] * tanker_cost_per_km
arcs_df.to_excel("new_Arcs.xlsx",index=False)

# Define pipeline cost table (values are from the provided table)
# Updated pipeline costs (per km/year and capacity in Mt/year)
pipeline_costs_updated = {
    4: {"CAPEX_per_km": 28000, "O&M_per_km": 980, "Capacity": 0.29},
    6: {"CAPEX_per_km": 29100, "O&M_per_km": 1019, "Capacity": 0.66},
    8: {"CAPEX_per_km": 34000, "O&M_per_km": 1190, "Capacity": 1.39},
    10: {"CAPEX_per_km": 36500, "O&M_per_km": 1278, "Capacity": 2.10},
    12: {"CAPEX_per_km": 39600, "O&M_per_km": 1386, "Capacity": 2.37},
    16: {"CAPEX_per_km": 46400, "O&M_per_km": 1624, "Capacity": 4.93},
    20: {"CAPEX_per_km": 52000, "O&M_per_km": 1820, "Capacity": 7.30},
}


# Function to calculate fixed costs per pipeline diameter
def calculate_pipeline_fixed_cost(distance, diameter):
    """
    Calculate fixed costs (CAPEX + O&M) for a given pipeline diameter and distance.
    """
    cost_info = pipeline_costs_updated[diameter]

    # Fixed costs (per year)
    capex_cost = distance * cost_info["CAPEX_per_km"]
    # o_and_m_cost = distance * cost_info["O&M_per_km"]
    # total_fixed_cost = capex_cost + o_and_m_cost

    return capex_cost
def calculate_pipeline_var_cost(distance, diameter):
    """
    Calculate variabele costs O&M for a given pipeline diameter and distance.
    """
    cost_info = pipeline_costs_updated[diameter]

    #variable costs per year
    o_and_m_cost = distance * cost_info["O&M_per_km"]


    return o_and_m_cost


# Add fixed cost columns to the arcs DataFrame
for diameter in pipeline_costs_updated.keys():
    arcs_df[f"Pipeline_Fixed_Cost_{diameter}"] = arcs_df["Distance (km)"].apply(
        lambda dist: calculate_pipeline_fixed_cost(dist, diameter)
    )
    arcs_df[f"Pipeline_Var_Cost_{diameter}"] = arcs_df["Distance (km)"].apply(
        lambda dist: calculate_pipeline_var_cost(dist, diameter)
    )

# Save the arcs to Excel
arcs_df.to_excel("valid_arcs.xlsx", index=False)
# print(f"Valid arcs saved to valid_arcs.xlsx")

print("Total number of valid arcs:", arcs_df.groupby(["From", "To"]).size())
