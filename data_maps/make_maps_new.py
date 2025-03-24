import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
file_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\all_countries_2025.xlsx"
data_points = pd.read_excel(file_path, sheet_name="emitters")
data_points_storage = pd.read_excel(file_path, sheet_name="storage_sites")
data_util = pd.read_excel(file_path, sheet_name="utilization_sites")
#arcss
arcs_file = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\model_results_HR_2025.xlsx"
arcs_df = pd.read_excel(arcs_file, "Transport Results")  # Adjust sheet_name if needed


selected_country = "HR"

# Filter emitters, storage, and utilization sites for this country
country_emitters = data_points[data_points["Country_Code"] == selected_country]
country_storage = data_points_storage[data_points_storage["Country_Code"] == selected_country]
country_utilization = data_util[data_util["Country_Code"] == selected_country]
print("countryyyyyy emiteerrssssssss",country_emitters)
print("countryyyyyy emiteerrssssssss",country_storage)

# Merge arcs with emitter data to get start coordinates
# Merge arcs_df with emitters, storage, and utilization sites to get 'From' coordinates
# Merge arcs_df with emitters, storage, and utilization sites to get 'From' coordinates
arcs_df = arcs_df.merge(country_emitters[['ID', 'LAT', 'LON']], left_on='From', right_on='ID', how='left')
arcs_df.rename(columns={'LAT': 'LAT_from_emitter', 'LON': 'LON_from_emitter'}, inplace=True)

arcs_df = arcs_df.merge(country_storage[['ID', 'LAT', 'LON']], left_on='From', right_on='ID', how='left')
arcs_df.rename(columns={'LAT': 'LAT_from_storage', 'LON': 'LON_from_storage'}, inplace=True)

arcs_df = arcs_df.merge(country_utilization[['ID', 'LAT', 'LON']], left_on='From', right_on='ID', how='left')
arcs_df.rename(columns={'LAT': 'LAT_from_util', 'LON': 'LON_from_util'}, inplace=True)

# # Merge arcs_df with emitters, storage, and utilization sites to get 'To' coordinates
arcs_df = arcs_df.merge(country_emitters[['ID', 'LAT', 'LON']], left_on='To', right_on='ID', how='left')
arcs_df.rename(columns={'LAT': 'LAT_to_emitter', 'LON': 'LON_to_emitter'}, inplace=True)

arcs_df = arcs_df.merge(country_storage[['ID', 'LAT', 'LON']], left_on='To', right_on='ID', how='left')
arcs_df.rename(columns={'LAT': 'LAT_to_storage', 'LON': 'LON_to_storage'}, inplace=True)

arcs_df = arcs_df.merge(country_utilization[['ID', 'LAT', 'LON']], left_on='To', right_on='ID', how='left')
arcs_df.rename(columns={'LAT': 'LAT_to_util', 'LON': 'LON_to_util'}, inplace=True)

# Check the resulting columns (for debugging)
print("Columns in arcs_df:", arcs_df.columns.tolist())

# Combine the candidate columns to create unified coordinate columns
arcs_df["LAT_from"] = (
    arcs_df["LAT_from_emitter"]
    .combine_first(arcs_df["LAT_from_storage"])
    .combine_first(arcs_df["LAT_from_util"])
)
arcs_df["LON_from"] = (
    arcs_df["LON_from_emitter"]
    .combine_first(arcs_df["LON_from_storage"])
    .combine_first(arcs_df["LON_from_util"])
)
arcs_df["LAT_to"] = (
    arcs_df["LAT_to_emitter"]
    .combine_first(arcs_df["LAT_to_storage"])
    .combine_first(arcs_df["LAT_to_util"])
)
arcs_df["LON_to"] = (
    arcs_df["LON_to_emitter"]
    .combine_first(arcs_df["LON_to_storage"])
    .combine_first(arcs_df["LON_to_util"])
)

#Keep only the required columns and drop rows with missing coordinates
# Instead of dropping them all, keep the build_ columns you need
arcs_df = arcs_df[[
    'From', 'To',
    'LAT_from', 'LON_from',
    'LAT_to', 'LON_to',
    'build_Pipeline_4', 'build_Pipeline_6',
    'build_Pipeline_8', 'build_Pipeline_16',
    'build_Truck'
]].dropna()
#arcs_df = arcs_df[['From', 'To', 'LAT_from', 'LON_from', 'LAT_to', 'LON_to']].dropna()
print(arcs_df)


# Print to verify correction
print("Corrected Emitter sample:\n", data_points[['LON', 'LAT']].head())
# Define colors for emitter activities
activity_colors = {
    "Manufacture of cement": "gray",
    "Manufacture of fertilisers and nitrogen compounds": "green",
    "Manufacture of basic iron and steel and of ferro-alloys": "black",
    "Manufacture of refined petroleum products": "yellow"
}

min_lon = min(country_emitters["LON"].min(), country_storage["LON"].min(), country_utilization["LON"].min()) - 0.5
max_lon = max(country_emitters["LON"].max(),  country_storage["LON"].max(), country_utilization["LON"].max()) + 0.5
min_lat = min(country_emitters["LAT"].min(),  country_storage["LAT"].min(),country_utilization["LAT"].min()) - 0.5
max_lat = max(country_emitters["LAT"].max(),  country_storage["LAT"].max(), country_utilization["LAT"].max()) + 0.5

# Adjust figure size to match the aspect ratio
width = (max_lon - min_lon) * 1.5  # Adjust multiplier for better width
height = (max_lat - min_lat) * 1.5  # Adjust multiplier for better height

# Set up the figure with appropriate width-to-height ratio
plt.figure(figsize=(width, height))
ax_map = plt.axes(projection=ccrs.PlateCarree())

# Set the extent to perfectly fit the dataset
ax_map.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
# # Create map
# fig = plt.figure(figsize=(12, 8))
# ax_map = plt.axes(projection=ccrs.PlateCarree())

# Add background features
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='50m',
    facecolor='lightgray'
)
ax_map.add_feature(land, zorder=1)
ax_map.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
ax_map.add_feature(cfeature.COASTLINE, alpha=0.7)
ax_map.stock_img()
# ax_map.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
# ax_map.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', zorder=1)
# ax_map.add_feature(cfeature.COASTLINE, alpha=0.7, zorder=0.7)
#ax_map.set_extent([20, 27, 33, 44], crs=ccrs.PlateCarree())
# Adjust map extent to fit the region better
#ax_map.set_extent([19, 30, 33, 45], crs=ccrs.PlateCarree())  # Widen longitude range

# Plot emitters
for _, point in country_emitters.iterrows():
    color = "red"
    #size = max(50, point['Emission (ton/year)'] / 200)  # Ensure visibility
    size=150
    ax_map.scatter(point['LON'], point['LAT'], s=size, color=color,
                   transform=ccrs.PlateCarree(), alpha=0.8, edgecolor="black", zorder=12)

# Plot utilization sites
ax_map.scatter(country_utilization['LON'], country_utilization['LAT'], s=100, color="green",
               transform=ccrs.PlateCarree(), alpha=0.7, edgecolor="black", label="Utilization Sites", zorder=13, marker='s')

# Plot storage sites
ax_map.scatter(country_storage['LON'], country_storage['LAT'], s=100, color="blue",
               transform=ccrs.PlateCarree(), alpha=0.7, edgecolor="black", label="Storage Sites", zorder=12, marker='s')
pipeline_cols = [
    'build_Pipeline_4',
    'build_Pipeline_6',
    'build_Pipeline_8',
    'build_Pipeline_16'
]

for _, arc in arcs_df.iterrows():
    from_lat, from_lon = arc["LAT_from"], arc["LON_from"]
    to_lat, to_lon     = arc["LAT_to"],   arc["LON_to"]

    # Check if any pipeline column is 1
    pipeline_used = any(arc[col] == 1 for col in pipeline_cols)
    truck_used    = (arc["build_Truck"] == 1)

    # Decide color: truck => blue, pipeline => black, else gray
    if truck_used:
        line_color = "blue"
    elif pipeline_used:
        line_color = "black"
    else:
        line_color = "gray"  # default if neither is used

    # Plot the arc line
    ax_map.plot(
        [from_lon, to_lon],
        [from_lat, to_lat],
        color=line_color,
        linewidth=0.001,              # adjust thickness as needed
        transform=ccrs.PlateCarree(),
        zorder=1,
        alpha=0.7
    )

    

    # Arrow direction
    dx = to_lon - from_lon
    dy = to_lat - from_lat

    # Plot arrow using quiver
    ax_map.quiver(
        from_lon, from_lat,
        dx, dy,
        angles="xy", scale_units="xy", scale=1,
        width=0.003,
        color=line_color,

        transform=ccrs.PlateCarree(),
        zorder=3
    )
# Update Your Legend
#Add legend entries for pipeline (black) and truck (blue). For example:

from matplotlib.lines import Line2D
# 1) Define the legend elements for nodes
legend_elements_nodes = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',   markersize=10, label='Emitters'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='Utilization'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue',  markersize=10, label='Storage'),
]

# 2) Place the first legend in the lower-left
legend1 = ax_map.legend(handles=legend_elements_nodes, loc='lower left')
ax_map.add_artist(legend1)  # Keep the first legend on the map

# 3) Define the legend elements for transport modes
legend_elements_transport = [
    Line2D([0], [0], color='black', linewidth=2, label='Pipeline'),
    Line2D([0], [0], color='blue',  linewidth=2, label='Truck/Tanker'),
]

# 4) Place the second legend in the lower-right
legend2 = ax_map.legend(handles=legend_elements_transport, loc='lower right')
ax_map.add_artist(legend2)  # Add the second legend as well

plt.show()
