import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from data import data, parameters, load_emitter_data
import math
import numpy as np


#get data from the file
file_path= r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\data\sources.xlsx"
data_points=load_emitter_data(file_path, "emitters")
#Index(['ID', 'Name', 'Emission (ton/year)', 'FIXCOST: capture cost (annual)',
       #'COST: capture cost per ton', 'LON', 'LAT', 'countryCode', 'Activity']

#coloring
activity_colors = {
    "Manufacture of cement": "white",
    "Manufacture of fertilisers and nitrogen compounds": "green",
    "Manufacture of basic iron and steel and of ferro-alloys": "black",
    "Manufacture of refined petroleum products": "yellow"  # You can replace with brown or red
}
# Aggregated emissions calculation
sector_totals = data_points.groupby("Activity")["Emission (ton/year)"].sum()
print("SECTOR TOTAL IS:",sector_totals)

# Define scaling for large circles
circle_scale = sector_totals.max() /8000  # Adjust for proportional scaling

# Define the desired order for sectors
sector_order = [
    "Manufacture of cement",
    "Manufacture of refined petroleum products",
    "Manufacture of basic iron and steel and of ferro-alloys",
    "Manufacture of fertilisers and nitrogen compounds"
]
sector_labels = {
    "Manufacture of cement": "Cement",
    "Manufacture of refined petroleum products": "Refinery",
    "Manufacture of basic iron and steel and of ferro-alloys": "Iron and Steel",
    "Manufacture of fertilisers and nitrogen compounds": "Fertilizers"
}

# Reorder `sector_totals` based on the desired order
sector_totals = sector_totals.reindex(sector_order)

# Dynamically calculate vertical range based on largest circle size
largest_circle_size = max(sector_totals) / circle_scale


# Create a figure with two subplots and adjust spacing
fig = plt.figure(figsize=(16, 8))
#gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.1)
#adjust the gs space for better fit
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0)

# Map subplot (left)
ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_map.stock_img()
land = cfeature.NaturalEarthFeature(
    category='physical',
    name='land',
    scale='50m',
    facecolor='lightgray'
)
ax_map.add_feature(land, zorder=1)
ax_map.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
ax_map.add_feature(cfeature.COASTLINE, alpha=0.7)
ax_map.set_extent([13.5, 30, 35, 49], crs=ccrs.PlateCarree())

# Plot individual emitter points
for index, point in data_points.iterrows():
    color = activity_colors.get(point["Activity"], "red")
    s = point['Emission (ton/year)'] / 1500  # Scale size dynamically

    ax_map.scatter(
        point['LON'], point['LAT'],
        s=s,
        color=color,
        transform=ccrs.PlateCarree(),
        alpha=0.7,
        edgecolor="black",
        zorder=19
    )

    # Add labels
    emission_label = f"{round(point['Emission (ton/year)'] / 1_000_000, 1)}mt"
    label_color = "black" if color == "white" else "white"
    ax_map.text(
        point["LON"], point["LAT"],
        emission_label,
        color=label_color,
        fontsize=6,
        ha='center',
        va='center',
        zorder=60,
        transform=ccrs.PlateCarree()
    )

# Aggregated emissions subplot (right)
ax_legend = fig.add_subplot(gs[0, 1])
ax_legend.axis("off")  # Turn off axis for the legend

# Plot the aggregated emission circles
x_pos = 0.3  # Fixed horizontal position for circles
# Set y-axis limits to clip circles within bounds
ax_legend.set_xlim(0, 1)  # Ensure horizontal centering
y_min=0.15
y_max=0.85
y_positions = [0.8,0.6,0.44,0.32]

ax_legend.set_ylim(y_min - 0.1, y_max + 0.1)  # Add padding for circle clipping

# Plot the aggregated emission circles
for i, (sector, total_emission) in enumerate(sector_totals.items()):
    print("TOtal emissions are:",round(total_emission/1000000,1))
    color = activity_colors.get(sector, "red")
    ax_legend.scatter(
        x_pos, y_positions[i],
        s=total_emission / circle_scale,  # Scale size based on total emissions
        color=color,
        alpha=0.7,
        edgecolor="black",
        clip_on=True  # Clip circles at the subplot edges
    )
    # Add text to label the total emissions
    ax_legend.text(
        x_pos , y_positions[i],
        f"{round(total_emission / 1000000,1)}mt",  # Format in thousands of tons
        fontsize=10,
        color='red',
        verticalalignment='center',
    ha = 'center',
    va = 'center',
    )
    ax_legend.text(
        x_pos + 0.35, y_positions[i],  # Slightly farther right for sector name
        sector_labels[sector],  # The sector name
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='left'
    )

# Titles
ax_map.set_title("Southeast Europe Emitters", fontsize=16)
ax_legend.set_title("Aggregated Emissions by Sector", fontsize=16, loc="left")

plt.show()
