import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------- Step 1: Filter Out Zero-Flow Arcs -------------------
def filter_results(results_df):
    return results_df[
        (results_df["Flow_Pipeline_4"] > 0) |
        (results_df["Flow_Pipeline_6"] > 0) |
        (results_df["Flow_Pipeline_8"] > 0) |
        (results_df["Flow_Pipeline_16"] > 0) |
        (results_df["Flow_Truck"] > 0)
        ].sort_values(
        by=["Flow_Pipeline_16", "Flow_Pipeline_8", "Flow_Pipeline_6", "Flow_Pipeline_4", "Flow_Truck"],
        ascending=False
    )


# ------------------- Step 2: Compute Key Statistics -------------------
def compute_summary(results_df, capture_results_df):
    total_CO2_transport = results_df[
        ["Flow_Pipeline_4", "Flow_Pipeline_6", "Flow_Pipeline_8", "Flow_Pipeline_16", "Flow_Truck"]
    ].sum()

    num_pipelines_built = results_df[
        ["build_Pipeline_4", "build_Pipeline_6", "build_Pipeline_8", "build_Pipeline_16"]].sum()
    num_truck_routes = results_df["build_Truck"].sum()

    print("\n🚀 Summary Statistics:")
    print("Total CO₂ Transported (tons):\n", total_CO2_transport)
    print("\nNumber of Pipelines Built:\n", num_pipelines_built)
    print("\nTruck Routes Used:", num_truck_routes)

    return total_CO2_transport, num_pipelines_built, num_truck_routes


# ------------------- Step 3: Generate LaTeX Table -------------------
def generate_latex_table(summary_results):
    pass


# ------------------- Step 4: Visualize Data -------------------
def plot_results(total_CO2_transport, num_pipelines_built, num_truck_routes):
    plt.figure(figsize=(8, 5))
    total_CO2_transport.plot(kind="bar", title="CO₂ Transported by Mode", ylabel="Tons", xlabel="Transport Mode")
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(6, 6))
    labels = ["Pipelines Built", "Truck Routes Used"]
    sizes = [num_pipelines_built.sum(), num_truck_routes]
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Pipeline vs. Truck Usage")
    plt.show()


# ------------------- Step 5: Process Results & Save -------------------
def process_results(results_df, capture_results_df, max_truck_data):
    # Step 1: Filter results
    filtered_results_df = filter_results(results_df)

    # Step 2: Compute key statistics
    total_CO2_transport, num_pipelines_built, num_truck_routes = compute_summary(filtered_results_df,
                                                                                 capture_results_df)

    # Step 3: Generate summary per country
    summary_results = []
    for country in max_truck_data.keys():
        if country == "Total":
            continue  # Skip total row

        country_arcs = [arc for arc in arcs if arc in arc_data]

        pipeline_cost = sum(
            arc_data[arc]["Pipeline_Fixed_Costs"][4] * y_pipeline_4[arc] +
            arc_data[arc]["Pipeline_Fixed_Costs"][6] * y_pipeline_6[arc] +
            arc_data[arc]["Pipeline_Fixed_Costs"][8] * y_pipeline_8[arc] +
            arc_data[arc]["Pipeline_Fixed_Costs"][16] * y_pipeline_16[arc] +
            y_truck[arc]  # Truck decision remains the same
            for arc in country_arcs
        )
        truck_cost = sum(arc_data[arc]["Truck_Fixed_Cost"] * y_truck[arc].x for arc in country_arcs)
        total_storage_cost = sum(data_storage["Total Cost"]) / sum(data_storage["Injection rate (2050) (t/year)"])
        total_capture = sum(x_capture[i].x for i, emitter in enumerate(emitters_data) if emitter in country_arcs)
        total_pipeline_length = sum(
            arc_data[arc]["Distance"] * (y_pipeline_4[arc] + y_pipeline_6[arc] + y_pipeline_8[arc] + y_pipeline_16[arc]+y_truck[arc]) for arc in country_arcs)

        summary_results.append({
            "Scenario": country,
            "Pipeline Construction Cost (€)": pipeline_cost,
            "Truck Transport Cost (€)": truck_cost,
            "Tanker Cost (€)": 0,
            "Total Storage Cost (€/ton)": total_storage_cost,
            "Total Capture (tons)": total_capture,
            "Total Pipeline Length (km)": total_pipeline_length
        })

    summary_df = pd.DataFrame(summary_results)

    # Step 4: Save filtered results
    with pd.ExcelWriter("filtered_model_results.xlsx") as writer:
        filtered_results_df.to_excel(writer, sheet_name="Transport Results", index=False)
        capture_results_df.to_excel(writer, sheet_name="Capture Results", index=False)

    # # Step 5: Save LaTeX table
    # latex_table = generate_latex_table(summary_df)
    # with open("ccus_results_table.tex", "w") as f:
    #     f.write(latex_table)

    print("\n📄 Filtered results saved to filtered_model_results.xlsx")
    print("📄 LaTeX table saved to ccus_results_table.tex")

    # Step 6: Visualize results
    plot_results(total_CO2_transport, num_pipelines_built, num_truck_routes)


import data

# Load data
arcs_file_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\valid_arcs.xlsx"
file_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\all_countries.xlsx"
data_utilizers = pd.read_excel(file_path, "utilization_sites")
data_storage = pd.read_excel(file_path, "storage_sites")
data_emitters=pd.read_excel(file_path,"emitters")
emitters_data= data_emitters['ID'].tolist()
storage_data=data_storage['ID'].tolist()
N_emitter=len(emitters_data)
# print("emitter_columns",data_emitters.columns)
# # Debugging output
# print("emitters_head",data_emitters.head())
# data checking
print("Storage Data Missing or Invalid:")
print(data_storage[["Total Cost", "Enviromental Impact", "Injection rate (2050) (t/year)"]].isnull().sum())

print("Utilizer Data Missing or Invalid:")
print(data_utilizers[["Capacity (Million ton)"]].isnull().sum())

arcs_df = pd.read_excel(arcs_file_path)
print(arcs_df)
# Create a dictionary for arc data with (From, To) as keys
arc_data = {
    (row["From"], row["To"]): {
        "Distance": row["Distance (km)"],
        "Truck_Fixed_Cost": row["Truck_Fixed_Cost"],
        "Truck_Var_Cost":row["Truck_Var_Cost"],
        "Pipeline_Fixed_Costs": {  # Pipeline costs for different diameters
            4: row["Pipeline_Fixed_Cost_4"],
            6: row["Pipeline_Fixed_Cost_6"],
            8: row["Pipeline_Fixed_Cost_8"],
            10: row["Pipeline_Fixed_Cost_10"],
            12: row["Pipeline_Fixed_Cost_12"],
            16: row["Pipeline_Fixed_Cost_16"],
            20: row["Pipeline_Fixed_Cost_20"],
        },
"Pipeline_Var_Costs": {  # Pipeline costs for different diameters
            4: row["Pipeline_Var_Cost_4"],
            6: row["Pipeline_Var_Cost_6"],
            8: row["Pipeline_Var_Cost_8"],
            10: row["Pipeline_Var_Cost_10"],
            12: row["Pipeline_Var_Cost_12"],
            16: row["Pipeline_Var_Cost_16"],
            20: row["Pipeline_Var_Cost_20"],
        }
    }
    for _, row in arcs_df.iterrows()
}
pipeline_capacity=[290000,660000,1390000,4930000 ]
# pipeline_costs_16=arc_data[("BG1","BG2")]["Pipeline_Fixed_Costs"][16]
# print("COSTS FOR 16 ARE:",pipeline_costs_16)
arcs = list(arc_data.keys())
#print(arcs)
if ('BG2', 'BG1') in arc_data:
    print("Key exists!")
else:
    print("Key does not exist in arc_data.")

arc = ('BG2', 'BG1')
print(arc_data[arc])  # Print the dictionary for the arc
print(arc_data[arc].get("Pipeline_Costs"))  # Check if 'Pipeline_Costs' exists
print(arc_data[arc]["Pipeline_Fixed_Costs"].get(16))  # Check if diameter 16 exists
for arc in arcs:
    distance = arc_data[arc]["Distance"]
    if pd.isna(distance) or not np.isfinite(distance):
        print(f"Invalid distance for arc {arc}: {distance}")
# Extract nodes from the arcs
nodes = list(set(arcs_df["From"]).union(set(arcs_df["To"])))
print("NODES",nodes)

modes = ['pipeline', 'tanker']
# Initialize the Gurobi model
model = gp.Model("CO2_Transport")

diameters = [4, 6, 8, 10, 12, 16, 20]  # Pipeline diameters
# Decision variables for pipelines (x and y for each diameter)
x_pipeline_16 = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_16")
x_pipeline_8 = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_8")
x_pipeline_6=model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_6")
x_pipeline_4=model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_4")
x_capture= model.addVars(N_emitter, vtype=GRB.CONTINUOUS, name="x_capture")
x_truck = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_tanker")
#decision variables for buildin/or not the pipelines
y_pipeline_16 = model.addVars(arcs,  vtype=GRB.BINARY, name="y_pipeline_16")
y_pipeline_8 = model.addVars(arcs, vtype=GRB.BINARY, name="y_pipeline_8")
y_pipeline_6 = model.addVars(arcs, vtype=GRB.BINARY, name="y_pipeline_6")
y_pipeline_4 = model.addVars(arcs, vtype=GRB.BINARY, name="y_pipeline_4")
y_capture= model.addVars(N_emitter, vtype=GRB.BINARY, name= "y_capture")
y_truck= model.addVars(arcs,vtype=GRB.BINARY, name= "y_truck")



# Decision variables for tankers
# x_tanker = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_tanker")
# y_tanker = model.addVars(arcs, vtype=GRB.BINARY, name="y_tanker")

# Minimize fixed and variable costs
model.setObjective(
    gp.quicksum(
        # Pipeline costs (variable and fixed for each diameter)
        arc_data[arc]["Pipeline_Fixed_Costs"][16] * y_pipeline_16[arc] +
        x_pipeline_16[arc] * arc_data[arc]["Pipeline_Var_Costs"][16]+ arc_data[arc]["Pipeline_Fixed_Costs"][8] * y_pipeline_8[arc] +
        x_pipeline_8[arc] * arc_data[arc]["Pipeline_Var_Costs"][8]+arc_data[arc]["Pipeline_Fixed_Costs"][6] * y_pipeline_6[arc] +
        x_pipeline_6[arc] * arc_data[arc]["Pipeline_Var_Costs"][6]+arc_data[arc]["Pipeline_Fixed_Costs"][4] * y_pipeline_4[arc] +
        x_pipeline_4[arc] * arc_data[arc]["Pipeline_Var_Costs"][4] +arc_data[arc]["Truck_Fixed_Cost"]*y_truck[arc]+arc_data[arc]["Truck_Var_Cost"]*x_truck[arc] # Example variable cost per km
        for arc in arcs
    ) +
    gp.quicksum(
        data_emitters.loc[i, "Fixed_Cost_New"] * y_capture[i] +
        data_emitters.loc[i, "Variable_Cost"] * x_capture[i]
        for i in range(len(emitters_data))
    )+

gp.quicksum(
    (row["Total Cost"] - row["Enviromental Impact"]) *
    gp.quicksum(
        x_pipeline_4[arc] + x_pipeline_6[arc] + x_pipeline_8[arc] + x_pipeline_16[arc] + x_truck[arc]
        for arc in arcs if arc[1] == row["ID"]
    )
    for _, row in data_storage.iterrows()
)-
gp.quicksum(
    row["Utilization_Cost"] *
    gp.quicksum(
        x_pipeline_4[arc] + x_pipeline_6[arc] + x_pipeline_8[arc] + x_pipeline_16[arc] + x_truck[arc]
        for arc in arcs if arc[1] == row["ID"]
    ) for _, row in data_utilizers.iterrows())
    , GRB.MINIMIZE )

# maximal amount of co2 transported
max_truck_data = {
    "Romania": {
        "max_truck_flow": 2.7e6,  # 2.7 million tons per year
        "num_trucks": 100  # Maximum trucks available
    },
    "Greece": {
        "max_truck_flow": 2.16e6,  # 2.16 million tons per year
        "num_trucks": 80
    },
    "Bulgaria": {
        "max_truck_flow": 1.62e6,  # 1.62 million tons per year
        "num_trucks": 60
    },
    "Croatia": {
        "max_truck_flow": 1.35e6,  # 1.35 million tons per year
        "num_trucks": 50
    }
}

# Compute total combined truck flow across all countries
total_max_truck_flow = sum(country["max_truck_flow"] for country in max_truck_data.values())
total_num_trucks = sum(country["num_trucks"] for country in max_truck_data.values())

# Add the total to the dictionary
max_truck_data["Total"] = {
    "max_truck_flow": total_max_truck_flow,
    "num_trucks": total_num_trucks
}
# Add global truck flow constraint
model.addConstr(
    gp.quicksum(x_truck[arc] for arc in arcs) <= max_truck_data["Total"]["max_truck_flow"],
    name="TotalTruckFlowLimit"
)

for arc in arcs:
    # Pipeline capacity constraints (upper bound only)
    model.addConstr(
        x_pipeline_16[arc] <= pipeline_capacity[3] * y_pipeline_16[arc],
        name=f"PipelineCapacityMax_{arc}_{16}"
    )
    model.addConstr(
        x_pipeline_8[arc] <= pipeline_capacity[2] * y_pipeline_8[arc],
        name=f"PipelineCapacityMax_{arc}_{8}"
    )
    model.addConstr(
        x_pipeline_6[arc] <= pipeline_capacity[1] * y_pipeline_6[arc],
        name=f"PipelineCapacityMax_{arc}_{6}"
    )
    model.addConstr(
        x_pipeline_4[arc] <= pipeline_capacity[0] * y_pipeline_4[arc],
        name=f"PipelineCapacityMax_{arc}_{4}"
    )

    # Ensure flow is zero if pipeline is not built
    model.addConstr(
        x_pipeline_16[arc] >= 0,  # Ensure flow is at least 0
        name=f"PipelineCapacityMin_{arc}_{16}"
    )
    model.addConstr(
        x_pipeline_8[arc] >= 0,
        name=f"PipelineCapacityMin_{arc}_{8}"
    )
    model.addConstr(
        x_pipeline_6[arc] >= 0,
        name=f"PipelineCapacityMin_{arc}_{6}"
    )
    model.addConstr(
        x_pipeline_4[arc] >= 0,
        name=f"PipelineCapacityMin_{arc}_{4}"
    )

    # Truck constraint with correct indexing for `y_truck`
    max_truck_capacity = (100* 25 * 60 * 5000) / (arc_data[arc]["Distance"] + 62.5)
    model.addConstr(
        x_truck[arc] <= max_truck_capacity * y_truck[arc],
        name=f"TankerCapacity_{arc}"
    )




#if  no truck then pipeline
M = 1e7  # Large enough to allow truck flow if no pipeline is built

# for arc in arcs:
#     model.addConstr(
#         x_truck[arc] <= max_truck_data["Total"]["max_truck_flow"] +\
#         M * (1 - (y_pipeline_4[arc] + y_pipeline_6[arc] + y_pipeline_8[arc] + y_pipeline_16[arc]+y_truck[arc])),
#         name=f"ForcePipelineIfTruckLimitReached_{arc}"
#     )


# Alternative flow conservation at emitters (net outflow minus inflow equals capture)
for emitter in emitters_data:
    inflow_pipeline = (
        gp.quicksum(x_pipeline_4[arc] for arc in arcs if arc[1] == emitter) +
        gp.quicksum(x_pipeline_6[arc] for arc in arcs if arc[1] == emitter) +
        gp.quicksum(x_pipeline_8[arc] for arc in arcs if arc[1] == emitter) +
        gp.quicksum(x_pipeline_16[arc] for arc in arcs if arc[1] == emitter)
    )
    outflow_pipeline = (
        gp.quicksum(x_pipeline_4[arc] for arc in arcs if arc[0] == emitter) +
        gp.quicksum(x_pipeline_6[arc] for arc in arcs if arc[0] == emitter) +
        gp.quicksum(x_pipeline_8[arc] for arc in arcs if arc[0] == emitter) +
        gp.quicksum(x_pipeline_16[arc] for arc in arcs if arc[0] == emitter)
    )
    model.addConstr(
        outflow_pipeline - inflow_pipeline == x_capture[emitters_data.index(emitter)],
        name=f"FlowConservationNet_Emitter_{emitter}"
    )

# Capture facility installation constraint at emitters
for i, row in data_emitters.iterrows():
    model.addConstr(
        x_capture[i] >= row["Upper_Bound"],
        name=f"CaptureConstraint_{row['ID']}"
    )
for i, row in data_emitters.iterrows():
    model.addConstr(
        x_capture[i] <= row["Emission (ton/year)"],
        name=f"CaptureConstraint_{row['ID']}"
    )
# Flow constraints for storage nodes: inflow must equal injection rate
for storage_node in storage_data:
    inflow_to_storage = (
        gp.quicksum(x_pipeline_4[arc] for arc in arcs if arc[1] == storage_node) +
        gp.quicksum(x_pipeline_6[arc] for arc in arcs if arc[1] == storage_node) +
        gp.quicksum(x_pipeline_8[arc] for arc in arcs if arc[1] == storage_node) +
        gp.quicksum(x_pipeline_16[arc] for arc in arcs if arc[1] == storage_node) +
        gp.quicksum(x_truck[arc] for arc in arcs if arc[1] == storage_node)
    )
    model.addConstr(
        inflow_to_storage <= data_storage.loc[data_storage["ID"] == storage_node, "Injection rate (2050) (t/year)"].values[0],
        name=f"StorageSink_{storage_node}"
    )

# Flow constraints for utilization nodes: inflow must equal capacity
for util_node in data_utilizers["ID"]:
    inflow_to_utilizer = (
        gp.quicksum(x_pipeline_4[arc] for arc in arcs if arc[1] == util_node) +
        gp.quicksum(x_pipeline_6[arc] for arc in arcs if arc[1] == util_node) +
        gp.quicksum(x_pipeline_8[arc] for arc in arcs if arc[1] == util_node) +
        gp.quicksum(x_pipeline_16[arc] for arc in arcs if arc[1] == util_node) +
        gp.quicksum(x_truck[arc] for arc in arcs if arc[1] == util_node)
    )
    model.addConstr(
        inflow_to_utilizer >= data_utilizers.loc[data_utilizers["ID"] == util_node, "Capacity (Million ton)"].values[0],
        name=f"UtilizerSink_{util_node}"
    )







# Identify intermediate nodes (nodes that are not emitters, storage, or utilization nodes)
intermediate_nodes = list(set(nodes) - set(emitters_data) - set(storage_data) - set(data_utilizers["ID"]))

total_possible_capture = sum(data_emitters["Upper_Bound"])
Q =  total_possible_capture  # Capture at least 60% of emissions


# CO2 reduction target constraint
model.addConstr(
    gp.quicksum(
        (x_pipeline_16[arc] + x_pipeline_8[arc] + x_pipeline_6[arc] + x_pipeline_4[arc] + x_truck[arc])
        for arc in arcs if (arc[1] in storage_data) and (arc[0] != arc[1])
    ) +
    gp.quicksum(
        (x_pipeline_16[arc] + x_pipeline_8[arc] + x_pipeline_6[arc] + x_pipeline_4[arc] + x_truck[arc])
        for arc in arcs if (arc[1] in data_utilizers["ID"].tolist()) and (arc[0] != arc[1])
    ) >= Q,
    name="CO2ReductionTarget"
)
# 5. Only one pipeline can be built for each arc
for arc in arcs:
    model.addConstr(
        y_pipeline_4[arc] + y_pipeline_6[arc] + y_pipeline_8[arc] + y_pipeline_16[arc] <=1,
        name=f"OnePipelinePerArc_{arc}"
    )
    # model.addConstr(
    #     y_truck[arc] ==0,
    #     name=f"OnePipelinePerArc_{arc}"
    # )

#only one mode of transportation
for arc in arcs:
    model.addConstr(
        y_pipeline_4[arc] + y_pipeline_6[arc] + y_pipeline_8[arc] + y_pipeline_16[arc]+y_truck[arc] <= 1,
        name=f"OnePipelinePerArc_{arc}"
    )
# Optimize model
model.optimize()

if model.status == GRB.UNBOUNDED:


    print(" Model is unbounded! Check for missing constraints.")

elif model.status == GRB.INF_OR_UNBD:
    print(" Model is either infeasible or unbounded! Running feasibility check...")
    model.setParam(GRB.Param.DualReductions, 0)
    model.optimize()
    if model.status == GRB.INFEASIBLE:
        print(" Confirmed: Model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print(" Confirmed: Model is unbounded.")

elif model.status == GRB.INTERRUPTED:
    print(" Optimization was interrupted before completion.")

else:
    print("Unknown issue. Model status:", model.status)
if model.status == GRB.INFEASIBLE:
    print(" Model is infeasible. Running infeasibility analysis...")

    # model.computeIIS()  # Compute the IIS (Minimal set of conflicting constraints)
    # model.write("infeasible_model.ilp")  # Save IIS to a file

    print("Model infeasibility analysis saved to infeasible_model.ilp.")
# Save results
if model.status == GRB.OPTIMAL:
    print("✅ Optimal solution found!")

    results = []
    for arc in arcs:
        results.append({
            "From": arc[0],
            "To": arc[1],
            "Flow_Pipeline_4": x_pipeline_4[arc].x,
            "Flow_Pipeline_6": x_pipeline_6[arc].x,
            "Flow_Pipeline_8": x_pipeline_8[arc].x,
            "Flow_Pipeline_16": x_pipeline_16[arc].x,
            "Flow_Truck": x_truck[arc].x,
            "build_Pipeline_4": y_pipeline_4[arc].x,
            "build_Pipeline_6": y_pipeline_6[arc].x,
            "build_Pipeline_8": y_pipeline_8[arc].x,
            "build_Pipeline_16": y_pipeline_16[arc].x,
            "build_Truck": y_truck[arc].x
        })

    capture_results = []
    for i, emitter in enumerate(emitters_data):
        capture_results.append({
            "Emitter": emitter,
            "Captured_CO2": x_capture[i].x,
            "Build_Capture": y_capture[i].x
        })

    results_df = pd.DataFrame(results)
    capture_results_df = pd.DataFrame(capture_results)

    # Call Data Processing
    process_results(results_df, capture_results_df, max_truck_data)
    print("📄 Summary statistics saved to summary_results.csv")


    # Save to Excel
    with pd.ExcelWriter("model_results.xlsx") as writer:
        results_df.to_excel(writer, sheet_name="Transport Results", index=False)
        capture_results_df.to_excel(writer, sheet_name="Capture Results", index=False)

    print("Results saved to model_results.xlsx")

    # Debugging output
    print("Captured CO2 per emitter:")
    print(capture_results_df)


else:
    print("No optimal solution found.")
    total_storage_capacity = sum(data_storage["Injection rate (2050) (t/year)"])
    total_utilization_capacity = sum(data_utilizers["Capacity (Million ton)"])   # Convert million tons

    print(f"Total Storage Capacity: {total_storage_capacity}")
    print(f"Total Utilization Capacity: {total_utilization_capacity}")
    print(f"CO₂ Reduction Target: {Q}")


