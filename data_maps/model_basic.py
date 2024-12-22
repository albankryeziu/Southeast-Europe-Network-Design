import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
# Load data
arcs_file_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\valid_arcs.xlsx"
arcs_df = pd.read_excel(arcs_file_path)
print(arcs_df)
# Create a dictionary for arc data with (From, To) as keys
arc_data = {
    (row["From"], row["To"]): {
        "Distance": row["Distance (km)"],
        "Tanker_Cost": row["Tanker_Cost"],
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
modes = ['pipeline', 'tanker']
# Initialize the Gurobi model
model = gp.Model("CO2_Transport")

diameters = [4, 6, 8, 10, 12, 16, 20]  # Pipeline diameters

# Decision variables for pipelines (x and y for each diameter)
x_pipeline_16 = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_16")
x_pipeline_8 = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_8")

y_pipeline_16 = model.addVars(arcs,  vtype=GRB.BINARY, name="y_pipeline_16")
y_pipeline_8 = model.addVars(arcs, vtype=GRB.BINARY, name="y_pipeline_8")


# Decision variables for tankers
x_tanker = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_tanker")
y_tanker = model.addVars(arcs, vtype=GRB.BINARY, name="y_tanker")

# Minimize fixed and variable costs
model.setObjective(
    gp.quicksum(
        # Pipeline costs (variable and fixed for each diameter)
        arc_data[arc]["Pipeline_Fixed_Costs"][16] * y_pipeline_16[arc] +
        x_pipeline_16[arc] * arc_data[arc]["Pipeline_Var_Costs"][16]  # Example variable cost per km
        for arc in arcs
    ) +
    gp.quicksum(
        # Tanker costs (variable only)
        arc_data[arc]["Tanker_Cost"] * x_tanker[arc]
        for arc in arcs
    ),
    GRB.MINIMIZE
)


for arc in arcs:
    model.addConstr(
            x_pipeline_16[arc]<= 1000000 * y_pipeline_16[arc],  # Example capacity for pipelines
            name=f"PipelineCapacity_{arc}_{16}"
        )
    model.addConstr(
        x_tanker[arc] <= 5000 * y_tanker[arc],  # Example capacity for tankers
        name=f"TankerCapacity_{arc}"
    )

#pipeline selection
# model.addConstr(
#     gp.quicksum(y_pipeline_16[arc]) <= 1,  # At most one pipeline per arc
#     name=f"SinglePipeline_{arc}"
# )

#flow constraints
for node in nodes:
    inflow_pipeline = gp.quicksum(
        x_pipeline_16[arc] for arc in arcs if arc[1] == node for d in diameters
    )
    outflow_pipeline = gp.quicksum(
        x_pipeline_16[arc] for arc in arcs if arc[0] == node for d in diameters
    )
    inflow_tanker = gp.quicksum(x_tanker[arc] for arc in arcs if arc[1] == node)
    outflow_tanker = gp.quicksum(x_tanker[arc] for arc in arcs if arc[0] == node)

    # Total inflow = total outflow
    model.addConstr(
        inflow_pipeline + inflow_tanker == outflow_pipeline + outflow_tanker,
        name=f"FlowConservation_{node}"
    )

# CO2 reduction target constraint
model.addConstr(
    gp.quicksum(x_pipeline_16[arc] for arc in arcs) +
    gp.quicksum(x_tanker[arc] for arc in arcs) >= 100000,  # Example target
    name="CO2ReductionTarget"
)
model.optimize()

#get solutions and save them somewhere
# Check if an optimal solution was found
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")

    # Create an empty list to store the results
    results = []

    # Extract pipeline flow and infrastructure decisions
    for arc in arcs:
        results.append({
                "From": arc[0],
                "To": arc[1],
                "Mode": "Pipeline",
                "Diameter": 16,
                "Flow": x_pipeline_16[arc].x,  # Extract flow value
                "Infrastructure": y_pipeline_16[arc].x,  # Extract binary decision
            })

    # Extract tanker flow and infrastructure decisions
    for arc in arcs:
        results.append({
            "From": arc[0],
            "To": arc[1],
            "Mode": "Tanker",
            "Diameter": None,  # No diameter for tanker
            "Flow": x_tanker[arc].x,  # Extract flow value
            "Infrastructure": y_tanker[arc].x,  # Extract binary decision
        })

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to an Excel file
    results_df.to_excel("model_results.xlsx", index=False)
    print("Results saved to model_results.xlsx")
else:
    print("No optimal solution found.")


