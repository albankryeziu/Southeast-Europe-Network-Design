import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# (Optional) Utility: filter out zero-flow arcs (same as your original function)
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


def run_model_for_country(country, base_path):
    """
    Runs the CO₂ transport optimization model for a given country using country-specific data files.

    Parameters:
      - country (str): Country code (e.g., "BG", "GR", "RO", "HR").
      - base_path (str): Folder where the country-specific Excel files are stored.

    Expected files (in base_path):
      - <country>_arcs.xlsx
      - <country>_emitters.xlsx
      - <country>_storage.xlsx
      - <country>_utilizers.xlsx

    Output:
      - Saves the model results into "model_results_<country>.xlsx" with two sheets:
          • Transport Results
          • Capture Results
    """
    # ------------------------------
    # 1. Load country-specific data
    # ------------------------------
    arcs_file = f"{base_path}/{country}_arcs.xlsx"
    emitters_file = f"{base_path}/{country}_emitters.xlsx"
    storage_file = f"{base_path}/{country}_storage.xlsx"
    utilizers_file = f"{base_path}/{country}_utilizers.xlsx"

    arcs_df = pd.read_excel(arcs_file)
    data_emitters = pd.read_excel(emitters_file)
    data_storage = pd.read_excel(storage_file)
    data_utilizers = pd.read_excel(utilizers_file)

    # Get emitter IDs and count
    emitters_data = data_emitters['ID'].tolist()
    storage_data = data_storage['ID'].tolist()
    N_emitter = len(emitters_data)

    # ------------------------------
    # 2. Create arc_data dictionary
    # ------------------------------
    arc_data = {
        (row["From"], row["To"]): {
            "Distance": row["Distance (km)"],
            "Truck_Fixed_Cost": row["Truck_Fixed_Cost"],
            "Truck_Var_Cost": row["Truck_Var_Cost"],
            "Pipeline_Fixed_Costs": {
                4: row["Pipeline_Fixed_Cost_4"],
                6: row["Pipeline_Fixed_Cost_6"],
                8: row["Pipeline_Fixed_Cost_8"],
                10: row["Pipeline_Fixed_Cost_10"],
                12: row["Pipeline_Fixed_Cost_12"],
                16: row["Pipeline_Fixed_Cost_16"],
                20: row["Pipeline_Fixed_Cost_20"],
            },
            "Pipeline_Var_Costs": {
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
    # Define valid destination nodes as the union of storage and utilization IDs
    # valid_destinations = set(storage_data).union(set(data_utilizers["ID"]))
    #
    # # Filter arcs: only keep arcs where the 'From' node is an emitter
    # # and the 'To' node is either a storage or a utilization node.
    # filtered_arcs = [arc for arc in arc_data.keys() if arc[0] in emitters_data and arc[1] in valid_destinations]
    #
    # # Now use filtered_arcs in your model instead of arcs.
    # arcs = filtered_arcs
    arcs = list(arc_data.keys())
    nodes = list(set(arcs_df["From"]).union(set(arcs_df["To"])))
    print("Nodes in model:", nodes)

    # ------------------------------
    # 3. Define model parameters
    # ------------------------------
    pipeline_capacity = [290000, 660000, 1390000, 4930000]

    # ------------------------------
    # 4. Build the Gurobi model
    # ------------------------------
    model = gp.Model(f"CO2_Transport_{country}")

    # Decision variables for pipelines and trucks
    x_pipeline_16 = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_16")
    x_pipeline_8 = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_8")
    x_pipeline_6 = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_6")
    x_pipeline_4 = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_pipeline_4")
    x_capture = model.addVars(N_emitter, vtype=GRB.CONTINUOUS, name="x_capture")
    x_truck = model.addVars(arcs, vtype=GRB.CONTINUOUS, name="x_truck")

    # Binary decision variables for building pipelines, capture facilities, trucks
    y_pipeline_16 = model.addVars(arcs, vtype=GRB.BINARY, name="y_pipeline_16")
    y_pipeline_8 = model.addVars(arcs, vtype=GRB.BINARY, name="y_pipeline_8")
    y_pipeline_6 = model.addVars(arcs, vtype=GRB.BINARY, name="y_pipeline_6")
    y_pipeline_4 = model.addVars(arcs, vtype=GRB.BINARY, name="y_pipeline_4")
    y_capture = model.addVars(N_emitter, vtype=GRB.BINARY, name="y_capture")
    y_truck = model.addVars(arcs, vtype=GRB.BINARY, name="y_truck")

    # ------------------------------
    # 5. Set the Objective Function
    # ------------------------------
    model.setObjective(
        gp.quicksum(
            # Pipeline costs (fixed and variable for various diameters) + truck costs
            arc_data[arc]["Pipeline_Fixed_Costs"][16] * y_pipeline_16[arc] +
            x_pipeline_16[arc] * arc_data[arc]["Pipeline_Var_Costs"][16] +
            arc_data[arc]["Pipeline_Fixed_Costs"][8] * y_pipeline_8[arc] +
            x_pipeline_8[arc] * arc_data[arc]["Pipeline_Var_Costs"][8] +
            arc_data[arc]["Pipeline_Fixed_Costs"][6] * y_pipeline_6[arc] +
            x_pipeline_6[arc] * arc_data[arc]["Pipeline_Var_Costs"][6] +
            arc_data[arc]["Pipeline_Fixed_Costs"][4] * y_pipeline_4[arc] +
            x_pipeline_4[arc] * arc_data[arc]["Pipeline_Var_Costs"][4] +
            arc_data[arc]["Truck_Fixed_Cost"] * y_truck[arc] +
            arc_data[arc]["Truck_Var_Cost"] * x_truck[arc]
            for arc in arcs
        ) +
        gp.quicksum(
            data_emitters.loc[i, "Fixed_Cost_New"] * y_capture[i] +
            data_emitters.loc[i, "Variable_Cost"] * x_capture[i]
            for i in range(N_emitter)
        ) +
        gp.quicksum(
            (row["Total Cost"] - row["Enviromental Impact"]) *
            gp.quicksum(
                x_pipeline_4[arc] + x_pipeline_6[arc] + x_pipeline_8[arc] + x_pipeline_16[arc] + x_truck[arc]
                for arc in arcs if arc[1] == row["ID"]
            )
            for _, row in data_storage.iterrows()
        ) -
        gp.quicksum(
            row["Utilization_Cost"] *
            gp.quicksum(
                x_pipeline_4[arc] + x_pipeline_6[arc] + x_pipeline_8[arc] + x_pipeline_16[arc] + x_truck[arc]
                for arc in arcs if arc[1] == row["ID"]
            )
            for _, row in data_utilizers.iterrows()
        ),
        GRB.MINIMIZE
    )

    # ------------------------------
    # 6. Add Constraints
    # ------------------------------
    # For this country, we set a maximum truck flow constraint.
    # Map the country code to the proper key if available:
    country_map = {"BG": "Bulgaria", "GR": "Greece", "RO": "Romania", "HR": "Croatia"}
    max_truck_data_all = {
        "Romania": {"max_truck_flow": 2.7e6, "num_trucks": 100},
        "Greece": {"max_truck_flow": 2.16e6, "num_trucks": 80},
        "Bulgaria": {"max_truck_flow": 1.62e6, "num_trucks": 60},
        "Croatia": {"max_truck_flow": 1.35e6, "num_trucks": 50}
    }
    if country in country_map:
        country_key = country_map[country]
        max_truck = max_truck_data_all[country_key]["max_truck_flow"]
    else:
        max_truck = 1e5  # fallback

    # model.addConstr(
    #     gp.quicksum(x_truck[arc] for arc in arcs) <= max_truck*y_truck[arc],
    #     name="TotalTruckFlowLimit"
    # )

    # Pipeline capacity constraints and truck capacity constraints
    for arc in arcs:
        model.addConstr(x_pipeline_16[arc] <= pipeline_capacity[3] * y_pipeline_16[arc],
                        name=f"PipelineCapacityMax_{arc}_16")
        model.addConstr(x_pipeline_8[arc] <= pipeline_capacity[2] * y_pipeline_8[arc],
                        name=f"PipelineCapacityMax_{arc}_8")
        model.addConstr(x_pipeline_6[arc] <= pipeline_capacity[1] * y_pipeline_6[arc],
                        name=f"PipelineCapacityMax_{arc}_6")
        model.addConstr(x_pipeline_4[arc] <= pipeline_capacity[0] * y_pipeline_4[arc],
                        name=f"PipelineCapacityMax_{arc}_4")
        model.addConstr(x_pipeline_16[arc] >= 0, name=f"PipelineCapacityMin_{arc}_16")
        model.addConstr(x_pipeline_8[arc] >= 0, name=f"PipelineCapacityMin_{arc}_8")
        model.addConstr(x_pipeline_6[arc] >= 0, name=f"PipelineCapacityMin_{arc}_6")
        model.addConstr(x_pipeline_4[arc] >= 0, name=f"PipelineCapacityMin_{arc}_4")

        max_truck_capacity = ( 2 * 60 * 5000) / (arc_data[arc]["Distance"] + 62.5)
        model.addConstr(x_truck[arc] <= max_truck*y_truck[arc])
        # model.addConstr(x_truck[arc] <= max_truck_capacity * y_truck[arc],
        #                 name=f"TankerCapacity_{arc}")

    M = 1e7  # A large constant (if needed in additional constraints)

    # Flow conservation at emitters: net outflow minus inflow equals capture
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
        model.addConstr(outflow_pipeline - inflow_pipeline == x_capture[emitters_data.index(emitter)],
                        name=f"FlowConservationNet_Emitter_{emitter}")

    # Capture facility installation constraints at emitters
    for i, row in data_emitters.iterrows():
        model.addConstr(x_capture[i] == row["Upper_Bound"], name=f"CaptureConstraint_{row['ID']}_lower")
        #model.addConstr(x_capture[i] <= row["Emission (ton/year)"], name=f"CaptureConstraint_{row['ID']}_upper")

    # Flow constraints for storage nodes: inflow must be no more than injection rate
    for storage_node in storage_data:
        inflow_to_storage = (
                gp.quicksum(x_pipeline_4[arc] for arc in arcs if arc[1] == storage_node) +
                gp.quicksum(x_pipeline_6[arc] for arc in arcs if arc[1] == storage_node) +
                gp.quicksum(x_pipeline_8[arc] for arc in arcs if arc[1] == storage_node) +
                gp.quicksum(x_pipeline_16[arc] for arc in arcs if arc[1] == storage_node) +
                gp.quicksum(x_truck[arc] for arc in arcs if arc[1] == storage_node)
        )
        injection_rate = data_storage.loc[data_storage["ID"] == storage_node, "Injection rate (2050) (t/year)"]
        if not injection_rate.empty:
            model.addConstr(inflow_to_storage <= injection_rate.values[0],
                            name=f"StorageSink_{storage_node}")

    # Flow constraints for utilization nodes: inflow must be at least the site capacity
    for util_node in data_utilizers["ID"]:
        inflow_to_utilizer = (
                gp.quicksum(x_pipeline_4[arc] for arc in arcs if arc[1] == util_node) +
                gp.quicksum(x_pipeline_6[arc] for arc in arcs if arc[1] == util_node) +
                gp.quicksum(x_pipeline_8[arc] for arc in arcs if arc[1] == util_node) +
                gp.quicksum(x_pipeline_16[arc] for arc in arcs if arc[1] == util_node) +
                gp.quicksum(x_truck[arc] for arc in arcs if arc[1] == util_node)
        )
        capacity_val = data_utilizers.loc[data_utilizers["ID"] == util_node, "Capacity (Million ton)"]
        if not capacity_val.empty:
            model.addConstr(inflow_to_utilizer <= capacity_val.values[0],
                            name=f"UtilizerSink_{util_node}")

    # (Optional) Identify intermediate nodes for information
    intermediate_nodes = list(set(nodes) - set(emitters_data) - set(storage_data) - set(data_utilizers["ID"]))
    print("Intermediate Nodes:", intermediate_nodes)

    total_possible_capture = sum(data_emitters["Upper_Bound"])
    Q = total_possible_capture  # CO₂ capture target

    # CO₂ reduction target constraint
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

    # Only one pipeline can be built per arc
    for arc in arcs:
        model.addConstr(
            y_pipeline_4[arc] + y_pipeline_6[arc] + y_pipeline_8[arc] + y_pipeline_16[arc] <= 1,
            name=f"OnePipelinePerArc_{arc}"
        )

    # Only one mode of transportation per arc (pipeline(s) or truck)
    for arc in arcs:
        model.addConstr(
            y_pipeline_4[arc] + y_pipeline_6[arc] + y_pipeline_8[arc] + y_pipeline_16[arc] + y_truck[arc] <= 1,
            name=f"OneModePerArc_{arc}"
        )

    # ------------------------------
    # 7. Optimize the model
    # ------------------------------
    model.optimize()

    if model.status == GRB.UNBOUNDED:
        print("⚠️ Model is unbounded! Check for missing constraints.")
    elif model.status == GRB.INF_OR_UNBD:
        print("⚠️ Model is either infeasible or unbounded! Running feasibility check...")
        model.setParam(GRB.Param.DualReductions, 0)
        model.optimize()
        if model.status == GRB.INFEASIBLE:
            print("⚠️ Confirmed: Model is infeasible.")
        elif model.status == GRB.UNBOUNDED:
            print("⚠️ Confirmed: Model is unbounded.")
    elif model.status == GRB.INTERRUPTED:
        print("⚠️ Optimization was interrupted before completion.")
    else:
        print("Model status:", model.status)

    if model.status == GRB.INFEASIBLE:
        print("⚠️ Model is infeasible. Consider running an IIS analysis.")
        # model.computeIIS()
        # model.write("infeasible_model.ilp")
    elif model.status == GRB.OPTIMAL:
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

        results_df_unfiltered = pd.DataFrame(results)
        results_df = filter_results(results_df_unfiltered)
        capture_results_df = pd.DataFrame(capture_results)

        output_file = f"{base_path}/model_results_{country}.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            results_df.to_excel(writer, sheet_name="Transport Results", index=False)
            capture_results_df.to_excel(writer, sheet_name="Capture Results", index=False)

        print(f"Results saved to {output_file}")
    else:
        print("No optimal solution found.")
        total_storage_capacity = sum(data_storage["Injection rate (2050) (t/year)"])
        total_utilization_capacity = sum(data_utilizers["Capacity (Million ton)"])
        print(f"Total Storage Capacity: {total_storage_capacity}")
        print(f"Total Utilization Capacity: {total_utilization_capacity}")
        print(f"CO₂ Reduction Target: {Q}")

def run_model_for_all_countries(country_list, base_path):
    """
    Iterates over a list of country codes and runs the CO₂ transport model for each country.

    Parameters:
      - country_list (list): List of country codes (e.g., ["BG", "GR", "RO", "HR"]).
      - base_path (str): Folder path where the country-specific Excel files are stored.

    It calls the `run_model_for_country` function for each country in the list.
    """
    for country in country_list:
        print(f"\n--- Running model for {country} ---")
        run_model_for_country(country, base_path)

# Example usage:
base_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel"
country_list = ["BG", "GR", "RO", "HR"]
run_model_for_all_countries(country_list, base_path)
