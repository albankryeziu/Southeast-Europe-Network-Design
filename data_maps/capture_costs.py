import pandas as pd

# Load the Excel file with all sheets
file_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\data\all_countries.xlsx"
excel_file = pd.ExcelFile(file_path)

# Load the specific sheet you want to update
data = pd.read_excel(file_path, sheet_name="emitters")

capture_costs = {
    "Cement": {
        "Capture Rate": 0.5,
        "Total Fixed Costs (€/ton)": 37,
        "Total Variable Costs (€/ton)": 59.12,
        "Total (€/ton)": 96,
    },
    "Iron and Steel": {
        "Capture Rate": 0.6,
        "Total Fixed Costs (€/ton)": 38,
        "Total Variable Costs (€/ton)": 62.36,
        "Total (€/ton)": 101,
    },
    "Refinery": {
        "Capture Rate": 0.8,
        "Total Fixed Costs (€/ton)": 49.2,
        "Total Variable Costs (€/ton)": 68.75,
        "Total (€/ton)": 118,
    },
    "Fertilizer": {
        "Capture Rate": 0.98,
        "Total Fixed Costs (€/ton)": 12,
        "Total Variable Costs (€/ton)": 14.72,
        "Total (€/ton)": 27,
    },
}

# Corrected function to categorize and calculate costs
def calculate_capture_cost(row):
    if row["Activity"]=="cement":
        capex= capture_costs["Cement"]["Total Fixed Costs (€/ton)"]
        opex=capture_costs["Cement"]["Total Variable Costs (€/ton)"]
        upper_bound=capture_costs["Cement"]["Capture Rate"]
    elif row["Activity"]=="iron&steel":
        capex = capture_costs["Iron and Steel"]["Total Fixed Costs (€/ton)"]
        opex = capture_costs["Iron and Steel"]["Total Variable Costs (€/ton)"]
        upper_bound = capture_costs["Iron and Steel"]["Capture Rate"]
    elif row["Activity"]=="fertilizer":
        capex = capture_costs["Fertilizer"]["Total Fixed Costs (€/ton)"]
        opex = capture_costs["Fertilizer"]["Total Variable Costs (€/ton)"]
        upper_bound = capture_costs["Fertilizer"]["Capture Rate"]
    else:
        capex = capture_costs["Refinery"]["Total Fixed Costs (€/ton)"]
        opex = capture_costs["Refinery"]["Total Variable Costs (€/ton)"]
        upper_bound = capture_costs["Refinery"]["Capture Rate"]



    return pd.Series([ capex, opex, upper_bound], index=["Fixed_Cost", "Variable_Cost", "Capture_Rate"])

# Apply the function to the DataFrame
data[["Fixed_Cost", "Variable_Cost", "Capture_Rate"]] = data.apply(calculate_capture_cost, axis=1)


with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    # Write only the updated sheet back to the file
    data.to_excel(writer, sheet_name="emitters", index=False)

print("Sheet successfully updated and saved!")
