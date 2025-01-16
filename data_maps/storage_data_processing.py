import pandas as pd

# Load the Excel file with all sheets
file_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\data\all_countries.xlsx"
excel_file = pd.ExcelFile(file_path)

# Load the specific sheet you want to update
data = pd.read_excel(file_path, sheet_name="storage_sites")

# Define storage cost parameters
storage_costs = {
    "Onshore - DOGF": {"CAPEX": [4, 3, 2], "OPEX": 2},
    "Offshore - DOGF": {"CAPEX": [8, 6, 4], "OPEX": 3},
    "Onshore - SA": {"CAPEX": [7, 5, 3], "OPEX": 4},
    "Offshore - SA": {"CAPEX": [14, 10, 6], "OPEX": 5},
}

# Corrected function to categorize and calculate costs
def calculate_storage_cost(row):
    type_value = str(row["Type"])  # Handle NaN values
    if "Hydrocarbon field" in type_value:
        if row["Offshore"] == "yes":
            category = "Offshore - DOGF"
        else:
            category = "Onshore - DOGF"
    else:
        if row["Offshore"] == "yes":
            category = "Offshore - SA"
        else:
            category = "Onshore - SA"

    # Determine the CAPEX based on injection rate
    injection_rate = row["Injection rate (2050) (t/year)"]
    if injection_rate <= 1_000_000:
        capex = storage_costs[category]["CAPEX"][0]
    elif injection_rate <= 3_000_000:
        capex = storage_costs[category]["CAPEX"][1]
    else:
        capex = storage_costs[category]["CAPEX"][2]

    # Calculate total storage cost per ton
    opex = storage_costs[category]["OPEX"]
    total_cost = capex + opex

    return pd.Series([category, capex, opex, total_cost], index=["Category", "CAPEX", "OPEX", "Total Cost"])

# Apply the function to the DataFrame
data[["Category", "CAPEX", "OPEX", "Total Cost"]] = data.apply(calculate_storage_cost, axis=1)

# Use openpyxl to preserve all sheets
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    # Write only the updated sheet back to the file
    data.to_excel(writer, sheet_name="storage_sites", index=False)

print("Sheet successfully updated and saved!")
