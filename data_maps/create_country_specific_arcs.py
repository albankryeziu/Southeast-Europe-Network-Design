import pandas as pd

# Load the full dataset
file_path = arcs_file_path = r"C:\Users\Alban\OneDrive - University of Groningen\Desktop\research\Master thesis\daniel\valid_arcs.xlsx"  # Update with the actual file path
df = pd.read_excel(file_path)

# Extract country prefix from the "From" column
df["From_Country"] = df["From"].str.extract(r"^([A-Z]+)")
df["To_Country"] = df["To"].str.extract(r"^([A-Z]+)")

# Filter only arcs where From and To are in the same country
df_filtered = df[df["From_Country"] == df["To_Country"]].drop(columns=["From_Country", "To_Country"])

# Split the data by country and save each as a separate Excel file
for country in df_filtered["From"].str.extract(r"^([A-Z]+)")[0].unique():
    country_df = df_filtered[df_filtered["From"].str.startswith(country)]
    country_file = f"{country}_arcs.xlsx"
    country_df.to_excel(country_file, index=False)
    print(f"✅ Saved {country_file}")
