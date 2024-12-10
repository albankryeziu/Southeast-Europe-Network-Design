# Create arcs based on rules
arcs = []

for _, row in node_pairs.iterrows():
    if row["SourceFile1"] == "emitters":  # Source node
        if row["SourceFile2"] == "emitters" and row["Node1"] != row["Node2"]:
            # Source to Source
            arcs.append((row["Node1"], row["Node2"], row["Distance (km)"]))
        elif row["SourceFile2"] in ["utilizers", "storage"]:
            # Source to Sink
            arcs.append((row["Node1"], row["Node2"], row["Distance (km)"]))
    # No connections from sinks to sources or sinks
    # (This part is excluded by default as we skip these combinations)

# Convert arcs to a DataFrame for easier manipulation and counting
arcs_df = pd.DataFrame(arcs, columns=["From", "To", "Distance (km)"])

# Save the arcs to Excel
arcs_df.to_excel("valid_arcs.xlsx", index=False)
print(f"Valid arcs saved to valid_arcs.xlsx")

# Print total number of valid arcs
print(f"Total number of valid arcs: {len(arcs)}")
