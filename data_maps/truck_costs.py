# Updated data for the new scenario

purchase_price = 120000  # EUR price of the new truck
resale_value = 20000  # EUR saling after 1000000km done
lifetime_km = 1000000  # km

# Calculate depreciation cost per km
depreciation_cost_per_km = (purchase_price - resale_value) / lifetime_km
annual_working_hours = 5000  # hours
average_speed = 50  # km/h

# Calculate total distance per year
total_distance_per_year = annual_working_hours * average_speed  # km

# Maintenance cost assumption (average)
maintenance_cost_per_km = 0.18  # EUR/km

# Calculate annual maintenance and depreciation costs
annual_depreciation_cost = depreciation_cost_per_km * total_distance_per_year
annual_maintenance_cost = maintenance_cost_per_km * total_distance_per_year

print(annual_depreciation_cost)
print(annual_maintenance_cost+annual_depreciation_cost)

annual_cost_per_km=(annual_maintenance_cost+annual_depreciation_cost)/total_distance_per_year
print(annual_cost_per_km)
