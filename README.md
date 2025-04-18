# SI-Traffic_Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

# ======================= Data Processing =======================
def clean_number(value):
    """Remove commas from a string and convert it to a numeric value."""
    return pd.to_numeric(value.replace(',', ''), errors='coerce')

# Define crossing names and Staten Island subset
crossing_names = [
    "All Crossings",
    "George Washington Bridge",
    "Lincoln Tunnel",
    "Holland Tunnel",
    "Goethals Bridge",
    "Outerbridge Crossing",
    "Bayonne Bridge"
]
staten_island = ["Goethals Bridge", "Outerbridge Crossing", "Bayonne Bridge"]
crossing_order = ["Goethals Bridge", "Outerbridge Crossing", "Bayonne Bridge"]

# Vehicle categories as in the CSV and desired display order (new_order)
original_categories = ["Automobiles", "Buses", "Truck"]
new_order = ["Truck", "Buses", "Automobiles"]
label_map = {"Truck": "TRUCK", "Buses": "BUS", "Automobiles": "AUTO"}

# --- Process 2024 Data ---
df2024_raw = pd.read_csv("traffic_2024.csv", header=None)
df2024_raw.columns = df2024_raw.iloc[1]   # Header row is index=1 for 2024
df2024_clean = df2024_raw.drop([0, 1]).reset_index(drop=True)

data_2024 = []
for cat in original_categories:
    df_cat = df2024_clean[df2024_clean["(Eastbound Traffic)"].str.contains(cat, case=False, na=False)].copy()
    df_cat['Crossing'] = crossing_names  # assign known order
    df_cat_si = df_cat[df_cat['Crossing'].isin(staten_island)].copy()
    df_cat_si['Jan_numeric'] = df_cat_si['Jan'].apply(clean_number)
    df_cat_si['Vehicle'] = cat
    data_2024.append(df_cat_si[['Crossing', 'Vehicle', 'Jan_numeric']])
df2024_final = pd.concat(data_2024, ignore_index=True)
df2024_final.rename(columns={'Jan_numeric': 'Jan_2024'}, inplace=True)

# --- Process 2025 Data ---
df2025_raw = pd.read_csv("traffic_2025.csv", header=None)
df2025_raw.columns = df2025_raw.iloc[2]   # Header row is index=2 for 2025
df2025_clean = df2025_raw.drop([0, 1, 2]).reset_index(drop=True)

data_2025 = []
for cat in original_categories:
    df_cat = df2025_clean[df2025_clean["(Eastbound Traffic)"].str.contains(cat, case=False, na=False)].copy()
    df_cat['Crossing'] = crossing_names
    df_cat_si = df_cat[df_cat['Crossing'].isin(staten_island)].copy()
    df_cat_si['Jan_numeric'] = df_cat_si['Jan'].apply(clean_number)
    df_cat_si['Vehicle'] = cat
    data_2025.append(df_cat_si[['Crossing', 'Vehicle', 'Jan_numeric']])
df2025_final = pd.concat(data_2025, ignore_index=True)
df2025_final.rename(columns={'Jan_numeric': 'Jan_2025'}, inplace=True)

# --- Merge & Compute Differences ---
df2024_comp = df2024_final[['Crossing', 'Vehicle', 'Jan_2024']]
df2025_comp = df2025_final[['Crossing', 'Vehicle', 'Jan_2025']]
df_compare = pd.merge(df2024_comp, df2025_comp, on=['Crossing', 'Vehicle'])
df_compare['Difference'] = df_compare['Jan_2025'] - df_compare['Jan_2024']
df_compare['Percent_Change'] = (df_compare['Difference'] / df_compare['Jan_2024']) * 100

print("Merged Comparison Data:")
print(df_compare)

# ======================= Pie Charts =======================
pie2024 = df2024_final.groupby('Vehicle')['Jan_2024'].sum()
pie2025 = df2025_final.groupby('Vehicle')['Jan_2025'].sum()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
pie_colors = ['#66c2a5', '#fc8d62', '#8da0cb']
axs[0].pie(pie2024, labels=pie2024.index, autopct='%1.1f%%', startangle=90, colors=pie_colors)
axs[0].set_title('Vehicle Distribution - January 2024\n(Staten Island Crossings)', fontsize=14)
axs[1].pie(pie2025, labels=pie2025.index, autopct='%1.1f%%', startangle=90, colors=pie_colors)
axs[1].set_title('Vehicle Distribution - January 2025\n(Staten Island Crossings)', fontsize=14)
plt.tight_layout()
plt.show()

# ======================= Overall Percentage Change Bar Chart (Real Totals) =======================
df_total = df_compare.groupby('Vehicle').agg({'Jan_2024': 'sum', 'Jan_2025': 'sum'}).reset_index()
df_total['Real_Percent_Change'] = ((df_total['Jan_2025'] - df_total['Jan_2024']) / df_total['Jan_2024']) * 100
df_total = df_total.set_index('Vehicle').reindex(new_order).reset_index()
df_total['VehicleLabel'] = df_total['Vehicle'].map(label_map)
bar_colors = {"Truck": "#2ca02c", "Buses": "#ff7f0e", "Automobiles": "#1f77b4"}
colors = [bar_colors[v] for v in df_total['Vehicle']]

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(df_total['VehicleLabel'], df_total['Real_Percent_Change'], color=colors, edgecolor='black')
ax.set_xlabel("Vehicle Type", fontsize=12)
ax.set_ylabel("Percentage Change (%)", fontsize=12)
ax.set_title("Overall Percentage Change (Real Totals) for January Traffic", fontsize=14)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height,
            f"{height:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
plt.tight_layout()
plt.show()

# ======================= Horizontal Bar Charts per Crossing (Percentage Change) =======================
# New label function: center labels inside the bars in white bold text.
def label_bar_center(ax, x_val, y_val, text):
    # Compute center of the bar: if x_val is 0, use 0; otherwise, halfway between 0 and x_val.
    x_center = x_val / 2 if x_val != 0 else 0
    ax.text(x_center, y_val, text, va='center', ha='center', fontsize=10,
            color='white', fontweight='bold', zorder=5)

fig, axes = plt.subplots(1, len(crossing_order), figsize=(18, 6), sharex=False)
# Adjust left margin so labels don't overlap axes.
plt.subplots_adjust(left=0.25)

bar_colors = {"Truck": "#2ca02c", "Buses": "#ff7f0e", "Automobiles": "#1f77b4"}

for i, cr in enumerate(crossing_order):
    ax = axes[i]
    df_cr = df_compare[df_compare['Crossing'] == cr].copy()
    # Reindex in desired order: Truck, Buses, Automobiles.
    df_cr = df_cr.set_index('Vehicle').reindex(new_order).reset_index()
    df_cr['VehicleLabel'] = df_cr['Vehicle'].map(label_map)
    
    y_pos = np.arange(len(df_cr))
    
    ax.axvline(0, color='black', linewidth=0.8, zorder=1)
    for j, row in df_cr.iterrows():
        x_val = row['Percent_Change']
        color = bar_colors.get(row['Vehicle'], 'gray')
        ax.barh(y_pos[j], x_val, color=color, edgecolor='black', zorder=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_cr['VehicleLabel'], fontsize=12)
    ax.set_xlabel("Percentage Change (%)", fontsize=12)
    ax.set_title(cr, fontsize=14)
    
    ax.relim()
    ax.autoscale_view()
    
    # Place labels at the center of each bar
    for j, row in df_cr.iterrows():
        x_val = row['Percent_Change']
        label_bar_center(ax, x_val, y_pos[j], f"{x_val:.1f}%")
        
plt.suptitle("Horizontal Bar Charts of % Change (Jan 2024 vs. Jan 2025) by Crossing & Vehicle", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ======================= Diverging Bar Chart (All Records) =======================
df_compare['Label'] = df_compare['Crossing'] + " (" + df_compare['Vehicle'] + ")"
df_diverge = df_compare.sort_values(by='Percent_Change')

fig, ax = plt.subplots(figsize=(8, 6))
div_colors = df_diverge['Percent_Change'].apply(lambda x: '#2ca02c' if x >= 0 else '#d62728')
ax.barh(df_diverge['Label'], df_diverge['Percent_Change'], color=div_colors, edgecolor='black')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel("Percentage Change (%)", fontsize=12)
ax.set_title("Diverging Bar Chart of % Change (Jan 2024 vs. Jan 2025)", fontsize=14)

for i, v in enumerate(df_diverge['Percent_Change']):
    offset = 1 if v >= 0 else -1
    ax.text(v + offset, i, f"{v:.1f}%", color='black', va='center', fontweight='bold')
plt.tight_layout()
plt.show()

# ======================= Interactive Map (Detailed Breakdown) =======================
map_data = {}
for cr in crossing_order:
    df_cr = df_compare[df_compare['Crossing'] == cr]
    lines = [f"<strong>{cr}</strong>"]
    for _, row in df_cr.iterrows():
        lines.append(
            f"{row['Vehicle']}: Jan 2024 = {row['Jan_2024']:,}, Jan 2025 = {row['Jan_2025']:,} ({row['Percent_Change']:.1f}%)"
        )
    map_data[cr] = "<br>".join(lines)

coords = {
    "Goethals Bridge": [40.63934925311288, -74.19869516103142],
    "Outerbridge Crossing": [40.52485869433162, -74.24691393916278],
    "Bayonne Bridge": [40.646164919143295, -74.14188795691581]
}

m = folium.Map(location=[40.63, -74.15], zoom_start=11)
for cr in crossing_order:
    lat, lon = coords.get(cr, [40.63, -74.15])
    popup_text = map_data.get(cr, "")
    folium.Marker(
        location=[lat, lon],
        popup=popup_text,
        tooltip=cr,
        icon=folium.Icon(color="darkblue", icon="info-sign")
    ).add_to(m)
m.save("SI_Vehicle_Breakdown_Map.html")
print("\nInteractive map saved as SI_Vehicle_Breakdown_Map.html")
