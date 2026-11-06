import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility ke liye seed set karein
np.random.seed(42)

# 1. Dataset Simulation
n_records = 500
shipment_ids = [f"SHIP-{1000 + i}" for i in range(n_records)]
transport_modes = np.random.choice(['Road', 'Rail', 'Air'], size=n_records, p=[0.55, 0.30, 0.15])
shipment_volume = np.round(np.random.uniform(5.0, 120.0, size=n_records), 2) # in m3 / weight units
distance_km = np.random.randint(50, 2200, size=n_records)

delivery_times = []
transport_costs = []

for mode, vol, dist in zip(transport_modes, shipment_volume, distance_km):
    if mode == 'Air':
        speed = np.random.normal(500, 30) # km per day equivalent
        base_rate = 1.8
        delay_chance = 0.08
    elif mode == 'Rail':
        speed = np.random.normal(200, 20)
        base_rate = 0.6
        delay_chance = 0.22
    else: # Road
        speed = np.random.normal(280, 25)
        base_rate = 1.0
        delay_chance = 0.18
        
    transit_time = (dist / max(speed, 50)) + np.random.uniform(0.5, 2.0)
    if np.random.rand() < delay_chance:
        transit_time += np.random.uniform(1.5, 4.0) # Logistics bottleneck / delay
        
    delivery_times.append(round(transit_time, 2))
    
    # Cost formula based on distance, volume, and mode
    cost = (dist * base_rate * 1.5) + (vol * 12.0) + np.random.normal(150, 30)
    transport_costs.append(round(max(cost, 200.0), 2))

carrier_status = np.random.choice(['On-Time', 'Delayed'], size=n_records, p=[0.78, 0.22])

df = pd.DataFrame({
    'Shipment_ID': shipment_ids,
    'Transport_Mode': transport_modes,
    'Distance_KM': distance_km,
    'Shipment_Volume': shipment_volume,
    'Delivery_Time_Days': delivery_times,
    'Transportation_Cost': transport_costs,
    'Status': carrier_status
})

# CSV file save karein
df.to_csv("logistics_data.csv", index=False)
print("Logistics Dataset Created Successfully!")

# 2. Exploratory Data Analysis (EDA) - Central Tendencies & Correlations
print("\n--- Summary Statistics (Central Tendencies) ---")
print(df[['Distance_KM', 'Shipment_Volume', 'Delivery_Time_Days', 'Transportation_Cost']].describe())

print("\n--- Correlation Matrix ---")
numeric_cols = ['Distance_KM', 'Shipment_Volume', 'Delivery_Time_Days', 'Transportation_Cost']
corr_matrix = df[numeric_cols].corr()
print(corr_matrix)

# 3. Data Visualizations
sns.set_theme(style="whitegrid")

# Visualization 1: Cost vs Distance by Transport Mode (Scatter Plot)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Distance_KM', y='Transportation_Cost', hue='Transport_Mode', palette='tab10', s=70)
plt.title("Transportation Cost vs. Distance by Mode", fontsize=13, weight='bold')
plt.xlabel("Distance (km)")
plt.ylabel("Transportation Cost ($ / INR)")
plt.tight_layout()
plt.savefig("viz1_cost_vs_distance.png", dpi=300)
plt.close()

# Visualization 2: Delivery Time Distribution across Modes (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Transport_Mode', y='Delivery_Time_Days', palette='Set2')
plt.title("Delivery Time Distribution Across Transport Modes", fontsize=13, weight='bold')
plt.xlabel("Transport Mode")
plt.ylabel("Delivery Time (Days)")
plt.tight_layout()
plt.savefig("viz2_delivery_distribution.png", dpi=300)
plt.close()

# Visualization 3: Correlation Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Logistics Performance Metrics", fontsize=13, weight='bold')
plt.tight_layout()
plt.savefig("viz3_correlation_heatmap.png", dpi=300)
plt.close()

# Visualization 4: Carrier Performance & On-Time Status (Count Plot)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Transport_Mode', hue='Status', palette='viridis')
plt.title("On-Time vs. Delayed Shipments by Transport Mode", fontsize=13, weight='bold')
plt.xlabel("Transport Mode")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig("viz4_shipment_status_by_mode.png", dpi=300)
plt.close()

print("All 4 visualizations generated and saved successfully!")