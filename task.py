# US Accidents (2016–2023) Analysis Script

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

# Load dataset
df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_March23.csv")

# Basic info
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Drop rows with missing critical values
df = df.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng", "Weather_Condition", "Visibility(mi)", "Severity"])

# Convert Start_Time to datetime
df["Start_Time"] = pd.to_datetime(df["Start_Time"])

# Feature engineering
df["Hour"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Start_Time"].dt.day_name()
df["Month"] = df["Start_Time"].dt.month_name()

# -------------------------------
# 1. Accidents by Hour of Day
plt.figure(figsize=(12,6))
sns.countplot(x="Hour", data=df, palette="viridis")
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 2. Top Weather Conditions
top_weather = df["Weather_Condition"].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_weather.index, y=top_weather.values, palette="coolwarm")
plt.title("Top 10 Weather Conditions During Accidents")
plt.xticks(rotation=45)
plt.ylabel("Number of Accidents")
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Visibility vs Severity
plt.figure(figsize=(10,6))
sns.boxplot(x="Severity", y="Visibility(mi)", data=df)
plt.title("Visibility vs Severity of Accidents")
plt.xlabel("Severity Level")
plt.ylabel("Visibility (miles)")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Accident Hotspots Heatmap
sample_df = df[["Start_Lat", "Start_Lng"]].dropna().sample(10000)

m = folium.Map(location=[39.5, -98.35], zoom_start=4)
HeatMap(data=sample_df.values.tolist(), radius=8).add_to(m)

# To display in Colab or Kaggle, use:
m.save("accident_heatmap.html")
print("Heatmap saved as 'accident_heatmap.html'")

# -------------------------------
# 5. Correlation with Severity
numeric_cols = df.select_dtypes(include=["float64", "int64"]).drop(columns=["Severity"])
correlations = numeric_cols.corrwith(df["Severity"]).sort_values()

plt.figure(figsize=(10,5))
sns.barplot(x=correlations.index, y=correlations.values, palette="magma")
plt.title("Correlation of Numeric Features with Severity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
