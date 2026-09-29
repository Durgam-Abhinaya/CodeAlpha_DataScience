import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"C:\\Users\\durga\\OneDrive\\Desktop\\codeAlpha\\Task_2\\dataset\\Unemployment in India.csv"
)

print("\n========== DATASET ==========")
print(df.head())


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

print("\n========== CLEANED COLUMN NAMES ==========")
print(df.columns.tolist())


# ============================================================
# DATA CLEANING
# ============================================================

print("\n========== DATA CLEANING ==========")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCompletely Empty Rows:")
print(df.isnull().all(axis=1).sum())

df = df.dropna(how="all")

print("\nShape after removing empty rows:")
print(df.shape)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)


# ============================================================
# DATE CONVERSION
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

print("\n========== DATE INFORMATION ==========")

print("Minimum Date:", df["Date"].min())
print("Maximum Date:", df["Date"].max())


# ============================================================
# REGION ANALYSIS
# ============================================================

print("\n========== REGION INFORMATION ==========")

print("Number of Regions:", df["Region"].nunique())

print("\nRegions:")
print(df["Region"].unique())


# ============================================================
# AREA DISTRIBUTION
# ============================================================

print("\n========== AREA DISTRIBUTION ==========")

print(df["Area"].value_counts())


# ============================================================
# UNEMPLOYMENT STATISTICS
# ============================================================

print("\n========== UNEMPLOYMENT STATISTICS ==========")

unemployment = df["Estimated Unemployment Rate (%)"]

print(unemployment.describe())

average_unemployment = unemployment.mean()

print(
    "\nAverage Unemployment Rate:",
    round(average_unemployment, 2),
    "%"
)


# ============================================================
# HIGHEST UNEMPLOYMENT
# ============================================================

print("\n========== HIGHEST UNEMPLOYMENT ==========")

max_index = unemployment.idxmax()

max_row = df.loc[max_index]

print(max_row)


# ============================================================
# LOWEST UNEMPLOYMENT
# ============================================================

print("\n========== LOWEST UNEMPLOYMENT ==========")

min_index = unemployment.idxmin()

min_row = df.loc[min_index]

print(min_row)


# ============================================================
# OVERALL UNEMPLOYMENT TREND
# ============================================================

monthly_unemployment = df.groupby(
    "Date"
)["Estimated Unemployment Rate (%)"].mean()

print("\n========== MONTHLY UNEMPLOYMENT ==========")

print(monthly_unemployment)


plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment.index,
    monthly_unemployment.values,
    marker="o"
)

plt.title("Overall Unemployment Rate Trend")

plt.xlabel("Date")

plt.ylabel("Unemployment Rate (%)")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_2\outputs\unemployment_trend.png"
)


plt.close()


# ============================================================
# MONTHLY TREND
# ============================================================

df["Month"] = df["Date"].dt.month

df["Month_Name"] = df["Date"].dt.strftime("%B")


monthly_avg = df.groupby(
    ["Month", "Month_Name"]
)["Estimated Unemployment Rate (%)"].mean()

monthly_avg = monthly_avg.sort_index()

print("\n========== MONTHLY AVERAGE ==========")

print(monthly_avg)


plt.figure(figsize=(10, 6))

plt.plot(
    monthly_avg.index.get_level_values("Month_Name"),
    monthly_avg.values,
    marker="o"
)

plt.title("Average Monthly Unemployment Rate")

plt.xlabel("Month")

plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.savefig(
   
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_2\outputs\unemployment_trend.png"

)

plt.close()


# ============================================================
# COVID-19 ANALYSIS
# ============================================================

pre_covid = df[
    df["Date"] < "2020-03-01"
]

covid_period = df[
    df["Date"] >= "2020-03-01"
]

pre_covid_avg = pre_covid[
    "Estimated Unemployment Rate (%)"
].mean()

covid_avg = covid_period[
    "Estimated Unemployment Rate (%)"
].mean()

covid_change = covid_avg - pre_covid_avg

print("\n========== COVID-19 IMPACT ==========")

print(
    "Pre-COVID Average:",
    round(pre_covid_avg, 2),
    "%"
)

print(
    "COVID Period Average:",
    round(covid_avg, 2),
    "%"
)

print(
    "Change:",
    round(covid_change, 2),
    "percentage points"
)


# ============================================================
# COVID-19 VISUALIZATION
# ============================================================

covid_monthly = df.groupby(
    "Date"
)["Estimated Unemployment Rate (%)"].mean()

plt.figure(figsize=(12, 6))

plt.plot(
    covid_monthly.index,
    covid_monthly.values,
    marker="o"
)

plt.axvline(
    pd.Timestamp("2020-03-01"),
    linestyle="--",
    label="COVID-19 Period Begins"
)

plt.title("Impact of COVID-19 on Unemployment Rate")

plt.xlabel("Date")

plt.ylabel("Unemployment Rate (%)")

plt.legend()

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.savefig(
   
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_2\outputs\unemployment_trend.png"

)

plt.close()


# ============================================================
# RURAL VS URBAN ANALYSIS
# ============================================================

print("\n========== RURAL VS URBAN ==========")

area_average = df.groupby(
    "Area"
)["Estimated Unemployment Rate (%)"].mean()

print(area_average)


plt.figure(figsize=(8, 6))

sns.barplot(
    data=df,
    x="Area",
    y="Estimated Unemployment Rate (%)"
)

plt.title("Average Unemployment Rate: Rural vs Urban")

plt.xlabel("Area")

plt.ylabel("Average Unemployment Rate (%)")

plt.tight_layout()

plt.savefig(
   
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_2\outputs\unemployment_trend.png"

)

plt.close()


# ============================================================
# REGIONAL UNEMPLOYMENT ANALYSIS
# ============================================================

print("\n========== REGIONAL UNEMPLOYMENT ==========")

regional_average = df.groupby(
    "Region"
)["Estimated Unemployment Rate (%)"].mean()

regional_average = regional_average.sort_values(
    ascending=False
)

print(regional_average)


plt.figure(figsize=(12, 8))

regional_average.sort_values().plot(
    kind="barh"
)

plt.title("Average Unemployment Rate by Region")

plt.xlabel("Average Unemployment Rate (%)")

plt.ylabel("Region")

plt.tight_layout()

plt.savefig(
    
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_2\outputs\unemployment_trend.png"
)

plt.close()


# ============================================================
# REGIONAL COVID-19 COMPARISON
# ============================================================

regional_pre_covid = pre_covid.groupby(
    "Region"
)["Estimated Unemployment Rate (%)"].mean()

regional_covid = covid_period.groupby(
    "Region"
)["Estimated Unemployment Rate (%)"].mean()


regional_covid_comparison = pd.DataFrame({
    "Pre-COVID": regional_pre_covid,
    "COVID Period": regional_covid
})


regional_covid_comparison["Change"] = (
    regional_covid_comparison["COVID Period"]
    -
    regional_covid_comparison["Pre-COVID"]
)


print("\n========== REGIONAL COVID-19 IMPACT ==========")

print(
    regional_covid_comparison.sort_values(
        "Change",
        ascending=False
    )
)


# ============================================================
# LABOUR PARTICIPATION VS UNEMPLOYMENT
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Estimated Labour Participation Rate (%)",
    y="Estimated Unemployment Rate (%)",
    hue="Area"
)

plt.title(
    "Labour Participation Rate vs Unemployment Rate"
)

plt.xlabel(
    "Labour Participation Rate (%)"
)

plt.ylabel(
    "Unemployment Rate (%)"
)

plt.tight_layout()

plt.savefig(
    
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_2\outputs\unemployment_trend.png"
)

plt.close()


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\n========== CORRELATION ANALYSIS ==========")

numeric_columns = [
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]

correlation = df[numeric_columns].corr()

print(correlation)


# ============================================================
# CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
  
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_2\outputs\unemployment_trend.png"

)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print(
    "Total observations:",
    len(df)
)

print(
    "Number of regions:",
    df["Region"].nunique()
)

print(
    "Average unemployment rate:",
    round(average_unemployment, 2),
    "%"
)

print(
    "Pre-COVID unemployment:",
    round(pre_covid_avg, 2),
    "%"
)

print(
    "COVID-period unemployment:",
    round(covid_avg, 2),
    "%"
)

print(
    "COVID-period change:",
    round(covid_change, 2),
    "percentage points"
)

print("\n========== ANALYSIS COMPLETED ==========")

print(
    "All graphs have been saved in:"
)

print(
    r"C:\Users\durga\OneDrive\Desktop\codeAlpha\Task_2_Unemployment\outputs"
)