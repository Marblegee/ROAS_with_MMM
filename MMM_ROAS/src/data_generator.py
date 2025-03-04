import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
n_days = (2024 - 2020 + 1) * 365  # Number of days from 2020-01-01 to 2024-12-31
channels = ['TV', 'Google Ads', 'Social Media', 'Radio', 'Print']

# Generate time series data
dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
seasonality = np.sin(2 * np.pi * dates.dayofyear / 365)  # Yearly seasonality effect
trend = 1 + 0.005 * np.arange(n_days)  # More realistic growing trend
macroeconomic_factor = 1 + 0.05 * np.sin(2 * np.pi * dates.dayofyear / 365)  # Economic cycle

# **MULTIPLICATION FACTOR FOR REALISTIC VALUES**
scaling_factor = 1000  # Multiply by 1000 to make values more realistic

# Generate synthetic marketing spend for each channel (rounded to 2 decimal places)
spend_data = {
    channel: np.round(np.abs(np.random.normal(loc=100, scale=30, size=n_days)) * scaling_factor, 2)
    for channel in channels
}
spend_df = pd.DataFrame(spend_data, index=dates)

# Generate base sales with a realistic growth pattern (rounded to 2 decimal places)
base_sales = np.round(
    (500 * trend * (1 + 0.2 * seasonality * macroeconomic_factor) + np.random.normal(0, 50, n_days)) * scaling_factor, 
    2
)

# Define marketing channel impact coefficients with diminishing returns
channel_effects = {
    'TV': 0.10,
    'Google Ads': 0.15,
    'Social Media': 0.12,
    'Radio': 0.07,
    'Print': 0.05,
}

# Introduce a log-based saturation effect to model diminishing returns
def diminishing_returns(spend, alpha=0.001):
    return np.log1p(spend * alpha)

# Compute sales uplift from marketing spend with diminishing returns (rounded to 2 decimal places)
marketing_effect = sum(
    np.round(diminishing_returns(spend_df[channel]) * channel_effects[channel], 2) 
    for channel in channels
)

# Final sales data with more realistic interaction between marketing and sales (rounded to 2 decimal places)
sales = np.round(base_sales + marketing_effect, 2)

# Create final dataset
df = spend_df.copy()
df['Seasonality'] = np.round(seasonality, 2)
df['Trend'] = np.round(trend, 2)
df['Macroeconomic'] = np.round(macroeconomic_factor, 2)
df['Sales'] = sales

# Save to CSV
df.to_csv("./data/synthetic_marketing_data_2020_2024.csv", index_label="DATE")

# Display data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Sales', color='blue')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Synthetic Sales Time Series (2020-2024)")
plt.legend()
plt.show()
