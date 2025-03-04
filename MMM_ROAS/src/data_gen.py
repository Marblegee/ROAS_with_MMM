import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
n_days = 365  # 1 year of daily data
channels = ['TV', 'Google Ads', 'Social Media', 'Radio', 'Print']

# Generate time series data
dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
seasonality = np.sin(2 * np.pi * dates.dayofyear / 365)  # Yearly seasonality effect
trend = np.linspace(1, 1.2, n_days)  # Slight increasing trend over time
macroeconomic_factor = 1 + 0.05 * np.sin(2 * np.pi * dates.dayofyear / 365)  # Economic cycle

# Generate synthetic marketing spend for each channel
spend_data = {
    channel: np.abs(np.random.normal(loc=100, scale=30, size=n_days)) for channel in channels
}
spend_df = pd.DataFrame(spend_data, index=dates)

# Generate base sales
base_sales = 500 + 50 * seasonality * macroeconomic_factor * trend + np.random.normal(0, 20, n_days)

# Define marketing channel impact coefficients
channel_effects = {
    'TV': 0.08,
    'Google Ads': 0.12,
    'Social Media': 0.10,
    'Radio': 0.05,
    'Print': 0.03,
}

# Compute sales uplift from marketing spend
marketing_effect = sum(spend_df[channel] * channel_effects[channel] for channel in channels)

# Final sales data
sales = base_sales + marketing_effect

# Create final dataset
df = spend_df.copy()
df['Seasonality'] = seasonality
df['Trend'] = trend
df['Macroeconomic'] = macroeconomic_factor
df['Sales'] = sales

# Save to CSV
df.to_csv('data/synthetic_marketing_data.csv')

# Display data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Sales')
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Synthetic Sales Time Series")
plt.legend()
plt.show()