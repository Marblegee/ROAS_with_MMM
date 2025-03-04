import os
import pandas as pd
import numpy as np
import pytimetk as tk
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt

from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM

def main():
    # DATA LOADING ----
    data = pd.read_csv("data/synthetic_marketing_data_2020_2024.csv", parse_dates=['DATE'])

    # Quick overview of the data
    print(data.info())
    print(data.head())

    # Standardize column names
    df = data.copy()
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]  

    # Reshape data for visualization
    df_melted = df.melt(
        id_vars=["date"], 
        value_vars=["sales", "tv", "google_ads", "social_media", "radio", "print"]
    )

    # Quick overview of the melted data
    print(df_melted.info())
    print(df_melted.head())

    # Calculate average spend on the original DataFrame
    ave_spend = df[['tv', 'google_ads', 'social_media', 'radio', 'print']].mean()
    print("Average Spend per Channel:")
    print(ave_spend)

    # FEATURE ENGINEERING ----
    df_features = df.assign(
        year=df["date"].dt.year,
        month=df["date"].dt.month,
        dayofyear=df["date"].dt.dayofyear,
        trend=np.arange(len(df)),  # Ensure trend is a numeric sequence
    )

    # Handle potential missing values
    df_features.fillna(0, inplace=True)

    # Compute spend proportions
    total_spend_per_channel = df_features[['tv', 'google_ads', 'social_media', 'radio', 'print']].sum(axis=0)
    spend_proportion = total_spend_per_channel / total_spend_per_channel.sum()

    # Prior distribution setup
    HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)
    n_channels = len(total_spend_per_channel)
    prior_sigma = HALFNORMAL_SCALE * n_channels * spend_proportion.values

    # my_config = {
    #     "progressbar" : True,
    #     "cores" : 10,
    # }

    # MODEL TRAINING ----
    my_config = {
        "progressbar": True,
        "cores": min(4, os.cpu_count() // 2),
        "chains": 2,
        "tune": 500,
        "draws": 1000,
        "target_accept": 0.95,
    }

    # Initialize Model
    mmm = DelayedSaturatedMMM(
        date_column="date",
        channel_columns=['tv', 'google_ads', 'social_media', 'radio', 'print'],
        control_columns=['trend', 'year', 'month'],
        adstock_max_lag=4,  # Reduced for faster training
        yearly_seasonality=2,
        sampler_config=my_config,
        nuts_sampler="jitter+adapt_diag",
    )

    X = df_features.drop(columns=["sales"])
    y = df_features["sales"]

    # Train with optimizations
    mmm.fit(X, y, target_accept=0.95, max_treedepth=12, init="adapt_diag", random_seed=42)

    # Save Model
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    mmm.save(os.path.join(model_dir, "mmm_base_model.pkl"))

    # Load Model & Analyze
    loaded_mmm = DelayedSaturatedMMM.load(os.path.join(model_dir, "mmm_base_model.pkl"))
    loaded_mmm.plot_components_contributions()
    loaded_mmm.graphviz()
    loaded_mmm.plot_direct_contribution_curves()

if __name__ == "__main__":
    main()
