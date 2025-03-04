import pandas as pd
import numpy as np
import pytimetk as tk

from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM

# data  = pd.read_csv("synthetic_marketing_data.csv", parse_dates =['Date'])
data  = pd.read_csv("mmm_example.csv", parse_dates =['date_week'])

data.glimpse()

# Quick check

mmm = DelayedSaturatedMMM(
    date_column="date_week",
    channel_columns=["x1", "x2"],
    control_columns=[
        "event_1",
        "event_2",
        "t"
    ],
    adstock_max_lag=8,
    yearly_seasonality=2,
    sampler_config={"cores": 1}
)

mmm.default_model_config

mmm.default_sampler_config

X = data.drop('y', axis=1)
y = data['y']

mmm.fit(X, y)

mmm.graphviz().render('mmm_model', format='png')

mmm.plot_components_contributions()
mmm.plot_prior_predictive()

# # Deployment

mmm.save("mmm_quick_practice_mode.pkl")

loaded_mmm = DelayedSaturatedMMM.load("mmm_quick_practice_mode.pkl")

loaded_mmm.plot_components_contributions()

print(loaded_mmm.idata)
