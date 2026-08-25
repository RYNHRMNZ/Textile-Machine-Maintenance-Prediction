# Textile-Machine-Maintenance-Prediction
A machine learning system that predicts whether textile manufacturing machines require maintenance, built to support predictive (rather than reactive or fixed-schedule) maintenance strategies in smart manufacturing environments.

## Overview

This project uses a classic ML model trained on tabular telemetry-style data (machine parameters such as temperature, vibration, operating hours, etc.) to classify whether a machine is likely to need maintenance soon. Predictions are surfaced through a real-time monitoring dashboard, allowing operators to catch early warning patterns before a breakdown occurs — reducing unplanned downtime and maintenance costs.

## Features
- ML model trained on machine telemetry data to predict maintenance needs
- Real-time dashboard for monitoring machine health indicators
- Alert system that flags patterns indicating potential upcoming failure

## Dataset
Trained on a public tabular dataset (1,000 rows) sourced from Kaggle, representing simulated/real machine operational parameters. *(Note: intended as a proof-of-concept; production use would require validation against real factory telemetry data.)*

## Tech Stack
- [fill in: e.g. Python, scikit-learn/XGBoost, Pandas]
- [fill in: dashboard framework — e.g. Streamlit, Flask, React]

## Motivation
Unplanned machine downtime is a major cause of production loss in textile manufacturing, particularly for small-to-medium manufacturers that lack access to expensive industrial IoT systems. This project explores a lightweight, accessible predictive maintenance approach as an alternative.

## How to use it
since this project is need a real-time data, we've made a sensor simulation that will be needed to run so the model that we've trained can run properly in real time
step 1 : open terminal or CMD for windows
step 2 : then run this command : 
    [ python -m streamlit run streamlit_app.py ]
    or
    [ streamlit run streamlit_app.py ]
  Note : 
    run this after you open the directory file
    
## Status
🚧 Work in progress / Proof of concept
