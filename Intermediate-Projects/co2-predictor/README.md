\# CO2 Emissions Predictor



\## Overview

This intermediate-level project predicts vehicle CO2 emissions using machine learning models like Linear Regression, Random Forest, and Gradient Boosting. It includes advanced features such as data exploration, feature engineering (polynomial features), hyperparameter tuning, cross-validation, model saving/loading, and command-line arguments for flexibility. The dataset (`co2.csv`) contains vehicle attributes like engine size, cylinders, fuel consumption, and CO2 output.



This project serves as a professional portfolio piece, demonstrating best practices in ML workflows, logging, and visualization.



\## Features

\- \*\*Data Exploration\*\*: Summary stats, missing value checks, correlation heatmaps, pairplots, and distributions.

\- \*\*Feature Engineering\*\*: Polynomial features and standardization.

\- \*\*Models\*\*: Linear Regression, Random Forest, Gradient Boosting with GridSearchCV.

\- \*\*Evaluation\*\*: MSE, MAE, R2, cross-validation, and prediction plots.

\- \*\*CLI Interface\*\*: Train, predict, plot via argparse.

\- \*\*Model Persistence\*\*: Save/load models using joblib.

\- \*\*Logging\*\*: Detailed logs for debugging and monitoring.



\## Requirements

\- Python 3.8+

\- Libraries: numpy, pandas, seaborn, matplotlib, scikit-learn, joblib



Install dependencies:

```bash

pip install numpy pandas seaborn matplotlib scikit-learn joblib





Dataset



Columns:



engine: Engine size (liters).

cylandr: Number of cylinders.

fuelcomb: Fuel consumption (L/100km).

out1: CO2 emissions (g/km, target variable).







Place co2.csv in the project directory.

How to Run



python co2\_predictor.py --train --model\_type random\_forest --plot



Make a prediction (after training):



python co2\_predictor.py --predict 3.0 6 10.5



Predicts CO2 for engine=3.0, cylinders=6, fuelcomb=10.5.



python co2\_predictor.py --plot



Custom paths:

Use --data\_path and --model\_path for custom file locations.



Example Output



Training logs metrics like MSE ~100-200, R2 ~0.95+ (depending on model).

Predictions are accurate based on strong correlations (e.g., fuelcomb ~0.99 with out1).



Improvements and Future Work



Integrate more datasets or features (e.g., vehicle type).

Add ensemble methods or neural networks.

Deploy as a web API with Flask/FastAPI.

Unit tests with pytest.



License

MIT License

