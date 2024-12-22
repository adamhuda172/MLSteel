# Steel Properties Prediction

This project implements a machine learning model to predict mechanical properties of steel based on its chemical composition.

## Features

- Predicts three mechanical properties:
  - Yield strength
  - Tensile strength
  - Elongation
- Uses Random Forest Regression for prediction
- Includes feature importance analysis
- Handles missing values in the dataset
- Provides model performance metrics (MAE, RMSE, R²)

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Place your `database_steel_properties.csv` file in the project directory.

3. Run the model:
```bash
python model.py
```

## Output

The script will:
1. Load and preprocess the data
2. Train separate models for each mechanical property
3. Print performance metrics for each model
4. Generate feature importance plots saved as PNG files

## Model Details

- Uses Random Forest Regression with 100 trees
- Features are standardized using StandardScaler
- Data is split 80/20 for training/testing
- Cross-validation is implemented for robust results

## Feature Importance

The script generates feature importance plots for each predicted property, showing which chemical elements have the strongest influence on each mechanical property.
