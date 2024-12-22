import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    # Load the data, skipping the first row which contains "steel_strength"
    df = pd.read_csv(file_path, skiprows=1)
    
    # Store the formula column separately if needed
    formulas = df['formula'].copy()
    
    # Remove the formula column as we'll use the individual element compositions
    if 'formula' in df.columns:
        df = df.drop('formula', axis=1)
    
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Convert all columns to float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Define chemical elements and target variables
    chemical_elements = ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']
    target_variables = ['yield strength', 'tensile strength', 'elongation']
    
    # Separate features for imputation
    features_for_imputation = df[chemical_elements].copy()
    
    # Initialize and fit the IterativeImputer
    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy='mean',
        min_value=0  # Chemical compositions cannot be negative
    )
    
    # Perform imputation on chemical elements
    imputed_features = imputer.fit_transform(features_for_imputation)
    
    # Update the dataframe with imputed values
    df[chemical_elements] = imputed_features
    
    # For target variables, drop rows with missing values
    df = df.dropna(subset=target_variables)
    
    return df

def split_features_targets(df):
    # Define feature columns (chemical compositions)
    feature_cols = ['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']
    
    # Define target columns
    target_cols = ['yield strength', 'tensile strength', 'elongation']
    
    X = df[feature_cols]
    y = df[target_cols]
    
    return X, y

def train_models(X, y):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Initialize models for each target variable
    models = {}
    for target in y.columns:
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Fit the model
        rf_model.fit(X_train, y_train[target])
        
        # Store the model
        models[target] = {
            'model': rf_model,
            'scaler': scaler,
            'feature_importance': pd.Series(
                rf_model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        }
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test[target], y_pred)
        rmse = np.sqrt(mean_squared_error(y_test[target], y_pred))
        r2 = r2_score(y_test[target], y_pred)
        
        print(f"\nMetrics for {target}:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        ax = models[target]['feature_importance'].plot(kind='bar')
        plt.title(f'Feature Importance for {target.capitalize()}', fontsize=12, pad=20)
        plt.xlabel('Chemical Elements', fontsize=10)
        plt.ylabel('Feature Importance Score', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of each bar
        for i, v in enumerate(models[target]['feature_importance']):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'feature_importance_{target.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return models

def main():
    # Load and preprocess the data
    file_path = "database_steel_properties.csv"
    df = load_and_preprocess_data(file_path)
    
    # Split features and targets
    X, y = split_features_targets(df)
    
    # Train the models and get predictions
    models = train_models(X, y)
    
    # Create a combined feature importance plot
    plt.figure(figsize=(15, 8))
    
    # Get feature importance for each target
    importance_data = pd.DataFrame({
        'Yield Strength': models['yield strength']['feature_importance'],
        'Tensile Strength': models['tensile strength']['feature_importance'],
        'Elongation': models['elongation']['feature_importance']
    })
    
    # Plot combined feature importance
    ax = importance_data.plot(kind='bar', width=0.8)
    plt.title('Feature Importance Comparison Across Properties', fontsize=14, pad=20)
    plt.xlabel('Chemical Elements', fontsize=12)
    plt.ylabel('Feature Importance Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Properties', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
