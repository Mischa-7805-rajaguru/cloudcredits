import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class CarPricePrediction:
    def __init__(self, data_path):
        """
        Initialize the Car Price Prediction model
        
        Parameters:
        data_path (str): Path to the CSV file containing car data
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load the car dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            print("\nDataset Info:")
            print(self.df.info())
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            print("Please ensure the CSV file exists at the specified location.")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.df is None:
            print("Please load data first using load_data() method")
            return
        
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print("\nDataset Shape:", self.df.shape)
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # Check data types
        print("\nData Types:")
        print(self.df.dtypes)
        
        # Price distribution
        if 'price' in self.df.columns:
            print(f"\nPrice Statistics:")
            print(f"Mean Price: ${self.df['price'].mean():.2f}")
            print(f"Median Price: ${self.df['price'].median():.2f}")
            print(f"Price Range: ${self.df['price'].min():.2f} - ${self.df['price'].max():.2f}")
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        if self.df is None:
            print("Please load data first using load_data() method")
            return False
        
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Make a copy for preprocessing
        df_processed = self.df.copy()
        
        # Print column info for debugging
        print("Column data types:")
        print(df_processed.dtypes)
        print("\nChecking for non-numeric values in each column:")
        
        # Handle missing values first
        print("\nHandling missing values...")
        
        # Check each column and handle appropriately
        for col in df_processed.columns:
            if col in ['price', 'Car_ID']:
                continue
                
            print(f"Processing column: {col}")
            
            # Check if column contains string values that should be numeric
            if df_processed[col].dtype == 'object':
                # Try to convert to numeric, if it fails, treat as categorical
                try:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    print(f"  - Converted {col} to numeric")
                except:
                    print(f"  - Keeping {col} as categorical")
        
        # Now separate numeric and categorical columns after conversion attempts
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        print(f"\nNumeric columns: {list(numeric_cols)}")
        print(f"Categorical columns: {list(categorical_cols)}")
        
        # Handle missing values for numeric columns
        for col in numeric_cols:
            if col not in ['price', 'Car_ID'] and df_processed[col].isnull().sum() > 0:
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                print(f"  - Filled {df_processed[col].isnull().sum()} missing values in {col} with median: {median_val}")
        
        # Handle missing values for categorical columns
        for col in categorical_cols:
            if col not in ['price', 'Car_ID'] and df_processed[col].isnull().sum() > 0:
                mode_val = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'unknown'
                df_processed[col].fillna(mode_val, inplace=True)
                print(f"  - Filled missing values in {col} with mode: {mode_val}")
        
        # Encode categorical variables
        print("\nEncoding categorical variables...")
        
        # Define expected categorical features (adapt based on your actual data)
        potential_categorical_features = ['carCompany', 'fueltype', 'aspiration', 'doornumber', 
                                        'carbody', 'drivewheel', 'enginelocation', 'enginetype',
                                        'cylindernumber', 'fuelsystem']
        
        # Only encode columns that exist and are actually categorical
        categorical_features_to_encode = []
        for feature in potential_categorical_features:
            if feature in df_processed.columns and df_processed[feature].dtype == 'object':
                categorical_features_to_encode.append(feature)
        
        # Also add any other object columns that weren't in our predefined list
        for col in categorical_cols:
            if col not in categorical_features_to_encode and col not in ['price', 'Car_ID']:
                categorical_features_to_encode.append(col)
        
        print(f"Categorical features to encode: {categorical_features_to_encode}")
        
        for feature in categorical_features_to_encode:
            try:
                le = LabelEncoder()
                # Clean the data first - remove any whitespace and convert to lowercase
                df_processed[feature] = df_processed[feature].astype(str).str.strip().str.lower()
                df_processed[feature] = le.fit_transform(df_processed[feature])
                self.label_encoders[feature] = le
                print(f"  - Encoded {feature}: {len(le.classes_)} unique values")
            except Exception as e:
                print(f"  - Error encoding {feature}: {str(e)}")
                # If encoding fails, drop the column
                df_processed.drop(feature, axis=1, inplace=True)
                print(f"  - Dropped problematic column: {feature}")
        
        # Separate features and target
        if 'price' not in df_processed.columns:
            print("Error: 'price' column not found in the dataset")
            return False
        
        # Convert price to numeric if it's not already
        try:
            df_processed['price'] = pd.to_numeric(df_processed['price'], errors='coerce')
            # Remove rows where price couldn't be converted
            price_nulls = df_processed['price'].isnull().sum()
            if price_nulls > 0:
                print(f"Warning: Removing {price_nulls} rows with invalid price values")
                df_processed = df_processed.dropna(subset=['price'])
        except Exception as e:
            print(f"Error processing price column: {str(e)}")
            return False
        
        # Define feature columns (excluding target and ID)
        feature_columns = [col for col in df_processed.columns if col not in ['price', 'Car_ID']]
        
        # Ensure all feature columns are numeric
        print("\nFinal data type check:")
        for col in feature_columns:
            if df_processed[col].dtype == 'object':
                print(f"Warning: {col} is still object type, attempting final conversion...")
                try:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                except:
                    print(f"Error: Could not convert {col} to numeric. Dropping column.")
                    feature_columns.remove(col)
        
        X = df_processed[feature_columns]
        y = df_processed['price']
        
        # Final check - ensure all X columns are numeric
        non_numeric_cols = X.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            print(f"Error: Still have non-numeric columns: {list(non_numeric_cols)}")
            print("Sample values from problematic columns:")
            for col in non_numeric_cols:
                print(f"{col}: {X[col].head().tolist()}")
            return False
        
        self.feature_names = feature_columns
        
        print(f"Features selected: {len(feature_columns)}")
        print(f"Feature names: {feature_columns}")
        print(f"Final dataset shape: {X.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        print("Scaling features...")
        try:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print("Scaling completed successfully!")
        except Exception as e:
            print(f"Error during scaling: {str(e)}")
            print("Data types in X_train:")
            print(self.X_train.dtypes)
            return False
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return True
    
    def feature_selection(self, k=10):
        """Select top k features using univariate feature selection"""
        print(f"\nPerforming feature selection (top {k} features)...")
        
        selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_test_selected = selector.transform(self.X_test_scaled)
        
        # Get selected feature names
        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        print(f"Selected features: {selected_features}")
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'Feature': self.feature_names,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        print("\nTop 10 Feature Scores:")
        print(feature_scores.head(10))
        
        return X_train_selected, X_test_selected, selected_features
    
    def train_model(self, use_feature_selection=True, k_features=15):
        """Train the linear regression model"""
        if self.X_train is None:
            print("Please preprocess data first using preprocess_data() method")
            return False
        
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Decide whether to use feature selection
        if use_feature_selection and len(self.feature_names) > k_features:
            X_train_final, X_test_final, selected_features = self.feature_selection(k_features)
            print(f"Using {len(selected_features)} selected features")
        else:
            X_train_final = self.X_train_scaled
            X_test_final = self.X_test_scaled
            print(f"Using all {len(self.feature_names)} features")
        
        # Train the model
        print("Training Linear Regression model...")
        self.model = LinearRegression()
        self.model.fit(X_train_final, self.y_train)
        
        # Store the final training and test sets
        self.X_train_final = X_train_final
        self.X_test_final = X_test_final
        
        print("Model training completed!")
        return True
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None:
            print("Please train the model first using train_model() method")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train_final)
        y_test_pred = self.model.predict(self.X_test_final)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print("Performance Metrics:")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"Training RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Training MAE: ${train_mae:.2f}")
        print(f"Test MAE: ${test_mae:.2f}")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.1:
            print("\nWarning: Possible overfitting detected!")
        else:
            print(f"\nModel generalization looks good (R² difference: {train_r2 - test_r2:.4f})")
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'y_test_pred': y_test_pred
        }
    
    def plot_results(self, results):
        """Plot model results"""
        plt.figure(figsize=(15, 10))
        
        # Actual vs Predicted
        plt.subplot(2, 2, 1)
        plt.scatter(self.y_test, results['y_test_pred'], alpha=0.7)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'Actual vs Predicted Prices\n(R² = {results["test_r2"]:.4f})')
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = self.y_test - results['y_test_pred']
        plt.scatter(results['y_test_pred'], residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        # Distribution of residuals
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        
        # Feature importance (coefficients)
        plt.subplot(2, 2, 4)
        if hasattr(self, 'X_train_final') and self.X_train_final.shape[1] <= 20:
            feature_names_plot = self.feature_names[:self.X_train_final.shape[1]]
            coefficients = self.model.coef_
            
            # Sort by absolute coefficient value
            coef_df = pd.DataFrame({
                'Feature': feature_names_plot,
                'Coefficient': coefficients
            })
            coef_df['Abs_Coef'] = abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Abs_Coef', ascending=True).tail(10)
            
            plt.barh(range(len(coef_df)), coef_df['Coefficient'])
            plt.yticks(range(len(coef_df)), coef_df['Feature'])
            plt.xlabel('Coefficient Value')
            plt.title('Top 10 Feature Coefficients')
        else:
            plt.text(0.5, 0.5, 'Too many features\nto display', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Feature Coefficients')
        
        plt.tight_layout()
        plt.show()
    
    def predict_single_car(self, car_features):
        """
        Predict price for a single car
        
        Parameters:
        car_features (dict): Dictionary containing car features
        """
        if self.model is None:
            print("Please train the model first using train_model() method")
            return None
        
        try:
            # Create a DataFrame with the input features
            input_df = pd.DataFrame([car_features])
            
            # Encode categorical features
            for feature, encoder in self.label_encoders.items():
                if feature in input_df.columns:
                    # Handle unknown categories
                    try:
                        input_df[feature] = encoder.transform([str(car_features[feature])])
                    except ValueError:
                        print(f"Warning: Unknown category '{car_features[feature]}' for feature '{feature}'")
                        # Use the most frequent category
                        input_df[feature] = encoder.transform([encoder.classes_[0]])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Default value for missing features
            
            # Select only the features used in training
            input_df = input_df[self.feature_names]
            
            # Scale the features
            input_scaled = self.scaler.transform(input_df)
            
            # If feature selection was used, apply the same selection
            if hasattr(self, 'X_train_final') and input_scaled.shape[1] > self.X_train_final.shape[1]:
                # This is a simplified approach - in practice, you'd want to store the selector
                input_scaled = input_scaled[:, :self.X_train_final.shape[1]]
            
            # Make prediction
            predicted_price = self.model.predict(input_scaled)[0]
            
            return predicted_price
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    def save_model(self, model_path):
        """Save the trained model"""
        import joblib
        
        if self.model is None:
            print("No model to save. Please train the model first.")
            return False
        
        try:
            # Save model and preprocessing objects
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names
            }, model_path)
            
            print(f"Model saved successfully to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_path):
        """Load a saved model"""
        import joblib
        
        try:
            saved_objects = joblib.load(model_path)
            self.model = saved_objects['model']
            self.scaler = saved_objects['scaler']
            self.label_encoders = saved_objects['label_encoders']
            self.feature_names = saved_objects['feature_names']
            
            print(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Example usage and main execution
def main():
    """Main function to demonstrate the car price prediction workflow"""
    
    # Initialize the model
    car_predictor = CarPricePrediction(r"C:\Users\ADMIN\Desktop\project\car prediction\car_data.csv")
    
    print("="*60)
    print("CAR PRICE PREDICTION SYSTEM")
    print("="*60)
    
    # Load data
    if not car_predictor.load_data():
        print("Failed to load data. Please check the file path and ensure the CSV file exists.")
        return
    
    # Explore data
    car_predictor.explore_data()
    
    # Preprocess data
    if not car_predictor.preprocess_data():
        print("Failed to preprocess data.")
        return
    
    # Train model
    if not car_predictor.train_model(use_feature_selection=True, k_features=15):
        print("Failed to train model.")
        return
    
    # Evaluate model
    results = car_predictor.evaluate_model()
    
    if results:
        # Plot results
        car_predictor.plot_results(results)
        
        # Save model
        model_save_path = r"C:\Users\ADMIN\Desktop\project\car prediction\car_price_model.pkl"
        car_predictor.save_model(model_save_path)
        
        # Example prediction for a single car
        print("\n" + "="*50)
        print("EXAMPLE PREDICTION")
        print("="*50)
        
        sample_car = {
            'symboling': 0,
            'carCompany': 'toyota',
            'fueltype': 'gas',
            'aspiration': 'std',
            'doornumber': 'four',
            'carbody': 'sedan',
            'drivewheel': 'fwd',
            'enginelocation': 'front',
            'wheelbase': 102.4,
            'carlength': 175.6,
            'carwidth': 66.5,
            'carheight': 54.1,
            'curbweight': 2548,
            'enginetype': 'ohc',
            'cylindernumber': 'four',
            'enginesize': 130,
            'fuelsystem': '2bbl',
            'boreratio': 3.19,
            'stroke': 3.40,
            'compressionratio': 8.5,
            'horsepower': 111,
            'peakrpm': 5000,
            'citympg': 21,
            'highwaympg': 27
        }
        
        predicted_price = car_predictor.predict_single_car(sample_car)
        if predicted_price:
            print(f"Predicted price for sample car: ${predicted_price:.2f}")
    
    print("\n" + "="*60)
    print("CAR PRICE PREDICTION COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()

# Additional utility functions for data preparation
def create_sample_dataset():
    """
    Create a sample dataset if you don't have one
    This is just for demonstration - replace with your actual data loading
    """
    
    print("Creating sample dataset...")
    
    # This is just sample data structure - replace with your actual data
    sample_data = {
        'Car_ID': range(1, 201),
        'symboling': np.random.randint(-2, 4, 200),
        'carCompany': np.random.choice(['toyota', 'honda', 'bmw', 'audi', 'mercedes'], 200),
        'fueltype': np.random.choice(['gas', 'diesel'], 200),
        'aspiration': np.random.choice(['std', 'turbo'], 200),
        'doornumber': np.random.choice(['two', 'four'], 200),
        'carbody': np.random.choice(['sedan', 'hatchback', 'wagon', 'coupe', 'convertible'], 200),
        'drivewheel': np.random.choice(['fwd', 'rwd', '4wd'], 200),
        'enginelocation': np.random.choice(['front', 'rear'], 200),
        'wheelbase': np.random.uniform(85, 120, 200),
        'carlength': np.random.uniform(140, 210, 200),
        'carwidth': np.random.uniform(60, 75, 200),
        'carheight': np.random.uniform(47, 60, 200),
        'curbweight': np.random.uniform(1500, 4500, 200),
        'enginetype': np.random.choice(['ohc', 'ohcv', 'l', 'rotor', 'ohcf'], 200),
        'cylindernumber': np.random.choice(['two', 'three', 'four', 'five', 'six', 'eight'], 200),
        'enginesize': np.random.uniform(60, 350, 200),
        'fuelsystem': np.random.choice(['2bbl', '4bbl', 'mpfi', 'spdi', 'idi'], 200),
        'boreratio': np.random.uniform(2.5, 4.0, 200),
        'stroke': np.random.uniform(2.0, 4.5, 200),
        'compressionratio': np.random.uniform(7.0, 23.0, 200),
        'horsepower': np.random.uniform(48, 300, 200),
        'peakrpm': np.random.uniform(4000, 6500, 200),
        'citympg': np.random.uniform(13, 49, 200),
        'highwaympg': np.random.uniform(16, 54, 200)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create a synthetic price based on some features (for demonstration)
    df['price'] = (
        df['horsepower'] * 50 +
        df['curbweight'] * 2 +
        df['enginesize'] * 30 +
        np.random.normal(0, 2000, 200)
    ).clip(lower=5000)
    
    # Save to CSV
    output_path = r"C:\Users\ADMIN\Desktop\project\car prediction\sample_car_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created and saved to: {output_path}")
    
    return df

# Uncomment the line below to create a sample dataset
# create_sample_dataset()
