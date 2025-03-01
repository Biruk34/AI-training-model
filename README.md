# Car Price Prediction Using Deep Learning
Author: Biruk Tamiru 
Tools Used: PyTorch, Pandas, Scikit-Learn, NumPy

Project Overview
This project builds a deep learning model to predict used car prices based on various features like year, kilometers driven, fuel type, and transmission type. The dataset is preprocessed, transformed, and trained using a neural network in PyTorch.

Dataset
The dataset used is "car data.csv" and contains the following columns:

Car_Name: Name of the car (Removed during preprocessing)
Year: Year of manufacture
Present_Price: Current market price of the car
Kms_Driven: Distance driven by the car (in km)
Owner: Number of previous owners
Fuel_Type: Type of fuel (Petrol/Diesel/CNG)
Seller_Type: Individual or dealership
Transmission: Manual or automatic
Selling_Price: Target Variable - Price at which the car was sold

⚙️ Project Workflow

1. Data Preprocessing
Dropped unnecessary columns (Car_Name)
Converted categorical variables to numerical using One-Hot Encoding
Standardized numerical features using StandardScaler

2.Splitting & Converting Data
Split into training (80%) and testing (20%) sets
Converted the data into PyTorch tensors

3.Model Architecture (Neural Network)
Input Layer: Takes in the processed features
Hidden Layers: Two fully connected layers with ReLU activation
Output Layer: Predicts car price

4.Training & Optimization
Used Mean Squared Error (MSE) Loss
Optimized using Adam Optimizer
Trained for 1000 epochs

5.Model Evaluation
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score (R-Squared) to measure model performance

Results
After training, the model achieves:
Low MAE & MSE – indicating accurate predictions
High R² Score (~0.98) – meaning the model explains 98% of variance in selling prices

Future Improvements
Try other ML models (Random Forest, XGBoost)
Deploy as a web app using Streamlit
Optimize model with hyperparameter tuning

License
This project is open-source under the MIT License.
