import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.sklearn.autolog()

df = pd.read_csv(r"D:\Asah 2025\Submission MSML\Membangun_model\salary_preprocessing.csv")

# Splitting dataset
X = df.drop('salary', axis=1)
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="RandomForestRegression") as run:

    # Modeling
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
     
    model.fit(X_train, y_train)