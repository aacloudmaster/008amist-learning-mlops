import mlflow
logged_model = 'C:/Users/Lenovo/Desktop/0800amist-learning-mlops-master/2_mlflow/mlartifacts/976319264970801313/3f8b2eab8fb84422b0c70d60570e0843/artifacts/rental_prediction_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

predicted_rental_price = loaded_model.predict(pd.DataFrame([[3,500]]))

print(predicted_rental_price)