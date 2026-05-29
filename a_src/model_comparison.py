import pandas as pd

# Add RMSE values manually after running models
comparison = pd.DataFrame({
    "Model": ["ARIMA", "Prophet", "LSTM"],
    "RMSE": [12.5, 10.2, 8.7]
})

comparison.to_csv("_data/model_comparison.csv", index=False)

print(" Model comparison saved as model_comparison.csv")