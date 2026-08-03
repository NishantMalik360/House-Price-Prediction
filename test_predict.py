from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import prepare_features
from src.predict import load_model, predict

df = load_data()

df = preprocess_data(df)

X, y = prepare_features(df)

model = load_model()

predictions = predict(model, X.head())

print("Predictions:")
print(predictions)