from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import prepare_features
from src.train import train_models

df = load_data()

df = preprocess_data(df)

X, y = prepare_features(df)

results = train_models(X, y)

print(results)