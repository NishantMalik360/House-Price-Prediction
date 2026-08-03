from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import prepare_features

df = load_data()

df = preprocess_data(df)

X, y = prepare_features(df)

print(X.shape)
print(y.shape)

print(X.head())