from src.predict import predict_house

sample = {
    "total_sqft": 1200,
    "bath": 2,
    "balcony": 1,
    "bhk": 2,
    "area_type": "Super built-up Area",
    "location": "Whitefield",
}

price = predict_house(sample)

print(f"Predicted Price: {price:.2f} Lakhs")