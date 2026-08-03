from src.data_loader import load_data
from src.preprocessing import (
    remove_duplicates,
    drop_society,
    handle_missing_values,
    clean_size,
)

df = load_data()

df = remove_duplicates(df)
df = drop_society(df)
df = handle_missing_values(df)
df = clean_size(df)

print(df.head())

print(df.columns)



#yeh part 2 hai


from src.preprocessing import (
    remove_duplicates,
    drop_society,
    handle_missing_values,
    clean_size,
    clean_total_sqft,
)

df = clean_total_sqft(df)

print(df["total_sqft"].dtype)

print(df["total_sqft"].head())


from src.data_loader import load_data
from src.preprocessing import preprocess_data

df = load_data()

df = preprocess_data(df)

print(df.head())

print()

print(df.shape)

print()

print(df.columns)