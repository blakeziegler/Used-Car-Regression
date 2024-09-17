import pandas as pd
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Load the data
df = pd.read_csv('test.csv')
'''
--------------------
	EXTRACTION
--------------------
'''
df['hp'] = df['engine'].str.extract(
    r'(\d+\.?\d*)\s?HP|(\d+\.?\d*)\s?Horsepower|(\d+\.?\d*)\s?bhp')[0]
df['tank_size'] = df['engine'].str.extract(r'(\d+\.?\d*)\s?L(?:iter)?')[0]

extracted = df['engine'].str.extract(r'(\d+)\s?Cylinder|V(\d+)|Straight (\d+)')

df['cyl'] = extracted.apply(lambda x: x.dropna(
).iloc[0] if not x.dropna().empty else None, axis=1)

df = df.drop(columns=['engine'])

luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land',
                 'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini',
                 'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
df['Is_Luxury_Brand'] = df['brand'].apply(
    lambda x: 1 if x in luxury_brands else 0)


'''
-----------------
	CLEANING
-----------------
'''

df['hp'] = pd.to_numeric(df['hp'], errors='coerce')
df['tank_size'] = pd.to_numeric(df['tank_size'], errors='coerce')
df['cyl'] = pd.to_numeric(df['cyl'], errors='coerce')

# Fill or handle missing values in 'accident' and 'clean_title'
df['accident'] = df['accident'].fillna('Unknown')
df['clean_title'] = df['clean_title'].fillna('Unknown')
df['fuel_type'] = df['fuel_type'].fillna('Unknown')


'''
----------------------------
	FEATURE ENGINEERING
----------------------------
'''

# BRAND
brand_mapping = {brand: idx for idx, brand in enumerate(df['brand'].unique())}

df['brand_encoded'] = df['brand'].map(brand_mapping)
df = df.drop(columns=['brand'])

# MODEL
model_counts = df['model'].value_counts()

threshold = 25

model_mapping = model_counts[model_counts > threshold].index

model_encoding = {model: idx for idx,
                  model in enumerate(model_mapping, start=1)}
model_encoding['Other'] = len(model_mapping) + 1  # Encoding for rare models

df['model_encoded'] = df['model'].apply(
    lambda x: model_encoding.get(x, model_encoding['Other']))
df = df.drop(columns=['model'])

# HP, TANK, CYL
df['hp'].fillna(df['hp'].mean(), inplace=True)
df['tank_size'].fillna(df['tank_size'].mean(), inplace=True)
df['cyl'].fillna(df['cyl'].mean(), inplace=True)

df['weight_to_power'] = df['hp'] / df['tank_size']


# MODEL YEAR


df['car_age'] = 2024 - df['model_year']
df = df.drop(columns=['model_year'])


# MILEAGE
df['mileage_scaled'] = scaler.fit_transform(df[['milage']])
df = df.drop(columns=['milage'])

# FUEL TYPE
fuel_mapping = {fuel_type: idx for idx,
                fuel_type in enumerate(df['fuel_type'].unique())}

df['fuel_encoded'] = df['fuel_type'].map(fuel_mapping)
df = df.drop(columns=['fuel_type'])

# TRANSMISSION
transmission_counts = df['transmission'].value_counts()

threshold = 25

transmission_mapping = transmission_counts[transmission_counts >
                                           threshold].index

transmission_encoding = {transmission: idx for idx,
                         transmission in enumerate(transmission_mapping, start=1)}
transmission_encoding['Other'] = len(
    transmission_mapping) + 1  # Encoding for rare transmissions

df['transmission_encoded'] = df['transmission'].apply(
    lambda x: transmission_encoding.get(x, transmission_encoding['Other']))

df = df.drop(columns=['transmission'])

# EXT COLOR
ext_counts = df['ext_col'].value_counts()

threshold = 25

ext_mapping = ext_counts[ext_counts > threshold].index

ext_encoding = {ext: idx for idx,
                         ext in enumerate(ext_mapping, start=1)}
ext_encoding['Other'] = len(
    ext_mapping) + 1  # Encoding for rare transmissions

df['ext_encoded'] = df['ext_col'].apply(
    lambda x: ext_encoding.get(x, ext_encoding['Other']))

df = df.drop(columns=['ext_col'])


# INT COLOR
int_counts = df['int_col'].value_counts()

threshold = 25

int_mapping = int_counts[int_counts > threshold].index

int_encoding = {int1: idx for idx,
                         int1 in enumerate(int_mapping, start=1)}
int_encoding['Other'] = len(
    int_mapping) + 1  # Encoding for rare transmissions

df['int_encoded'] = df['int_col'].apply(
    lambda x: int_encoding.get(x, int_encoding['Other']))

df = df.drop(columns=['int_col'])

# ACCIDENT
df['accident_none_reported'] = (df['accident'] == 'None reported').astype(int)
df['accident_damage_reported'] = (
    df['accident'] == 'At least 1 accident or damage reported').astype(int)

df = df.drop(columns=['accident'])

# CLEAN TITLE
df['clean_title_yes'] = (df['clean_title'] == 'Yes').astype(int)

df = df.drop(columns=['clean_title'])


'''
--------------------
    FINAL CLEAN
--------------------
'''

# Scaling and binning the hp column
df['hp_scaled'] = scaler.fit_transform(df[['hp']])
df = df.drop(columns=['hp'])

# Scaling and binning the tank_size column
df['tank_scaled'] = scaler.fit_transform(df[['tank_size']])
df = df.drop(columns=['tank_size'])


# Scale the cyl column

# Save the cleaned and encoded DataFrame to a new CSV file
print(df.head())
print(df.info())
print(df.min())
print(df.max())

df.to_csv('clean_test.csv', index=False)
