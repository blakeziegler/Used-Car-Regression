import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# Load the data
df = pd.read_csv('train.csv')

'''
--------------------
	EXTRACTION
--------------------
'''
df['hp'] = df['engine'].str.extract(r'(\d+\.?\d*)\s?HP|(\d+\.?\d*)\s?Horsepower|(\d+\.?\d*)\s?bhp')[0]
df['tank_size'] = df['engine'].str.extract(r'(\d+\.?\d*)\s?L(?:iter)?')[0]

extracted = df['engine'].str.extract(r'(\d+)\s?Cylinder|V(\d+)|Straight (\d+)')

df['cyl'] = extracted.apply(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None, axis=1)

df = df.drop(columns=['engine'])


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
df['fuel_type'] = df['fuel_type'].fillna('Electric')


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

model_encoding = {model: idx for idx, model in enumerate(model_mapping, start=1)}
model_encoding['Other'] = len(model_mapping) + 1  # Encoding for rare models

df['model_encoded'] = df['model'].apply(lambda x: model_encoding.get(x, model_encoding['Other']))
df = df.drop(columns=['model'])

# MODEL YEAR

scaler = StandardScaler()
mean_year = df['model_year'].mean()
df['year_from_mean'] = df['model_year'] - mean_year
df['year_scaled'] = scaler.fit_transform(df[['year_from_mean']])
df = df.drop(columns=['model_year'])
df = df.drop(columns=['year_from_mean'])

# MILEAGE
df['mileage_scaled'] = scaler.fit_transform(df[['milage']])
df = df.drop(columns=['milage'])

# FUEL TYPE
fuel_mapping = {fuel_type: idx for idx, fuel_type in enumerate(df['fuel_type'].unique())}

df['fuel_encoded'] = df['fuel_type'].map(fuel_mapping)
df = df.drop(columns=['fuel_type'])

# TRANSMISSION
transmission_counts = df['transmission'].value_counts()

threshold = 25

transmission_mapping = transmission_counts[transmission_counts > threshold].index

transmission_encoding = {transmission: idx for idx, transmission in enumerate(transmission_mapping, start=1)}
transmission_encoding['Other'] = len(transmission_mapping) + 1  # Encoding for rare transmissions

df['transmission_encoded'] = df['transmission'].apply(lambda x: transmission_encoding.get(x, transmission_encoding['Other']))

df = df.drop(columns=['transmission'])

# EXT COLOR
ext_col_mapping = {ext_col: idx for idx, ext_col in enumerate(df['ext_col'].unique())}

df['ext_col_mapping'] = df['ext_col'].map(ext_col_mapping)
df = df.drop(columns=['ext_col'])

# INT COLOR
int_col_mapping = {int_col: idx for idx, int_col in enumerate(df['int_col'].unique())}

df['int_col_mapping'] = df['int_col'].map(int_col_mapping)
df = df.drop(columns=['int_col'])

# ACCIDENT
df['accident_none_reported'] = (df['accident'] == 'None reported').astype(int)
df['accident_damage_reported'] = (df['accident'] == 'At least 1 accident or damage reported').astype(int)

df = df.drop(columns=['accident'])

# CLEAN TITLE
df['clean_title_yes'] = (df['clean_title'] == 'Yes').astype(int)

df = df.drop(columns=['clean_title'])


'''
--------------------
    FINAL CLEAN
--------------------
'''

# Fill missing values with grouped means
df['hp'] = df.groupby(['fuel_encoded', 'brand_encoded', 'transmission_encoded', pd.cut(df['year_scaled'], bins=10)], observed=False)['hp'].transform(lambda x: x.fillna(x.mean()))
df['tank_size'] = df.groupby(['fuel_encoded', 'brand_encoded','transmission_encoded', pd.cut(df['year_scaled'], bins=10)], observed=False)['tank_size'].transform(lambda x: x.fillna(x.mean()))
df['cyl'] = df.groupby(['fuel_encoded', 'brand_encoded', 'transmission_encoded', pd.cut(df['year_scaled'], bins=10)], observed=False)['cyl'].transform(lambda x: x.fillna(x.mean()))

# Fill any remaining NaN values with overall column mean
df['hp'].fillna(df['hp'].mean(), inplace=True)
df['tank_size'].fillna(df['tank_size'].mean(), inplace=True)
df['cyl'].fillna(df['cyl'].mean(), inplace=True)

# Scaling and binning the hp column
df['hp_scaled'] = scaler.fit_transform(df[['hp']])
df = df.drop(columns=['hp'])

# Scaling and binning the tank_size column
scaler2 = MinMaxScaler()
df['tank_scaled'] = scaler2.fit_transform(df[['tank_size']])
df = df.drop(columns=['tank_size'])


# Scale the cyl column
df['cyl_scaled'] = scaler.fit_transform(df[['cyl']])
df = df.drop(columns=(['cyl']))

# Filter out outliers in mileage_scaled

# Save the cleaned and encoded DataFrame to a new CSV file
print(df.head())
print(df.info())
df.to_csv('clean_train.csv', index=False)
