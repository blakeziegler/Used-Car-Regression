import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
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


luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land',
                 'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini',
                 'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
df['Is_Luxury_Brand'] = df['brand'].apply(
    lambda x: 1 if x in luxury_brands else 0)

df['transmission_type'] = df['transmission'].apply(lambda x: 
                                                   'manual' if 'm/t' in x or 'manual' in x or  'mt' in x else 
                                                   'automatic' if 'a/t' in x or 'automatic' in x or  'at' in x else 
                                                   'CVT' if 'CVT' in x else 
                                                   'Other')

df = df.drop(columns=['transmission'])

def extract_fuel_type(engine_info):
    if pd.isna(engine_info):
        return np.nan
    if 'Gasoline' in engine_info:
        return 'Gasoline'
    elif 'Hybrid' in engine_info:
        return 'Hybrid'
    elif 'Flex Fuel' in engine_info or 'E85' in engine_info:
        return 'Flex Fuel'
    elif 'Diesel' in engine_info:
        return 'Diesel'
    elif 'Electric' in engine_info:
        return 'Electric'
    else:
        return np.nan

df['extracted_fuel_type'] = df['engine'].apply(extract_fuel_type)
df['fuel_type'].fillna(df['extracted_fuel_type'], inplace=True)
df.drop(columns=['extracted_fuel_type'], inplace=True)
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


df['hp'] = df.groupby(['brand', pd.cut(df['model_year'], bins=5)], observed=False)['hp'].transform(lambda x: x.fillna(x.mean()))
df['tank_size'] = df.groupby(['brand', pd.cut(df['model_year'], bins=5)], observed=False)['tank_size'].transform(lambda x: x.fillna(x.mean()))
df['cyl'] = df.groupby(['brand', pd.cut(df['model_year'], bins=5)], observed=False)['cyl'].transform(lambda x: x.fillna(x.mean()))

# HP, TANK, CYL
df['hp'].fillna(df['hp'].mean(), inplace=True)
df['tank_size'].fillna(df['tank_size'].mean(), inplace=True)
df['cyl'].fillna(df['cyl'].mean(), inplace=True)


# MODEL YEAR


df['car_age'] = 2024 - df['model_year']
df = df.drop(columns=['model_year'])

df['clean_title_yes'] = (df['clean_title'] == 'Yes').astype(int)

df = df.drop(columns=['clean_title'])

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['brand'] = encoder.fit_transform(df['brand'])
df['int_col'] = encoder.fit_transform(df['int_col'])
df['ext_col'] = encoder.fit_transform(df['ext_col'])
df['transmission_type'] = encoder.fit_transform(df['transmission_type'])
df['fuel_type'] = encoder.fit_transform(df['fuel_type'])
df['model'] = encoder.fit_transform(df['model'])
df['accident'] = encoder.fit_transform(df['accident'])


df = df.drop(columns=['engine'])


'''
--------------------
    FINAL CLEAN
--------------------
'''

print(df.head())
print(df.info())
print(df.min())
print(df.max())

df.to_csv('clean_test.csv', index=False)
