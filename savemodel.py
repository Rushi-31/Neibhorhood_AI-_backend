import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('pd copy.csv')

# Drop rows where target is missing
df = df.dropna(subset=['Neighborhood', 'Sublocality'])

# Create new target column: neighborhood = Neighborhood + Sublocality
df['FullNeighborhood'] = df['Neighborhood'].astype(str) + ', ' + df['Sublocality'].astype(str)

# Define input and output
target = 'FullNeighborhood'
input_features = [
    'City', 'Property Type', 'Price (INR)', 'Area (sq ft)', 'Bedrooms',
    'Price per sq ft', 'Monthly Rent (INR)', 'Schools Nearby', 'Hospitals Nearby',
    'Public Transport', 'Amenities', 'Community Sentiment', 'Crime Rate',
    'Professions', 'General Income (INR)'
]

X = df[input_features]
y = df[target]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preprocessing setup
categorical_features = [
    'City', 'Property Type', 'Public Transport', 'Amenities',
    'Community Sentiment', 'Crime Rate', 'Professions'
]

numerical_features = [
    'Price (INR)', 'Area (sq ft)', 'Bedrooms', 'Price per sq ft',
    'Monthly Rent (INR)', 'Schools Nearby', 'Hospitals Nearby',
    'General Income (INR)'
]

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', categorical_transformer, categorical_features),
    ('num', numerical_transformer, numerical_features)
])

# Final model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model_pipeline.fit(X, y_encoded)

# Save model and encoder
joblib.dump(model_pipeline, 'neighborhood_model.pkl')
joblib.dump(label_encoder, 'neighborhood_encoder.pkl')

print("✅ Model and encoder saved successfully.")
