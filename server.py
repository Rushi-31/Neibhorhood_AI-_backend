from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# -----------------------------
# Load Neighborhood Dataset
# -----------------------------
try:
    neighborhood_df = pd.read_csv('datasets/pd copy.csv')
except FileNotFoundError:
    neighborhood_df = pd.DataFrame()

# -----------------------------
# Helper Function
# -----------------------------
def parse_income(value):
    if pd.isna(value):
        return np.nan
    value = value.replace(",", "").replace("+", "")
    if "-" in value:
        low, high = map(int, value.split("-"))
        return (low + high) / 2
    return int(value)

# -----------------------------
# Preprocess Neighborhood Dataset
# -----------------------------
if not neighborhood_df.empty:
    neighborhood_df['Income_Estimate'] = neighborhood_df['General Income (INR)'].apply(parse_income)
    neighborhood_df = neighborhood_df.dropna(subset=['Income_Estimate'])

    # Combine Neighborhood and Sublocality
    if 'Sublocality' in neighborhood_df.columns:
        neighborhood_df['Combined_Neighborhood'] = neighborhood_df['Neighborhood'].astype(str).str.strip() + ", " + neighborhood_df['Sublocality'].astype(str).str.strip()
    else:
        neighborhood_df['Combined_Neighborhood'] = neighborhood_df['Neighborhood'].astype(str).str.strip()

    # Define features
    input_features = ['Professions', 'Income_Estimate', 'Community Sentiment', 'Public Transport']
    output_features = ['Combined_Neighborhood', 'Crime Rate', 'Schools Nearby', 'Hospitals Nearby', 'Professions']

    categorical_features = ['Professions', 'Community Sentiment', 'Public Transport']
    numerical_features = ['Income_Estimate']

    # Transformers
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

    # Fit model
    X_processed = preprocessor.fit_transform(neighborhood_df[input_features])
    knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn_model.fit(X_processed)

# -----------------------------
# Load Property Dataset
# -----------------------------
try:
    property_df = pd.read_csv("datasets/Pune_property_data.csv")
    property_df = property_df[property_df['price'] != 'price']
    property_df['price'] = property_df['price'].replace('[\₹,]', '', regex=True)
    property_df['price'] = pd.to_numeric(property_df['price'], errors='coerce')
    property_df = property_df.dropna(subset=['price'])
    property_df['locality'] = property_df['locality'].astype(str).str.strip()
except FileNotFoundError:
    property_df = pd.DataFrame()

# -----------------------------
# ROUTES
# -----------------------------

@app.route('/')
def home():
    return "Hello I am ready!"
    
@app.route('/recommend', methods=['POST'])
def recommend():
    if neighborhood_df.empty:
        return jsonify({"error": "Neighborhood data not found or could not be loaded."}), 500

    try:
        user_input = request.get_json()
        user_df = pd.DataFrame([user_input])
        user_processed = preprocessor.transform(user_df[input_features])
        distances, indices = knn_model.kneighbors(user_processed, n_neighbors=5)

        results = neighborhood_df.iloc[indices[0]][output_features].reset_index(drop=True)
        results = results.rename(columns={"Combined_Neighborhood": "Neighborhood"})
        results['Similarity Score'] = (1 / (1 + distances[0])).round(3)
        return jsonify(results.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend_properties', methods=['POST'])
def recommend_properties():
    if property_df.empty:
        return jsonify({"error": "Property data not found or could not be loaded."}), 500

    data = request.get_json()
    input_locality = data.get("locality", "").strip().lower()
    transaction_type = data.get("type", "sale").strip().lower()

    filtered = property_df[property_df['locality'].str.lower() == input_locality]

    if filtered.empty:
        return jsonify({"error": f"No properties found in locality '{input_locality}'"}), 404

    if transaction_type in ["sale", "rent"]:
        filtered = filtered[filtered['transaction_type'].str.lower() == transaction_type]
    else:
        return jsonify({"error": "Invalid transaction type. Use 'sale' or 'rent'."}), 400

    if filtered.empty:
        return jsonify({"error": f"No {transaction_type} properties found in locality '{input_locality}'"}), 404

    threshold = filtered['price'].quantile(0.25)
    budget_properties = filtered[filtered['price'] <= threshold]

    recommended = budget_properties.sort_values(by='price').head(10)

    display_columns = [
        'area', 'bhk', 'locality', 'price', 'projectname', 'status',
        'facing', 'property_type', 'transaction_type', 'floor_details',
        'amenities', 'possession_date'
    ]
    selected_columns = [col for col in display_columns if col in recommended.columns]
    output = recommended[selected_columns].to_dict(orient='records')

    return jsonify({
        "locality": input_locality.title(),
        "type": transaction_type.title(),
        "recommended_properties": output
    })

# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
