from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import ast

app = Flask(__name__)

# Load Dataset
data = pd.read_csv("recipe_final (1).csv")

# Preprocess Ingredients
vectorizer = TfidfVectorizer()
X_ingredients = vectorizer.fit_transform(data['ingredients_list'])

# Normalize Numerical Features
numerical_cols = ['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']
scaler = StandardScaler()
X_numerical = scaler.fit_transform(data[numerical_cols])

# Combine Features (using sparse matrix hstack for memory efficiency)
X_combined = hstack([X_numerical, X_ingredients])

# Train KNN Model
knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)

def clean_ingredients(ingredients_str):
    try:
        lst = ast.literal_eval(ingredients_str)
        if isinstance(lst, list):
            return ", ".join(lst)
    except Exception:
        pass
    return ingredients_str.replace('[', '').replace(']', '').replace("'", "").replace('"', '')

def get_recipe_image(recipe_name):
    name_lower = recipe_name.lower()
    categories = {
        'smoothie': 'https://images.unsplash.com/photo-1553530666-ba11a7da3888?w=500&auto=format&fit=crop&q=80',
        'juice': 'https://images.unsplash.com/photo-1622483767028-3f66f32aef97?w=500&auto=format&fit=crop&q=80',
        'salad': 'https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=500&auto=format&fit=crop&q=80',
        'pasta': 'https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=500&auto=format&fit=crop&q=80',
        'pizza': 'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=500&auto=format&fit=crop&q=80',
        'cake': 'https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=500&auto=format&fit=crop&q=80',
        'cookie': 'https://images.unsplash.com/photo-1499636136210-6f4ee915583e?w=500&auto=format&fit=crop&q=80',
        'muffin': 'https://images.unsplash.com/photo-1607958996333-41aef7caefaa?w=500&auto=format&fit=crop&q=80',
        'bread': 'https://images.unsplash.com/photo-1509440159596-0249088772ff?w=500&auto=format&fit=crop&q=80',
        'soup': 'https://images.unsplash.com/photo-1547592180-85f173990554?w=500&auto=format&fit=crop&q=80',
        'steak': 'https://images.unsplash.com/photo-1544025162-d76694265947?w=500&auto=format&fit=crop&q=80',
        'chicken': 'https://images.unsplash.com/photo-1598908314732-07113901949e?w=500&auto=format&fit=crop&q=80',
        'beef': 'https://images.unsplash.com/photo-1544025162-d76694265947?w=500&auto=format&fit=crop&q=80',
        'fish': 'https://images.unsplash.com/photo-1519708227418-c8fd9a32b7a2?w=500&auto=format&fit=crop&q=80',
        'seafood': 'https://images.unsplash.com/photo-1534422298391-e4f8c172dddb?w=500&auto=format&fit=crop&q=80',
        'burger': 'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=500&auto=format&fit=crop&q=80',
        'sandwich': 'https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=500&auto=format&fit=crop&q=80',
        'rice': 'https://images.unsplash.com/photo-1512058564366-18510be2db19?w=500&auto=format&fit=crop&q=80',
        'curry': 'https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=500&auto=format&fit=crop&q=80',
        'taco': 'https://images.unsplash.com/photo-1551504734-5ee1c4a1479b?w=500&auto=format&fit=crop&q=80',
        'dessert': 'https://images.unsplash.com/photo-1551024601-bec78aea704b?w=500&auto=format&fit=crop&q=80',
        'drink': 'https://images.unsplash.com/photo-1513558161293-cdaf765ed2fd?w=500&auto=format&fit=crop&q=80',
        'beverage': 'https://images.unsplash.com/photo-1513558161293-cdaf765ed2fd?w=500&auto=format&fit=crop&q=80',
        'pie': 'https://images.unsplash.com/photo-1519869325930-281384150729?w=500&auto=format&fit=crop&q=80',
        'pancake': 'https://images.unsplash.com/photo-1528207776546-365bb710ee93?w=500&auto=format&fit=crop&q=80',
        'egg': 'https://images.unsplash.com/photo-1525351484163-7529414344d8?w=500&auto=format&fit=crop&q=80',
        'toast': 'https://images.unsplash.com/photo-1525351484163-7529414344d8?w=500&auto=format&fit=crop&q=80',
        'waffle': 'https://images.unsplash.com/photo-1562376502-6f769499c886?w=500&auto=format&fit=crop&q=80',
        'oatmeal': 'https://images.unsplash.com/photo-1517881917430-e70dfb3610aa?w=500&auto=format&fit=crop&q=80',
    }
    for key, url in categories.items():
        if key in name_lower:
            return url
    generic_images = [
        'https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1504674900247-0877df9cc836?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1498837167922-ddd27525d352?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1476224203421-9ac39bcb3327?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1482049016688-2d3e1b311543?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1484723091739-30a097e8f929?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1473093295043-cdd812d0e601?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=500&auto=format&fit=crop&q=80',
        'https://images.unsplash.com/photo-1555939594-58d7cb561ad1?w=500&auto=format&fit=crop&q=80'
    ]
    idx = abs(hash(recipe_name)) % len(generic_images)
    return generic_images[idx]

def recommend_recipes(input_features):
    # Pass input features as DataFrame with column names to avoid StandardScaler warnings
    input_df = pd.DataFrame([input_features[:7]], columns=numerical_cols)
    input_features_scaled = scaler.transform(input_df)
    
    input_ingredients_transformed = vectorizer.transform([input_features[7]])
    input_combined = hstack([input_features_scaled, input_ingredients_transformed])
    
    distances, indices = knn.kneighbors(input_combined)
    recommendations = data.iloc[indices[0]].copy()
    recommendations['ingredients_list_clean'] = recommendations['ingredients_list'].apply(clean_ingredients)
    recommendations['image_url'] = recommendations['recipe_name'].apply(get_recipe_image)
    
    return recommendations[['recipe_name', 'ingredients_list_clean', 'image_url']].head(5)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        calories = float(request.form['calories'])
        fat = float(request.form['fat'])
        carbohydrates = float(request.form['carbohydrates'])
        protein = float(request.form['protein'])
        cholesterol = float(request.form['cholesterol'])
        sodium = float(request.form['sodium'])
        fiber = float(request.form['fiber'])
        ingredients = request.form['ingredients']
        input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
        recommendations = recommend_recipes(input_features)
        return render_template('index.html', recommendations=recommendations.to_dict(orient='records'))
    return render_template('index.html', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)

