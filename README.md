# Recipe & Food Recommendation System

A machine learning-based recipe recommendation system built with **Python**, **Flask**, and **Scikit-Learn**. The application utilizes the **K-Nearest Neighbors (KNN)** algorithm to recommend recipes matching both user-specified nutritional targets (Calories, Fat, Carbohydrates, Protein, Cholesterol, Sodium, Fiber) and preferred ingredients.

## 🚀 Key Features

* **Advanced Recommendation Engine:** Uses a hybrid KNN approach combining normalized numerical features (using `StandardScaler`) and vectorized ingredient lists (using `TfidfVectorizer`).
* **Highly Memory Optimized:** Combined features are represented as memory-efficient **sparse matrices** via `scipy.sparse`, bypassing massive RAM footprints and ensuring instant recommendations.
* **Stunning Dynamic UI:** Integrates real-time, high-quality recipe images fetched dynamically from **Unsplash** based on recipe titles, replacing broken database URLs.
* **Polished Ingredient Display:** Pre-processes database outputs to render ingredients in a clean, human-readable comma-separated format.
* **Modern CSS Styling:** Form inputs are clean and modern, and result cards are equipped with smooth hover animations, image shadow transitions, and auto-truncation text layouts.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn, NumPy, Pandas, SciPy (Sparse Matrices)
* **Frontend:** Bootstrap 4, Jinja2 Templates, Vanilla CSS

---

## 📁 Repository Structure

```
├── app.py                     # Main Flask application and ML logic
├── recipe_final (1).csv       # Raw recipe and nutrient dataset (48k+ entries)
├── templates/
│   └── index.html             # Polished frontend user interface
├── requirements.txt           # Python library dependencies
├── .gitignore                 # Version control ignores (ignores venv/, checkpoints, etc.)
├── recipe_recommendation_system.ipynb  # ML prototyping Jupyter Notebook
└── Data Preparing.ipynb       # Data preprocessing Jupyter Notebook
```

---

## ⚙️ Installation & Local Setup

Follow these steps to run the application locally:

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd Recipe-Recommendation-Food-Recommendation-Python-Machine-Learning-main
```

### 2. Set Up a Virtual Environment (Recommended)
On Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
On Windows:
```powershell
.\venv\Scripts\python.exe app.py
```
On macOS/Linux:
```bash
python app.py
```

### 5. Access the Web App
Open your web browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 💡 How It Works (Machine Learning Pipeline)

1. **TF-IDF Vectorization:** The textual ingredient lists are converted into a sparse TF-IDF matrix using `TfidfVectorizer` to model ingredient similarities.
2. **Feature Scaling:** The numerical features (`calories`, `fat`, `carbohydrates`, `protein`, `cholesterol`, `sodium`, `fiber`) are normalized using `StandardScaler` to ensure all nutrients contribute equally to the distance calculation.
3. **Sparse Horizontal Stacking:** The normalized numerical values and sparse TF-IDF features are horizontally stacked into a single sparse matrix to prevent high RAM consumption.
4. **KNN Search:** A `NearestNeighbors` model is fitted on the combined sparse matrix using the Euclidean distance metric. When a user enters target nutrients and ingredients, the model returns the top 3 nearest recipes.
