import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

# 1️ Load dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\heart.csv")
print(df.head())

# 2️ Encode categorical features
df["Sex"] = df["Sex"].map({"M":1, "F":0})
df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y":1, "N":0})
df_encoded = df.rename(columns={"Sex":"is_male","ExerciseAngina":"ExerciseAngina_Y"})

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df_encoded, columns=["ChestPainType","RestingECG","ST_Slope"], drop_first=True)

df_encoded = df_encoded.astype(int)

# 3️ Features and target
X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]
<<<<<<< HEAD
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)


# 4️ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️ Define models with hyperparameter grids
models = {
    "KNN": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3,5,7,9], "weights": ["uniform","distance"], "p": [1,2]}
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {"max_depth": [3,5,7,None], "min_samples_split": [2,5,10]}
    },
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {"C":[0.01,0.1,1,10], "penalty":["l2"], "solver":["lbfgs"]}
    },
    "SVM": {
        "model": SVC(probability=True),
        "params": {"C":[0.1,1,10], "kernel":["linear","rbf"]}
    },
    "NaiveBayes": {
        "model": GaussianNB(),
        "params": {}
    }
}

# 7️ Train models with hyperparameter tuning
best_models = {}
results = []

for name, m in models.items():
    print(f"\nTraining {name}...")
    model = m["model"]
    params = m["params"]
    
    if params:
        grid = GridSearchCV(model, params, cv=5, scoring="f1")
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)
    
    y_pred = best_model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall
    })
    
    best_models[name] = best_model

# 8️ Print all results
for r in results:
    print(f"\nModel: {r['Model']}")
    print(f"Accuracy: {r['Accuracy']:.4f}")
    print(f"F1 Score: {r['F1 Score']:.4f}")
    print(f"Precision: {r['Precision']:.4f}")
    print(f"Recall: {r['Recall']:.4f}")

# 9️ Save the best KNN model, scaler, and columns for Streamlit
joblib.dump(best_models["KNN"], "knn_heart.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.to_list(), "columns.pkl")
print("\nBest KNN model, scaler, and columns saved successfully!")

















