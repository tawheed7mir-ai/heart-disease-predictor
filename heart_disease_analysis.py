import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score,classification_report



df =pd.read_csv(r"C:\Users\DELL\Downloads\heart.csv")
print(df.head())
le = LabelEncoder()

df["Sex"] = df["Sex"].map({"M":1,"F":0})
df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y":1,"N":0})
df_encoded = df.rename(columns={"Sex":"is_male","ExerciseAngina":"ExerciseAngina_Y"})

df_encoded = pd.get_dummies(data = df,columns=["ChestPainType" ,"RestingECG","ST_Slope"],drop_first=True)

df_encoded = df_encoded.astype(int)

X = df_encoded.drop("HeartDisease",axis=1)
y = df_encoded["HeartDisease"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)


models = {"logistic regression":LogisticRegression(),
         "KNN":KNeighborsClassifier(),
          "naive bayes":GaussianNB(),
           "Decision tree":DecisionTreeClassifier() ,
           "SVM":SVC(probability=True)}
result = []
for name,model in models.items():
    model.fit(x_train_scaled,y_train)
    y_pred = model.predict(x_test_scaled)
    acc = accuracy_score(y_test,y_pred)
    f1 =f1_score(y_test,y_pred)
    result.append({"model":name,
                   "accuracy":acc,
                    "f1 score":f1})

# print(result)
import joblib 
joblib.dump(models["KNN"],"knn_heart.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(X.columns.to_list(),"columns.pkl")






