"""Fraud classifier with several common leaks (example input for pariksha)."""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("transactions.csv")
X, y = df.drop(columns="is_fraud"), df["is_fraud"]

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = SelectKBest(f_classif, k=20).fit_transform(X, y)
X_res, y_res = SMOTE().fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)
print("test F1:", f1_score(y_test, model.predict(X_test)))

# "Final" model on everything, then report its accuracy
final = RandomForestClassifier(n_estimators=300)
final.fit(X_res, y_res)
print("accuracy:", accuracy_score(y_res, final.predict(X_res)))
