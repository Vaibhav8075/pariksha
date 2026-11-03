"""The same fraud classifier with the leaks fixed: pariksha reports nothing here."""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("transactions.csv")
X, y = df.drop(columns="is_fraud"), df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = make_pipeline(
    StandardScaler(),
    SelectKBest(f_classif, k=20),
    SMOTE(random_state=42),
    RandomForestClassifier(n_estimators=300, random_state=42),
)
model.fit(X_train, y_train)
print("train F1:", f1_score(y_train, model.predict(X_train)))
print("test F1:", f1_score(y_test, model.predict(X_test)))
