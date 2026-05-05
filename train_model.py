import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

# Load dataset
data = pd.read_csv("data/diabetes.csv")

# Feature / target split
X = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/diabetes_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model training complete and saved.")

