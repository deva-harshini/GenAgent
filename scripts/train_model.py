from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

def train_model():
    print("Training model...")

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)

    # Metrics
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/iris_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved to models/iris_model.pkl")

if __name__ == "__main__":
    train_model()
