# logistic_regression.py
import numpy as np

def sigmoid(z):
    """
    Squashes any real number into a range between 0 and 1.
    """
    # np.clip prevents overflow errors if z gets dangerously large or small
    z = np.clip(z, -250, 250) 
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the binary classification model.
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Trains the model using Gradient Descent to minimize Binary Cross-Entropy loss.
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent Loop
        for _ in range(self.n_iterations):
            # The Linear Model (same as Day 1)
            linear_model = np.dot(X, self.weights) + self.bias
            
            # The Activation Function (this makes it Logistic!)
            y_predicted = sigmoid(linear_model)

            # Gradients (Mathematically identical to Linear Regression)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        """
        Returns the raw probability of the positive class (1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predicts binary class labels (0 or 1) based on a probability threshold.
        """
        probabilities = self.predict_proba(X)
        # Convert probabilities to strictly 0 or 1
        return np.array([1 if p >= threshold else 0 for p in probabilities])


# --- Quick Test ---
if __name__ == "__main__":
    # Fake medical data: [Blood Pressure, Cholesterol]
    # Class 0 = Healthy, Class 1 = At Risk
    X = np.array([
        [110, 180], [120, 190], [115, 185], # Healthy (Lower numbers)
        [150, 240], [160, 250], [155, 245]  # At Risk (Higher numbers)
    ])
    
    # Standardize the data (Crucial for gradient descent to converge quickly!)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std

    y = np.array([0, 0, 0, 1, 1, 1])

    print("🧠 Training Logistic Regression...")
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_scaled, y)

    # Test with a borderline patient
    X_test = np.array([[135, 210]])
    X_test_scaled = (X_test - X_mean) / X_std
    
    prob = model.predict_proba(X_test_scaled)[0]
    pred = model.predict(X_test_scaled)[0]

    print(f"🎯 Probability of Risk: {prob * 100:.2f}%")
    print(f"🏥 Final Diagnosis Class: {pred}")