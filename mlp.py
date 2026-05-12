# mlp.py
import numpy as np

def sigmoid(z):
    """Squashes output directly between 0 and 1."""
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(a):
    """
    Derivative of the sigmoid function. 
    Note: Mathematically simplifies to a * (1 - a) where 'a' is the already-activated output.
    """
    return a * (1.0 - a)

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initializes the Neural Network architecture weights and biases.
        """
        self.lr = learning_rate
        
        # Initialize weights using a scaled standard normal distribution
        # Shape of W1: (input_size, hidden_size)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        # Shape of W2: (hidden_size, output_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """
        Pushes input matrix X through the network layers.
        Caches intermediate dot products and activations required for backprop.
        """
        # Layer 1: Input -> Hidden
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)
        
        # Layer 2: Hidden -> Output
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        
        return self.A2

    def backward(self, X, y):
        """
        Applies the chain rule to compute weight gradients and execute optimizer steps.
        """
        m = X.shape[0]  # Number of training samples
        
        # --- Output Layer Gradients ---
        # The derivative of Binary Cross-Entropy Loss combined with the Sigmoid derivative simplifies elegantly:
        dZ2 = self.A2 - y
        dW2 = (1.0 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1.0 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # --- Hidden Layer Gradients ---
        # Backpropagate the output error matrix into the hidden layer nodes
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.A1)
        
        dW1 = (1.0 / m) * np.dot(X.T, dZ1)
        db1 = (1.0 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # --- Gradient Descent Weight Updates ---
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y, epochs=10000):
        """Executes the training loop over specified iterations."""
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            
            # Output convergence metrics periodically
            if epoch % 2000 == 0:
                loss = self._binary_cross_entropy(y, self.A2)
                print(f"🔄 Epoch {epoch} | Loss: {loss:.4f}")

    def _binary_cross_entropy(self, y_true, y_pred):
        """Calculates scalar log loss safely preventing mathematical undefined bounds."""
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def predict(self, X, threshold=0.5):
        """Evaluates raw inference probabilities against a definitive threshold."""
        probabilities = self.forward(X)
        return (probabilities >= threshold).astype(int)


# --- Quick Test ---
if __name__ == "__main__":
    # The classic XOR dataset setup
    # Single-layer linear models (Linear/Logistic Regression) mathematically fail to separate this logic
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # Explicit targets vector
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    print("🧠 Booting up bare-metal Neural Network for XOR logic verification...")
    # Architecture configuration: 2 input dimensions, 4 hidden units, 1 final output neuron
    nn = MultiLayerPerceptron(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    
    nn.fit(X, y, epochs=10000)
    
    predictions = nn.predict(X)
    probabilities = nn.forward(X)
    
    print("\n🎯 Final Evaluation Mapping:")
    for i in range(len(X)):
        print(f"Input: {X[i].tolist()} | Output Prob: {probabilities[i][0]:.4f} | Prediction: {predictions[i][0]} (Expected: {y[i][0]})")