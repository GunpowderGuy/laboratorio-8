import numpy as np
import matplotlib.pyplot as plt

# ---------------------
# Funciones de activación
# ---------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# ---------------------
# Carga y preprocesamiento de datos
# ---------------------
def load_data(path="airfoil_self_noise.dat"):
    data = np.loadtxt(path)
    np.random.seed(42)
    np.random.shuffle(data)
    X, y = data[:, :-1], data[:, -1:]
    # división 80/20
    n_train = int(0.8 * X.shape[0])
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    # normalización basada en train
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test  = (X_test  - mu) / sigma
    return X_train, y_train, X_test, y_test

# ---------------------
# Definición del modelo
# ---------------------
class NeuralNetwork:
    def __init__(self, dims, lr=1e-3):
        # dims = [n_input, n_h1, n_h2, 1]
        self.lr = lr
        self.W1 = np.random.randn(dims[0], dims[1]) * 0.1
        self.b1 = np.zeros((1, dims[1]))
        self.W2 = np.random.randn(dims[1], dims[2]) * 0.1
        self.b2 = np.zeros((1, dims[2]))
        self.W3 = np.random.randn(dims[2], dims[3]) * 0.1
        self.b3 = np.zeros((1, dims[3]))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.z3  # salida lineal

    def compute_loss(self, pred, y):
        m = y.shape[0]
        loss = np.mean((pred - y)**2)
        return loss

    def backward(self, X, y, pred):
        m = y.shape[0]
        # grad de MSE
        dL_dout = 2*(pred - y) / m
        
        # capa 3
        dL_dW3 = self.a2.T @ dL_dout
        dL_db3 = np.sum(dL_dout, axis=0, keepdims=True)
        dL_da2 = dL_dout @ self.W3.T
        
        # capa 2
        dL_dz2 = dL_da2 * relu_derivative(self.z2)
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        dL_da1 = dL_dz2 @ self.W2.T
        
        # capa 1
        dL_dz1 = dL_da1 * relu_derivative(self.z1)
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        # actualizar parámetros
        self.W3 -= self.lr * dL_dW3
        self.b3 -= self.lr * dL_db3
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

    def fit(self, X, y, epochs=100):
        history = []
        for epoch in range(1, epochs+1):
            pred = self.forward(X)
            loss = self.compute_loss(pred, y)
            history.append(loss)
            self.backward(X, y, pred)
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} – Loss: {loss:.4f}")
        return history

    def evaluate(self, X, y):
        pred = self.forward(X)
        mse = np.mean((pred - y)**2)
        mae = np.mean(np.abs(pred - y))
        return mse, mae

# ---------------------
# Script principal
# ---------------------
def main():
    # cargar datos
    X_train, y_train, X_test, y_test = load_data()

    # crear y entrenar modelo
    dims = [X_train.shape[1], 64, 64, 1]
    net = NeuralNetwork(dims, lr=1e-3)
    history = net.fit(X_train, y_train, epochs=100)

    # evaluar
    mse, mae = net.evaluate(X_test, y_test)
    print(f"\nTest MSE: {mse:.4f} — Test MAE: {mae:.4f}")

    # graficar curva de aprendizaje
    plt.figure()
    plt.plot(history, label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Guardado → training_history.png")

if __name__ == "__main__":
    main()

