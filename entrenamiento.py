import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
class LogisticRegression:
    def __init__(self):
        self.theta = None

    @staticmethod
    def sigmoid(x):
        exp_values = np.exp(-x)
        return 1 / (1 + np.clip(exp_values, a_min=None, a_max=1e308))

    def train(self, X, y, learning_rate, epochs):
        m, n = len(X), len(X[0])
        self.theta = [0] * n  # Inicializar parámetros

        for epoch in range(epochs):
            for i in range(m):
                # Calcular el valor predicho
                z = sum(self.theta[j] * X[i][j] for j in range(n))
                h = self.sigmoid(z)

                # Calcular el error
                error = h - y[i]

                # Actualizar parámetros usando el descenso de gradiente estocástico
                for j in range(n):
                    self.theta[j] = self.theta[j] - learning_rate * error * X[i][j]

    def predict(self, X):
        if self.theta is None:
            raise ValueError("El modelo no ha sido entrenado.")
        return [1 if self.sigmoid(sum(self.theta[j] * X[i][j] for j in range(len(X[0])))) >= 0.5 else 0 for i in range(len(X))]

# Función para graficar la predicción y el entrenamiento
def plot_prediction_training(X, y, theta):
    plt.figure(figsize=(12, 8))

    # Graficar los puntos de entrenamiento
    plt.scatter([X[i][1] for i in range(len(X)) if y[i] == 0], [X[i][2] for i in range(len(X)) if y[i] == 0], color='blue', label='No Diabetes', marker='o')
    plt.scatter([X[i][1] for i in range(len(X)) if y[i] == 1], [X[i][2] for i in range(len(X)) if y[i] == 1], color='red', label='Diabetes', marker='x')

    # Graficar la línea de predicción
    x_line = np.linspace(min([X[i][1] for i in range(len(X))]), max([X[i][1] for i in range(len(X))]), 100)
    y_line = -(theta[0] + theta[1] * x_line) / theta[2]
    plt.plot(x_line, y_line, color='green', label='Línea de predicción')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Predicción y Entrenamiento de Regresión Logística')
    plt.legend()
    plt.show()

# Función para graficar la función sigmoide
def plot_sigmoid_function(theta, X, y_pred):
    # Generar datos para graficar la función sigmoide
    x_vals = np.linspace(-7, 7, 100)  # Puedes ajustar el rango según tus datos
    y_vals = LogisticRegression.sigmoid(x_vals)

    # Graficar la función sigmoide
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='Función Sigmoide', color='blue')

    # Graficar los puntos de la predicción
    for i in range(len(X)):
        color = 'red' if y_pred[i] == 1 else 'blue'
        marker = 'x' if y_pred[i] == 1 else 'o'
        plt.scatter(sum(theta[j] * X[i][j] for j in range(len(X[0]))), LogisticRegression.sigmoid(sum(theta[j] * X[i][j] for j in range(len(X[0])))), color=color, marker=marker)

    plt.title('Gráfica de la Función Sigmoide con Datos de Predicción')
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.axhline(0.5, color='red', linestyle='--', label='Umbral 0.5')
    plt.legend()
    plt.grid(True)
    plt.show()

# Cargar el conjunto de datos
df = pd.read_csv('diabetes.csv')

# Separar las características (X) y la variable objetivo (y)
X = df.drop(columns=['Outcome']).values.tolist()
y = df['Outcome'].tolist()

# Agregar un término de sesgo a X (sesgo/bias term)
X = [[1] + fila for fila in X]

# Parámetros de entrenamiento
tasa_aprendizaje = 1
epochs = 1000

# Crear instancia del modelo
modelo = LogisticRegression()

# Entrenar el modelo
modelo.train(X, y, tasa_aprendizaje, epochs)

# Guardar el modelo entrenado en un archivo
with open('modelo_entrenado.pkl', 'wb') as file:
    pickle.dump(modelo, file)

# Entrenar el modelo
modelo.train(X, y, tasa_aprendizaje, epochs)
theta_entrenado = modelo.theta
print("Parámetros del modelo entrenado:", theta_entrenado)

# Realizar predicciones en el conjunto de entrenamiento
y_pred = modelo.predict(X)

# Calcular la exactitud
exactitud = sum(1 for pred, real in zip(y_pred, y) if pred == real) / len(y)
print("Exactitud del modelo en el conjunto de entrenamiento:", exactitud)
print("El modelo ha sido entrenado con exito")
# Graficar la predicción y el entrenamiento
plot_prediction_training(X, y, theta_entrenado)

# Graficar la función sigmoide con los datos de predicción
plot_sigmoid_function(theta_entrenado, X, y_pred)