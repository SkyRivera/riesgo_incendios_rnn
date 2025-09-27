from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class RiskModel:
    def __init__(self, input_dim_i: int):
        # Crear el modelo de red neuronal,
        # Se recomienda usar 5 capas ocultas con 10 neuronas cada una
        self.modelo = Sequential()
        self.modelo.add(Dense(10, input_dim=input_dim_i, activation='relu'))
        self.modelo.add(Dense(10, input_dim=input_dim_i, activation='relu'))
        self.modelo.add(Dense(10, input_dim=input_dim_i, activation='relu'))
        self.modelo.add(Dense(4, activation='softmax'))

    def compile(self):
        # Compilar el modelo
        self.modelo.compile(loss='sparse_categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])

    def get_model(self):
        self.compile()
        return self.modelo
