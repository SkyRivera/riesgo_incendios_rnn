class Trainer:
    def __init__(self, modelo):
        self.modelo = modelo

    # Se recomienda usar batch_size=1
    # Se recomienda usar epochs=10
    def train(self, input_trainer, output_trainer, epoch, batch_sizes):
        # Entrenar el modelo
        self.modelo.fit(input_trainer,
                        output_trainer,
                        epochs=epoch,
                        batch_size=batch_sizes)
        return self.modelo
