import numpy as np
from constants import CATEGORIAS


class Evaluator:
    def __init__(self, model, input_test, output_test):
        self.model = model
        self.input_test = input_test
        self.output_test = output_test

    # Se evalua la precisión del modelo con los datos de prueba
    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.input_test, self.output_test)
        return {"loss": loss, "accuracy": accuracy}
