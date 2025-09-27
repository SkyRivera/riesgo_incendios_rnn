import generate_data
import model
import numpy as np
import constants
import evaluator
import trainer

if __name__ == "__main__":
    # Generar datos
    data_generator = generate_data.GenerateData()
    datos = data_generator.read_raw_data()
    entrada_entrenamiento, entrada_prueba, salida_entrenamiento, salida_prueba = data_generator.split_data(
        datos)

    # Crear y compilar modelo
    modelo = model.RiskModel(5).get_model()

    # Entrenar modelo
    modelo = trainer.Trainer(modelo).train(
        entrada_entrenamiento, salida_entrenamiento, 10, 1)

    # Evaluar modelo
    evaluator = evaluator.Evaluator(modelo, entrada_prueba, salida_prueba)
    resultados = evaluator.evaluate()
    print(f"Resultados de la evaluación: {resultados}")

    # Guardar modelo entrenado si se requires
    # modelo.save(constants.model_routes)
