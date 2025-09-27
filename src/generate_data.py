import yaml
import pandas as pd
import csv
from pathlib import Path
from constants import CATEGORIAS
from sklearn.model_selection import train_test_split


class GenerateData:
    # Cargar configuración desde config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    def __init__(self):
        self.raw_data_path = Path(
            self.config["data"]["raw_path"]) / self.config["data"]["original_dataset"]

    # Leer datos crudos desde un archivo CSV
    def read_raw_data(self):
        archivo_csv = self.raw_data_path
        datos = pd.read_csv(archivo_csv, header=None)
        return datos

    # Dividir datos en conjuntos de entrenamiento y prueba
    def split_data(self, datos):
        # Dividir en entradas (intput) y salidas (output)
        input = datos.iloc[:, :-1].values
        output = datos.iloc[:, -1].map(CATEGORIAS).values

        # Dividir en conjuntos de entrenamiento y prueba
        entrada_entrenamiento, entrada_prueba, salida_entrenamiento, salida_prueba = train_test_split(
            input, output, test_size=0.2, random_state=42)

        return entrada_entrenamiento, entrada_prueba, salida_entrenamiento, salida_prueba

    # Reformatear salidas para que sean matrices 2D
    def reshape_outputs(self, salida_entrenamiento, salida_prueba):
        # Reformatear salida_entrenamiento y salida_prueba para que sean matrices 2D
        salida_entrenamiento = salida_entrenamiento.reshape(-1, 1)
        salida_prueba = salida_prueba.reshape(-1, 1)
        return salida_entrenamiento, salida_prueba

    # Escribir archivos CSV para datos procesados
    def write_data_files(self, entrada_entrenamiento, salida_entrenamiento, entrada_prueba, salida_prueba):
        # Definir rutas de archivos para datos de entrenamiento y prueba
        output_file_training_input = Path(
            self.config["data"]["processed_path"]) / self.config["data"]["training_inputs_dataset"]
        output_file_training_output = Path(
            self.config["data"]["processed_path"]) / self.config["data"]["training_output_dataset"]
        output_file_testing_input = Path(
            self.config["data"]["processed_path"]) / self.config["data"]["testing_inputs_dataset"]
        output_file_testing_output = Path(
            self.config["data"]["processed_path"]) / self.config["data"]["testing_output_dataset"]

        # Escribir los archivos para el entrenamiento y la prueba
        with open(output_file_training_input, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(entrada_entrenamiento)

        with open(output_file_training_output, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(salida_entrenamiento)

        with open(output_file_testing_input, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(entrada_prueba)

        with open(output_file_testing_output, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(salida_prueba)

    # Método principal para generar datos procesados
    def generate(self):
        # Leer datos crudos
        datos = self.read_raw_data()

        # Dividir datos en conjuntos de entrenamiento y prueba
        entrada_entrenamiento, entrada_prueba, salida_entrenamiento, salida_prueba = self.split_data(
            datos)

        # Reformatear salidas
        salida_entrenamiento, salida_prueba = self.reshape_outputs(
            salida_entrenamiento, salida_prueba)

        # Escribir archivos de datos procesados
        self.write_data_files(
            entrada_entrenamiento, salida_entrenamiento, entrada_prueba, salida_prueba)

        print("✅ Datos generados y guardados en la carpeta data/processed/")
