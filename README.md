# riesgo_incendios_rnn

Red neuronal en Python para predecir riesgo de incendios forestales con factores ambientales como base el semáforo de Karl Lewinsking.

# Sistema de Predicción de Riesgo de Incendios Forestales

Este proyecto implementa una **red neuronal artificial** en Python para predecir el nivel de riesgo de incendios forestales con base en factores ambientales como **temperatura, humedad, viento, vegetación y cielo**.

**Análisis:**

parametros según el semáforo de Karl Lewinsking

Cielo: nublado = 0, medio nublado = 1, soleado = 3
Humedad: hum > 50, 50 > hum > 40, 40 > hum > 20, hum < 20
Temperatura: tem < 25, 25 < tem < 30, 30 < tem < 36, 36 < tem
Vegetación: semihumeda = 0, seca = 1, muy seca = 2
Viento: vel < 10, 10 < vel < 20, 20 < vel 30, 30 < vel
