# Reporte Técnico — Clasificación de Dígitos MNIST con CNN

**Asunto:** Implementación, entrenamiento y evaluación de una Red Neuronal Convolucional (CNN) para clasificación de imágenes del conjunto MNIST utilizando Python y TensorFlow/Keras.

**Autor:** Tarea académica — Procesamiento de Imágenes / Aprendizaje Profundo  
**Fecha:** Noviembre 2026  
**Entorno:** Python 3.12, TensorFlow 2.21.0 (CPU), NumPy, Matplotlib, scikit-learn

---

## 1. Propósito del código

El script `mnist_cnn.py` tiene como objetivo demostrar de principio a fin cómo construir, entrenar y evaluar una **Red Neuronal Convolucional (CNN) sencilla** capaz de reconocer dígitos escritos a mano (0–9) a partir de imágenes en escala de grises de 28×28 píxeles.

El código está organizado en cinco bloques que responden directamente a los requisitos de la tarea:

| Requisito | Sección del código |
|-----------|-------------------|
| 1. Descargar MNIST | Carga automática de datos con `keras.datasets.mnist.load_data()` |
| 2. Implementar CNN sencilla | Definición del modelo `Sequential` con capas convolucionales y densas |
| 3. Entrenar y reportar métricas | Entrenamiento con `modelo.fit`, evaluación y matriz de confusión |
| 4. Mostrar ejemplos correctos e incorrectos | Visualización de 5 aciertos y 5 fallos del modelo |
| 5. Analizar y proponer mejoras | Texto de análisis con observaciones y dos mejoras sugeridas |

---

## 2. ¿Para qué funciona el código?

Este programa permite:

- **Descargar** automáticamente el dataset MNIST desde internet (primera ejecución).
- **Preprocesar** las imágenes (normalización a [0, 1] y ajuste de forma para convoluciones).
- **Construir** una CNN con dos bloques convolucionales, pooling, aplanado, capa densa, regularización por Dropout y capa de salida softmax.
- **Entrenar** la red durante 10 épocas usando el optimizador Adam y entropía cruzada categórica dispersa.
- **Evaluar** el modelo tanto en datos de entrenamiento como en datos de prueba no vistos.
- **Generar** gráficos de la matriz de confusión, historial de entrenamiento (precisión y pérdida por época) y ejemplos visuales de clasificación correcta e incorrecta.
- **Guardar** todos los resultados (texto, imágenes y el modelo entrenado) en la carpeta `resultados/`.

En resumen: es un pipeline completo de clasificación de imágenes supervisada, listo para ejecutar y para usarse como base de proyectos más avanzados.

---

## 3. Requisitos técnicos

| Componente | Versión usada | Propósito |
|------------|---------------|-----------|
| Python | 3.12.3 | Lenguaje de programación |
| TensorFlow / Keras | 2.21.0 (CPU) | Framework de deep learning |
| NumPy | — | Manipulación de arrays y operaciones numéricas |
| Matplotlib | — | Visualización de gráficos e imágenes |
| scikit-learn | — | Matriz de confusión y reporte de clasificación |

Para instalar las dependencias:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Flujo de ejecución paso a paso

### Paso 1 — Configuración y semillas aleatorias

Se fija la semilla `RANDOM_SEED = 42` en NumPy y TensorFlow para que los resultados sean **reproducibles**. Se definen los hiperparámetros:

- `EPOCHS = 10`
- `BATCH_SIZE = 128`
- `VALIDATION_SPLIT = 0.1` (10% del entrenamiento reservado para validación)
- `OUTPUT_DIR = "resultados"`

### Paso 2 — Descarga y preparación de MNIST

Se cargan 60,000 imágenes de entrenamiento y 10,000 de prueba. Cada imagen pasa por:

1. Conversión a `float32`.
2. Normalización dividiendo entre 255 (valores de píxel en [0, 1]).
3. Agregado de dimensión de canal: `(28, 28)` → `(28, 28, 1)`, requerido por las capas `Conv2D`.

### Paso 3 — Arquitectura de la CNN

Modelo secuencial con 8 capas:

| # | Capa | Salida | Parámetros | Función |
|---|------|--------|------------|---------|
| 1 | Conv2D (32 filtros, 3×3, ReLU) | (None, 26, 26, 32) | 320 | Detecta bordes y texturas básicas |
| 2 | MaxPooling2D (2×2) | (None, 13, 13, 32) | 0 | Reduce dimensión espacial |
| 3 | Conv2D (64 filtros, 3×3, ReLU) | (None, 11, 11, 64) | 18,496 | Detecta patrones más complejos |
| 4 | MaxPooling2D (2×2) | (None, 5, 5, 64) | 0 | Reduce dimensión espacial |
| 5 | Flatten | (None, 1600) | 0 | Aplana para capas densas |
| 6 | Dense (128, ReLU) | (None, 128) | 204,928 | Clasificador intermedio |
| 7 | Dropout (0.5) | (None, 128) | 0 | Regularización contra sobreajuste |
| 8 | Dense (10, softmax) | (None, 10) | 1,290 | Probabilidades por cada dígito 0–9 |

**Total de parámetros:** 225,034 (todos entrenables).

Compilación:
- Optimizador: Adam, tasa de aprendizaje 1e-3.
- Pérdida: `sparse_categorical_crossentropy`.
- Métrica: `accuracy`.

### Paso 4 — Entrenamiento y evaluación

- Se entrena con `modelo.fit()` durante 10 épocas.
- Se evalúa en entrenamiento y prueba con `modelo.evaluate()`.
- Se generan predicciones con `modelo.predict()` y se calcula la clase predicha mediante `argmax`.
- Se construye la **matriz de confusión** (filas = etiqueta real, columnas = etiqueta predicha) y el **reporte de clasificación** (precisión, recall, f1-score por clase).
- Se guardan gráficos: matriz de confusión como mapa de calor y curvas de precisión/pérdida por época.

### Paso 5 — Ejemplos visuales

Se identifican los índices de aciertos y fallos en el conjunto de prueba. Se generan dos imágenes:

- `ejemplos_correctos.png`: 5 dígitos bien clasificados (título en verde).
- `ejemplos_incorrectos.png`: 5 dígitos mal clasificados (título en rojo), mostrando la etiqueta real, la predicha y la confianza del modelo.

Se guarda un archivo `ejemplos.txt` con los detalles numéricos de cada ejemplo.

### Paso 6 — Análisis y mejoras

Se imprime y guarda en `analisis.txt` un texto con:
- Resumen de arquitectura y parámetros.
- Métricas obtenidas.
- Observaciones sobre errores (dígitos visualmente similares).
- Dos mejoras propuestas (ver sección 8).

### Paso 7 — Guardado del modelo

El modelo entrenado se serializa en `resultados/modelo_mnist_cnn.keras` para poder cargarlo y reutilizarlo sin volver a entrenar.

---

## 5. Resultados obtenidos

| Métrica | Valor |
|---------|-------|
| Precisión de entrenamiento | 99.64% |
| Precisión de prueba | 99.15% |
| Pérdida de entrenamiento | 0.0120 |
| Pérdida de prueba | 0.0262 |

### Matriz de confusión (resumen visual)

```
Real \ Pred  0    1    2    3    4    5    6    7    8    9
0           977   0    0    0    0    0    1    1    1    0
1             0 1133   1    1    0    0    0    0    0    0
2             4   0 1019   0    1    0    0    7    1    0
3             0   0    0 1001   0    3    0    2    4    0
4             0   0    0    0  978   0    0    1    1    2
5             1   0    0    4    0  883   1    1    2    0
6             3   3    0    0    3    5  943   0    1    0
7             0   2    4    0    0    0    0 1019   1    2
8             0   0    1    1    0    0    0    2  968   2
9             1   0    0    1    4    3    0    2    4  994
```

La diagonal principal contiene los aciertos. Los valores fuera de la diagonal representan errores, concentrados principalmente en pares de dígitos visualmente similares (ej. 2↔7, 3↔8, 4↔9, 5↔6, 6↔0).

---

## 6. Archivos generados

Todos los resultados se escriben en la carpeta `resultados/`:

| Archivo | Contenido |
|---------|-----------|
| `arquitectura.txt` | Resumen de capas y conteo de parámetros |
| `metricas.txt` | Precisión, pérdida y reporte de clasificación por clase |
| `analisis.txt` | Análisis de resultados y dos mejoras propuestas |
| `ejemplos.txt` | Índices, etiquetas y confianza de ejemplos correctos/incorrectos |
| `matriz_confusion.png` | Mapa de calor de la matriz de confusión |
| `historial_entrenamiento.png` | Curvas de precisión y pérdida por época |
| `ejemplos_correctos.png` | 5 muestras bien clasificadas |
| `ejemplos_incorrectos.png` | 5 muestras mal clasificadas |
| `modelo_mnist_cnn.keras` | Modelo entrenado listo para cargar y usar |

---

## 7. Análisis de resultados

- El modelo alcanza **99.15% de precisión en prueba**, muy cercano al entrenamiento (99.64%), lo que indica un buen balance entre aprendizaje y generalización.
- La pequeña brecha entre entrenamiento y prueba sugiere un **ligero sobreajuste**, mitigado parcialmente por el `Dropout` del 0.5.
- La mayoría de los errores ocurren en dígitos escritos de forma ambigua o con trazos delgados/desalineados, y en pares visualmente parecidos.

---

## 8. Dos mejoras propuestas

### Mejora 1 — Aumentar la capacidad y profundidad del modelo

Agregar un tercer bloque convolucional (por ejemplo, `Conv2D(128, 3×3) + MaxPooling2D`) o incorporar **capas residuales (skip connections)** tipo ResNet. Esto permite que la red aprenda representaciones más abstractas y jerárquicas de los dígitos, mejorando la precisión sin necesidad de aumentar drásticamente el número de parámetros en las capas densas.

### Mejora 2 — Aumento de datos (data augmentation) y ajuste de hiperparámetros

Utilizar `tf.keras.preprocessing.image.ImageDataGenerator` (o las capas de preprocessing de Keras 3) para aplicar **rotaciones leves, desplazamientos y zoom** durante el entrenamiento, generando variantes sintéticas que hagan al modelo más robusto. Combinarlo con callbacks como `EarlyStopping` (detener si la validación no mejora) y `ReduceLROnPlateau` (reducir la tasa de aprendizaje cuando se estanca) para converger a un mejor mínimo y reducir el sobreajuste.

---

## 9. Cómo ejecutar el código

```bash
# 1. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el script
python mnist_cnn.py

# 4. Revisar los resultados en la carpeta resultados/
ls resultados/
```

Nota: en la primera ejecución se descarga automáticamente el dataset MNIST (requiere conexión a internet).

---

## 10. Estructura del proyecto

```
3P/
├── mnist_cnn.py        # Código fuente completo y comentado
├── requirements.txt    # Dependencias de Python
├── README.md           # Guía rápida de uso
├── REPORTE.md          # Este documento (reporte técnico)
├── venv/               # Entorno virtual de Python
└── resultados/         # Salidas generadas al ejecutar el script
    ├── arquitectura.txt
    ├── metricas.txt
    ├── analisis.txt
    ├── ejemplos.txt
    ├── matriz_confusion.png
    ├── historial_entrenamiento.png
    ├── ejemplos_correctos.png
    ├── ejemplos_incorrectos.png
    └── modelo_mnist_cnn.keras
```

---

## 11. Conclusión

El código implementa de forma completa y autocontenida una CNN para clasificación de dígitos MNIST. Cumple con todos los requisitos solicitados: descarga de datos, implementación de la red, entrenamiento, reporte de métricas y matriz de confusión, visualización de ejemplos correctos e incorrectos, y análisis con mejoras propuestas. El modelo obtenido es ligero (225,034 parámetros) y eficaz (99.15% en prueba), constituyendo una base sólida para experimentos más avanzados de visión por computadora.
