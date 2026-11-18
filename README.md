# Clasificación de Dígitos MNIST con una CNN en TensorFlow/Keras

Proyecto académico que implementa una Red Neuronal Convolucional (CNN) sencilla para clasificar dígitos escritos a mano del conjunto de datos **MNIST** utilizando **Python y TensorFlow**.

---

## 📋 Tabla de contenidos

- [Descripción](#descripción)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso / Cómo ejecutar](#uso-como-ejecutar)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Arquitectura del modelo](#arquitectura-del-modelo)
- [Resultados obtenidos](#resultados-obtenidos)
- [Archivos generados](#archivos-generados)
- [Análisis y mejoras propuestas](#analisis-y-mejoras-propuestas)
- [Notas sobre el editor (Zed)](#notas-sobre-el-editor-zed)

---

## 📖 Descripción

El script `mnist_cnn.py` cumple con los siguientes puntos:

1. **Descarga** automática del conjunto de datos MNIST (60,000 imágenes de entrenamiento y 10,000 de prueba).
2. **Implementa** una CNN sencilla con dos bloques convolucionales y un clasificador denso.
3. **Entrena** el modelo y reporta:
   - Arquitectura utilizada
   - Número de parámetros
   - Precisión de entrenamiento
   - Precisión de prueba
   - Matriz de confusión
4. **Muestra** cinco ejemplos correctamente clasificados y cinco incorrectamente clasificados.
5. **Analiza** los resultados y propone dos mejoras posibles al modelo.

---

## ⚙️ Requisitos

- Python 3.10 o superior
- Entorno virtual (recomendado)
- Librerías:
  - `tensorflow-cpu` (o `tensorflow` si tienes GPU)
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

---

## 🚀 Instalación

Desde la carpeta del proyecto:

```bash
# Crear entorno virtual
python3 -m venv venv

# Activarlo (Linux/macOS)
source venv/bin/activate

# Instalar dependencias
pip install tensorflow-cpu numpy matplotlib scikit-learn
```

---

## 💻 Uso / Cómo ejecutar

Con el entorno virtual activado:

```bash
python mnist_cnn.py
```

El script:
1. Descarga MNIST automáticamente (primera ejecución).
2. Entrena el modelo durante 10 épocas.
3. Genera gráficos y métricas en la carpeta `resultados/`.
4. Guarda el modelo entrenado como `modelo_mnist_cnn.keras`.

---

## 📁 Estructura del proyecto

```
3P/
├── mnist_cnn.py          # Script principal
├── README.md             # Este archivo
├── venv/                 # Entorno virtual de Python
└── resultados/           # Salidas generadas al ejecutar el script
    ├── arquitectura.txt         # Resumen de capas y parámetros
    ├── metricas.txt             # Precisión, pérdida y reporte de clasificación
    ├── analisis.txt             # Análisis de resultados y mejoras
    ├── ejemplos.txt             # Índices y etiquetas de ejemplos correctos/incorrectos
    ├── matriz_confusion.png     # Matriz de confusión (Test)
    ├── historial_entrenamiento.png  # Curvas de precisión y pérdida por época
    ├── ejemplos_correctos.png   # 5 ejemplos bien clasificados
    ├── ejemplos_incorrectos.png # 5 ejemplos mal clasificados
    └── modelo_mnist_cnn.keras   # Modelo entrenado guardado
```

---

## 🧠 Arquitectura del modelo

Modelo `Sequential` de Keras:

| Capa | Tipo | Salida | Parámetros |
|------|------|--------|------------|
| 1 | Conv2D (32 filtros, 3×3, ReLU) | (None, 26, 26, 32) | 320 |
| 2 | MaxPooling2D (2×2) | (None, 13, 13, 32) | 0 |
| 3 | Conv2D (64 filtros, 3×3, ReLU) | (None, 11, 11, 64) | 18,496 |
| 4 | MaxPooling2D (2×2) | (None, 5, 5, 64) | 0 |
| 5 | Flatten | (None, 1600) | 0 |
| 6 | Dense (128 unidades, ReLU) | (None, 128) | 204,928 |
| 7 | Dropout (0.5) | (None, 128) | 0 |
| 8 | Dense (10 unidades, softmax) | (None, 10) | 1,290 |

- **Optimizador:** Adam (learning_rate=1e-3)
- **Función de pérdida:** `sparse_categorical_crossentropy`
- **Métrica:** `accuracy`

**Total de parámetros:** 225,034 (todos entrenables)

---

## 📊 Resultados obtenidos

| Métrica | Valor |
|---------|-------|
| Épocas entrenadas | 10 |
| Tamaño de lote | 128 |
| Precisión de entrenamiento | **99.64%** |
| Precisión de prueba | **99.15%** |
| Pérdida de entrenamiento | 0.0120 |
| Pérdida de prueba | 0.0262 |

### Matriz de confusión (conjunto de prueba)

Filas = etiqueta real, Columnas = etiqueta predicha:

```
[[977   0   0   0   0   0   1   1   1   0]
 [  0 1133   1   1   0   0   0   0   0   0]
 [  4   0 1019   0   1   0   0   7   1   0]
 [  0   0   0 1001   0   3   0   2   4   0]
 [  0   0   0   0  978   0   0   1   1   2]
 [  1   0   0   4   0  883   1   1   2   0]
 [  3   3   0   0   3   5  943   0   1   0]
 [  0   2   4   0   0   0   0 1019   1   2]
 [  0   0   1   1   0   0   0   2  968   2]
 [  1   0   0   1   4   3   0   2   4  994]]
```

Los errores se concentran principalmente en pares de dígitos visualmente similares (por ejemplo: 2↔7, 3↔8, 4↔9, 5↔6, 6↔0).

---

## 📸 Archivos de imagen generados

Todos se guardan en `resultados/`:

- `matriz_confusion.png` — Matriz de confusión del test.
- `historial_entrenamiento.png` — Curvas de precisión y pérdida por época (entrenamiento vs validación).
- `ejemplos_correctos.png` — 5 dígitos correctamente clasificados (etiqueta en verde).
- `ejemplos_incorrectos.png` — 5 dígitos incorrectamente clasificados (etiqueta en rojo).

---

## 🔍 Análisis y mejoras propuestas

### Observaciones
- El modelo alcanza **99.15% de precisión en prueba**, muy cercano al entrenamiento (99.64%), lo que indica un buen desempeño con poco sobreajuste gracias al `Dropout` del 0.5.
- Existe una pequeña brecha entre entrenamiento y prueba, lo que sugiere un **ligero sobreajuste**.
- Los errores ocurren principalmente en dígitos escritos de forma ambigua, con trazos delgados o desalineados.

### Dos mejoras posibles

1. **Aumentar la capacidad y profundidad del modelo:**  
   Agregar un tercer bloque convolucional (por ejemplo, `Conv2D(128, 3×3) + MaxPooling`) o usar capas residuales (skip connections) para que la red aprenda características más abstractas sin sobreajustar excesivamente.

2. **Aumento de datos (data augmentation) y ajuste de hiperparámetros:**  
   Usar `ImageDataGenerator` con rotaciones leves, desplazamientos y zoom para generar más variantes de entrenamiento. Combinarlo con `EarlyStopping` y `ReduceLROnPlateau` para detener el entrenamiento cuando la precisión de validación deje de mejorar y reducir la tasa de aprendizaje, lo que ayuda a converger a un mejor mínimo y reducir el sobreajuste.

---

## 🛠 Notas sobre el editor (Zed)

Si Zed te marca errores en rojo en `mnist_cnn.py`, es probable que no esté usando el entorno virtual donde se instalaron las dependencias.

**Solución:** Abre `settings.json` de Zed (`Ctrl + ,`) y añade:

```json
{
  "languages": {
    "python": {
      "language-servers": ["pyright"],
      "pyright": {
        "executionEnvironment": "/home/cris/Documentos/7Semestre/Imagenes/3P/venv/bin/python",
        "pythonPath": "/home/cris/Documentos/7Semestre/Imagenes/3P/venv/bin/python"
      }
    }
  }
}
```

Luego reinicia el servidor de lenguaje (`Ctrl + Shift + P` → "Restart Language Server") o cierra y vuelve a abrir el archivo.

> El código está correcto y ya se ejecutó sin errores; los marcados del editor son solo problemas de resolución de importaciones, no fallos del script.

---

## 📄 Licencia

Proyecto académico — Uso libre con fines educativos.
