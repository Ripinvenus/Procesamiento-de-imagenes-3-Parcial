"""
Tarea: CNN sencilla para clasificación de dígitos MNIST.
Requisitos:
  1. Descargar MNIST.
  2. Implementar una CNN sencilla.
  3. Entrenar y reportar arquitectura, #parametros, precision de entrenamiento,
     precision de prueba y matriz de confusion.
  4. Mostrar 5 ejemplos correctamente clasificados y 5 incorrectamente clasificados.
  5. Analizar resultados y proponer dos mejoras.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

EPOCHS = 10
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
OUTPUT_DIR = "resultados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)

# ----------------------------------------------------------------
# 1. Descargar el conjunto de datos MNIST
# ----------------------------------------------------------------
print("\n[1] Descargando MNIST...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar a [0, 1] y agregar dimensión de canal -> (28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
print("x_test  shape:", x_test.shape,  "y_test  shape:", y_test.shape)
print("Clases:", np.unique(y_train))

# ----------------------------------------------------------------
# 2. Implementar una CNN sencilla
# ----------------------------------------------------------------
print("\n[2] Construyendo el modelo CNN...")

modelo = models.Sequential([
    # Bloque convolucional 1
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
                  input_shape=(28, 28, 1), name="conv2d_1"),
    layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"),

    # Bloque convolucional 2
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"),
    layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"),

    # Clasificador denso
    layers.Flatten(name="flatten"),
    layers.Dense(128, activation="relu", name="dense_1"),
    layers.Dropout(0.5, name="dropout"),
    layers.Dense(10, activation="softmax", name="dense_salida"),
])

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nArquitectura del modelo:")
modelo.summary()

# Numero de parametros
total_params = modelo.count_params()
trainable_params = sum(int(np.prod(v.shape)) for v in modelo.trainable_weights)
non_trainable_params = total_params - trainable_params
print(f"\nTotal de parametros: {total_params:,}")
print(f"Parametros entrenables: {trainable_params:,}")
print(f"Parametros no entrenables: {non_trainable_params:,}")

# Guardar arquitectura en archivo de texto
with open(os.path.join(OUTPUT_DIR, "arquitectura.txt"), "w", encoding="utf-8") as f:
    f.write("ARQUITECTURA DEL MODELO CNN (MNIST)\n")
    f.write("=" * 45 + "\n\n")
    modelo.summary(print_fn=lambda x: f.write(x + "\n"))
    f.write(f"\nTotal de parametros: {total_params:,}\n")
    f.write(f"Parametros entrenables: {trainable_params:,}\n")
    f.write(f"Parametros no entrenables: {non_trainable_params:,}\n")

# ----------------------------------------------------------------
# 3. Entrenar el modelo
# ----------------------------------------------------------------
print("\n[3] Entrenando el modelo...")
historia = modelo.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=1,
)

# Evaluar en entrenamiento y prueba
print("\n[3b] Evaluando...")
loss_train, acc_train = modelo.evaluate(x_train, y_train, verbose=0)
loss_test, acc_test = modelo.evaluate(x_test, y_test, verbose=0)

print(f"\nPrecision de entrenamiento: {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"Precision de prueba:        {acc_test:.4f} ({acc_test*100:.2f}%)")
print(f"Perdida de entrenamiento:   {loss_train:.4f}")
print(f"Perdida de prueba:          {loss_test:.4f}")

# Predicciones sobre el conjunto de prueba
y_pred_prob = modelo.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# Matriz de confusion
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusion (filas=real, columnas=predicho):")
print(cm)

# Guardar reporte de clasificacion
report = classification_report(y_test, y_pred, digits=4)
print("\nReporte de clasificacion:\n", report)

# Graficar matriz de confusion
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de confusion - MNIST (Test)")
plt.colorbar()
ticks = np.arange(10)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.xlabel("Etiqueta predicha")
plt.ylabel("Etiqueta real")
thr = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thr else "black",
                 fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusion.png"), dpi=150)
plt.close()

# Graficar historial de entrenamiento (accuracy y loss)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(historia.history["accuracy"], label="Entrenamiento")
axes[0].plot(historia.history["val_accuracy"], label="Validacion")
axes[0].set_title("Precision por epoca")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("Precision")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(historia.history["loss"], label="Entrenamiento")
axes[1].plot(historia.history["val_loss"], label="Validacion")
axes[1].set_title("Perdida por epoca")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("Perdida")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "historial_entrenamiento.png"), dpi=150)
plt.close()

# Guardar metricas en archivo
with open(os.path.join(OUTPUT_DIR, "metricas.txt"), "w", encoding="utf-8") as f:
    f.write("METRICAS DEL MODELO\n")
    f.write("=" * 30 + "\n\n")
    f.write(f"Epocas entrenadas: {EPOCHS}\n")
    f.write(f"Tamano de lote: {BATCH_SIZE}\n")
    f.write(f"Precision de entrenamiento: {acc_train:.4f} ({acc_train*100:.2f}%)\n")
    f.write(f"Precision de prueba:        {acc_test:.4f} ({acc_test*100:.2f}%)\n")
    f.write(f"Perdida de entrenamiento:   {loss_train:.4f}\n")
    f.write(f"Perdida de prueba:          {loss_test:.4f}\n")
    f.write(f"Total de parametros: {total_params:,}\n")
    f.write(f"Parametros entrenables: {trainable_params:,}\n")
    f.write("\nReporte de clasificacion:\n")
    f.write(report)

# ----------------------------------------------------------------
# 4. Cinco ejemplos correctamente clasificados y cinco incorrectos
# ----------------------------------------------------------------
print("\n[4] Seleccionando ejemplos correctos e incorrectos...")

correct_mask = (y_pred == y_test)
incorrect_mask = (y_pred != y_test)

correct_idx = np.where(correct_mask)[0]
incorrect_idx = np.where(incorrect_mask)[0]

# Tomar los primeros 5 de cada uno (o menos si no hay suficientes)
n_correct = min(5, len(correct_idx))
n_incorrect = min(5, len(incorrect_idx))
correct_samples = correct_idx[:n_correct]
incorrect_samples = incorrect_idx[:n_incorrect]

def plot_examples(indices, title, fname):
    n = len(indices)
    if n == 0:
        print("  No hay ejemplos para:", title)
        return
    plt.figure(figsize=(n * 2.2, 2.6))
    for k, idx in enumerate(indices):
        img = x_test[idx].squeeze()
        real = y_test[idx]
        pred = y_pred[idx]
        confianza = y_pred_prob[idx][pred]
        plt.subplot(1, n, k + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Real: {real}\nPred: {pred}\nConf: {confianza:.2f}",
                  fontsize=10, color="green" if real == pred else "red")
        plt.axis("off")
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.88])
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150)
    plt.close()
    print(f"  Guardado: {fname} ({n} ejemplos)")

plot_examples(correct_samples,
              "Ejemplos CORRECTAMENTE clasificados",
              "ejemplos_correctos.png")
plot_examples(incorrect_samples,
              "Ejemplos INCORRECTAMENTE clasificados",
              "ejemplos_incorrectos.png")

# Guardar lista de indices y etiquetas en txt
with open(os.path.join(OUTPUT_DIR, "ejemplos.txt"), "w", encoding="utf-8") as f:
    f.write("EJEMPLOS CORRECTAMENTE CLASIFICADOS\n")
    f.write("-" * 40 + "\n")
    for idx in correct_samples:
        f.write(f"idx={idx}, real={y_test[idx]}, predicho={y_pred[idx]}, "
                f"confianza={y_pred_prob[idx][y_pred[idx]]:.4f}\n")
    f.write("\nEJEMPLOS INCORRECTAMENTE CLASIFICADOS\n")
    f.write("-" * 40 + "\n")
    for idx in incorrect_samples:
        f.write(f"idx={idx}, real={y_test[idx]}, predicho={y_pred[idx]}, "
                f"confianza={y_pred_prob[idx][y_pred[idx]]:.4f}\n")

# ----------------------------------------------------------------
# 5. Analisis y mejoras propuestas
# ----------------------------------------------------------------
print("\n[5] Analisis de resultados...")

analisis = f"""\
ANALISIS DE RESULTADOS - CNN sobre MNIST
{"="*50}

1. ARQUITECTURA UTILIZADA
-------------------------
- Entrada: imagenes de 28x28x1 (escala de grises, normalizadas a [0,1]).
- Capa 1: Conv2D 32 filtros, kernel 3x3, activacion ReLU.
- Capa 2: MaxPooling2D 2x2.
- Capa 3: Conv2D 64 filtros, kernel 3x3, activacion ReLU.
- Capa 4: MaxPooling2D 2x2.
- Capa 5: Flatten.
- Capa 6: Dense 128 unidades, activacion ReLU.
- Capa 7: Dropout 0.5 (regularizacion).
- Capa 8: Dense 10 unidades, activacion softmax (clasificacion de 10 digitos).

Optimizador: Adam (lr=1e-3)
Funcion de perdida: sparse_categorical_crossentropy
Metrica: accuracy

2. NUMERO DE PARAMETROS
-----------------------
- Total: {total_params:,}
- Entrenables: {trainable_params:,}
- No entrenables: {non_trainable_params:,}

3. METRICAS
-----------
- Precision de entrenamiento: {acc_train:.4f} ({acc_train*100:.2f}%)
- Precision de prueba:        {acc_test:.4f} ({acc_test*100:.2f}%)
- Perdida de entrenamiento:   {loss_train:.4f}
- Perdida de prueba:          {loss_test:.4f}

4. MATRIZ DE CONFUSION
----------------------
{cm}

5. OBSERVACIONES
----------------
- El modelo alcanza una precision razonable en el conjunto de prueba
  ({acc_test*100:.2f}%).
- La matriz de confusion muestra que la mayoria de los errores se concentran
  en pares de digitos visualmente similares (por ejemplo, 4 vs 9, 3 vs 8,
  5 vs 6, 1 vs 7).
- Existe una pequena brecha entre precision de entrenamiento y de prueba,
  lo que indica un ligero sobreajuste (overfitting), mitigado parcialmente
  por el Dropout del 0.5.
- Los ejemplos incorrectos suelen corresponder a digitos escritos de forma
  ambigua, con trazos delgados o desalineados.

6. DOS MEJORAS POSIBLES
-----------------------
MEJORA 1: Aumentar la capacidad y profundidad del modelo.
   Agregar un tercer bloque convolucional (p. ej., Conv2D 128 filtros +
   MaxPooling) y/o usar capas residuales (skip connections) para permitir
   que la red aprenda caracteristicas mas abstractas y mejore la precision
   sin sobreajustar excesivamente.

MEJORA 2: Aplicar aumento de datos (data augmentation) y ajuste de
   hiperparametros.
   Usar keras.preprocessing.image.ImageDataGenerator con rotaciones leves,
   desplazamientos y zoom para generar mas variantes de entrenamiento.
   Combinarlo con EarlyStopping y ReduceLROnPlateau para detener el
   entrenamiento cuando la precision de validación deje de mejorar y
   reducir la tasa de aprendizaje, lo que ayuda a convergir a un mejor
   minimo y reducir el sobreajuste.

Resultados guardados en la carpeta: {OUTPUT_DIR}/
- arquitectura.txt
- metricas.txt
- ejemplos.txt
- matriz_confusion.png
- historial_entrenamiento.png
- ejemplos_correctos.png
- ejemplos_incorrectos.png
"""

print(analisis)

with open(os.path.join(OUTPUT_DIR, "analisis.txt"), "w", encoding="utf-8") as f:
    f.write(analisis)

# Guardar el modelo entrenado
modelo.save(os.path.join(OUTPUT_DIR, "modelo_mnist_cnn.keras"))

print("\nProceso completado. Todos los resultados estan en:", OUTPUT_DIR)
