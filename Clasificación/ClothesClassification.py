from __future__ import absolute_import, division, print_function, unicode_literals


# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerías auxiliares
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Importar el set de datos de moda de MNIST

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Al cargar el set de datos retorna cuatro arreglos en Numpy:
# Train_images y train_labels son arreglos que el modelo va a utilizar para aprender
# Test_images, test_labels son aquellos items con los que vamos a probar el modelo

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']

# Explorar el formato del set de datos antes de entrenar el modelo
#print(train_images.shape)
#len(train_labels)

#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

# Escalar los valores en un rango de 0 a 1 antes de alimentar el modelo de la
# red neuronal

train_images = train_images / 255.0
test_images = test_images / 255.0

#plt.figure(figsize=(10,10))
#for i in range(36):
#    plt.subplot(6,6,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i],cmap= plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

# Construir el modelo

# 1. Configuración de las capas
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #Transforma la matriz en un arreglo unidimensional
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

# 2. Compile el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Entrenar el modelo
model.fit(train_images, train_labels, epochs=10)

# 4. Evaluar la exactitud
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest Accuracy: ', test_acc)

# Resulta que la exactitud sobre el tes de datos es un poco menor que la exactitud
# sobre el test de entrenamiento. Esta diferencia entre el entrenamiento y el test
# se debe al overfitting (Un modelo de aprendizaje de máquina ML tiene un rendimiento
# peor sobre un set de datos nuevo, que nunca antes ha visto comparado con el del entrenamiento)

# 5. Hacer predicciones
predictions = model.predict(test_images)

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()