# Test Classification with Tensoflow Hub: Movie reviews
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# Dowload the IMDB dataset
train_data, validation_data, test_data = tfds.load(
    name= "imdb_reviews",
    split=('train[:60%]','train[:60%]','test'),
    as_supervised=True
)

# Explore data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch[0])
print(train_labels_batch[0])

# Build the model
# Para tratar texto vamos a utilizar la capa cargada a continuación que está
# precargada para tokenizar texto por sentencias en inglés.
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
print(hub_layer(train_examples_batch[:3]))

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
# El shuffle sirve para barajar 
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data.batch(512),verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

# This fairly naive approach achieves an accuracy of about 85%. With more
# advanced approaches, the model should get closer to 95%