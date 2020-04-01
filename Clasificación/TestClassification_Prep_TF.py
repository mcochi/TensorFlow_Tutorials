import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar

import numpy as numpy
print(tf.__version__)

# Download IMDB datasets
# The IMBD movie reviews dataset comes packaged in tfds. It has already
# benn preprocessed os that the reviews have been converted to sequences
# of integers, where each integer represents a specific word in a dictionary
# To load your own text: https://www.tensorflow.org/tutorials/load_data/text


(train_data, test_data),info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised= True,
    with_info=True
)

# Try the enocer: The dataset info includes the text encoder

encoder = info.features['text'].encoder
print('Vocabulari size: {}'.format(encoder.vocab_size))

# This text encoder will reversesibly encode any string:
sample_string = 'Hello Tensorflow'
encoded_string = encoder.encode(sample_string)
original_string = encoder.decode(encoded_string)
print('The original string: {}'.format(original_string))
assert original_string == sample_string

# The encoder encodes string by breaking it into subwords or characters if the
# word is not in its dictionary. So the more a string ressembles the dataset,
# the shorter the encoded representation will be.

# Explore data

for train_example, train_label in train_data.take(1):
    print(train_example.numpy().size)
    print('Encoded Text: ', train_example[:20].numpy())
    print('Label',train_label.numpy)

print(encoder.decode(train_example))


# Prepare data for taining
# You will wat to create batches of training data for your model. The reviews
# are all different lengths, so use padded_batch to zero pad the sequences while batching

BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, padded_shapes=([None],[]))
)

test_batches = (
    test_data
    .padded_batch(32, padded_shapes=([None],[]))
)

for example_batch, label_batch in train_batches.take(2):
    print(example_batch.shape)
    print(example_batch)
    print(label_batch)

# Después de barajear la información de train_data, hace batch de 32 de las descripciones.
# Pone cada una de ellas en cada fila y rellena para que todas las descripciones de dicho
# Batch tengan la misma longitud. 

# Build the model

# In this case, the input data consists on an array de word-indices. The labels
# to predict are either 0 or 1. Let's build a continuous bag of words style model

model = keras.Sequential(
    [
        keras.layers.Embedding(encoder.vocab_size, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1)
    ]
)

model.summary()

# Embedding: This layer takes the integer-encoded vocabulary and looks up
# the embedding vector for each word-index. 
# GlobalAveragePooling1d: Returns a fixed-lenghth output vector for each example by
# averaging over the sequence dimension. This allows the model to handle input
# of variable length, in the simples way possible

model.compile(optimizer='adam',
    loss=tf.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(train_batches,epochs=10, validation_data=test_batches, validation_steps=30)

# Evaluate the model
loss, accuracy = model.evaluate(test_batches)
print("Loss:", loss)
print("Accuracy:", accuracy)



