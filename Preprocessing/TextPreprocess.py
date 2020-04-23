#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load Text
# This tutorial provides an example of how to use tf.data.TextLineDataset
# to load examples from text files. TextLineDataset is desinged to 
# create a dataset from a text file, in which each example is a line of text
# from the original file. This is potentially useful for any text
# data that is primarily line-based (for example, poetry or error logs)

# In this tutorial, we'll use three different English translations of the same work
# and a train model to identify the translator given a single line of text


# In[3]:


get_ipython().system('pip install -q tf-nightly')
import tensorflow as tf

import tensorflow_datasets as tfds
import os 


# In[5]:


# The texts of the three translations are by:

# 1. William Cowper
# 2. Edward, Earl of Derby
# 3. Samuel Butler

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

for name in FILE_NAMES:
  text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)

parent_dir = os.path.dirname(text_dir)

parent_dir


# In[6]:


# Load text into datasets
# Iterate through the files, loading each one into its own datasets.

def labeler(example, index):
  return example, tf.cast(index, tf.int64)  

labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
  lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
  labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
  labeled_data_sets.append(labeled_dataset)


# In[8]:


BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000


# In[9]:


all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
  all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
  
all_labeled_data = all_labeled_data.shuffle(
    BUFFER_SIZE, reshuffle_each_iteration=False)

for ex in all_labeled_data.take(5):
  print(ex)


# In[ ]:


# Como hemos visto, cada una de las líneas de cada traducción han sido 
# labeladas con [0,1,2] en función del autor de la traducción del texto.
# Recordar que el último objetivo será predecir quién es el traductor 
#de las líneas


# In[10]:


# Encode text lines as numbers
# Machine learning models work on numbers, not words, so the 
# String need to be converted into lists of numbers. To do that,
# map each unique word to a unique integer

# Build Vocabulary
# First, build a vocabulary by tokenizing the text into a collection
# of indiviual unique words. These are a few ways to do this in 
# both tensorFlow and Python. For this tutorial

# 1. Iterate over each example's numpy value
# 2. Use tfds.features.text.Tokenizer to split it into tokens
# 3. Collect these tokens into a Python set, to remove duplicates
# 4. Get the size of the vocabulary for later use

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
vocab_size


# In[11]:


# Construimos el vocabulario a través de todas las líneas que hemos
# extraído

# Encode examples
# Create an encoder by passing the vocabulary_set to 
# tfds.features.text.TokenTextEncoder. The encoder's encode
# method takes in a string of text and returns a list of integers

encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
example_text = next(iter(all_labeled_data))[0].numpy()
print(example_text)


# In[12]:


encoded_example = encoder.encode(example_text)
print(encoded_example)


# In[13]:


# Now run the encoder on the dataset by wrapping it in tf.py_function

def encode(text_tensor,label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
  # py_func doesn't set the shape of the returned tensors.
  encoded_text, label = tf.py_function(encode, 
                                       inp=[text, label], 
                                       Tout=(tf.int64, tf.int64))

  # `tf.data.Datasets` work best if all components have a shape set
  #  so set the shapes manually: 
  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)


# In[14]:


# Split the dataset into test and train batches
# Se añade padding porque en principio todas las filas no tienen
# porque tener el mismo número de elementos
train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))


# In[15]:


sample_text, sample_labels = next(iter(test_data))

sample_text[0], sample_labels[0]


# In[16]:


# Since we have introduced a new token encoding (the zero used for padding), the vocabulary size has increased by one.
vocab_size += 1


# In[17]:


# Build the model
model = tf.keras.Sequential()
# The first layer converts integer representations to dense vector embeddings. See the word embeddings tutorial or more details.
model.add(tf.keras.layers.Embedding(vocab_size, 64))

# The next layer is a Long Short-Term Memory layer, which lets the model understand words in their context with other words. A bidirectional wrapper on the LSTM helps it to learn about the datapoints in relationship to the datapoints that came before it and after it.
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))

# One or more dense layers.
# Edit the list in the `for` line to experiment with layer sizes.
for units in [64, 64]:
  model.add(tf.keras.layers.Dense(units, activation='relu')) # Simplemente está metiendo capas densas de 64 nodos

# Output layer. The first argument is the number of labels.
model.add(tf.keras.layers.Dense(3))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[18]:


# Train the model
model.fit(train_data, epochs=3, validation_data=test_data)

eval_loss, eval_acc = model.evaluate(test_data)

print('\nEval loss: {:.3f}, Eval accuracy: {:.3f}'.format(eval_loss, eval_acc))


# In[ ]:




