#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Text generation with RNN
# This tutorial demonstrates how to generate text using a character-based RNN.
# We will work with a dataset of Shakespeare's writing from
# Andrej Karpathy's The Unreasonable Effectiveness of Recurrent
# Neural Networks

# Given a sequence of characters from this data, train a model to predict
# the next character in the sequence

import tensorflow as tf
import numpy as np
import os
import time


# In[4]:


# Download the Shakespeare dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the data
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))
# Take a look at the first 250 characters in text
print(text[:250])


# In[5]:


# The unique characters in the file
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))


# In[6]:


# Process the text
# Before training, we need to map strings to a numerical representations. 
# Create two lookup tables: One mapping characterers to numbers, and another for numbers to characters

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')


# In[7]:


print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))


# # The Prediction task
# https://medium.com/mlreview/making-ai-art-with-style-transfer-using-keras-8bb5fa44b216
# 
# Given a character, or a sequence of characters, what's is the most probable next character? The input to the model will be a sequence of characters and we train the model to predict the output- the following character at each time step.
# 
# Since RNNs maintain an internal state that depends on the previously see elements, given all the characters computed util this moment,what is the next character

# # Create training examples
# 
# Next divide the text into example sequences. Each input sequences will contain seq_length characters from the text. For each input sequence, the correspondign targets contain the same length of text, except shifted one character to the right.
# 
# So break the text into chunks of seq_length+1. For example, say seq_length is 4 and our text is "Hello". The input sequence would be "Hell" and the target sequence "ello".
# 
# To do this firs use the tf.data.Datasets.from_tensor_slices function to convert the text vector into a stream of character indices
# 

# In[9]:


# The maximun lenght sentece we want for a single input in characters

seq_length = 100
examples_per_epoch = len(text)

# Create training examples /target

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(10):
    print(idx2char[i.numpy()])


# In[13]:


# The batch method lets us easyly convert these individuals characters
# to sequence of desired size

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))


# In[12]:


def split_input_target(chunk):
    input_text = chunk [:-1]
    target_text = chunk [1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


# Each index of these vectors are processed as one time step. For the input at time 0, the model receives the index for "F" and trys to predict the index for "i" as the next character. At the next timestep, it does the same thing but the RNN considers the previous step contexts in addition to the current input character.

# In[14]:


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# In[15]:


# Create training Batches

# Batch size 
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset

# De este modo cogemos BATCH de 64 elementos con 100 elementos cada uno, que era la longitud de la sentencia, uno de input y otro de target


# # Build the model
# Use tf.keras.Sequential to define the model. For this simple example three layers are used to define our model:
# * tf.keras.layers.Embedding: The input layer. A trainable lookup table that will map the numbers of each character to a vector with embedding_dim dimensions;
# * tf.keras.layers.GRU: A type of RNN with size units=rnn_units (You can also use a LSTM layer here)
# * tf.keras.layers.Dense: The output layer, with vocab_size outputs

# In[16]:


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


# For each character the model looks up the embedding, runs the GRU one timestep with the embedding as input, and applies the dense layer to generate logits predicting the log-liklihood of the next character

# In[17]:


#Try the model
# Check the shape of the output

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    


# Tal y como se puede comprobar, el output es un cubo, y en la tercer dimensión nos da la probabilidad del siguiente caracter 

# In[18]:


model.summary()


# To get actual predictions from the model we need to sample from the output distribution, to get character indices. This distribution is defined by the logits over the character vocabulary

# In[19]:


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices


# In[20]:


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


# In[21]:


# Train the model
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


# In[22]:


model.compile(optimizer='adam', loss=loss)


# In[23]:


# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# # Generate Text
# ## Restore the latest checkpoint

# In[24]:


tf.train.latest_checkpoint(checkpoint_dir)


# In[25]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()


# ## The prediction Loop
# The following code block generates the text:
# 
# * It starts by choosing a start string, initializing the RNN state and setting the number of characters to generate
# 
# * Get the prediction distribution of the next character using the start and the RNN state
# 
# * Then, use a categorical distribution to calculate the index of the predicted character. Use this predicted character as our next input to the model
# 
# * The RNN state returned by the model is bed back into the model so that is now has more context, instead than only one character. After predictig the next character, the modified RNN states are agin fed back into the model, which is how it learns as it gets more context from the previously predicted characters

# In[35]:


def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 3500

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 0.8

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"ERNESTO: "))


# In[ ]:




