#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Save and load models
# Model progress can be saved during training. This means a model
# can resume where it left off and avoid long training times.
# When publishing research models and techniques, most machine
# learning practitioners share:

# 1. Code to create the model, and
# 2. The trained weigths, or parameters for the model

# Required to save models in HDF5 format
#pip install -q pyyaml h5py  


# In[2]:


import os
import tensorflow as tf
from tensorflow import keras
print(tf.version.VERSION)


# In[3]:


# Get an example dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# In[4]:


# Define a model: Sart by building a simple sequential model
# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()


# In[5]:


# Checkpoint callback usage

# Save checkpoints during training
# You can use a trained model without having to retrain it, or
# pick-up training where you left off. The tf.keras.callbacks.ModelChecpoint
# callback allows to continually save the model both during and at the end
# of the training

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.


# In[7]:


# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# In[9]:


# Then load the weigths from the checkpoint and re-evaluate

# Loads the weigths
model.load_weights(checkpoint_path)

#Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose="2")
print("Restored model, acuracy:{:5.2f}%".format(100*acc))


# In[11]:


# Checkpoitn callbacks option

# The callback provides several options to provide unique names
# fro checkpoints and adjust the checkpoint frequency. Train
# anew model, and save uniquely named checkpoints once every five
# epochs

# Incluude the epoch in the file name (uses 'str.format')
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"

# Create a callback that saves the model's weight every 5 epochs

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5) # Indica que vamos a guardar los valores cada 5 peridos

# Create a new model instance
model = create_model()

# Save the weights using the checkpoint_path format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images,
         train_labels,
         epochs=50,
         callbacks=[cp_callback],
         validation_data=(test_images,test_labels),
         verbose=0)


# In[12]:


latest = tf.train.latest_checkpoint(checkpoint_dir)
latest


# In[15]:


# Create a new model instance

model = create_model()

#Load the previously saved weigths
model.load_weights("training_2/cp-0050.ckpt")

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# In[16]:


# What are these files?
# The above code stores the weights to a collection of checkpoint
# formatted files that contain only the trained weights in a binary format
# Checkpoint contain:

# 1. One or more shards that contain your model's weights
# 2. An index file that indicates which weights are stored in which shard

# Manually save weights

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose = 2)
print("Restored model, accuracy : {:5.2f}%".format(100*acc))


# In[17]:


# Save the entire model

# Call model.save to save a model's architecture, weigths and training configuration
# in a single file/folder. This allows you to export a model so it can be 
# used without access to orgininal Python code.

# Entire model can be saved in two different file formats (SavedModel and HDF5)

# SavedModel format

# The SavedModel format is another way to serialize models. Models
# saved in this format can be restored useing tf.keras.models.load_model

# Create and train a new model instance
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel
get_ipython().system('mkdir -p saved_model')
model.save('saved_model/my_model')


# In[18]:


get_ipython().system('ls saved_model/my_model')


# In[20]:


# Reload a fresh keras model from the saved model
new_model = tf.keras.models.load_model('saved_model/my_model')

#check its architecture
new_model.summary()


# In[22]:


# Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose= 2)
print("Restored model, accuracy : {:5.2f}%".format(100*acc))
print(new_model.predict(test_images).shape)


# In[23]:


# HDF5 format
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file
# The '.h5' extension indicates that the model should be shaved to HDF5
model.save('my_model.h5')


# In[25]:


# Now recreate the model from that file:

# Recreate the exact same model, including its weigths and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the new model architecture
new_model.summary()


# In[26]:


loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))


# In[ ]:



# Saving custom objects:

# The key difference between HDF5 and SavedModel is that HDF5
# uses object configs to save the model architecture,
# while SavedModel saves the executino graph. 

