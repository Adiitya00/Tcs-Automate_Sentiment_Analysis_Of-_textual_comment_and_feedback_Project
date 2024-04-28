#!/usr/bin/env python
# coding: utf-8

# **Importing required Libraries.**

# In[ ]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np


# **Loading the dataset**

# The IMDB movie reviews dataset comes packaged in `tfds`. It has already been preprocessed so that the reviews (sequences of words) have been converted to sequences of integers, where each integer represents a specific word in a dictionary. Let's start the project!

# In[ ]:


(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k', 
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised = True,
    # Also return the `info` structure. 
    with_info = True)


# In[ ]:


encoder = info.features['text'].encoder


# In[ ]:


print ('Vocabulary size: {}'.format(encoder.vocab_size))


# In[ ]:


for train_example, train_label in train_data.take(1):
  print('Encoded text:', train_example[:10].numpy())
  print('Label:', train_label.numpy())


# In[ ]:


BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32))

test_batches = (
    test_data
    .padded_batch(32))


# **Let's Build the Model**

# In[ ]:


model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1)])

model.summary()


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(train_batches,
                    epochs = 10,
                    validation_data = test_batches,
                    validation_steps = 30)


# **Test For Accuracy**

# In[ ]:


loss, accuracy = model.evaluate(test_batches)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


# **Prediction Functions**

# In[ ]:


def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec


# In[ ]:


def sample_predict(sample_pred_text, pad):
  encoded_sample_pred_text = encoder.encode(sample_pred_text)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
  print("Prediction Score: ", predictions)
  output = ""
  if predictions[0][0] >= 0.5: output = "POSITIVE"
  elif predictions[0][0] <= -1: output = "NEGATIVE"
  else: output = "NEUTRAL"

  return output


# **Prediction with Sample Sentiments of 3 idiots movie from bollywood**

# In[ ]:


sample_pred_text = ('The movie was not good. The animation and the graphics were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


sample_pred_text = ('The movie was cool. The animation and the graphics were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


sample_pred_text = ('This movie is awesome. The acting was incredicable. Highly recommend')
predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


sample_pred_text = ('This movie was so so. The acting was medicore. Kind of recommend')
predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


# 3-IDIOTS 5 STAR COMMENT

sample_pred_text = ("""Three Idiots was an amazing film that really impressed me of how good Bollywood films can be. My emotions throughout the film was like a roller coaster, going from sad to jubilant in a matter of seconds. 
I mainly recommend this film to anyone who is unsure of watching a Bollywood film, 
yet I also recommend this film to every other person in the entirety of the earth.""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


# 3-IDIOTS 3 STAR COMMENT

sample_pred_text = ("""Some of the acting is good, the main song is catchy and colours look nice. 
But I do not see any other positives. It can probably deserve a 4 or 5, but I will just give it a 3 
because I really don't understand how can this movie be so high in the top 250.
I think the top 250 should give some weight to the credibility of the voters.""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


# 3-IDIOTS 1 STAR COMMENT

sample_pred_text = ("""I feel like a fool wasting three hours on what practically is the Indian equivalent of the Three Stooges. 
With some context, this was the first Bollywood film I had seen, and it doesn't make me want to rush back and view more. 
The jokes are pretty unfunny, almost inline with unintelligent slapstick comedy and weird foreign comedy. 
Nothing in this film made me want to continue viewing; poor acting, a poor script
and horrible cinematography are among the worst I've ever seen.""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


sample_pred_text = ("""3 Idiots is actually a movie for above 16 years of age and
 not at all suitable to be watched along with young kids. 
 Apart from good theme and underline message, nothing in the movie is actually suitable for viewing by children under 16 years of age. 
 May be it is due to commercial side of the movie in which producers have to gather the attention of people
and make it more appealing to the grown up viewers.""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


sample_pred_text = ("""Rajkumar Hirani and Abhijat Joshi, who wrote the story, 
infuse a great amount of life into the film. ‘3 Idiots’ is the kind of film which takes you on a roller coaster ride right from the word go. 
It also tries to address the root of the problem which is plaguing the Indian Education System. 
The method of teaching has turned colleges into pressure cookers and the students are made to compete rather than excel. 
What’s really good about the film is it has the right dose of funny moments which are brilliantly enacted by all its characters. 
Aamir Khan brings the much needed star power to the film.""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


history_dict = history.history
history_dict.keys()


# **Plotting Accuracy & Loss Function Graphs**

# In[ ]:


import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

