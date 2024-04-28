#!/usr/bin/env python
# coding: utf-8

# **Importing required Libraries.**

# In[ ]:


import tensorflow_datasets as tfds
import tensorflow as tf


# **Loading the dataset**

# The IMDB movie reviews dataset comes packaged in `tfds`. It has already been preprocessed so that the reviews (sequences of words) have been converted to sequences of integers, where each integer represents a specific word in a dictionary. Let's start the project!

# In[ ]:


dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


# In[ ]:


train_examples_batch, train_labels_batch = next(iter(train_dataset))
print(train_examples_batch)
print(train_labels_batch)


# ### Text Encoding
# The dataset info includes the encoder (a `tfds.features.text.SubwordTextEncoder`).
# This text encoder will reversibly encode any string, falling back to byte-encoding if necessary.

# In[ ]:


encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))


# In[ ]:


sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))


# In[ ]:


assert original_string == sample_string
for index in encoded_string:
  print('{} ----> {}'.format(index, encoder.decode([index])))


# Create batches of training data for your model. The reviews are all different lengths, so use `padded_batch` to zero pad the sequences while batching.

# In[ ]:


BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE)

test_dataset = test_dataset.padded_batch(BATCH_SIZE)


# **Build Model with an LSTM layer**
# 
# Creating a `tf.keras.Sequential` model and start with an embedding layer. An embedding layer stores one vector per word. When called, it converts the sequences of word indices to sequences of vectors. These vectors are trainable. After training (on enough data), words with similar meanings often have similar vectors.
# 
# This index-lookup is much more efficient than the equivalent operation of passing a one-hot encoded vector through a `tf.keras.layers.Dense layer`.
# 
# A recurrent neural network (RNN) processes sequence input by iterating through the elements. RNNs pass the outputs from one timestep to their input—and then to the next.
# 
# The `tf.keras.layers.Bidirectional` wrapper can also be used with an RNN layer. This propagates the input forward and backwards through the RNN layer and then concatenates the output. This helps the RNN to learn long range dependencies.
# Keras sequential model here since all the layers in the model only have single input and produce single output.

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.summary()


# Since this is a binary classification problem and the model outputs logits (a single-unit layer with a linear activation), we'll use the `binary_crossentropy` loss function. It is better for dealing with probabilities—it measures the "distance" between probability distributions, or in our case, between the ground-truth distribution and the predictions.

# In[ ]:


model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(1e-4),
              metrics = ['accuracy'])


# Let's Train the model for 10 `epochs`. This is 10 iterations over all samples in the `train_dataset` tensors. 

# In[ ]:


history = model.fit(train_dataset, epochs = 10, validation_data = test_dataset, validation_steps = 30)


# **Test For Accuracy**

# In[ ]:


test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))


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

sample_pred_text = ("""Three Idiots was an amazing film that really impressed me of how good Bollywood films can be. 
My emotions throughout the film was like a roller coaster, going from sad to jubilant in a matter of seconds. 
I mainly recommend this film to anyone who is unsure of watching a Bollywood film, yet
 I also recommend this film to every other person in the entirety of the earth.""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


# 3- IDIOTS 3 STAR COMMENT

sample_pred_text = ("""Some of the acting is good, the main song is catchy and colours look nice. But I do not see any other positives.

It can probably deserve a 4 or 5, but I will just give it a 3 because I really don't understand how can this movie be so high in the top 250.

I think the top 250 should give some weight to the credibility of the voters.""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


# 3-IDIOTS 2 STAR COMMENT

sample_pred_text = ("""To me a movie is an escapism from reality. 
I want to be thrilled and entertained. But '3 idiots' was reminding me of the reality and stating the obvious philosophies. 
It was predictable. Everything was so nice and cheesy. Oh hug me I hug you bullshits repeated over and over again. 
I love nice people and nice things happening BUT THATS IN REALITY. In the movie I wanted something different......""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


# 3-IDIOTS 1 STAR COMMENT

sample_pred_text = ("""I feel like a fool wasting three hours on what practically is the Indian equivalent of the Three Stooges. 
With some context, this was the first Bollywood film I had seen, and it doesn't make me want to rush back and view more. 
The jokes are pretty unfunny, almost inline with unintelligent slapstick comedy and weird foreign comedy. 
Nothing in this film made me want to continue viewing; poor acting, 
a poor script and horrible cinematography are among the worst I've ever seen.""")

predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


# 3-IDIOTS COMMENT

sample_pred_text = ("""3 Idiots is actually a movie for above 16 years of age and not at all suitable to be watched along with young kids. 
Apart from good theme and underline message, nothing in the movie is actually suitable for viewing by children under 16 years of age. 
May be it is due to commercial side of the movie in which producers have to gather the attention of people and
 make it more appealing to the grown up viewers.""")
predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# In[ ]:


sample_pred_text = ("""Rajkumar Hirani and Abhijat Joshi, who wrote the story, infuse a great amount of life into the film.
 ‘3 Idiots’ is the kind of film which takes you on a roller coaster ride right from the word go. 
 It also tries to address the root of the problem which is plaguing the Indian Education System. 
 The method of teaching has turned colleges into pressure cookers and the students are made to compete rather than excel. 
 What’s really good about the film is it has the right dose of funny moments which are brilliantly enacted by all its characters. 
 Aamir Khan brings the much needed star power to the film.""")
predictions = sample_predict(sample_pred_text, pad = False)
print(predictions)


# **Plotting Accuracy & Loss Function Graphs**

# In[ ]:


import matplotlib.pyplot as plt

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])
  plt.show()

plot_graphs(history, 'accuracy')


# In[ ]:


plot_graphs(history, 'loss')

