
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import time

import utils

# Define paramaters for the model
learning_rate = 0.001
batch_size = 75
n_epochs = 50
n_train = 60000
n_test = 10000

# Step 1: Read in data
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)


# In[2]:


# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data.shuffle(10000)
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices(test)
test_data = test_data.shuffle(10000)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data) # initializer for train_data
test_init = iterator.make_initializer(test_data) # initializer for test_data


# In[3]:


# create one iterator and initialize it with different datasets
iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data) # initializer for train_data
test_init = iterator.make_initializer(test_data) # initializer for test_data


# In[4]:


#Step3: Initialize weights and biases for 2 hidden layers and output layer

k=200#No. of neurons in first hidden layer
l=100#No. of neurons in second hidden layer
m= 10#No. of neurons in output layer

w1 = tf.get_variable(initializer=tf.random_normal(shape = [img.shape[1].value,k],
                                                 mean = 0, stddev= 0.01),name = 'Weight1')

b1 = tf.get_variable(initializer=tf.zeros(shape = [k]), name = 'Bias1')

w2 = tf.get_variable(initializer=tf.random_normal(shape = [k,l],
                                                 mean = 0, stddev= 0.01),name = 'Weight2')

b2 = tf.get_variable(initializer=tf.zeros(shape = [l]), name = 'Bias2')

w3 = tf.get_variable(initializer=tf.random_normal(shape = [l,m],
                                                 mean = 0, stddev= 0.01),name = 'Weight3')

b3 = tf.get_variable(initializer=tf.zeros(shape = [m]), name = 'Bias3')


# In[5]:


#Step 4 :Defining the hidden layers
Y1= tf.nn.relu(tf.matmul(img,w1)+b1)
Y2 = tf.nn.relu(tf.matmul(Y1,w2)+b2)


# In[6]:


#Step 5: Defining the output layer
logits = tf.matmul(Y2,w3)+b3


# In[7]:


# Step 6: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logits)
loss = tf.reduce_mean(entropy)


# In[8]:


# Step 7: define optimizer
# using Adam Optimizer with pre-defined learning rate to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Step 8: calculate accuracy with test set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())
with tf.Session() as sess:
   
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # train the model n_epochs times
    for i in range(n_epochs): 	
        sess.run(train_init)	# drawing samples from train_data
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
    print('Total time: {0} seconds'.format(time.time() - start_time))

    # test the model
    sess.run(test_init) # drawing samples from test_data
    total_correct_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy)
            total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
        pass

    print('Accuracy {0}'.format(total_correct_preds/n_test))
writer.close()

