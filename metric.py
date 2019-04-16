# generate the metrics for the naturally generated images. 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack


with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

#batch_size = config['training_batch_size']

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()
saver = tf.train.Saver()
model_file = tf.train.latest_checkpoint('C:\\Users\\Administrator\\Desktop\\mnist_challenge-master\\models\\natural')

#options: ./models/adv_trained.


'''
# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])
'''
# generate the embeddings of the training set and the testing set.

'''
with tf.Session() as sess:

    saver.restore(sess, model_file)
    
    eval_batch_size = 500


    num_batches = int(math.ceil( len(mnist.train.images) / eval_batch_size))
    print (len(mnist.train.images) )
    train_hidden = []
    print(num_batches)

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, len(mnist.train.images))

      x_batch = mnist.train.images[bstart:bend, :]
      y_batch = mnist.train.labels[bstart:bend]


      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      #dict_train = {mode}
      hidden = sess.run([model.h_fc1], feed_dict=dict_adv)
      if ibatch == 0:

      	train_hidden.append(hidden)
      	train_hidden = np.asarray(train_hidden)
      	train_hidden = train_hidden.reshape((500,1024))
      	print(train_hidden.shape)
      else:
      	train_hidden = np.concatenate((train_hidden, np.asarray(hidden).reshape((500,1024))), axis =0 )

    print(len(train_hidden))

    np.save('hidden_mnist_adv_train.npy', train_hidden)

    print('finish....')

with tf.Session() as sess:

    saver.restore(sess, model_file)
    
    eval_batch_size = 500


    num_batches = int(math.ceil( len(mnist.test.images) / eval_batch_size))
    print(num_batches)
    train_hidden = []

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, len(mnist.test.images))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]
      

      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      #dict_train = {mode}

      hidden = sess.run([model.h_fc1], feed_dict=dict_adv)

      if ibatch == 0:

      	train_hidden.append(hidden)
      	train_hidden = np.asarray(train_hidden)
      	train_hidden = train_hidden.reshape((500,1024))
      	print(train_hidden.shape)
      else:
      	train_hidden = np.concatenate((train_hidden, np.asarray(hidden).reshape((500,1024))), axis =0 )

    print(len(train_hidden))

    np.save('hidden_mnist_adv_test.npy', train_hidden)

    print('finish....')
'''

# computing the distance in the natural images. 

'''
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

test_data = np.load('hidden_mnist_adv_test.npy')
train_data = np.load('hidden_mnist_adv_train.npy')

distance_all = []
index = 0
for j in range(len(test_data)):
	distance = []
	for i in range(len(train_data)):

		if mnist.test.labels[j] == mnist.train.labels[i]: # should be cautious the images are stored in order with the orginal testing images. 
			op1=np.linalg.norm(train_data[i]-test_data[j])
		#np.sqrt(np.sum(np.square(j-i)))
			distance.append(op1)
	sorted(distance)
	temp = distance[0:5]
	hhh = averagenum(temp)
	distance_all.append(hhh)
	print('finish {} sample...'.format(index))
	print(hhh)
	index += 1
np.save('mnist_adv_test_rank.npy', distance_all)
'''


'''
#prepare the perturbed images.
#----------------1-0---------------------
x_batch = mnist.test.images
y_batch = mnist.test.labels

#print(x_batch[0])

np.savez('mnist_1_0_test',x_test = x_batch, y_test = y_batch)
'''


'''

#------------------0.9-0------------------
np.savez('mnist_0.9_0_test', x_test = x_batch * 0.9, y_test = y_batch)

#------------------0.9-0.05--------------------

np.savez('mnist_0.9_0.05_test', x_test = x_batch * 0.9 + 0.05, y_test = y_batch)

#-------------------0.8-0.1-------------

np.savez('mnist_0.8_0.1_test', x_test = x_batch * 0.8+ 0.1, y_test = y_batch)
#-------------0.8-0-----------------

np.savez('mnist_0.8_0_test', x_test = x_batch * 0.8, y_test = y_batch)

#------------0.7-0.15--------------

np.savez('mnist_0.7_0.15_test', x_test = x_batch * 0.7+0.15, y_test = y_batch)

#
np.savez('mnist_0.7_0_test', x_test = x_batch * 0.7 ,y_test = y_batch)
'''


'''
# generate the fake images using cw2 attack or pgd attack.
#from cw_attack import *

f = np.load('./mnist_1_0_test.npz')

data_test, labels_test = f['x_test'], f['y_test']

attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

with tf.Session() as sess:

    saver.restore(sess, model_file)
    
    eval_batch_size = 500

    num_batches = int(math.ceil( len(mnist.test.images) / eval_batch_size))
    print(num_batches)
    adv = []

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, len(mnist.test.images))

      x_batch = data_test[bstart:bend, :] # note that here is the perturbed images not the original images. 
      #print(x_batch.shape)
      y_batch = labels_test[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      print(x_batch_adv.shape)

      if ibatch == 0:

      	adv.append(x_batch_adv)
      	adv = np.asarray(adv)
      	adv = adv.reshape((500,784))
      	print(adv.shape)
      else:
      	adv = np.concatenate((adv, np.asarray(x_batch_adv).reshape((500,784))), axis =0 )
	
    print(len(adv))

    np.save('adv_1_0_mnist_adv.npy', adv)

    print('finish....')
'''
#np.save('temp_1_0.npy',data_test)

'''

# verifying the distance between the testing images and the training set.
def subtract (a_list,b_list):
    ret_list = []
    for item in a_list:
        if item not in b_list:
            ret_list.append(item)
    for item in b_list:
        if item not in a_list:
            ret_list.append(item)
    return ret_list

prediction = np.load('prediction_0.8_0.npy')
distance = np.load('mnist_adv_test_rank.npy')
print(np.max(distance))
print(np.min(distance))
true_index = []

print(distance)
for i in range(len(prediction)):

	if not prediction[i]:
		true_index.append(i)

dist_false = [] # false matching pair
for j in true_index:
	dist_false.append(distance[j])

dist_true =[] # true matching pair

for k in subtract( list(range(len(prediction))), true_index):
	dist_true.append(distance[k])

print(np.mean(dist_false))
print(np.mean(dist_true))

import matplotlib.pyplot as plt


# plot the result like something in figure 1, 2 and 3
min_dist = np.floor( np.min(distance) )
max_dist = np.floor( np.max(distance) )

pool = {}
# remmbering the indexes


for the in range(int(min_dist), int(max_dist+1)):
	pool[the] = []

	for data in range(len(distance)):
		if distance[data] >= the and distance[data] <= the+1:
			pool[the].append(data)

pool2 = {}

for k in pool.keys():
	rate = 0
	for j in pool[k]:
		if not prediction[j]:
			rate+=1
	accuracy = float(rate) / float(len(pool[k]))
	pool2[k]=accuracy

# mind the order

final_x = list(pool.keys())
final_y = []

for k in final_x:
	final_y.append(pool2[k])

plt.plot(final_x, final_y)
plt.show()
'''


#compute the KL-divergence of training and testing images. 

