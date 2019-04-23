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
#from cw2 import *

with open('config.json') as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

#batch_size = config['training_batch_size']

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model()
saver = tf.train.Saver()
model_file = tf.train.latest_checkpoint('C:\\Users\\Administrator\\Desktop\\Implementation-of-ICLR2019-paper-blind-spot-attack\\Implementation-of-ICLR2019-paper-blind-spot-attack\\models\\secret')

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

data = np.load('adv_1_0_mnist_adv.npy')

with tf.Session() as sess:

    saver.restore(sess, model_file)
    
    eval_batch_size = 500


    num_batches = int(math.ceil( len(mnist.test.images) / eval_batch_size))
    print(num_batches)
    train_hidden = []

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, len(mnist.test.images))

      x_batch = data[bstart:bend, :]#mnist.test.images
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

    np.save('hidden_mnist_adv_test_temp.npy', train_hidden)

    print('finish....')


# computing the distance in the natural images. 

'''


'''
def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

test_data = np.load('hidden_mnist_adv_test_temp.npy')

#test_data = np.load('adv_1_0_mnist_adv.npy')
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
np.save('mnist_adv_test_rank_temp.npy', distance_all)

'''

'''
#prepare the perturbed images.
#----------------1-0---------------------


#print(x_batch[0])

np.savez('mnist_1_0_test',x_test = x_batch, y_test = y_batch)
'''
x_batch = mnist.test.images
#print(x_batch[0])
y_batch = mnist.test.labels

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



# generate the fake images using cw2 attack or pgd attack.
#from cw_attack import *
'''
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

prediction = np.load('prediction_1_0.npy')
distance = np.load('mnist_adv_test_rank_temp.npy')
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

# plot the exemplary successfullt attacked images when the distance is large (feature extrator is adv_trained model)
'''
result = np.load('prediction_1_0.npy')


import matplotlib.pyplot as plt
import operator


f = np.load('./mnist_1_0_test.npz')
data_test, labels_test = f['x_test'], f['y_test']


#f = np.load('./mnist_0.9_0.05_test.npz')
distance = np.load('mnist_adv_test_rank.npy') # the distances. 

dist_dict = {}
for i in range(len(distance)):
	dist_dict[i] = distance[i]

temp = sorted(dist_dict.items(), key = operator.itemgetter(1))

#print(temp)
image_for_generate = []
label_for_generate = []

#print(temp[9995:10000])
# selecting the maximum 5 samples.
for k in temp[9900:10000]:

	if not result[k[0]] and len(image_for_generate) < 10:
		image_for_generate.append( data_test[ k[0] ])
		label_for_generate.append( labels_test [ k[0] ])

assert len(image_for_generate) <= 10




attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

with tf.Session() as sess:

    saver.restore(sess, model_file)
    
    #eval_batch_size = 1

    #num_batches = int(math.ceil( len(mnist.test.images) / eval_batch_size))
    #print(num_batches)
    #adv = []

    #for ibatch in range(num_batches):
      #bstart = ibatch * eval_batch_size
      #bend = min(bstart + eval_batch_size, len(mnist.test.images))

    x_batch = np.asarray(image_for_generate) # note that here is the perturbed images not the original images. 
      #print(x_batch.shape)
    y_batch = np.asarray(label_for_generate)

    x_batch_adv = attack.perturb(x_batch, y_batch, sess)

    print(x_batch_adv.shape)

    
	
    #print(len(adv))

    np.save('1_0_plot_adv.npy', x_batch_adv)

    print('finish....')



print(label_for_generate)
a = np.load('1_0_plot_adv.npy')
print(a.shape)
all = []
for i in range(len(a)):
	b = a[i].reshape((28,28))
	all.append(b)
all = np.asarray(all)
print(all.shape)

#for obj in range(len(all)):
 # plt.gray()
 # plt.imsave( 'C:\\Users\\Administrator\\Desktop\\Implementation-of-ICLR2019-paper-blind-spot-attack\\Implementation-of-ICLR2019-paper-blind-spot-attack\\image\\'+\
 #   str(obj) + '.png',all[obj] )


def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

show_images(all, titles= label_for_generate)


'''
'''
with tf.Session() as sess:
  saver.restore(sess, model_file)
  x_batch = image_for_generate
  y_batch = label_for_generate
  
  dict_adv = {model.x_input: x_batch, model.y_input: y_batch}
      

  hidden = sess.run([model.y_pred], feed_dict=dict_adv)

  print(hidden)

  print('finish.....')

'''





