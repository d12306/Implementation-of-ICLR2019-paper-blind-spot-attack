"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      x += self.a * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
      x = np.clip(x, 0, 1) # ensure valid pixel range

    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))





# attack code

#generate the cw l-infinity attack

from art.classifiers import TFClassifier
from art.attacks import CarliniL2Method, CarliniLInfMethod
from art.utils import load_mnist, random_targets, master_seed, get_classifier_tf, get_classifier_kr

f = np.load('./mnist_0.9_0.05_test.npz')

data_test, labels_test = f['x_test'], f['y_test']
with tf.Session() as sess:

    saver.restore(sess, model_file)
    
    eval_batch_size = 500

    num_batches = int(math.ceil( len(mnist.test.images) / eval_batch_size))
    print(num_batches)
    adv = []

    tfc = model.cw2(sess)


    clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=100, eps=0.3)

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, len(mnist.test.images))

      x_batch = data_test[bstart:bend, :] # note that here is the perturbed images not the original images. 
      #print(x_batch.shape)
      x_batch = x_batch.reshape((len(x_batch), 28,28,1))
      y_batch = labels_test[bstart:bend]

      params = {'y': random_targets(y_batch, tfc.nb_classes)}
      x_batch_adv = clinfm.generate(x_batch, **params)
      

      #x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      print(x_batch_adv.shape)

      if ibatch == 0:

        adv.append(x_batch_adv)
        adv = np.asarray(adv)
        adv = adv.reshape((500,784))
        print(adv.shape)
      else:
        adv = np.concatenate((adv, np.asarray(x_batch_adv).reshape((500,784))), axis =0 )
  
    print(len(adv))

    np.save('adv_0.9_0.05_mnist_cw2.npy', adv)

    print('finish....')

    