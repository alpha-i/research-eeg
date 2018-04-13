import os
from timeit import default_timer as timer

import logging
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matlab_data import make_eeg_data_provider, load_normal_test_batch, load_abnormal_test_batch

logger = logging.getLogger('tipper')
logger.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

OUTPUT_DIR = './outputs/'

N_EPOCHS = 128
N_SENSORS = 16
N_TIMESTEPS = 64
DATA_LENGTH = N_TIMESTEPS * N_SENSORS
PRINT_EPOCH_INTERVAL = 10


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


data_provider = make_eeg_data_provider(feature_length=DATA_LENGTH)

normal_test_batch = load_normal_test_batch(feature_length=DATA_LENGTH)
abnormal_test_batch = load_abnormal_test_batch(feature_length=DATA_LENGTH)

X = tf.placeholder(tf.float32, shape=[None, DATA_LENGTH])
Z = tf.placeholder(tf.float32, shape=[None, 100])

D_W1 = tf.Variable(xavier_init([DATA_LENGTH, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))
D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_b1, D_b2]

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, DATA_LENGTH]))
G_b2 = tf.Variable(tf.zeros(shape=[DATA_LENGTH]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    G_signal = (G_prob - 0.5) * 2

    return G_signal


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample_data = sample.reshape(N_SENSORS, N_TIMESTEPS)
        for i in range(N_SENSORS):  # Vertical offsets for visual clarity
            sample_data[i, :] += i * 1.5

        plt.plot(np.transpose(sample_data))

    return fig


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))

# Use this for testing known anomalous data
D_loss_preictal = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.zeros_like(D_logit_real)))

D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

batch_size = data_provider.batch_size
Z_dim = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
it = 0
n_batches = data_provider.number_of_batches

for epoch in range(N_EPOCHS):
    data_provider.shuffle_data()

    start_time = timer()

    if epoch % PRINT_EPOCH_INTERVAL == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        figure_filename = os.path.join(
            OUTPUT_DIR,
            '{}.png'.format(str(i).zfill(3))
        )

        plt.savefig(figure_filename, bbox_inches='tight')
        i += 1
        plt.close(fig)

    for batch_number in range(n_batches):

        it += 1

        # small amount of noise desirable since raw data is rounded to integers
        batch_data = data_provider.get_noisy_batch(batch_number, noise_amplitude=0.01)

        if it == 1:
            label_shape = 0 if batch_data.labels is None else batch_data.labels.shape
            logging.info("Training {} batches of size {} and {}".format(
                n_batches,
                batch_data.features.shape,
                label_shape
            ))

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: batch_data.features, Z: sample_Z(batch_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

    if epoch % PRINT_EPOCH_INTERVAL == 0:
        # See how the D performs against the test samples

        time_epoch = timer() - start_time

        D_test_loss = sess.run(D_loss_real, feed_dict={X: normal_test_batch})
        D_preictal_loss = sess.run(D_loss_preictal, feed_dict={X: abnormal_test_batch})
        D_false_preictal_loss = sess.run(D_loss_real, feed_dict={X: abnormal_test_batch})

        msg = "Epoch {} of {} ...  in {:.2f} seconds."
        logging.info(msg.format(epoch + 1, N_EPOCHS, time_epoch))
        logging.info('Iter: {}'.format(it))
        logging.info('G loss: {:.4}'.format(G_loss_curr))
        logging.info('D train loss: {:.4}'.format(D_loss_curr))
        logging.info('D test loss: {:.4}'.format(D_test_loss))
        logging.info('D false negative test loss: {:.4}'.format(D_false_preictal_loss))
        logging.info('D seizure test loss: {:.4}'.format(D_preictal_loss))




# Finally let's see how well the discriminator performs on a) more normal data (b) weird data and (c) generated data

    # INFO:root:Epoch 121 of 128 ...  in 6.21 seconds.
    # INFO:root:Iter: 226512
    # INFO:root:G loss: 1.016
    # INFO:root:D train loss: 1.235
    # INFO:root:D test loss: 0.6048
    # INFO:root:D false negative test loss: 0.7878
    # INFO:root:D seizure test loss: 1.287

# for it in range(1000000):
#     if it % 1000 == 0:
#         samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
#
#         fig = plot(samples)
#         plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#         i += 1
#         plt.close(fig)
#
#     X_mb, _ = data_provider.next_batch(batch_size)
#
#     _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, Z_dim)})
#     _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})
#
#     if it % 1000 == 0:
#         print('Iter: {}'.format(it))
#         print('D loss: {:.4}'. format(D_loss_curr))
#         print('G_loss: {:.4}'.format(G_loss_curr))
#         print()
