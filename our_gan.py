import os, sys

sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.datasets
from tensorflow import layers
from keras.datasets import mnist
import time

num_epochs = 1

BATCH_SIZE = 64
TRAINING_RATIO = 5  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
OUTPUT_DIM = 784
disc_iters = 5
num_labels = 10
latent_dim = 128
DIM = 64
channel_first = False


def generate_images(images, epoch):
	# output gen: (-1,1) --> (-127.5, 127.5) --> (0, 255)
	# shape 10x784
	# plt.figure()
	plt.figure(figsize=(100, 10))
	test_image_stack = np.squeeze((np.array(images, dtype=np.float32) * 127.5) + 127.5)
	for i in range(10):
		new_image = test_image_stack[i].reshape(28, 28)
		plt.subplot(1, 10, i + 1)
		plt.axis("off")
		plt.imshow(new_image)
		plt.axis("off")
	plt.axis("off")
	plt.savefig("epoch_" + str(epoch) + ".png")


def generator(n_samples, noise_with_labels, reuse=None):
	"""
    :param n_samples:         number of samples
    :param noise_with_labels: latent noise + labels
    :return:                  generated images
    """
	with tf.variable_scope('Generator', reuse=reuse):  # Needed for later, in order to get variables of discriminator
		# ----- Layer1, Dense, Batch, Leaky ----- #
		alpha = 0.01
		output = layers.dense(inputs=noise_with_labels, units=4 * 4 * 4 * 64)
		output = layers.batch_normalization(output)
		output = tf.maximum(alpha * output, output)
		if channel_first:
			# size: 128 x 7 x 7
			output = tf.reshape(output, (-1, 4 * 64, 4, 4))
			bn_axis = 1  # [0, 2, 3]  # first
		else:
			# size: 7 x 7 x 128
			output = tf.reshape(output, (-1, 4, 4, 4 * 64))
			bn_axis = -1  # [0, 1, 2]  # last

		# ----- Layer2, deConv, Batch, Leaky ----- #
		output = layers.conv2d_transpose(output, filters=4 * DIM, kernel_size=(5, 5), strides=2, padding='same')
		output = layers.batch_normalization(output, axis=bn_axis)
		output = tf.maximum(alpha * output, output)
		if channel_first:
			output = output[:, :, :7, :7]
		else:
			output = output[:, :7, :7, :]

		# ----- Layer3, deConv, Batch, Leaky ----- #
		output = layers.conv2d_transpose(output, filters=2 * DIM, kernel_size=(5, 5), strides=2, padding='same')
		output = layers.batch_normalization(output, axis=bn_axis)
		output = tf.maximum(alpha * output, output)

		# ----- Layer4, deConv, Batch, Leaky ----- #
		output = layers.conv2d_transpose(output, filters=DIM, kernel_size=(5, 5), strides=2, padding='same')
		output = layers.batch_normalization(output, axis=bn_axis)
		output = tf.maximum(alpha * output, output)

		# ----- Layer5, deConv, Batch, Leaky ----- #
		output = layers.conv2d_transpose(output, filters=1, kernel_size=(5, 5), strides=1, padding='same')
		output = tf.nn.tanh(output)

		output = tf.reshape(output, [-1, OUTPUT_DIM])
		print('Generator output size:')
		print(output)

	return output


def discriminator(images, reuse=None):
	"""
    :param images:    images that are input of the discriminator
    :return:          likeliness of the image
    """
	with tf.variable_scope('Discriminator', reuse=reuse):  # Needed for later, in order to get variables of generator
		if channel_first:
			output = tf.reshape(images, [-1, 1, 28, 28])
		else:
			output = tf.reshape(images, [-1, 28, 28, 1])

		# ----- Layer1, Conv, Leaky ----- #
		alpha = 0.01
		output = layers.conv2d(output, filters=DIM, kernel_size=(5, 5), strides=2, padding='same')
		output = tf.maximum(alpha * output, output)

		# ----- Layer2, Conv, Leaky ----- #
		output = layers.conv2d(output, filters=2 * DIM, kernel_size=(5, 5), strides=2, padding='same')
		output = tf.maximum(alpha * output, output)

		# ----- Layer3, Conv, Leaky ----- #
		output = layers.conv2d(output, filters=4 * DIM, kernel_size=(5, 5), strides=2, padding='same')
		output = tf.maximum(alpha * output, output)
		output = tf.reshape(output, [-1, 4 * 4 * 4 * DIM])

		# ----- Layer4, Dense, Linear ----- #
		output = layers.dense(output, units=11)

		print('Discriminator output size:')
		print(output)

	scores_out = tf.identity(output[:, :1], name='scores_out')
	labels_out = tf.identity(output[:, 1:], name='labels_out')
	return scores_out, labels_out


def get_trainable_variables():
	"""
    :return: trainable variables (d_vars, g_vars)
    """
	tvars = tf.trainable_variables()
	d_vars = [var for var in tvars if 'Discriminator' in var.name]
	g_vars = [var for var in tvars if 'Generator' in var.name]
	return d_vars, g_vars


# --------------------------------- Placeholders ------------------------------- #

# GENERATOR
# ----- Noise + Labels(G) ----- #
input_generator = tf.placeholder(tf.float32, shape=[BATCH_SIZE, latent_dim + num_labels])
test_input = tf.placeholder(tf.float32, shape=[10, latent_dim + num_labels])

# DISCRIMINATOR
# ------ Real Samples(D) ------ #
real_samples = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])

# -------- Labels(D) ---------- #
labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, num_labels])

# ----------------------------------- Outputs ----------------------------------- #
fake_samples = generator(BATCH_SIZE, input_generator)
test_samples = generator(10, test_input, reuse=True)

disc_real_score, disc_real_labels = discriminator(real_samples)
disc_fake_score, disc_fake_labels = discriminator(fake_samples, reuse=True)

# Trainable variables
d_vars, g_vars = get_trainable_variables()

# ---------------------------------- Losses ------------------------------------ #

# ----- Gen Loss ----- #

# wasserstein
gen_wasserstein_loss = -tf.reduce_mean(disc_fake_score)  # WASSERSTEIN

# labels
labels_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
                                                               logits=disc_fake_labels)
generator_loss = gen_wasserstein_loss + labels_penalty_fakes

# ----- Disc Loss ----- #

# wasserstein
disc_wasserstein_loss = tf.reduce_mean(disc_fake_score) - tf.reduce_mean(disc_real_score)

# labels
labels_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
                                                               logits=disc_fake_labels)
labels_penalty_real = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
                                                              logits=disc_real_labels)

# gradient penalty
alpha = tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
differences = fake_samples - real_samples
interpolates = real_samples + alpha * differences
gradients = tf.gradients(discriminator(interpolates, reuse=True)[0], [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

# sum losses
discriminator_loss = disc_wasserstein_loss + labels_penalty_fakes + labels_penalty_real + gradient_penalty

# ---------------------------------- Optimizers ----------------------------------- #
generator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
                                             beta1=0.5,
                                             beta2=0.9).minimize(generator_loss, var_list=g_vars)

discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
                                                 beta1=0.5,
                                                 beta2=0.9).minimize(discriminator_loss, var_list=d_vars)

# -------------------------------- Load Dataset ---------------------------------- #
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.reshape(X_train, newshape=[-1, OUTPUT_DIM])
X_test = np.reshape(X_test, newshape=[-1, OUTPUT_DIM])
X_train = np.concatenate((X_train, X_test), axis=0)
X_train = (X_train - 127.5) / 127.5

y_train = np.concatenate((y_train, y_test), axis=0)
y_hot = np.zeros((y_train.shape[0], 10))
b = np.arange(y_train.shape[0])
y_hot[b, y_train] = 1
y_train = y_hot

# ------------------------------------ Train ------------------------------------- #
with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	indices = np.arange(X_train.shape[0])

	# big batch size
	macro_batches_size = BATCH_SIZE * disc_iters
	# num of batches
	num_macro_batches = int((X_train.shape[0]) // macro_batches_size)
	discriminator_history = []
	generator_history = []

	# EPOCHS FOR
	for iteration in range(num_epochs):
		start_time = time.time()
		print("epoch: ", iteration)
		np.random.shuffle(indices)
		X_train = X_train[indices]
		y_train = y_train[indices]

		# MACRO BATCHES FOR
		for i in range(num_macro_batches):  # macro batches
			# print("macro_batch: ", i)
			discriminator_macro_batches = X_train[i * macro_batches_size:(i + 1) * macro_batches_size]
			labels_macro_batches = y_train[i * macro_batches_size:(i + 1) * macro_batches_size]
			noise_macro_batches = np.random.rand(macro_batches_size, latent_dim)
			disc_cost_sum = 0

			# (MICRO) BATCHES FOR
			for j in range(disc_iters):  # batches
				# print("micro batches: ", j)
				# DISCRIMINATOR TRAINING
				img_samples = discriminator_macro_batches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
				img_labels = labels_macro_batches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
				noise = noise_macro_batches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]

				discriminator_labels_with_noise = np.concatenate((img_labels, noise), axis=1)
				disc_cost, _ = session.run([discriminator_loss,
				                            discriminator_optimizer],
				                           feed_dict={input_generator: discriminator_labels_with_noise,
				                                      real_samples: img_samples,
				                                      labels: img_labels})
				disc_cost_sum += disc_cost
			# END FOR MICRO BATCHES
			discriminator_history.append(np.mean(disc_cost_sum))
			# GENERATOR TRAINING
			generator_noise = np.random.rand(BATCH_SIZE, latent_dim)
			fake_labels = np.random.randint(low=0, high=9, size=[BATCH_SIZE, ])
			fake_labels_onehot = np.zeros((BATCH_SIZE, 10))
			fake_labels_onehot[np.arange(BATCH_SIZE), fake_labels] = 1
			generator_labels_with_noise = np.concatenate((fake_labels_onehot,
			                                              generator_noise), axis=1)
			gen_cost, _ = session.run([generator_loss,
			                           generator_optimizer],
			                           feed_dict={input_generator: generator_labels_with_noise,
			                                     labels: fake_labels_onehot})
			generator_history.append(gen_cost)
			# END FOR MACRO BATCHES

		test_noise = np.random.rand(10, latent_dim)
		sorted_labels = np.eye(10)
		sorted_labels_with_noise = np.concatenate((sorted_labels, test_noise), axis=1)

		generated_img = session.run([test_samples], feed_dict={test_input: sorted_labels_with_noise})

		generate_images(generated_img, iteration)
		print(" time: ", time.time() - start_time)

		if iteration % 10 == 0 or iteration == (num_epochs - 1):
			# SAVE & PRINT LOSSES
			plt.figure()
			gen_line = plt.plot(generator_history, label="Generator Loss")
			disc_line = plt.plot(discriminator_history, label="Discriminator Loss")
			plt.legend([gen_line, disc_line], ["Generator Loss", "Discriminator Loss"])
			plt.savefig("all_losses.png")

			loss_file = open('gen_loss.txt', 'w')
			for item in generator_history:
				loss_file.write("%s\n" % item)

			loss_file = open('disc_loss.txt', 'w')
			for item in discriminator_history:
				loss_file.write("%s\n" % item)

		# END FOR EPOCHS
	# END SESSION

