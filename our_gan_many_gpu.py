import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from tensorflow import layers
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.get_backend()
matplotlib.rcParams['backend'] = "Qt4Agg"


from tqdm import tqdm
im_tqdm = True

try:
	# memory footprint support libraries/code
	import psutil
	import humanize
	import os
	import GPUtil as GPU
	GPUs = GPU.getGPUs()
	# XXX: only one GPU on Colab and isn’t guaranteed
	gpu = GPUs[0]

	def printm():
		process = psutil.Process(os.getpid())
		print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
		print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
except:
	pass



# import sklearn.datasets

# --------- SETTINGS ---------

# dataset
mnist_data         = False
fashion_mnist_data = True
cifar10_data       = False

# gan architecture
num_epochs              = 50
BATCH_SIZE              = 64
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
disc_iters              = 5   # Number of discriminator updates each generator update. The paper uses 5.
latent_dim              = 128
DIM                     = 64  # number of filters
label_increment         = 0.33

# CONV Parameters
kernel_size = (5, 5)
strides     = 2
size_init   = 4
leakage     = 0.01   # leaky constant

# number of GPUs
N_GPU = 1

# verbose
always_get_loss = True
always_show_fig = True

# --------- VARIANT PARAMETERS ---------

if mnist_data:
	print('mnist dataset')
	from keras.datasets import mnist
	resolution_image = 28
	num_labels = 10
	channels = 1
	channel_first = False


if fashion_mnist_data:
	print('fashion mnist dataset')
	from keras.datasets import fashion_mnist
	resolution_image = 28
	num_labels = 10
	channels = 1
	channel_first = False

if cifar10_data:
	print('cifar10 dataset')
	from keras.datasets import cifar10
	resolution_image = 32
	num_labels = 10
	channels = 3
	channel_first = False

print('resolution image: ', resolution_image)
print('channels:         ', channels)
print('strides:          ', strides)
print('kernel size:      ', kernel_size)

OUTPUT_DIM = int(resolution_image**2)*channels
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPU)]


def generate_images(images, epoch):
	# output gen: (-1,1) --> (-127.5, 127.5) --> (0, 255)
	# shape 10x784
	# plt.figure()
	plt.figure(figsize=(100, 10))
	test_image_stack = np.squeeze((np.array(images, dtype=np.float32) * 127.5) + 127.5)
	for i in range(num_labels):
		if channels > 1:
			new_image = test_image_stack[i].reshape(resolution_image, resolution_image, channels)
		else:
			new_image = test_image_stack[i].reshape(resolution_image, resolution_image)
		plt.subplot(1, num_labels, i + 1)
		plt.axis("off")
		plt.imshow(new_image)
		plt.axis("off")
	plt.axis("off")
	plt.savefig("epoch_" + str(epoch) + ".png")
	if always_show_fig:
		plt.show()

	try:
		printm()
	except:
		pass


def generator(n_samples, noise_with_labels, reuse=None):
	"""
    :param n_samples:         number of samples
    :param noise_with_labels: latent noise + labels
    :return:                  generated images
    """

	# number conv layer number of time we need to duplicate image
	# starting from size_init to image_resolution ----> if image_res 2 -> 1 layer, 4 -> 2layer ...
	# then log2

	# num filters --> arriva a 1 per ultimo layer
	# if 1 layer --> 1, if 2 layers --> 2, 3 layers --> 4
	# then 2^conv_layer-1

	# for cifar 10:
	# image res 32, size init 4 --> 32/4 = 8 --> log2 8 = 3
	# filters = 2^3 = 4

	# for mnist:
	# image res 28, size init 4 --> 28/4 = 7 (wrong, but) --> log2 7 = 2.8 (ceil!)--> 3 ok
	# n_filter = 4

	# (if image size is a power of 2 --> you can n_filter = image_res/n_filter)

	n_conv_layer = int(np.ceil(np.log2(resolution_image/size_init)))
	n_filters = int(2**(n_conv_layer-1))


	print('n-conv layer generator: ', n_conv_layer)
	print('n-filters generator:    ', n_filters)


	with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):  # Needed for later, in order to get variables of discriminator

		# ----- Layer1, Dense, Batch, Leaky ----- #
		print('units dense generator: ', channels * (size_init * size_init) * (n_filters * DIM))

		output = layers.dense(inputs=noise_with_labels,
		                      units=channels * (size_init * size_init) * (n_filters * DIM))


		output = layers.batch_normalization(output)
		output = tf.maximum(leakage * output, output)

		print(output)

		if channel_first:
			# size: 128 x 7 x 7
			print('channel first TRUE')
			output = tf.reshape(output, (-1, n_filters * DIM * channels, size_init, size_init))
			bn_axis = 1  # [0, 2, 3]  # first
		else:
			# size: 7 x 7 x 128
			print('channel first FALSE')
			output = tf.reshape(output, (-1, size_init, size_init, n_filters * DIM * channels))
			bn_axis = -1  # [0, 1, 2]  # last
		print('after reshape:')
		print(output)

		# ----- LoopLayers, deConv, Batch, Leaky ----- #
		for i in range(n_conv_layer):
			print('iter G: ', i, ' - tot filters: ', n_filters * DIM * channels, ' - n_filters: ', n_filters)

			print('before conv2d_transpose:')
			print(output)

			output = layers.conv2d_transpose(output,
			                                 filters=n_filters * DIM * channels,
			                                 kernel_size=kernel_size,
			                                 strides=strides,
			                                 padding='same')
			print('after conv2d: ')
			print(output)

			output = layers.batch_normalization(output, axis=bn_axis)
			output = tf.maximum(leakage * output, output)

			n_filters = int(n_filters/2)

			if resolution_image == 28 and 2*size_init*(1+i) == 8:
				if channel_first:
					print('cut mnist - channel first TRUE')
					output = output[:, :, :7, :7]
				else:
					print('cut mnist - channel first FALSE')
					output = output[:, :7, :7, :]
				print(output)

		# ----- LastLayer, deConv, Batch, Leaky ----- #

		print('n filters layer G out: ', channels)
		output = layers.conv2d_transpose(output,
		                                 filters=1*channels,
		                                 kernel_size=kernel_size,
		                                 strides=1,
		                                 padding='same')
		print('after conv2d: ')
		print(output)

		output = tf.nn.tanh(output)
		output = tf.reshape(output, [-1, OUTPUT_DIM])

		print('Generator after reshape output size:')
		print(output)

	return output


def discriminator(images, reuse=None, n_conv_layer=3):
	"""
    :param images:    images that are input of the discriminator
    :return:          likeliness of the image
    """
	n_conv_layer = int(np.ceil(np.log2(resolution_image / size_init)))
	n_filters = 1

	with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):  # Needed for later, in order to
																	# get variables of generator
		print('Input for discriminator:')
		print(images)

		if channel_first:
			print('channel first: TRUE')
			output = tf.reshape(images, [-1, channels, resolution_image, resolution_image])
		else:
			print('channel first: FALSE')
			output = tf.reshape(images, [-1, resolution_image, resolution_image, channels])

		print('Input for discriminator, after reshape:')
		print(images)

		# ----- LoopLayers, Conv, Leaky ----- #
		for i in range(n_conv_layer):
			print('iter D: ', i, ' - tot filters: ', n_filters*DIM, ' - n_filters: ',n_filters)

			output = layers.conv2d(output,
			                       filters=n_filters*DIM,
			                       kernel_size=kernel_size,
			                       strides=strides,
			                       padding='same')

			print('output after conv2d: ')
			print(output)

			output = tf.maximum(leakage * output, output)
			n_filters = int(n_filters*2)


		output = tf.reshape(output, [-1, size_init * size_init * (int(n_filters/2) * DIM)])
		print('output reshaped for dens layer: ')
		print(output)

		# ----- Layer4, Dense, Linear ----- #
		output = layers.dense(output, units=num_labels+1)
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


# -------------------------------- Load Dataset ---------------------------------- #
if mnist_data:
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
if fashion_mnist_data:
	(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
if cifar10_data:
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = np.reshape(X_train, newshape=[-1, OUTPUT_DIM])
X_test = np.reshape(X_test, newshape=[-1, OUTPUT_DIM])
X_train = np.concatenate((X_train, X_test), axis=0)
X_train = (X_train - 127.5) / 127.5

y_train = np.concatenate((y_train, y_test), axis=0)
y_hot = np.zeros((y_train.shape[0], num_labels))
b = np.arange(y_train.shape[0])
y_hot[b, y_train] = 1
y_train = y_hot

# ------------------------------------------------------------------------------ #

# TENSORFLOW SESSION
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

	label_weights = tf.placeholder(tf.float32, shape=())
	test_input = tf.placeholder(tf.float32, shape=[num_labels, latent_dim + num_labels])

	print('----------------- G: TEST SAMPLES    -----------------')

	test_samples = generator(num_labels, test_input, reuse=True)

	all_input_generator = tf.placeholder(tf.float32, shape=[BATCH_SIZE, latent_dim + num_labels])
	all_real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
	all_real_labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, num_labels])

	binder_real_data = tf.split(all_real_data, len(DEVICES))
	binder_real_labels = tf.split(all_real_labels, len(DEVICES))
	binder_input_generator = tf.split(all_input_generator, len(DEVICES))

	generator_loss_list = []
	discriminator_loss_list = []

	BATCH_SIZE = int(BATCH_SIZE // len(DEVICES))

	for device_index, (device, one_device_real_data, one_device_real_labels, one_device_input_generator) in enumerate(zip(DEVICES, binder_real_data, binder_real_labels, binder_input_generator)):
		# device_index is easy incremental int
		# device = DEVICE[i]
		# real_data_conv = split_real_data_conv[i]

		print(device_index)

		# choose what GPU
		with tf.device(device):

			# --------------------------------- Placeholders ------------------------------- #

			# GENERATOR
			# ----- Noise + Labels(G) ----- #
			# input_generator = tf.placeholder(tf.float32, shape=[BATCH_SIZE, latent_dim + num_labels])
			input_generator = one_device_input_generator

			# DISCRIMINATOR
			# ------ Real Samples(D) ------ #
			# real_samples = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
			real_samples = tf.cast(one_device_real_data, tf.float32)

			# -------- Labels(D) ---------- #
			# labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, num_labels])
			labels = one_device_real_labels

			# ----------------------------------- Outputs ----------------------------------- #
			print('----------------- G: FAKE SAMPLES    -----------------')
			fake_samples = generator(BATCH_SIZE, input_generator, reuse=True)

			print('----------------- D: DISC REAL SCORE -----------------')
			disc_real_score, disc_real_labels = discriminator(real_samples, reuse=True)
			print('----------------- D: DISC REAL SCORE -----------------')
			disc_fake_score, disc_fake_labels = discriminator(fake_samples, reuse=True)

			# ---------------------------------- Losses ------------------------------------ #

			# ----- Gen Loss ----- #

			# wasserstein
			gen_wasserstein_loss = -tf.reduce_mean(disc_fake_score)  # WASSERSTEIN

			# labels
			labels_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
			                                                               logits=disc_fake_labels)
			generator_loss = gen_wasserstein_loss + labels_penalty_fakes * label_weights

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
			fake_labels_weight = 0.1
			discriminator_loss = disc_wasserstein_loss + fake_labels_weight * labels_penalty_fakes * label_weights + labels_penalty_real * label_weights + gradient_penalty

			generator_loss_list.append(generator_loss)
			discriminator_loss_list.append(discriminator_loss)
	# end gpu iter

	# Trainable variables
	d_vars, g_vars = get_trainable_variables()

	# get total average cost of total BATCH
	generator_loss_list = tf.add_n(generator_loss_list) / len(DEVICES)
	discriminator_loss_list = tf.add_n(discriminator_loss_list) / len(DEVICES)
	# ---------------------------------- Optimizers ----------------------------------- #
	generator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
	                                             beta1=0.5,
	                                             beta2=0.9).minimize(generator_loss_list, var_list=g_vars)

	discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,
	                                                 beta1=0.5,
	                                                 beta2=0.9).minimize(discriminator_loss_list, var_list=d_vars)

	# ------------------------------------ Train ---------------------------------------------- #
	# with tf.Session() as session:

	# restore BATCH_SIZE
	BATCH_SIZE = int(BATCH_SIZE*len(DEVICES))

	# run session
	session.run(tf.global_variables_initializer())
	indices = np.arange(X_train.shape[0])

	# big batch size
	macro_batches_size = BATCH_SIZE * disc_iters

	# num of batches
	num_macro_batches = int((X_train.shape[0]) // macro_batches_size)
	discriminator_history = []
	generator_history = []

	labels_incremental_weight = 0
	# EPOCHS FOR
	for epoch in range(num_epochs):

		start_time = time.time()
		print("epoch: ", epoch)
		np.random.shuffle(indices)
		X_train = X_train[indices]
		y_train = y_train[indices]

		# MACRO BATCHES FOR
		for i in tqdm(range(num_macro_batches)):  # macro batches
			# print("macro_batch: ", i)
			discriminator_macro_batches = X_train[i * macro_batches_size:(i + 1) * macro_batches_size]
			labels_macro_batches = y_train[i * macro_batches_size:(i + 1) * macro_batches_size]
			noise_macro_batches = np.random.rand(macro_batches_size, latent_dim)
			disc_cost_sum = 0

			if not im_tqdm and i % (num_macro_batches // 10) == 0:
				print(100*i // num_macro_batches, '%')

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
				                           feed_dict={all_input_generator: discriminator_labels_with_noise,
				                                      all_real_data: img_samples,
				                                      all_real_labels: img_labels,
				                                      label_weights: labels_incremental_weight })
				disc_cost_sum += disc_cost
			# END FOR MICRO BATCHES
			discriminator_history.append(np.mean(disc_cost_sum))

			# GENERATOR TRAINING
			generator_noise = np.random.rand(BATCH_SIZE, latent_dim)
			fake_labels = np.random.randint(low=0, high=num_labels-1, size=[BATCH_SIZE, ])
			fake_labels_onehot = np.zeros((BATCH_SIZE, 10))
			fake_labels_onehot[np.arange(BATCH_SIZE), fake_labels] = 1
			generator_labels_with_noise = np.concatenate((fake_labels_onehot,
			                                              generator_noise), axis=1)
			gen_cost, _ = session.run([generator_loss,
			                           generator_optimizer],
			                          feed_dict={all_input_generator: generator_labels_with_noise,
			                                     all_real_labels: fake_labels_onehot,
			                                     label_weights: labels_incremental_weight})
			generator_history.append(gen_cost)
		# END FOR MACRO BATCHES

		test_noise = np.random.rand(num_labels, latent_dim)
		sorted_labels = np.eye(num_labels)
		sorted_labels_with_noise = np.concatenate((sorted_labels, test_noise), axis=1)

		generated_img = session.run([test_samples],
		                            feed_dict={test_input: sorted_labels_with_noise,
		                                       label_weights: labels_incremental_weight})

		generate_images(generated_img, epoch)
		print(" time: ", time.time() - start_time)

		if (epoch % 10 == 0 or epoch == (num_epochs - 1) or always_get_loss):
			# SAVE & PRINT LOSSES
			plt.figure()
			gen_line = plt.plot(generator_history)  # , label="Generator Loss")
			disc_line = plt.plot(discriminator_history)  # , label="Discriminator Loss")
			# plt.legend([gen_line, disc_line], ["Generator Loss", "Discriminator Loss"])
			plt.savefig("all_losses.png")
			plt.show()

			loss_file = open('gen_loss.txt', 'w')
			for item in generator_history:
				loss_file.write("%s\n" % item)

			loss_file = open('disc_loss.txt', 'w')
			for item in discriminator_history:
				loss_file.write("%s\n" % item)

		# labels weight settings
		labels_incremental_weight += label_increment
		labels_incremental_weight = max(labels_incremental_weight,1)

		# END FOR EPOCHS
# END SESSION
