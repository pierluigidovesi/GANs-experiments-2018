import os, sys
sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from tensorflow import layers
import time
import matplotlib.pyplot as plt

# import tqdm only if previously installed
try:
	from tqdm import tqdm
	im_tqdm = True
except:
	im_tqdm = False

# --------- SETTINGS ---------

# max time allowed
timer = 11000            # seconds

# dataset
mnist_data   = False     # 28 28 (1)
fashion_data = False     # 28 28 (1)
cifar10_data = True      # 32 32  3

# gan architecture
num_epochs = 50          # tot epochs
BATCH_SIZE = 64          # micro batch size
grad_pen_w = 10          # in the paper 10
disc_iters = 5           # Number of discriminator updates each generator update. The paper uses 5.
latent_dim = 128         # input dim
const_filt = 64          # number of filters
label_incr = 0.1         # increment of labels weight (saturate in 1)

# CONV Parameters
kernel_size = (5, 5)     # conv kenel size
strides     = 2          # conv strides
size_init   = 4          # in the paper 4
leakage     = 0.01       # leaky relu constant

# number of GPUs
N_GPU = 1                # need to change if many gpu!

# verbose
sample_repetitions = 2   # to get more rows of images of same epoch in same plot
always_get_loss = True   # get loss each epoch
always_show_fig = False  # real time show test samples each epoch (do not work in backend)

# --------- DEPENDENT PARAMETERS AND PRINTS---------

print('1. DATASET SETTINGS')
if mnist_data:
	print('mnist dataset')
	from keras.datasets import mnist

	resolution_image = 28
	num_labels = 10
	channels = 1
	channel_first = False
	channel_first_disc = False

if fashion_data:
	print('fashion mnist dataset')
	from keras.datasets import fashion_mnist

	resolution_image = 28
	num_labels = 10
	channels = 1
	channel_first = False
	channel_first_disc =False

if cifar10_data:
	print('cifar10 dataset')
	from keras.datasets import cifar10

	resolution_image = 32
	num_labels = 10
	channels = 3
	channel_first = False
	channel_first_disc = False

OUTPUT_DIM = int(resolution_image ** 2) * channels
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPU)]

# ---- PRINT ----

print('res_image:   ', resolution_image)
print('num_labels:  ', num_labels)
print('channels:    ', channels)
print('ch_first:    ', channel_first)
print('ch_first_d:  ', channel_first_disc)

print('2. GAN ARCHITECTURE')
print('num_epochs:  ', num_epochs)
print('BATCH_SIZE:  ', BATCH_SIZE)
print('grad_pen_w:  ', grad_pen_w)
print('disc_iters:  ', disc_iters)
print('latent_dim:  ', latent_dim)
print('const_filt:  ', const_filt)
print('label_incr:  ', label_incr)

print('3. CONV PARAMETERS')
print('kernel_size: ', kernel_size)
print('strides:     ', strides)
print('size_init:   ', size_init)
print('leakage:     ',leakage)

print('USED GPUs:   ', N_GPU)



def generate_images(images, epoch):
	# output gen: (-1,1) --> (-127.5, 127.5) --> (0, 255)
	# shape 10x784

	plt.figure(figsize=(10*num_labels, 10*sample_repetitions))
	test_image_stack = np.squeeze((np.array(images, dtype=np.float32) * 127.5) + 127.5)

	for j in range(sample_repetitions):
		for i in range(num_labels):
			if channels > 1:
				new_image = test_image_stack[i+j*num_labels].reshape(resolution_image, resolution_image, channels)
			else:
				new_image = test_image_stack[i+j*num_labels].reshape(resolution_image, resolution_image)

			plt.subplot(sample_repetitions, num_labels, 1 + i + j*num_labels)
			plt.imshow(new_image)
			plt.axis("off")

	plt.axis("off")
	plt.savefig("epoch_" + str(epoch) + ".png")
	if always_show_fig:
		plt.show()


def generator(n_samples, noise_with_labels, reuse=None):
	"""
    :param n_samples:         number of samples
    :param noise_with_labels: latent noise + labels
    :return:                  generated images
    """

	# CONV "THEORY"
	# number conv layer number of time we need to duplicate image
	# starting from size_init to image_resolution --> if image_res 2 -> 1 layer, 4 -> 2layer ...
	# then log2

	# num filters --> arriva a 1 per ultimo layer
	# if 1 layer --> 1, if 2 layers --> 2, 3 layers --> 4
	# then 2^conv_layer-1

	# for cifar 10:
	# image res 32, size init 4 --> 32/4 = 8 --> log2 8 = 3
	# filters = 2^3 = 4

	# for mnist:
	# image res 28, size init 4 --> 28/4 = 7 (wrong, but) --> log2 7 = 2.8 (ceil!)--> 3 ok
	# n_filter = 4 (POSSIBLE REDUCTION IS SPECIFIED)

	# (if image size is a power of 2 --> you can: n_filter = image_res/n_filter)

	# get number of layers and filters

	n_conv_layer = int(np.ceil(np.log2(resolution_image / size_init)))
	n_filters = int(2 ** (n_conv_layer - 1))

	print(' G: n-conv layer generator: ', n_conv_layer)
	print(' G: n-filters generator:    ', n_filters)

	with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):  # Needed for later, in order to
																# get variables of discriminator
		# ----- Layer1, Dense, Batch, Leaky ----- #
		print(' G: units dense generator: ', channels * (size_init * size_init) * (n_filters * const_filt))

		output = layers.dense(inputs=noise_with_labels,
		                      units=channels * (size_init * size_init) * (n_filters * const_filt))

		output = layers.batch_normalization(output)
		output = tf.maximum(leakage * output, output)

		print(' G: dense layer')
		print(output)

		if channel_first:
			# size: 128 x 7 x 7
			output = tf.reshape(output, (-1, n_filters * const_filt * channels, size_init, size_init))
			bn_axis = 1  # [0, 2, 3]  # first
		else:
			# size: 7 x 7 x 128
			output = tf.reshape(output, (-1, size_init, size_init, n_filters * const_filt * channels))
			bn_axis = -1  # [0, 1, 2] # last
		print(' G: channel reshape:')
		print(output)

		# ----- LoopLayers, deConv, Batch, Leaky ----- #

		for i in range(n_conv_layer):

			if resolution_image == 28 and size_init * (1 + i) == 8:

				if channel_first:
					output = output[:, :, :7, :7]
				else:
					output = output[:, :7, :7, :]
				print(' G: cut mnist, iteration: ', i)
				print(output)

			print(' G: conv2d_transpose iter', i, ' - tot filters: ',
			      n_filters * const_filt * channels, ' - n_filters: ', n_filters)

			output = layers.conv2d_transpose(output,
			                                 filters=n_filters * const_filt * channels,
			                                 kernel_size=kernel_size,
			                                 strides=strides,
			                                 padding='same')
			print(output)

			output = layers.batch_normalization(output, axis=bn_axis)
			output = tf.maximum(leakage * output, output)

			n_filters = int(n_filters / 2)

		# ----- LastLayer, deConv, Batch, Leaky ----- #

		print(' G: last conv2d_transpose layer - n filters layer: ', channels)
		output = layers.conv2d_transpose(output,
		                                 filters=1 * channels,
		                                 kernel_size=kernel_size,
		                                 strides=1,
		                                 padding='same')
		print(output)

		output = tf.nn.tanh(output)
		output = tf.reshape(output, [-1, OUTPUT_DIM])

		print(' G: output reshape')
		print(output)

	return output


def discriminator(images, reuse=None, n_conv_layer=3):
	"""
    :param images:    images that are input of the discriminator
    :return:          likeliness of the image
    """

	if channel_first_disc == True:
		channels_key = 'channels_first'
	else:
		channels_key = 'channels_last'

	n_conv_layer = int(np.ceil(np.log2(resolution_image / size_init)))
	n_filters = 1

	print(' D: n-conv layer discriminator: ', n_conv_layer)
	print(' D: n-filters discriminator:    ', n_filters)

	with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):  # Needed for later, in order to
																	# get variables of generator
		print(' D: input')
		print(images)

		if channel_first_disc:
			print('channel first disc ON')
			output = tf.reshape(images, [-1, channels, resolution_image, resolution_image])
		else:
			output = tf.reshape(images, [-1, resolution_image, resolution_image, channels])

		print(' D: channel reshape')
		print(output)

		# ----- LoopLayers, Conv, Leaky ----- #

		for i in range(n_conv_layer):
			print(' D: conv2d iter: ', i, ' - n_filters: ', n_filters)

			output = layers.conv2d(output,
			                       filters=n_filters * const_filt,
			                       kernel_size=kernel_size,
			                       strides=strides,
			                       padding='same',
			                       data_format=channels_key)

			print(output)

			output = tf.maximum(leakage * output, output)
			n_filters = int(n_filters * 2)

		output = tf.reshape(output, [-1, size_init * size_init * (int(n_filters / 2) * const_filt)])
		print(' D: reshape linear layer')
		print(output)

		# ----- Layer4, Dense, Linear ----- #
		output = layers.dense(output, units=num_labels + 1)
		print(' D: dense layer output')
		print(output)

	scores_out = tf.identity(output[:, :1], name='scores_out')
	labels_out = tf.identity(output[:, 1:], name='labels_out')
	return scores_out, labels_out


def get_trainable_variables(): # used in optimizer/minimize (training)
	"""
    :return: trainable variables (d_vars, g_vars)
    """
	tvars = tf.trainable_variables()
	d_vars = [var for var in tvars if 'Discriminator' in var.name]
	g_vars = [var for var in tvars if 'Generator' in var.name]
	return d_vars, g_vars


# -------------------------------- Load Dataset ---------------------------------- #

# get data
if mnist_data:
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
if fashion_data:
	(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
if cifar10_data:
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("DATASET DIMENSIONS:")
print(X_train.shape)

# reshape and merge train and test data
X_train = np.reshape(X_train, newshape=[-1, OUTPUT_DIM])
X_test = np.reshape(X_test, newshape=[-1, OUTPUT_DIM])
X_train = np.concatenate((X_train, X_test), axis=0)
X_train = (X_train - 127.5) / 127.5

# merge and one hot train and test labels
y_train = np.concatenate((y_train, y_test), axis=0)
y_hot = np.zeros((y_train.shape[0], num_labels))
b = np.arange(y_train.shape[0])
y_hot[b, y_train] = 1
y_train = y_hot

# ------------------------------------------------------------------------------ #

# TENSORFLOW SESSION
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

	# TEST SAMPLE GENERATION SESSION
	print('----------------- G: TEST SAMPLES    -----------------')
	test_input = tf.placeholder(tf.float32, shape=[sample_repetitions * num_labels, latent_dim + num_labels])
	test_samples = generator(num_labels, test_input, reuse=True)

	# TRAINING SESSION
	label_weights = tf.placeholder(tf.float32, shape=())

	all_input_generator = tf.placeholder(tf.float32, shape=[BATCH_SIZE, latent_dim + num_labels])
	all_real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
	all_real_labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, num_labels])

	# split over GPUs
	binder_real_data = tf.split(all_real_data, len(DEVICES))
	binder_real_labels = tf.split(all_real_labels, len(DEVICES))
	binder_input_generator = tf.split(all_input_generator, len(DEVICES))

	# list used for mean over GPUs
	generator_loss_list = []
	discriminator_loss_list = []

	# split BATCH_SIZE
	BATCH_SIZE = int(BATCH_SIZE // len(DEVICES))

	# for device_index, (device, one_device_real_data, one_device_real_labels, one_device_input_generator)
	# in enumerate(zip(DEVICES, binder_real_data, binder_real_labels, binder_input_generator)):

	# for each GPU, select relative sub-batch of data
	for device_index, (device, real_samples, labels, input_generator) in enumerate(
			zip(DEVICES, binder_real_data, binder_real_labels, binder_input_generator)):
		# device_index is easy incremental int
		# device = DEVICE[i]
		# real_data_conv = split_real_data_conv[i]

		print('GPU device_index: ', device_index)

		# choose what GPU
		with tf.device(device):

			# ----------------------------------- Outputs ----------------------------------- #
			print('----------------- G: FAKE SAMPLES    -----------------')
			fake_samples = generator(BATCH_SIZE, input_generator, reuse=True)

			print('----------------- D: DISC REAL SCORE -----------------')
			disc_real_score, disc_real_labels = discriminator(real_samples, reuse=True)

			print('----------------- D: DISC FAKE SCORE -----------------')
			disc_fake_score, disc_fake_labels = discriminator(fake_samples, reuse=True)

			# ---------------------------------- Losses ------------------------------------ #

			# ----- Gen Loss ----- #

			# wasserstein
			gen_wasserstein_loss = -tf.reduce_mean(disc_fake_score)  # WASSERSTEIN

			# labels
			labels_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
			                                                               logits=disc_fake_labels)
			gen_labels_loss = labels_penalty_fakes * label_weights

			# total gen loss
			generator_loss = gen_wasserstein_loss + gen_labels_loss

			# ----- Disc Loss ----- #

			# wasserstein
			disc_wasserstein_loss = tf.reduce_mean(disc_fake_score) - tf.reduce_mean(disc_real_score)

			# labels
			labels_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
			                                                               logits=disc_fake_labels)
			labels_penalty_real = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
			                                                              logits=disc_real_labels)
			fake_labels_weight = 0  # this should be a placeholder

			# tot labels loss
			disc_labels_loss = (fake_labels_weight * labels_penalty_fakes + labels_penalty_real) * label_weights

			# gradient penalty
			alpha = tf.random_uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
			differences = fake_samples - real_samples
			interpolates = real_samples + alpha * differences
			gradients = tf.gradients(discriminator(interpolates, reuse=True)[0], [interpolates])[0]
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
			gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)*grad_pen_w

			# sum losses
			discriminator_loss = disc_wasserstein_loss + disc_labels_loss + gradient_penalty

			generator_loss_list.append(generator_loss)
			discriminator_loss_list.append(discriminator_loss)
	# end gpu iter

	# Trainable variables
	d_vars, g_vars = get_trainable_variables()

	# get total average cost of total BATCH
	generator_loss_list = tf.add_n(generator_loss_list) / len(DEVICES)
	discriminator_loss_list = tf.add_n(discriminator_loss_list) / len(DEVICES)
	# ---------------------------------- Optimizers ----------------------------------- #
	generator_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4,
	                                             beta1=0.5,
	                                             beta2=0.9).minimize(generator_loss_list, var_list=g_vars)

	discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=2e-4,
	                                                 beta1=0.5,
	                                                 beta2=0.9).minimize(discriminator_loss_list, var_list=d_vars)

	# ------------------------------------ Train ---------------------------------------------- #
	print(' - - - - - - - - - - TRAIN - - - - - - - - - - ')
	# with tf.Session() as session:

	# restore BATCH_SIZE
	BATCH_SIZE = int(BATCH_SIZE * len(DEVICES))

	# run session
	session.run(tf.global_variables_initializer())

	# set dataset index
	indices = np.arange(X_train.shape[0])

	# big batch size
	macro_batches_size = BATCH_SIZE * disc_iters

	# num of batches
	num_macro_batches = int((X_train.shape[0]) // macro_batches_size)

	# init losses history
	discriminator_history = []
	generator_history = []

	# init label weight
	labels_incremental_weight = 0

	# EPOCHS FOR
	init_time = time.time()
	for epoch in range(num_epochs):

		start_time = time.time()
		print(" ----------> epoch: ", epoch)

		#shuffle dataset
		#np.random.shuffle(indices)
		#X_train = X_train[indices]
		#y_train = y_train[indices]

		# MACRO BATCHES FOR
		for i in tqdm(range(num_macro_batches)):  # macro batches

			# divide dataset
			discriminator_macro_batches = X_train[i * macro_batches_size:(i + 1) * macro_batches_size]
			# get labels
			labels_macro_batches = y_train[i * macro_batches_size:(i + 1) * macro_batches_size]
			# generate noise
			noise_macro_batches = np.random.rand(macro_batches_size, latent_dim)

			# init disc cost vector (to be used in disc_iters)
			d_cost_vector = []

			# verbose when tqdm OFF
			if not im_tqdm and i % (num_macro_batches // 10) == 0:
				print(100 * i // num_macro_batches, '%')

			# (MICRO) BATCHES FOR
			for j in range(disc_iters):  # batches

				# DISCRIMINATOR TRAINING

				# divide dataset in batches
				img_samples = discriminator_macro_batches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
				# get labels
				img_labels = labels_macro_batches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
				# get noise
				noise = noise_macro_batches[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
				# create latent space
				discriminator_labels_with_noise = np.concatenate((img_labels, noise), axis=1)

				# train disc
				disc_cost, dw_cost, d_gradpen, d_lab_cost, _ = session.run([discriminator_loss,
				                                                            disc_wasserstein_loss,
				                                                            gradient_penalty,
				                                                            disc_labels_loss,
				                                                            discriminator_optimizer],
				                                                           feed_dict={all_input_generator: discriminator_labels_with_noise,
				                                                                      all_real_data: img_samples,
				                                                                      all_real_labels: img_labels,
				                                                                      label_weights: labels_incremental_weight})

				# append losses means (each loss has BATCH_SIZE element)
				d_cost_vector.append([np.mean(disc_cost), np.mean(dw_cost), np.mean(d_gradpen), np.mean(d_lab_cost)])

			# END FOR MICRO BATCHES
			# append disc loss over disc_iters
			discriminator_history.append(np.mean(d_cost_vector, 0))

			# GENERATOR TRAINING

			# generate noise
			generator_noise = np.random.rand(BATCH_SIZE, latent_dim)
			# generate random labels and make them one hot
			fake_labels = np.random.randint(low=0, high=num_labels - 1, size=[BATCH_SIZE, ])
			fake_labels_onehot = np.zeros((BATCH_SIZE, 10))
			fake_labels_onehot[np.arange(BATCH_SIZE), fake_labels] = 1
			# concatenate to create latent space
			generator_labels_with_noise = np.concatenate((fake_labels_onehot,
			                                              generator_noise), axis=1)

			# train gen
			gen_cost, gw_cost, g_lab_cost, _ = session.run([generator_loss,
			                                                gen_wasserstein_loss,
			                                                gen_labels_loss,
			                                                generator_optimizer],
			                                               feed_dict={all_input_generator: generator_labels_with_noise,
			                                                          all_real_labels: fake_labels_onehot,
			                                                          label_weights: labels_incremental_weight})

			# append directly in gen loss history (with mean beacuse of BATCH_SIZE)
			generator_history.append([np.mean(gen_cost), np.mean(gw_cost), np.mean(g_lab_cost)])
		# END FOR MACRO BATCHES

		# generate test latent space (with sample_repetitions to create more rows of samples)
		test_noise = np.random.rand(num_labels * sample_repetitions, latent_dim)
		sorted_labels = np.tile(np.eye(num_labels), sample_repetitions).transpose()
		sorted_labels_with_noise = np.concatenate((sorted_labels, test_noise), axis=1)

		# recall generator
		generated_img = session.run([test_samples],
		                            feed_dict={test_input: sorted_labels_with_noise})

		# print test img
		generate_images(generated_img, epoch)

		if epoch % 10 == 0 or epoch == (num_epochs - 1) or always_get_loss:
			# SAVE & PRINT LOSSES

			# generator vs discriminator loss
			plt.figure()
			disc_line = plt.plot(np.asarray([item[0] for item in discriminator_history]), label='DISC')
			gen_line  = plt.plot(np.asarray([item[0] for item in generator_history]), label='GEN')
			plt.legend()
			plt.savefig("GD_losses.png")

			# discriminator losses
			plt.figure()
			disc_sum  = plt.plot(np.array([item[0] for item in discriminator_history]), label='ALL')
			disc_w    = plt.plot(np.array([item[1] for item in discriminator_history]), label='WASS')
			disc_grad = plt.plot(np.array([item[2] for item in discriminator_history]), label='GRAD')
			disc_lab  = plt.plot(np.array([item[3] for item in discriminator_history]), label='LAB')
			plt.legend()
			plt.savefig("D_losses.png")

			# generator losses
			plt.figure()
			gen_sum = plt.plot(np.array([item[0] for item in generator_history]), label='ALL')
			gen_w   = plt.plot(np.array([item[1] for item in generator_history]), label='WASS')
			gen_lab = plt.plot(np.array([item[2] for item in generator_history]), label='LAB')
			plt.legend()
			plt.savefig("G_losses.png")

			if always_show_fig:
				plt.show()  # it works only in interactive mode

			# save txt logs
			loss_file = open('gen_losses.txt', 'w')
			for item in generator_history:
				loss_file.write("%s\n" % item)

			loss_file = open('disc_losses.txt', 'w')
			for item in discriminator_history:
				loss_file.write("%s\n" % item)

		# increment/saturate label weight
		labels_incremental_weight += label_incr  # now 0
		labels_incremental_weight = min(labels_incremental_weight, 1)

		total_time = time.time() - init_time
		print(' cycle time:  ', time.time() - start_time, " - total time: ", total_time)
		print(' gen cost   = ', np.mean([item[0] for item in generator_history[-num_macro_batches:]]))
		print(' disc cost  = ', np.mean([item[0] for item in discriminator_history[-num_macro_batches:]]))

		if total_time >= timer:
			epoch = num_epochs
			print('time out!')

	# END FOR EPOCHS
# END SESSION
