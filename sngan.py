import inception_score
import os, sys
import pickle

sys.path.append(os.getcwd())
import numpy as np
import tensorflow as tf
from tensorflow import layers
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import tqdm only if previously installed
try:
    from tqdm import tqdm
    im_tqdm = True
except:
    im_tqdm = False

# --------- SETTINGS ---------

# max time allowed
timer = 3600*30       # seconds

# random seed
seed = 200
np.random.seed(seed)

# Dataset
mnist_data   = False  # 28 28 (1)
fashion_data = False  # 28 28 (1)
cifar10_data = True   # 32 32 (3)

# GAN architecture
sngan      = True     # spectral normalization
num_epochs = 300      # tot epochs
batch_size = 64       # micro batch size
disc_iters = 10       # Number of discriminator updates each generator update. The paper uses 5.
latent_dim = 128      # input dim (paper 128, but suggested 64)
is_n_batch = 80       # number of batches for EACH class for Inception Score evaluation

# Losses parameters
wasserst_w = 1        # wasserstain weight (always 1)
grad_pen_w = 10       # in the paper 10
learn_rate = 2e-4     # in the paper 1/2e-4
beta1_opti = 0.5      # in the paper 0.5
beta2_opti = 0.9      # in the paper 0.9
label_incr = 1        # increment of labels weight (saturate in 1)
label_satu = 1        # max label weight

# CONV Parameters
const_filt  = 96      # number of filters (paper 64) [96 maybe better]
kernel_size = (5, 5)  # conv kenel size
strides     = 2       # conv strides
size_init   = 4       # in the paper 4
leakage     = 0.01    # leaky relu constant

# number of GPUs
N_GPU = 2             # need to change if many gpu!

# verbose
is_freq = 10
fixed_noise = True       # always use same noise for image samples
sample_repetitions = 10  # to get more rows of images of same epoch in same plot (always put highest value)
always_get_loss = True   # get loss each epoch
always_show_fig = False  # real time show test samples each epoch (do not work in backend)
check_in_out = False     # print disc images and values
version = "gan"



# --------- DEPENDENT PARAMETERS AND PRINTS---------

if mnist_data:
    from keras.datasets import mnist

    resolution_image = 28
    num_labels = 10
    channels = 1
    channel_first_gen = False
    channel_first_disc = False

if fashion_data:
    from keras.datasets import fashion_mnist

    resolution_image = 28
    num_labels = 10
    channels = 1
    channel_first_gen = False
    channel_first_disc = False

if cifar10_data:
    from keras.datasets import cifar10

    resolution_image = 32
    num_labels = 10
    channels = 3
    channel_first_gen = False
    channel_first_disc = False

OUTPUT_DIM = int(resolution_image ** 2) * channels
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPU)]


# ---- PRINT ----
def print_log():
    print('1. DATASET SETTINGS')
    if mnist_data:
        print('mnist dataset')
    if fashion_data:
        print('fashion mnist dataset')
    if cifar10_data:
        print('cifar10 dataset')
    print('res_image:   ', resolution_image)
    print('num_labels:  ', num_labels)
    print('channels:    ', channels)
    print('ch_first:    ', channel_first_gen)
    print('ch_first_d:  ', channel_first_disc)

    print('2. GAN ARCHITECTURE')
    print('sngan:       ', sngan)
    print('num_epochs:  ', num_epochs)
    print('batch_size:  ', batch_size)
    print('disc_iters:  ', disc_iters)
    print('latent_dim:  ', latent_dim)
    print('is_n_batch:  ', is_n_batch)

    print('3. LOSSES PARAMETERS')
    print('wasserst_w:  ', wasserst_w)
    print('grad_pen_w:  ', grad_pen_w)
    print('learn rate:  ', learn_rate)
    print('beta1_opti:  ', beta1_opti)
    print('beta2_opti:  ', beta2_opti)
    print('label_incr:  ', label_incr)
    print('label_satu:  ', label_satu)

    print('4. CONV PARAMETERS')
    print('const_filt:  ', const_filt)
    print('kernel_size: ', kernel_size)
    print('strides:     ', strides)
    print('size_init:   ', size_init)
    print('leakage:     ', leakage)

    print('5. MISCELLANEOUS')
    print('used GPUs:   ', DEVICES)
    print('random seed: ', seed)
    print('sample rep:  ', sample_repetitions)
    print()


# print settings
print_log()

# save txt logs
orig_stdout = sys.stdout
f = open('settings_log.txt', 'w')
sys.stdout = f
print_log()
sys.stdout = orig_stdout
f.close()


def generate_images(images, epoch, repetitions=1):
    # output gen: (-1,1) --> (-127.5, 127.5) --> (0, 255)
    # shape 10x784

    names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    plt.figure(figsize=(10 * num_labels, 10 * repetitions))
    plt.suptitle("Epoch " + str(epoch))
    test_image_stack = np.squeeze((np.array(images, dtype=np.float32) * 0.5) + 0.5)

    for j in range(repetitions):
        for i in range(num_labels):
            if channels > 1:
                new_image = test_image_stack[i + j * num_labels].reshape(resolution_image, resolution_image, channels)
            else:
                new_image = test_image_stack[i + j * num_labels].reshape(resolution_image, resolution_image)

            plt.subplot(repetitions, num_labels, 1 + i + j * num_labels)

            if j == 0 and label_incr > 0:
                plt.title(names[i], fontsize=100)

            plt.imshow(new_image)
            plt.axis("off")

    plt.axis("off")
    plt.savefig("sample_epoch_" + str(epoch) + ".png")
    if always_show_fig:
        plt.show()
    plt.close('all')


def generator(n_samples, noise_with_labels, reuse=None):

    # (if image size is a power of 2 --> you can: n_filter = image_res/n_filter)
    # get number of layers and filters

    n_conv_layer = int(np.ceil(np.log2(resolution_image / size_init)))
    n_filters = int(2 ** (n_conv_layer - 1))

    print(' G: n-conv layer generator: ', n_conv_layer)
    print(' G: n-filters generator:    ', n_filters)

    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):  # Needed for later, in order to

        # ----- Layer1, Dense, Batch, Leaky ----- #
        print(' G: units dense generator: ', channels * (size_init * size_init) * (n_filters * const_filt))

        output = layers.dense(inputs=noise_with_labels,
                              units=channels * (size_init * size_init) * (n_filters * const_filt))

        output = layers.batch_normalization(output)
        output = tf.maximum(leakage * output, output)

        print(' G: dense layer')
        print(output)

        if channel_first_gen:
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

                if channel_first_gen:
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
            output = tf.maximum(leakage * output, output)  # relu

            n_filters = int(n_filters / 2)

        # ----- LastLayer, deConv, Batch, Leaky ----- #

        print(' G: last conv2d_transpose layer - n filters layer: ', channels)
        output = layers.conv2d_transpose(output,
                                         filters=1 * channels,
                                         kernel_size=kernel_size,  # 1 or kernel_size???? to check
                                         strides=1,
                                         padding='same')
        print(output)

        output = tf.nn.tanh(output)
        output = tf.reshape(output, [-1, OUTPUT_DIM])

        print(' G: output reshape')
        print(output)

    return output


def spectral_norm(w, i, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u" + str(i), [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
       power iteration
       Usually iteration = 1 will be enough
       """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def discriminator(images, reuse=None, n_conv_layer=3):

    if channel_first_disc:
        channels_key = 'channels_first'
        channels_code = "NCHW"
    else:
        channels_key = 'channels_last'
        channels_code = "NHWC"

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

            # GETTING THE WEIGHTS THAT WILL BE NORMALIZED.
            print(output)
            print("Kernel print:" + "(" + str(i) + ")")
            w = tf.get_variable("kernel" + str(i),
                                shape=[kernel_size[0], kernel_size[1], output.get_shape()[-1], n_filters * const_filt])
            print(w)
            print("constant print:" + "(" + str(i) + ")")
            b = tf.get_variable("constant" + str(i), [n_filters * const_filt], initializer=tf.constant_initializer(0.0))
            print(b)

            if sngan:
                pesi = spectral_norm(w, i)
            else:
                pesi = w

            output = tf.nn.conv2d(input=output,
                                  filter=pesi,
                                  strides=[1, strides, strides, 1],
                                  padding='SAME',
                                  data_format=channels_code) + b

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

    print(' D: scores output')
    print(scores_out)
    print(' D: labels output')
    print(labels_out)

    return scores_out, labels_out


def get_trainable_variables():  # used in optimizer/minimize (training)
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
X_train_original = (X_train - 127.5) / 127.5

X_train = np.reshape(X_train, newshape=[-1, OUTPUT_DIM])
X_test = np.reshape(X_test, newshape=[-1, OUTPUT_DIM])
X_train = np.concatenate((X_train, X_test), axis=0)
X_train = (X_train - 127.5) / 127.5

# merge and one hot train and test labels
if mnist_data or fashion_data:
    y_train = np.concatenate((y_train, y_test), axis=0)

if cifar10_data:
    y_train = np.concatenate((y_train[:, 0], y_test[:, 0]), axis=0)

y_hot = np.zeros((y_train.shape[0], num_labels))
b = np.arange(y_train.shape[0])
y_hot[b, y_train] = 1
y_train = y_hot

# ========================== TENSORFLOW SESSION =================================== #

# TF Session
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # TEST SAMPLE GENERATION SESSION
    print('----------------- G: TEST SAMPLES    -----------------')
    test_input = tf.placeholder(tf.float32, shape=[sample_repetitions * num_labels, latent_dim + num_labels])
    test_samples = generator(num_labels, test_input, reuse=True)


    # Inception Score SAMPLES
    print('----------------- G: Inception Score SAMPLES    -----------------')
    is_input = tf.placeholder(tf.float32, shape=[batch_size * num_labels, latent_dim + num_labels])
    is_samples = generator(num_labels, is_input, reuse=True)

    # TRAINING SESSION
    label_weights = tf.placeholder(tf.float32, shape=())

    all_input_generator = tf.placeholder(tf.float32, shape=[batch_size, latent_dim + num_labels])
    all_real_data = tf.placeholder(tf.float32, shape=[batch_size, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.float32, shape=[batch_size, num_labels])

    # split over GPUs
    binder_real_data = tf.split(all_real_data, len(DEVICES))
    binder_real_labels = tf.split(all_real_labels, len(DEVICES))
    binder_input_generator = tf.split(all_input_generator, len(DEVICES))

    # list used for mean over GPUs
    generator_loss_list = []
    discriminator_loss_list = []

    gradient_penalty_list = []
    disc_wasserstein_loss_list = []
    disc_labels_loss_list = []

    gen_wasserstein_loss_list = []
    gen_labels_loss_list = []

    real_accuracy_list = []
    fake_accuracy_list = []

    # split batch_size
    batch_size = int(batch_size // len(DEVICES))

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
            fake_samples = generator(batch_size, input_generator, reuse=True)

            print('----------------- D: DISC REAL SCORE -----------------')
            disc_real_score, disc_real_labels = discriminator(real_samples, reuse=True)

            print('----------------- D: DISC FAKE SCORE -----------------')
            disc_fake_score, disc_fake_labels = discriminator(fake_samples, reuse=True)

            # ---------------------------------- Losses ------------------------------------ #

            # ----- Gen Loss ----- #

            # wasserstein
            gen_wasserstein_loss = -tf.reduce_mean(disc_fake_score) * wasserst_w  # WASSERSTEIN

            # labels
            labels_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
                                                                           logits=disc_fake_labels)
            gen_labels_loss = labels_penalty_fakes * label_weights

            # total gen loss
            generator_loss = gen_wasserstein_loss + gen_labels_loss

            # ----- Disc Loss ----- #

            # wasserstein
            disc_wasserstein_loss = (tf.reduce_mean(disc_fake_score) - tf.reduce_mean(disc_real_score)) * wasserst_w

            # labels
            labels_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
                                                                           logits=disc_fake_labels)
            labels_penalty_real = tf.nn.softmax_cross_entropy_with_logits(labels=labels,  # (deprecated)
                                                                          logits=disc_real_labels)
            fake_labels_weight = 0  # this should be a placeholder

            # tot labels loss
            disc_labels_loss = (fake_labels_weight * labels_penalty_fakes + labels_penalty_real) * label_weights

            # gradient penalty
            alpha = tf.random_uniform(shape=[batch_size, 1], minval=0., maxval=1.)
            differences = fake_samples - real_samples
            interpolates = real_samples + alpha * differences
            gradients = tf.gradients(discriminator(interpolates, reuse=True)[0], [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2) * grad_pen_w

            # sum losses
            discriminator_loss = disc_wasserstein_loss + disc_labels_loss + gradient_penalty

            # append all losses
            generator_loss_list.append(generator_loss)
            discriminator_loss_list.append(discriminator_loss)
            # single losses:
            #  - disc
            gradient_penalty_list.append(gradient_penalty)
            disc_wasserstein_loss_list.append(disc_wasserstein_loss)
            disc_labels_loss_list.append(disc_labels_loss)
            # - gen
            gen_wasserstein_loss_list.append(gen_wasserstein_loss)
            gen_labels_loss_list.append(gen_labels_loss)

            # ---------- ACCURACY --------

            # disc accuracy on REAL img
            real_correct_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(disc_real_labels, 1))
            real_accuracy = tf.reduce_mean(tf.cast(real_correct_pred, tf.float32))
            real_accuracy_list.append(real_accuracy)

            # disc accuracy of FAKE img ---> i.e. gen accuracy
            fake_correct_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(disc_fake_labels, 1))
            fake_accuracy = tf.reduce_mean(tf.cast(fake_correct_pred, tf.float32))
            fake_accuracy_list.append(fake_accuracy)

    # end gpu iter

    # get total average cost of total BATCH (over the gpus)
    generator_loss_mean = tf.add_n(generator_loss_list) / len(DEVICES)
    discriminator_loss_mean = tf.add_n(discriminator_loss_list) / len(DEVICES)

    # single average losses:
    #  - disc
    gradient_penalty_mean = tf.add_n(gradient_penalty_list) / len(DEVICES)
    disc_wasserstein_loss_mean = tf.add_n(disc_wasserstein_loss_list) / len(DEVICES)
    disc_labels_loss_mean = tf.add_n(disc_labels_loss_list) / len(DEVICES)
    # - gen
    gen_wasserstein_loss_mean = tf.add_n(gen_wasserstein_loss_list) / len(DEVICES)
    gen_labels_loss_mean = tf.add_n(gen_labels_loss_list) / len(DEVICES)

    # get total average accuracy
    real_accuracy_mean = tf.add_n(real_accuracy_list) / len(DEVICES)
    fake_accuracy_mean = tf.add_n(fake_accuracy_list) / len(DEVICES)

    # ---------------------------------- Optimizers ----------------------------------- #
    # Trainable variables
    d_vars, g_vars = get_trainable_variables()

    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                                 beta1=beta1_opti,
                                                 beta2=beta2_opti).minimize(generator_loss_mean, var_list=g_vars)

    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                                     beta1=beta1_opti,
                                                     beta2=beta2_opti).minimize(discriminator_loss_mean,
                                                                                var_list=d_vars)

    # ------------------------------------ Train ---------------------------------------------- #
    print(' - - - - - - - - - - TRAIN - - - - - - - - - - ')
    # with tf.Session() as session:


    """
    # TF Saver
    saver = tf.train.Saver()
    # continue training
    # save_path = saver.save(session, "/tmp/model.ckpt")
    # ckpt = tf.train.latest_checkpoint('./model')

    try:
        saver.restore(session)
        print('saver: variables restored!')
    except:
        print('saver: nothing to restore.')
    """

    # restore batch_size
    batch_size = int(batch_size * len(DEVICES))

    #  - - - - RUN - - - - - -
    session.run(tf.global_variables_initializer())

    # set dataset index
    indices = np.arange(X_train.shape[0])

    # big batch size
    macro_batches_size = batch_size * disc_iters

    # num of batches
    num_macro_batches = int((X_train.shape[0]) // macro_batches_size)

    # init losses history
    discriminator_history = []
    generator_history = []
    is_history = []

    # init label weight
    labels_incremental_weight = 0

    # increment/saturate label weight
    labels_incremental_weight += label_incr
    labels_incremental_weight = min(labels_incremental_weight, label_satu)

    # EPOCHS FOR
    init_time = time.time()
    for epoch in range(num_epochs):

        start_time = time.time()
        print()
        print(" ----------> epoch: ", epoch, '- iterations: ', num_macro_batches * epoch)

        # shuffle dataset
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        # MACRO BATCHES FOR
        for i in tqdm(range(num_macro_batches)):  # macro batches

            # divide dataset
            discriminator_macro_batches = X_train[i * macro_batches_size:(i + 1) * macro_batches_size]
            # get labels
            labels_macro_batches = y_train[i * macro_batches_size:(i + 1) * macro_batches_size]
            # generate noise
            noise_macro_batches = np.random.randn(macro_batches_size, latent_dim)

            # init disc cost vector (to be used in disc_iters)
            d_cost_vector = []

            # verbose when tqdm OFF
            if not im_tqdm and i % (num_macro_batches // 10) == 0:
                print(100 * i // num_macro_batches, '%')

            # (MICRO) BATCHES FOR
            for j in range(disc_iters):  # batches

                # DISCRIMINATOR TRAINING

                # divide dataset in batches
                img_samples = discriminator_macro_batches[j * batch_size:(j + 1) * batch_size]
                # get labels
                img_labels = labels_macro_batches[j * batch_size:(j + 1) * batch_size]
                # get noise
                noise = noise_macro_batches[j * batch_size:(j + 1) * batch_size]
                # create latent space
                discriminator_labels_with_noise = np.concatenate((img_labels, noise), axis=1)

                #print(np.sum(discriminator_labels_with_noise[:,:10], axis=0)/np.sum(discriminator_labels_with_noise[:,:10]))

                # train disc
                # disc_cost, dw_cost, d_gradpen, d_lab_cost, disc_accuracy, gen_accuracy, _
                # disc_run_out = []
                disc_run_out = session.run([discriminator_loss_mean,
                                            disc_wasserstein_loss_mean,
                                            gradient_penalty_mean,
                                            disc_labels_loss_mean,
                                            real_accuracy_mean,
                                            fake_accuracy_mean,
                                            discriminator_optimizer],
                                           feed_dict={all_input_generator: discriminator_labels_with_noise,
                                                      all_real_data: img_samples,
                                                      all_real_labels: img_labels,
                                                      label_weights: labels_incremental_weight})

                # append losses means (each loss has batch_size element)
                # d_cost_vector.append([np.mean(disc_cost), np.mean(dw_cost), np.mean(d_gradpen), np.mean(d_lab_cost)])

                disc_run_out = disc_run_out[:-1]
                d_cost_vector.append([np.mean(elem) for elem in disc_run_out])

            # END FOR MICRO BATCHES
            # append disc loss over disc_iters

            discriminator_history.append(np.mean(d_cost_vector, 0))

            # GENERATOR TRAINING

            # generate noise
            generator_noise = np.random.randn(batch_size, latent_dim)
            # generate random labels and make them one hot
            fake_labels = np.random.randint(low=0, high=num_labels - 1, size=[batch_size, ])
            fake_labels_onehot = np.zeros((batch_size, 10))
            fake_labels_onehot[np.arange(batch_size), fake_labels] = 1
            # concatenate to create latent space
            generator_labels_with_noise = np.concatenate((fake_labels_onehot,
                                                          generator_noise), axis=1)

            # train gen
            # gen_cost, gw_cost, g_lab_cost, _
            gen_run_out = session.run([generator_loss_mean,
                                       gen_wasserstein_loss_mean,
                                       gen_labels_loss_mean,
                                       generator_optimizer],
                                      feed_dict={all_input_generator: generator_labels_with_noise,
                                                 all_real_labels: fake_labels_onehot,
                                                 label_weights: labels_incremental_weight})

            gen_run_out = gen_run_out[:-1]
            # append directly in gen loss history (with mean because of batch_size)

            # generator_history.append([np.mean(gen_cost), np.mean(gw_cost), np.mean(g_lab_cost)])
            generator_history.append([np.mean(elem) for elem in gen_run_out])
        # END FOR MACRO BATCHES

        # generate test latent space (with sample_repetitions to create more rows of samples)
        if not fixed_noise or epoch == 0:
            test_noise = np.random.randn(num_labels * sample_repetitions, latent_dim)
            sorted_labels = np.tile(np.eye(num_labels), sample_repetitions).transpose()
            sorted_labels_with_noise = np.concatenate((sorted_labels, test_noise), axis=1)

        # recall generator
        generated_img = session.run([test_samples],
                                    feed_dict={test_input: sorted_labels_with_noise})
        # print test img
        plot_rows = sample_repetitions
        generate_images(generated_img, epoch, repetitions=plot_rows)

        # plot images
        if check_in_out:
            # generate_images(img_samples, 100+epoch, repetitions=6)
            print(' max value real img (last batch): ', img_samples.max())
            print(' min value real img (last batch): ', img_samples.min())
            # print('labels feeded for epoch: ', epoch)
            # print(img_labels)
            print(' max value generated img (all):   ', np.max(generated_img))
            print(' min value generated img (all):   ', np.min(generated_img))

        if epoch % 10 == 0 or epoch == (num_epochs - 1) or always_get_loss:
            # SAVE & PRINT LOSSES

            # generator vs discriminator loss
            plt.figure()
            disc_line = plt.plot(np.asarray([item[0] for item in discriminator_history]), label='DISC')
            gen_line = plt.plot(np.asarray([item[0] for item in generator_history]), label='GEN')
            plt.legend()
            plt.savefig("GD_losses.png")

            # discriminator losses
            plt.figure()
            disc_sum = plt.plot(np.array([item[0] for item in discriminator_history]), label='ALL')
            disc_w = plt.plot(np.array([item[1] for item in discriminator_history]), label='WASS')
            disc_grad = plt.plot(np.array([item[2] for item in discriminator_history]), label='GRAD')
            disc_lab = plt.plot(np.array([item[3] for item in discriminator_history]), label='LAB')
            plt.legend()
            plt.savefig("D_losses.png")

            # generator losses
            plt.figure()
            gen_sum = plt.plot(np.array([item[0] for item in generator_history]), label='ALL')
            gen_w = plt.plot(np.array([item[1] for item in generator_history]), label='WASS')
            gen_lab = plt.plot(np.array([item[2] for item in generator_history]), label='LAB')
            plt.legend()
            plt.savefig("G_losses.png")

            # discriminator losses
            plt.figure()
            disc_acc = plt.plot(np.array([item[4] for item in discriminator_history]), label='DISC')
            gen_acc = plt.plot(np.array([item[5] for item in discriminator_history]), label='GEN')
            plt.legend()
            plt.savefig("accuracy.png")

            if always_show_fig:
                plt.show()  # it works only in interactive mode

            plt.close('all')

            # save txt logs
            ''' loss_file = open('gen_losses.txt', 'w')
            for item in generator_history:
                loss_file.write("%s\n" % item)

            loss_file = open('disc_losses.txt', 'w')
            for item in discriminator_history:
                loss_file.write("%s\n" % item)
            '''

            with open("gen_losses.txt", "wb") as gl:
                pickle.dump(generator_history, gl)

            with open("disc_losses.txt", "wb") as dl:
                pickle.dump(discriminator_history, dl)

        total_time = time.time() - init_time
        print(' cycle time:  ', time.time() - start_time, " - total time: ", total_time)
        print(' gen  cost  = ', np.mean([item[0] for item in generator_history[-num_macro_batches:]]))
        print(' disc cost  = ', np.mean([item[0] for item in discriminator_history[-num_macro_batches:]]))

        print(' gen  accu  = ', np.mean([item[5] for item in discriminator_history[-num_macro_batches:]]))
        print(' disc accu  = ', np.mean([item[4] for item in discriminator_history[-num_macro_batches:]]))



        #########################################################

        if epoch%is_freq==0:

            print('Inception Score - image generation...')
            is_img = []
            for i in tqdm(range(is_n_batch)):
                # is input stuff
                is_noise = np.random.randn(num_labels * batch_size, latent_dim)
                is_labels = np.tile(np.eye(num_labels), batch_size).transpose()
                is_labels_with_noise = np.concatenate((is_labels, is_noise), axis=1)

                # recall generator
                is_img.append(session.run([is_samples],
                                          feed_dict={is_input: is_labels_with_noise}))

            # Inception Score
            is_image = np.array(is_img).reshape(-1, resolution_image, resolution_image, channels)
            print('Inception score images shape: ', is_image.shape)
            is_result = inception_score.main(is_image.transpose([0, 3, 1, 2]))
            print('INCEPTION SCORE: mean: ', is_result[0], ' std: ', is_result[1])

            is_plot = plt.errorbar(np.array([i for i in range(len(is_history))]),
                                   np.array([element[0] for element in is_history]),
                                   np.array([element[1] for element in is_history]),
                                   linestyle='-', marker='o')

            plt.savefig("is_plot.png")

            is_history.append([elem for elem in is_result])
            with open("is_history.txt", "wb") as fp:
                pickle.dump(is_history, fp)

        #########################################################

        inputs = {"test_input": test_input}
        outputs = {"test_samples": test_samples}

        tf.saved_model.simple_save(session, '',inputs=inputs, outputs=outputs)

        # save_path = saver.save(session, "/tmp/model.ckpt")

        """
        if not os.path.exists('./model'):
            os.makedirs('./model')

        # saver.save(session, './model/saving_' + str(epoch))  # 2 seconds needed
        saver.save(session, './model/model.ckpt')
        saver.save(session, './model.ckpt')
        """

        if total_time >= timer:
            print(' - - - - - TIME OUT! - - - - - ')
            break

    # END FOR EPOCHS

        """
        print('Inception Score - image generation...')
        is_img = []
        for i in tqdm(range(is_n_batch)):
            # is input stuff
            is_noise = np.random.randn(num_labels * batch_size, latent_dim)
            is_labels = np.tile(np.eye(num_labels), batch_size).transpose()
            is_labels_with_noise = np.concatenate((is_labels, is_noise), axis=1)
    
            # recall generator
            is_img.append(session.run([is_samples],
                                      feed_dict={is_input: is_labels_with_noise}))
        """
    # END SESSION
"""
# Inception Score
is_image = np.array(is_img).reshape(-1, resolution_image, resolution_image, channels)
print('Inception score images shape: ', is_image.shape)
is_mean, is_std = inception_score.main(is_image.transpose([0, 3, 1, 2]))
print('INCEPTION SCORE: mean: ', is_mean, ' std: ', is_std)
"""