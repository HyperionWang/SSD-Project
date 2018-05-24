import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import os

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


class HyperParamters:
    def __init__(self):
        self.L2_REG = 1e-4
        self.KEEP_PROB = 0.5
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 100
        self.BATCH_SIZE = 16
        self.IMAGE_SIZE = (160, 576)
        self.NUM_CLASSES = 2


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    return input, keep_prob, layer3, layer4, layer7


print('Start to load VGG Model...')


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    param = HyperParamters()
    L2_reg_Value = param.L2_REG
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_reg_Value))
    output1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding='same',
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_reg_Value))
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_reg_Value))
    output2 = tf.add(output1, layer4_1x1)
    output3 = tf.layers.conv2d_transpose(output2, num_classes, 4, 2, padding='same',
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_reg_Value))

    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1, padding='same',
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_reg_Value))
    output4 = tf.add(output3, layer3_1x1)
    output5 = tf.layers.conv2d_transpose(output4, num_classes, 16, 8, padding='same',
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_reg_Value))

    return output5


print('Start to test the layer')


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    option = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    return logits, option, cross_entropy_loss


print("Start the optimizer ... ")


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    param = HyperParamters()
    for epoch in range(epochs):
        start = time.time()
        for image, label in get_batches_fn(batch_size):
            out, loss = sess.run([train_op, cross_entropy_loss],
                                 feed_dict={input_image: image, correct_label: label, keep_prob: param.KEEP_PROB,
                                            learning_rate: param.LEARNING_RATE})

        print("Current run time: %s" % str(time.time() - start))
        # print("Epoch: {}".format(epoch), "of {}".format(epochs), "current loss is: {:.2f}".format(loss))
        print("Epoch: %s / %s" % (epoch, epochs))
        print("The loss is:")
        print(loss)


print("Start test training module ... ")
tests.test_train_nn(train_nn)


def run():
    param = HyperParamters()
    num_classes = param.NUM_CLASSES
    image_shape = param.IMAGE_SIZE
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    print("Start training ...")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_images, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, num_classes))
        learning_rate = tf.placeholder(dtype=tf.float32)
        logits, train_op, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate,
                                                        num_classes)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_nn(sess, param.EPOCHS, param.BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_images,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_images)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
