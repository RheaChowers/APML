import tensorflow as tf


def mlp(x: tf.Tensor, nlabels):
    """
    multi layer perceptrone: x -> linear > relu > linear.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param nlabels: the dimension of the output.
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels). the model return logits (before softmax).
    """
    # batch, dim1, dim2 = x.shape
    layer_dimension = 100
    new_dim = x.shape[1] * x.shape[2]
    reshaped_x = tf.reshape(x, [tf.shape(x)[0], new_dim])  # flatten
    
    # regularizer
    reg = tf.contrib.layers.l2_regularizer(0.001)

    # variables
    W1 = tf.get_variable('weights1', shape=[new_dim, layer_dimension], regularizer=reg, trainable=True)
    b1 = tf.get_variable('bias1', shape=[layer_dimension], regularizer=reg, trainable=True)
    W2 = tf.get_variable('weights2', shape=[layer_dimension, nlabels], regularizer=reg, trainable=True)
    b2 = tf.get_variable('bias2', shape=[nlabels], regularizer=reg, trainable=True)
    
    # layers
    first_linear = tf.add(tf.matmul(reshaped_x, W1), b1, name="first_lin")  # batch x 100
    relu_result = tf.nn.relu(first_linear, name="relu")  # batch x 100
    second_linear = tf.add(tf.matmul(relu_result, W2), b2, name="second_lin")  # batch x nlabels

    return second_linear


def conv_net(x: tf.Tensor, nlabels):
    """
    convnet.
    in the convolution use 3x3 filteres with 1x1 strides, 20 filters each time.
    in the  maxpool use 2x2 pooling.
    :param x: symbolic tensor with shape (batch, 28, 28)
    :param nlabels: the dimension of the output.
    :return: a symbolic tensor, the result of the mlp, with shape (batch, nlabels). the model return logits (before softmax).
    """
    # constants
    filter_size = 3
    maxpool_size = 2
    num_filters = 20
    strides = [1]*4
    maxpool_ksize = [1, maxpool_size, maxpool_size, 1]
    _, height, width = x.shape
    final_dimension = num_filters * (height - 2*maxpool_size + 2 - 2*filter_size + 2) * (
                      width  - 2*maxpool_size + 2 - 2*filter_size + 2)  # assume strides=1
    
    # variables
    reshape_x = tf.reshape(x, shape=[tf.shape(x)[0], height, width, 1])
    conv1_filter = tf.get_variable(name="first_filter",shape=[filter_size, filter_size,
                                                              1, num_filters], trainable=True)
    conv2_filter = tf.get_variable(name="second_filter",shape=[filter_size, filter_size,
                                                              num_filters, num_filters], trainable=True)
    W = tf.get_variable(name='weights', shape=[final_dimension, nlabels], trainable=True)
    b = tf.get_variable(name='bias', shape=[nlabels], trainable=True) 

    # layers
    conv1 = tf.nn.conv2d(reshape_x, filter=conv1_filter, strides=strides, padding='VALID', name="First_conv")  # output [batch, 26, 26, 20]
    maxpool1 = tf.nn.max_pool(value=conv1, ksize=maxpool_ksize, strides=strides, padding='VALID', name="maxpool1")  # output [batch, 25, 25, 20]
    conv2 = tf.nn.conv2d(maxpool1, filter=conv2_filter, strides=strides, padding='VALID', name="second_conv")  # output [batch, 23, 23, 20] 
    maxpool2 = tf.nn.max_pool(value=conv2, ksize=maxpool_ksize, strides=strides, padding='VALID', name="maxpool2")  # output [batch, 22, 22, 20] 

    reshape_conv = tf.reshape(maxpool2, shape=[tf.shape(x)[0], final_dimension])
    
    predictions = tf.matmul(reshape_conv, W) + b  # batch x nlabels
    
    return predictions