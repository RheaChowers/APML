import tensorflow as tf
import numpy as np
import operations as op
import matplotlib.pyplot as plt
from math import ceil


def load_data():
    from sklearn.datasets import load_boston
    boston_dataset = load_boston()
    X = np.array(boston_dataset.data)
    y = np.array(boston_dataset.target)
    return X, y


def model(x: tf.Tensor):
    """
    linear regression model: y_predict = W*x + b
    please use your matrix multiplication implementation.
    :param x: symbolic tensor with shape (batch, dim)
    :return:  a tuple that contains: 1.symbolic tensor y_predict, 2. list of the variables used in the model: [W, b]
                the result shape is (batch)
    """
    _, dimension = x.shape
    W = tf.get_variable(name='weights', shape=[dimension, 1],  dtype=tf.float32, trainable=True)
    b = tf.get_variable(name='bias', shape=[1, ],  dtype=tf.float32, trainable=True)
    y_predict = op.matmul_tf(x, W) + b
    return y_predict, [W, b]


def train(epochs, learning_rate, batch_size):
    """
    create linear regression using model() function from above and train it on boston houses dataset using batch-SGD.
    please normalize your data as a pre-processing step.
    please use your mse-loss implementation.
    :param epochs: number of epochs
    :param learning_rate: the learning rate of the SGD
    :return: list contains the mean loss from each epoch.
    """
    x, y = load_data()
    
    # preprocess
    x = tf.keras.utils.normalize(x, axis=1)
    num_samples, dimension = x.shape
    n_batches = ceil(num_samples/batch_size)

    # create model
    X = tf.placeholder(tf.float32, shape=[None, dimension],name="X")
    Y = tf.placeholder(tf.float32, shape=[None], name="Y")
    predictions, weights = model(X)
    [W, b] = weights
    loss = op.mse_tf(Y, predictions)
    gradients = tf.gradients(ys=[loss], xs=[W, b], name="gradients")
    weights_update = tf.assign_sub(W, learning_rate*gradients[0])
    bias_update = tf.assign_sub(b, learning_rate*gradients[1])
    update = tf.group(weights_update, bias_update, name="grad_des")
    
    # define some constants
    epoch_loss = []
    epoch_cost = 0
    
    print("Starting Training Loop....\n")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for i in range(n_batches):
                start = batch_size * i
                end = start + batch_size
                if i == n_batches - 1:
                    end = num_samples
                x_batch = x[start:end]
                y_batch = y[start:end]
                cost, _ = sess.run([loss, update], feed_dict={X: x_batch, Y: y_batch})  # cost is of 1 x batch_size

                epoch_cost += (np.average(cost) / n_batches)  # average over all samples, and over batches in epoch
            
            epoch_loss.append(epoch_cost)
            # print("Epoch Number: {}, Loss: {:.4f}".format(epoch, epoch_cost))
            epoch_cost = 0

            # shuffle data for next epoch
            randomize = np.arange(num_samples)
            np.random.shuffle(randomize)
            x, y = x[randomize], y[randomize]


    return epoch_loss


def main():
    learn_rate = 0.0001
    plot = False
    losses = train(50, learn_rate, 32)
    if plot:
        title = "Linear Regression Loss with Learning Rate = "+str(learn_rate)
        plt.plot(losses)
        plt.title(title)
        plt.show()
        


if __name__== "__main__":
  main()

