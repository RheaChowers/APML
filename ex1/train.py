import tensorflow as tf
import numpy as np
from model import mlp, conv_net
from math import ceil
import matplotlib.pyplot as plt
mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def load_data():
    """
    Loads the fashion mnist dataset. returns the number of labels, training set, training labels, test set and test
    labels.
    """
    # load data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # preprocess
    train_images = tf.keras.utils.normalize(train_images, axis=1)
    test_images = tf.keras.utils.normalize(test_images, axis=1)
    nlabels = np.unique(train_labels).shape[0]

    return nlabels, train_images, train_labels.astype(np.int32), test_images, test_labels.astype(np.int32)


def plot(train_loss, train_acc, test_loss, test_acc, show=True):
    """
    creates a okit
    """
    plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.plot(train_acc, label="train_acc")
    plt.plot(test_loss, label="test_loss")
    plt.plot(test_acc, label="test_acc")
    plt.legend()
    if show:
        plt.show()


def train(model_fn, batch_size, learning_rate=None):
    """
    load FashionMNIST data.
    create model using model_fn, and train it on FashionMNIST.
    :param model_fn: a function to create the model (should be one of the functions from model.py)
    :param batch_size: the batch size for the training
    :param learning_rate: optional parameter - option to specify learning rate for the optimizer.
    :return:
    """
    epochs = 100
    nlabels, train_images, train_labels, test_images, test_labels = load_data()
    train_samples, height, width = train_images.shape
    test_samples, _, _ = test_images.shape

    # create training model
    X = tf.placeholder(tf.float32, shape=[None, height, width], name="input")
    Y = tf.placeholder(tf.int32, shape=[None], name="labels")
    predictions = model_fn(X, nlabels)
    sftmx = tf.nn.softmax(predictions)
    total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=predictions, name="loss")
    avg_loss = tf.reduce_mean(total_loss)
    loss = avg_loss + tf.losses.get_regularization_loss()  # mean batch loss + regularization
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    final_preds = tf.argmax(predictions, axis=1, output_type=tf.int32)
    zero_one = tf.cast(tf.equal(final_preds, Y), dtype=tf.float32)
    accuracy = tf.reduce_mean(zero_one)


    # for finding an adversarial image, we'll try and saturate deactivated neurons
    epsilon = [0.01, 0.05, 0.1, 0.5]
    image_grad = tf.gradients(ys=avg_loss, xs=[X])
    signed_grad = tf.sign(image_grad)

    # define some constants
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    n_batches = ceil(train_samples/batch_size)
    avg_loss, avg_acc = 0, 0
    start, end = 0, 0
    find_adversarial = False  # Turn true to find adversarial image

    print("Starting Training Loop....\n")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for i in range(n_batches):
                start = batch_size * i
                end = start + batch_size
                if i == n_batches - 1:
                    end = train_samples
                model_labels, cost, acc, _ = sess.run([sftmx, loss, accuracy, optimizer],
                                        feed_dict={X: train_images[start:end], Y: train_labels[start:end]})
                
                avg_loss += (cost / n_batches)
                avg_acc += (acc / n_batches)
            
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)

            # test at the end of the epoch
            cost, acc = sess.run([loss, accuracy], feed_dict={X: test_images, Y: test_labels})  # cost is of 1 x batch_size
            test_loss.append(cost)
            test_acc.append(acc)

            print("Train: Epoch Number: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch, train_loss[epoch], train_acc[epoch]))
            print("Test: Epoch Number: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch, test_loss[epoch], test_acc[epoch]))
            
            # shuffle training data for next epoch
            if epoch != (epochs - 1):
                randomize = np.arange(train_samples)
                np.random.shuffle(randomize)
                train_images, train_labels = train_images[randomize], train_labels[randomize]
                randomize = np.arange(test_samples)
                np.random.shuffle(randomize)
                test_images, test_labels = test_images[randomize], test_labels[randomize]
                avg_loss, avg_acc = 0, 0


        # find the adversarial
        if find_adversarial:
            indices = find_true(train_images[start:end], train_labels[start:end], model_labels)
            model_probs = model_labels[indices]
            grad = sess.run([signed_grad], feed_dict={X:train_images[indices], Y:train_labels[indices]})
            image_grad = np.reshape(grad[0], (grad[0].shape[1], grad[0].shape[2], grad[0].shape[3]))
            for eps in epsilon:
                image = train_images[indices]
                labels = train_labels[indices]
                img_eps = image + eps * image_grad
                probs = sess.run([sftmx], feed_dict={X:img_eps, Y:labels})
                new_probs = probs[0]
                new_label = np.argmax(new_probs, axis=1)
                adversarial_prob = np.max(new_probs, axis=1)
                for i in range(len(indices)):
                    if labels[i] != new_label[i]:
                        plt.subplot(1, 2, 1)
                        plt.imshow(image[i], cmap='gray')
                        plt.title('{}, w.p. {:.4f}'.format(mnist_classes[labels[i]], model_probs[i,labels[i]]))
                        plt.subplot(1, 2, 2)
                        plt.imshow(img_eps[i], cmap='gray')
                        plt.title('Adversarial: {}, w.p. {:.4f}'.format(mnist_classes[new_label[i]], adversarial_prob[i]))

    # plot the results
    plot(train_loss, train_acc, test_loss, test_acc)


def find_true(x, y, prediction_softmax):
    """
    Finds an image that the model correctly labeled and returns indices of x and y
    """
    y_predict = np.argmax(prediction_softmax, axis=1)
    indices = []
    for i in range(x.shape[0]):
        if y[i] == y_predict[i]:
            indices.append(i)
    return indices        


def main():
    train(mlp, 64, 0.001)
    # train (conv_net, 64, 0.001)

if __name__== "__main__":
    main()

