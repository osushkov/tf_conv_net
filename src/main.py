
import idx
import tensorflow as tf
import numpy as np
import math


def loadInputOutput(inputsPath, outputsPath):
    inputs = idx.readImages(inputsPath)
    outputs = idx.readLabels(outputsPath)

    assert(inputs.shape[0] == outputs.shape[0])

    permutation = np.random.permutation(inputs.shape[0])
    return (inputs[permutation,:], outputs[permutation,:])

def weight(shape):
    stddev = 1.0 / math.sqrt(shape[0])
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def convOp(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def poolOp(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

trainImages, trainLabels = loadInputOutput("data/train_images.idx3", "data/train_labels.idx1")
testImages, testLabels = loadInputOutput("data/test_images.idx3", "data/test_labels.idx1")

sess = tf.Session()

input = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])

image = tf.reshape(input, [-1, 28, 28, 1])

W1 = weight([5, 5, 1, 30])
b1 = bias([30])

conv1 = tf.nn.elu(convOp(image, W1) + b1)
pool1 = poolOp(conv1)

W2 = weight([5, 5, 30, 60])
b2 = bias([60])

conv2 = tf.nn.elu(convOp(pool1, W2) + b2)
pool2 = poolOp(conv2)

fcW = weight([7*7*60, 1000])
fcB = bias([1000])

flattened = tf.reshape(pool2, [-1, 7*7*60])
fc = tf.nn.elu(tf.matmul(flattened, fcW) + fcB)

keepP = tf.placeholder(tf.float32)
fcDrop = tf.nn.dropout(fc, keepP)

outputW = weight([1000, 10])
outputB = bias([10])

output = tf.matmul(fcDrop, outputW) + outputB
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))
trainStep =  tf.train.AdamOptimizer().minimize(crossEntropy)

prediction = tf.equal(tf.argmax(target, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

sess.run(tf.initialize_all_variables())

BATCH_SIZE = 500
for i in xrange(2000):
    offset = (i * BATCH_SIZE) % (trainImages.shape[0] - BATCH_SIZE)

    feedDict = {
        input : trainImages[offset:offset+BATCH_SIZE],
        target : trainLabels[offset:offset+BATCH_SIZE],
        keepP : 0.6,
    }

    _, loss_value = sess.run([trainStep, crossEntropy], feed_dict=feedDict)
    print("loss: %f" % loss_value)


# Need to calculate the accuracy in batches since we cant really git the whole test
# dataset into memory.
accuracySum = 0.0
numSamples = 0

for offset in xrange(0, testImages.shape[0] - BATCH_SIZE, BATCH_SIZE):
    feedDict = {
        input : testImages[offset:offset+BATCH_SIZE],
        target : testLabels[offset:offset+BATCH_SIZE],
        keepP : 1.0,
    }

    trainAccuracy = sess.run(accuracy, feed_dict=feedDict)

    accuracySum += trainAccuracy
    numSamples += 1

print("train accuracy: %f" % (accuracySum / numSamples))


sess.close()
