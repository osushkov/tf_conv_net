
import idx
import tensorflow as tf
import numpy as np

def loadInputOutput(inputsPath, outputsPath):
  inputs = idx.readImages(inputsPath)
  outputs = idx.readLabels(outputsPath)

  assert(inputs.shape[0] == outputs.shape[0])

  permutation = np.random.permutation(inputs.shape[0])
  return (inputs[permutation,:], outputs[permutation,:])

trainImages, trainLabels = loadInputOutput("data/train_images.idx3", "data/train_labels.idx1")
testImages, testLabels = loadInputOutput("data/test_images.idx3", "data/test_labels.idx1")

sess = tf.Session()

input = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

output = tf.matmul(input, W) + b
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))
trainStep = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

prediction = tf.equal(tf.argmax(target, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

sess.run(tf.initialize_all_variables())

BATCH_SIZE = 100
for i in range(1000):
    offset = (i * BATCH_SIZE) % (trainImages.shape[0] - BATCH_SIZE)

    feedDict = {
        input : trainImages[offset:offset+BATCH_SIZE],
        target : trainLabels[offset:offset+BATCH_SIZE],
    }

    _, loss_value = sess.run([trainStep, crossEntropy], feed_dict=feedDict)
    print("loss: %f" % loss_value)

trainAccuracy = sess.run(accuracy, feed_dict={input:testImages, target:testLabels})
print("train accuracy: %f" % trainAccuracy)


sess.close()
