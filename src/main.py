
import idx
import tensorflow as tf

labels = idx.readLabels("data/test_labels.idx1")
images = idx.readImages("data/test_images.idx3")

sess = tf.Session()

input = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

output = tf.matmul(input, W) + b
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, target))
trainStep = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

sess.run(tf.initialize_all_variables())

for i in range(1000):
    _, loss_value = sess.run([trainStep, crossEntropy], feed_dict={input:images, target:labels})
    print("loss: %f" % loss_value)

sess.close()
