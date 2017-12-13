from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import time
# import mnist
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels


# 檢視結構
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("---")

# 檢視一個觀測值
# print(np.argmax(y_train[0, :])) # 第一張訓練圖片的真實答案


# 定義一個添加層的函數
def add_layer(inputs, input_tensors, output_tensors, activation_function = None):
    with tf.name_scope('Weights'):
        W = tf.Variable(tf.random_normal([input_tensors, output_tensors]))
    with tf.name_scope('Biases'):
        b = tf.Variable(tf.random_normal([output_tensors]))
    with tf.name_scope('Formula'):
        formula = tf.add(tf.matmul(inputs, W), b)
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_function(formula)
    return outputs


INPUT = x_train.shape[1]
HIDDEN = int(INPUT/5)
OUTPUT = y_train.shape[1]

with tf.name_scope('Input'):
    x_feeds = tf.placeholder(tf.float32, shape=[None, INPUT], name='input')

with tf.name_scope('Label'):
    y_feeds = tf.placeholder(tf.float32, shape=[None, OUTPUT], name='label')

with tf.name_scope('Layer1'):
    hidden_layer1 = add_layer(inputs=x_feeds, input_tensors=INPUT, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
    hidden_layer2 = add_layer(inputs=hidden_layer1, input_tensors=HIDDEN, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
    hidden_layer3 = add_layer(inputs=hidden_layer2, input_tensors=HIDDEN, output_tensors=HIDDEN,
                              activation_function=tf.nn.sigmoid)
    hidden_layer4 = add_layer(inputs=hidden_layer3, input_tensors=HIDDEN, output_tensors=HIDDEN,
                              activation_function=tf.nn.sigmoid)
    hidden_layer5 = add_layer(inputs=hidden_layer4, input_tensors=HIDDEN, output_tensors=HIDDEN,
                              activation_function=tf.nn.sigmoid)

with tf.name_scope('Output'):
    output_layer = add_layer(inputs=hidden_layer5, input_tensors=HIDDEN, output_tensors=OUTPUT, activation_function=tf.nn.softmax)

with tf.name_scope('Loss'):
    loss = tf.losses.mean_squared_error(labels=y_feeds, predictions=output_layer)
    tf.summary.scalar('loss', loss)

with tf.name_scope('Training'):
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss)

# Accuracy
with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.arg_max(output_layer, 1), tf.arg_max(y_feeds, 1))
    acc = tf.cast(correct_prediction, tf.float32)


init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))

writer = tf.summary.FileWriter("TensorBoard/", graph=sess.graph)

sess.run(init)

acc_list = []
train_loss_lost = []
test_loss_list = []

for iteration in range(11):
    print('iteration:', iteration)
    print(time.strftime("%D,%H:%M:%S"))
    train_loss = 0
    for k in range(x_train.shape[0]):
        _, cost = sess.run([train_op, loss], feed_dict={x_feeds: [x_train[k]], y_feeds: [y_train[k]]})
        train_loss += cost
    l = (train_loss / x_train.shape[0])
    print('train_loss:', l)

    # for batch_num in range(0, x_train.shape[0], 1000):
    #     batch_x, batch_y = mnist.train.next_batch(1000)
    #     sess.run(train_op, feed_dict={x_feeds: batch_x, y_feeds: batch_y})

    print(time.strftime("%D,%H:%M:%S"))
    test_acc = 0
    test_cost = 0
    for test in range(x_test.shape[0]):
        test_acc += sess.run([acc], feed_dict={x_feeds: [x_test[test]], y_feeds: [y_test[test]]})[0]
        test_cost += sess.run([loss], feed_dict={x_feeds: [x_test[test]], y_feeds: [y_test[test]]})[0]
    a = (test_acc / x_test.shape[0])
    tl = (test_cost/x_test.shape[0])
    print('accuracy:', a)
    print('test_loss:', tl)
    print('')
    acc_list.append(a)
    train_loss_lost.append(l)
    test_loss_list.append(tl)

np.savetxt('./training/acc' + '_deep3' + '.txt', acc_list, delimiter=',')
np.savetxt('./training/train_loss' + '_deep3' + '.txt', train_loss_lost, delimiter=',')
np.savetxt('./training/test_loss' + '_deep3' + '.txt', test_loss_list, delimiter=',')
sess.close()
#
