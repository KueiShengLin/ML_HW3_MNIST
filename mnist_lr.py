import mnist_reader
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def mnist_data_read():
    training_data = list(mnist_reader.read(dataset='training', path='.\\MNIST_data'))
    testing_data = list(mnist_reader.read(dataset='testing', path='.\\MNIST_data'))

    train_label = []
    train_pixels = []
    test_label = []
    test_pixels = []

    for data in training_data:
        train_label.append(data[0])
        pix = data[1].reshape(28*28)
        train_pixels.append(pix)

    for data in testing_data:
        test_label.append(data[0])
        pix = data[1].reshape(28 * 28)
        test_pixels.append(pix)

    return np.asarray(train_pixels), np.asarray(train_label), np.asarray(test_pixels), np.asarray(test_label)


def graphing():
    acc = np.loadtxt('./training/lr_acc' + '.txt', delimiter=',')
    loss = np.loadtxt('./training/lr_loss' + '.txt', delimiter=',')
    plt.title('Accuracy & Loss')
    plt.xlabel('iteration')
    plt.ylabel('A&L')
    plt.xlim(1, 10)
    plt.ylim(0, 1.1)
    iteration_time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.plot(iteration_time, acc, color='C0', label='accuracy')
    plt.plot(iteration_time, loss, '--', color='C0', label='loss')
    plt.legend()
    plt.show()


def logistic_regression():
    score_list = []

    clf = LogisticRegression(C=50 / x_train.shape[0], multi_class='multinomial', solver='sag', verbose=10, tol=0.1, max_iter=1)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    score_list.append(score)

    for i in range(2, 11):
        clf.set_params(max_iter=i)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        score_list.append(score)
        print(score)

    np.savetxt('./training/lr_acc' + '.txt', score_list, delimiter=',')
    predict = clf.predict(x_test)


x_train, y_train, x_test, y_test = mnist_data_read()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("---")

logistic_regression()
graphing()

#
