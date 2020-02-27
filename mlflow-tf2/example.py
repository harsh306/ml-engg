"""
Mlflow Tracking with Tensorflow 2.0
"""

import mlflow
import tensorflow as tf


def input():
    # data scaling example
    x = tf.constant([-10, 0, 2, 30, 8, 10], dtype=tf.float32)
    y = tf.constant([5, 0, -1, -15, -4, -5], dtype=tf.float32)
    return x, y


def setup_mlflow():
    mlflow.set_tracking_uri('file:///Users/hpathak/PycharmProjects/ml-fast-preprocessing/mlruns/')
    mlflow.set_experiment('Example')
    mlflow.start_run(run_name='alpha')


def get_hparams():
    dict = {'lr_rate': 0.0005}
    return dict


def model():
    W1 = tf.Variable(initial_value=-0.35, trainable=True)
    b1 = tf.Variable(initial_value=0.1, trainable=True)
    return W1, b1


def run():
    hparams = get_hparams()
    x, y = input()
    W1, b1 = model()
    for i in range(50):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(W1)
            tape.watch(x)
            tape.watch(y)
            tape.watch(b1)
            layer = tf.math.multiply(W1, x) + b1
            loss = tf.reduce_sum(tf.math.square(tf.math.subtract(y, layer)))

        # compute gradients
        dW1 = tape.gradient(loss, W1)
        db1 = tape.gradient(loss, b1)

        # perform gradient descent
        W1.assign_sub(hparams['lr_rate'] * dW1)
        b1.assign_sub(hparams['lr_rate'] * db1)

    # track mlfow
    mlflow.log_params(hparams)
    mlflow.log_metric('mse_loss', loss.numpy())

    mlflow.end_run()


if __name__ == '__main__':
    setup_mlflow()
    run()
