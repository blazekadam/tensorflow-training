import sys; sys.path.append('..')  # allow to import other stuff
import random
import string
import os
import os.path as path

import numpy as np
import tensorflow as tf

from course2.assignments.data import load_data, batch_data
from course2.assignments.net import build_facial_keypoints_ff


def train() -> None:
    """Train CNN net for facial key-points detection."""
    with tf.Session().as_default() as sess:
        # prepare log directories
        dirname = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        valid_dirname = path.join('log', dirname+'-valid')
        train_dirname = path.join('log', dirname+'-train')
        os.makedirs(valid_dirname)
        os.makedirs(train_dirname)
        print('Summaries will be written to  {} and {} respectively'.format(train_dirname, valid_dirname))

        # create placeholders, model and loss summary
        print('Creating model')
        images = tf.placeholder(dtype=tf.float32, shape=(None, 96, 96, 1), name='images')
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 4, 2), name='targets')
        predictions = build_facial_keypoints_ff(images)
        flat_targets = tf.layers.flatten(targets)
        loss = tf.reduce_mean(tf.nn.l2_loss(predictions-flat_targets))
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        tf.summary.scalar('loss', loss)

        # load and split data
        print('Loading data ')
        images_data, targets_data = load_data('training.csv')
        train_images = images_data[:-200]
        train_targets = targets_data[:-200]
        valid_images = images_data[-200:]
        valid_targets = targets_data[-200:]

        # initialize variables and collect summaries
        sess.run(tf.global_variables_initializer())
        train_summaries = tf.summary.merge_all()

        # draw predictions to the 1st image and create its image summary
        blue_np = np.zeros((1, 96, 96, 3))
        blue_np[0, :, :, 2] = 255
        blue = tf.constant(blue_np, dtype=tf.float32)
        color_images = tf.image.grayscale_to_rgb(images)
        example = color_images[0:1, :, :, :]
        example_predictions = tf.reverse(tf.clip_by_value(tf.cast(tf.round(predictions[0]), tf.int64), 0, 95), [0])
        indices = tf.SparseTensor(tf.reshape(example_predictions, [4, 2]), values=tf.ones((4,)), dense_shape=(96, 96))
        indices = tf.reshape(tf.sparse_to_dense(indices.indices, (96, 96), indices.values, validate_indices=False),
                             (1, 96, 96))
        indices = tf.stack([indices, indices, indices], -1)
        example = tf.where(tf.greater(indices, 0), blue, example)
        visualization_summary = tf.summary.image('example_image', example)

        # create summary writers
        print('Writing graph')
        train_writer = tf.summary.FileWriter(train_dirname, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_dirname)

        print('Entering main loop')
        step = 0
        for _ in range(5):  # five epochs should be enough
            for batch_images, batch_targets in batch_data(train_images, train_targets):
                # train step
                _, summaries = sess.run([train_op, train_summaries], {images: batch_images, targets: batch_targets})
                train_writer.add_summary(summaries, step)

                # valid visualization
                summaries, visual = sess.run([train_summaries, visualization_summary],
                                             {images: valid_images, targets: valid_targets})
                valid_writer.add_summary(summaries, step)
                valid_writer.add_summary(visual, step)

                # report progress
                step += 1
                print('.', end='', flush=True)

            print('\n- epoch done -')


if __name__ == '__main__':
    train()
