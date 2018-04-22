import tensorflow as tf

with tf.Graph().as_default():
    with tf.Session() as sess:

        with tf.variable_scope('my_scope'):
            # my_scope/variable:0
            tf.get_variable('variable', shape=(4,), dtype=tf.float32, initializer=tf.constant_initializer(2.))

            # this would raise an error (variable with the same name was already defined)
            # tf.get_variable('variable', shape=(4,), dtype=tf.float32, initializer=tf.constant_initializer(2.))

        with tf.variable_scope('another_scope'):
            # another_scope/variable:0
            tf.get_variable('variable', shape=(4,), dtype=tf.float32, initializer=tf.constant_initializer(3.))

            with tf.variable_scope('nested'):
                # another_scope/nested/variable:0
                tf.get_variable('variable', shape=(4,), dtype=tf.float32, initializer=tf.constant_initializer(4.))

        with tf.variable_scope('reusing_scope', reuse=tf.AUTO_REUSE):
            v = tf.get_variable('variable', shape=(4,), dtype=tf.float32, initializer=tf.constant_initializer(5.))

            # this will return 'v' as it was already defined
            # an error would be raise if the shape or dtype is changed
            v2 = tf.get_variable('variable', shape=(4,), dtype=tf.float32)

            assert v == v2

        sess.run(tf.global_variables_initializer())

