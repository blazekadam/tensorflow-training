import tensorflow as tf

with tf.Graph().as_default():
    with tf.Session() as sess:

        # create a trainable variable
        var = tf.get_variable('variable', shape=(4,),
                              dtype=tf.float32, initializer=tf.constant_initializer(2.))

        # we should see the the variable defined above
        print(tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
