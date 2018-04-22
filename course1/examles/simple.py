import tensorflow as tf

with tf.Graph().as_default():
    with tf.Session() as sess:
        a = tf.constant((1., 2., 3., 4.))  # constant node with constant output tensor

        # variable which can be updated by some other op
        var = tf.get_variable('variable', shape=(4,),
                              dtype=tf.float32, initializer=tf.constant_initializer(2.))

        # create a mul op (nothing is really computed)
        b = a*var

        # b.eval()  # this would cause an error - variable is not initialized

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # run the computation
        print(b.eval())
