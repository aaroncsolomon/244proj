import numpy as np
import tensorflow as tf

# TODO create a cluster spec matching your server spec

cluster_spec = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

task_input = tf.placeholder(tf.float64, 100)

# First part: compute mean of half of the input data
with tf.device("/job:local/task:0"):
    local_input = tf.slice(task_input, [50], [50])
    local_mean = tf.reduce_mean(local_input)

# TODO do another half of the computation using another device

with tf.device("/job:local/task:1"):
    local_input_2 = tf.slice(task_input, [0], [50])
    local_mean_2 = tf.reduce_mean(local_input)
    myoutput = tf.reduce_mean([local_mean, local_mean_2])


# TODO compute the overall result by combining both results

# TODO Fill in the session specification
with tf.Session("grpc://localhost:2223") as sess:
    # Sample some data for the computation
    data = np.random.random(100)

    # TODO run the session to compute the overall using your workers
    # and the input data. Output the result.
    print(sess.run(myoutput, {task_input : data}))
    print(np.mean(data))

