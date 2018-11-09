import tensorflow as tf

hello = tf.constant("Hello, Tensor!")
sess = tf.Session()
print(sess.run(hello))


c = tf.constant("Hello dist TF")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
sess.run(c)