import tensorflow as tf
import numpy as np

# Generate samples of a function we are trying to predict:
samples = 100
xs = np.linspace(-5, 5, samples)
# We will attempt to fit this function
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, samples)

# First, create TensorFlow placeholders for input data (xs) and
# output (ys) data. Placeholders are inputs to the computation graph.
# When we run the graph, we need to feed values for the placerholders into the graph.
# TODO: create placeholders for inputs and outputs

inputs = tf.placeholder(tf.float32)
outputs = tf.placeholder(tf.float32)


# We will try minimzing the mean squared error between our predictions and the
# output. Our predictions will take the form X*W + b, where X is input data,
# W are ou weights, and b is a bias term:
# minimize ||(X*w + b) - y||^2
# To do so, you will need to create some variables for W and b. Variables
# need to be initialised; often a normal distribution is used for this.
#

weights = tf.Variable(tf.random_normal([1]))
bias = tf.Variable(tf.random_normal([1]))


# Next, you need to create a node in the graph combining the variables to predict
# the output: Y = X * w + b. Find the appropriate TensorFlow operations to do so.

predictions = tf.add(tf.multiply(inputs, weights), bias)

# Finally, we need to define a loss that can be minimized using gradient descent:
# The loss should be the mean squared difference between predictions
# and outputs.
loss = tf.losses.mean_squared_error(outputs, predictions)

# Use gradient descent to optimize your variables
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
tcall = optimizer.minimize(loss)

# We create a session to use the graph and initialize all variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# Optimisation loop
epochs = 1000
previous_loss = 0.0
training_cost = 10

condition = tf.less(np.abs(previous_loss - training_cost), 0.000001)
session.run(tcall, {inputs: xs, outputs: ys})
training_cost = session.run(loss, {inputs: xs, outputs: ys})

i, out = tf.while_loop()

with session as sess:
    print(sess.run(i))


    print('Training cost = {}'.format(training_cost))

    # Termination condition for the optimization loop



    previous_loss = training_cost

# TODO try plotting the predictions by using the model to predict outputs, e.g.:
import matplotlib.pyplot as plt
predictions = session.run(predictions, {inputs:xs})
plt.plot(xs, predictions)
plt.show()
