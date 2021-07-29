import tensorflow as tf

# linear regression function -  y = Wx+b
# Data
x_train = [1.0, 2.0, 3.0, 4.0]
# Expected data
y_train = [-1.0, -2.0, -3.0, -4.0]
# Node creation
# disabling eager execution for placeholder node error
tf.compat.v1.disable_eager_execution()
W = tf.Variable(initial_value=[1.0], dtype=tf.float32)
b = tf.Variable(initial_value=[1.0], dtype=tf.float32)

x = tf.compat.v1.placeholder(dtype=tf.float32)
y_input = tf.compat.v1.placeholder(dtype=tf.float32)

y_output = W * x + b

loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_input))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train_step = optimizer.minimize(loss=loss)

session = tf.compat.v1.Session()
session.run(tf.compat.v1.global_variables_initializer())

print(session.run(fetches=loss, feed_dict={x: x_train, y_input: y_train}))

# training 2000 times
for _ in range(2000):
    session.run(fetches=train_step, feed_dict={x: x_train, y_input: y_train})

print(session.run(fetches=[loss, W, b], feed_dict={x: x_train, y_input: y_train}))

print(session.run(fetches=y_output, feed_dict={x: [5.0, 10.0, 15.0]}))
