#this is test program
import tensorflow as tf
sess = tf.Session()

hello = tf.constant("Hello world")
print(sess.run(hello))
