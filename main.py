from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#importing inputdata from other file dont ask where it is from
import tensorflow as tf
#mnist = input_data.read_data_sets('/tmp/data/', one_hot = True)
#basic parameters
alpha = 0.01 #just a guesss
iterations = 30 #not jobless
batch_size = 100
display_step = 2 #youll know later

#placeholders x,y. act like containers for i/p and o/p data
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float',[None, 10])
#initialize weights
w = tf.Variable(tf.zeros([784,10]))
#add bias
b = tf.Variable(tf.zeros([10]))
#remember we are not doing a nn

#scope1. more like a job1
#wx+b
with tf.name_scope('wxb') as scope:
    model = tf.nn.softmax(tf.matmul(x,w)+b)
    #softmax function(0-1)!

#summarizing for data visuals(not compulsory)
w_h = tf.summary.histogram('weights',w)
b_h = tf.summary.histogram('biases',b)

#job2. formulate cost function to check error rate
with tf.name_scope('cost_function') as scope:
    cost_function = -tf.reduce_sum(y*tf.log(model))
    #idk should come back later
    #for summary
    tf.summary.scalar("cost_function", cost_function)

#job3. start gradient descent algo on model with cf as metric
with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost_function)

#time to init all variables coz this is tensorflow
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer.run()

#merge all summary variables to make it handy
merged_summary_op = tf.summary.merge_all()


#start tensorflow session
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('/home/pavankoushik/tensor/mnist/log', graph = sess.graph)
    #start training
    for iteration in range(iterations):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #for all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
            avg_cost += sess.run(cost_function, feed_dict={x:batch_xs,y:batch_ys})/total_batch
            summary_str = sess.run(merged_summary_op, feed_dict={x:batch_xs, y:batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch+i)
        #for each two iterations we are displaying summary
        if iteration%display_step == 0:
            print("iteration: %04d, cost = %9f"%(iteration+1, avg_cost))


    #done
    predictions = tf.equal(tf.argmax(model, 1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(predictions, 'float'))
    print(acc.eval({x:mnist.test.images, y:mnist.test.labels}))

