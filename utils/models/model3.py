import tensorflow as tf
import time

def SVHNmodel(input_x, input_y, label_length, is_training=False, drop_rate=0.5):
    if is_training == False:
        drop_rate = 0
    with tf.variable_scope('hidden_layer_1'):
        conv1 = tf.layers.conv2d(inputs=input_x, filters=384, strides=4, kernel_size=[5, 5], padding='same')
        norm1 = tf.layers.batch_normalization(conv1)
        act1 = tf.nn.relu(norm1)
        pool1 = tf.layers.max_pooling2d(inputs=act1, pool_size=[4,4], strides=4, padding='same')
        drop1 = tf.layers.dropout(pool1, rate=drop_rate)
        layer1 = drop1
        print(layer1.shape)

    # one locally connected hidden layer
    # The fully connected layers contain 3,072 units each
    flatten = tf.reshape(layer1, [-1, 4*4*384]) #3072 get from the paper
    with tf.variable_scope('densely_hidden_layer_1'):
        dense1 = tf.layers.dense(inputs=flatten, units=7680, activation=tf.nn.relu)
        layer2 = dense1

    with tf.variable_scope('L'):
        fc1 = tf.layers.dense(inputs=layer2, units=7, name='fc1')
        L = fc1

    with tf.variable_scope('S1'):
        fc2 = tf.layers.dense(inputs=layer2, units=11, name='fc2')
        S1 = fc2

    with tf.variable_scope('S2'):
        fc3 = tf.layers.dense(inputs=layer2, units=11, name='fc3')
        S2 = fc3

    with tf.variable_scope('S3'):
        fc4 = tf.layers.dense(inputs=layer2, units=11, name='fc4')
        S3 = fc4

    with tf.variable_scope('S4'):
        fc5 = tf.layers.dense(inputs=layer2, units=11, name='fc5')
        S4 = fc5

    with tf.variable_scope('S5'):
        fc6 = tf.layers.dense(inputs=layer2, units=11, name='fc6')
        S5 = fc6

    with tf.name_scope("loss"):
        # No regularization in the paper
        L_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_length, logits=L), name='L_loss')
        S1_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y[:,0,:], logits=S1), name='S1_loss')
        S2_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y[:,1,:], logits=S2), name='S2_loss')
        S3_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y[:,2,:], logits=S3), name='S3_loss')
        S4_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y[:,3,:], logits=S4), name='S4_loss')
        S5_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y[:,4,:], logits=S5), name='S5_loss')
        all_loss = L_loss + S1_loss + S2_loss + S3_loss + S4_loss + S5_loss
        
        tf.summary.scalar('SVHNmodel_loss', all_loss)

    digits = tf.stack([S1, S2, S3, S4, S5], axis=1, name='outdig')
    return L, digits, all_loss


def train_step(loss, learning_rate):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return step


def matrix2num(matrix_label):
    """
    :param matrix_length: n x 7 matrix
    :param matrix_label: n x 5 x 11
    :return:
    """
    with tf.name_scope('matrix2num'):
        digit = tf.argmax(matrix_label, axis=2) # n x 5
    return digit


def training(x_train, train_label, train_label_length,
             x_val, val_label, val_label_length,
             learning_rate=1e-4,
             batch_size=100,
             drop_rate=0.2,
             epoch=20):
    with tf.name_scope('inputs'):
        input_x = tf.placeholder(shape=[None, 64, 64, 3], dtype=tf.float32, name='input_x')
        input_y = tf.placeholder(shape=[None, 5, 11], dtype=tf.int64, name='input_y')
        label_length = tf.placeholder(shape=[None, 7], dtype=tf.int64, name='label_length')
        is_training = tf.placeholder(tf.bool, name='is_training')
        
    L, digits, all_loss = SVHNmodel(input_x, input_y, label_length, is_training, drop_rate)
    iters = int(x_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))
    
    step = train_step(all_loss, learning_rate)
    output_label = matrix2num(digits)
    input_label = matrix2num(val_label[0:100])
    
    iter_total = 0
    best_acc = 0
    cur_model_name = 'SVHNmodel_{}'.format(int(time.time()))
    with tf.Session() as sess:
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log3/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))
            for itr in range(iters):
                iter_total += 1
                training_batch_x = x_train[itr * batch_size: (1 + itr) * batch_size]
                train_batch_label = train_label[itr * batch_size: (1 + itr) * batch_size]
                train_batch_label_length = train_label_length[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, all_loss], feed_dict={input_x: training_batch_x,
                                                                    input_y: train_batch_label,
                                                                    label_length: train_batch_label_length, is_training:True})
              
                if iter_total % 300 == 0:
                    o, merge_result = sess.run([output_label,merge], feed_dict={input_x: x_val[0:100], input_y: val_label[0:100],
                                                                                label_length: val_label_length[0:100], is_training:False})
                    error_num = 0
                    for i in range(100):
                        e = tf.count_nonzero(o[i] - input_label[i])
                        #if i % 10 == 0:
                            #print('=====is training=======')
                        if e.eval() != 0:
                            error_num += 1
                    tf.summary.scalar('SVHNmodel_error_num', error_num)
                    valid_eve = error_num

                    valid_acc = 100 - (valid_eve * 100)/100
                    print(' validation accuracy iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                    writer.add_summary(merge_result, iter_total)
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model3/{}'.format(cur_model_name))

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("the number of total parameters are {}.".format(total_parameters))
    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    
    
    
 