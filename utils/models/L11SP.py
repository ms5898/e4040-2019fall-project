import tensorflow as tf
import time

def SVHNmodel(input_x, input_y, label_length, is_training=False, drop_rate=0.5):
    if is_training == False:
        drop_rate = 0
    with tf.variable_scope('hidden_layer_1'):
        conv1 = tf.layers.conv2d(inputs=input_x, filters=48, kernel_size=[5, 5], padding='same')
        norm1 = tf.layers.batch_normalization(conv1)
        act1 = tf.nn.relu(norm1)
        pool1 = tf.layers.max_pooling2d(inputs=act1, pool_size=[2,2], strides=2, padding='same')
        drop1 = tf.layers.dropout(pool1, rate=drop_rate)
        layer1 = drop1

    with tf.variable_scope('hidden_layer_2'):
        conv2 = tf.layers.conv2d(inputs=layer1, filters=21, kernel_size=[5, 5], padding='same')
        norm2 = tf.layers.batch_normalization(conv2)
        act2 = tf.nn.relu(norm2)
        pool2 = tf.layers.max_pooling2d(inputs=act2, pool_size=[2,2], strides=1, padding='same')
        drop2 = tf.layers.dropout(pool2, rate=drop_rate)
        layer2 = drop2

    with tf.variable_scope('hidden_layer_3'):
        conv3 = tf.layers.conv2d(inputs=layer2, filters=42, kernel_size=[5, 5], padding='same')
        norm3 = tf.layers.batch_normalization(conv3)
        act3 = tf.nn.relu(norm3)
        pool3 = tf.layers.max_pooling2d(inputs=act3, pool_size=[2,2], strides=2, padding='same')
        drop3 = tf.layers.dropout(pool3, rate=drop_rate)
        layer3 = drop3

    with tf.variable_scope('hidden_layer_4'):
        conv4 = tf.layers.conv2d(inputs=layer3, filters=53, kernel_size=[5, 5], padding='same')
        norm4 = tf.layers.batch_normalization(conv4)
        act4 = tf.nn.relu(norm4)
        pool4 = tf.layers.max_pooling2d(inputs=act4, pool_size=[2,2], strides=1, padding='same')
        drop4 = tf.layers.dropout(pool4, rate=drop_rate)
        layer4 = drop4

    with tf.variable_scope('hidden_layer_5'):
        conv5 = tf.layers.conv2d(inputs=layer4, filters=64, kernel_size=[5, 5], padding='same')
        norm5 = tf.layers.batch_normalization(conv5)
        act5 = tf.nn.relu(norm5)
        pool5 = tf.layers.max_pooling2d(inputs=act5, pool_size=[2,2], strides=2, padding='same')
        drop5 = tf.layers.dropout(pool5, rate=drop_rate)
        layer5 = drop5

    with tf.variable_scope('hidden_layer_6'):
        conv6 = tf.layers.conv2d(inputs=layer5, filters=64, kernel_size=[5, 5], padding='same')
        norm6 = tf.layers.batch_normalization(conv6)
        act6 = tf.nn.relu(norm6)
        pool6 = tf.layers.max_pooling2d(inputs=act6, pool_size=[2,2], strides=1, padding='same')
        drop6 = tf.layers.dropout(pool6, rate=drop_rate)
        layer6 = drop6

    with tf.variable_scope('hidden_layer_7'):
        conv7 = tf.layers.conv2d(inputs=layer6, filters=64, kernel_size=[5, 5], padding='same')
        norm7 = tf.layers.batch_normalization(conv7)
        act7 = tf.nn.relu(norm7)
        pool7 = tf.layers.max_pooling2d(inputs=act7, pool_size=[2,2], strides=2, padding='same')
        drop7 = tf.layers.dropout(pool7, rate=drop_rate)
        layer7 = drop7

    with tf.variable_scope('hidden_layer_8'):
        conv8 = tf.layers.conv2d(inputs=layer7, filters=64, kernel_size=[5, 5], padding='same')
        norm8 = tf.layers.batch_normalization(conv8)
        act8 = tf.nn.relu(norm8)
        pool8 = tf.layers.max_pooling2d(inputs=act8, pool_size=[2,2], strides=1, padding='same')
        drop8 = tf.layers.dropout(pool8, rate=drop_rate)
        layer8 = drop8
        print(layer8.shape)

    # one locally connected hidden layer
    # The fully connected layers contain 3,072 units each
    flatten = tf.reshape(layer8, [-1, 1024]) #3072 get from the paper
    with tf.variable_scope('densely_hidden_layer_1'):
        dense1 = tf.layers.dense(inputs=flatten, units=2000, activation=tf.nn.relu)
        layer9 = dense1
    with tf.variable_scope('densely_hidden_layer_2'):
        dense2 = tf.layers.dense(inputs=layer9, units=2000, activation=tf.nn.relu)
        layer10 = dense2

    with tf.variable_scope('L'):
        fc1 = tf.layers.dense(inputs=layer10, units=7, name='fc1')
        L = fc1

    with tf.variable_scope('S1'):
        fc2 = tf.layers.dense(inputs=layer10, units=11, name='fc2')
        S1 = fc2

    with tf.variable_scope('S2'):
        fc3 = tf.layers.dense(inputs=layer10, units=11, name='fc3')
        S2 = fc3

    with tf.variable_scope('S3'):
        fc4 = tf.layers.dense(inputs=layer10, units=11, name='fc4')
        S3 = fc4

    with tf.variable_scope('S4'):
        fc5 = tf.layers.dense(inputs=layer10, units=11, name='fc5')
        S4 = fc5

    with tf.variable_scope('S5'):
        fc6 = tf.layers.dense(inputs=layer10, units=11, name='fc6')
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
        input_x = tf.placeholder(shape=[None, 54, 54, 3], dtype=tf.float32, name='input_x')
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
        writer = tf.summary.FileWriter("log11SP/{}".format(cur_model_name), sess.graph)
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
              
                if iter_total % 500 == 0:
                    o, merge_result = sess.run([output_label,merge], feed_dict={input_x: x_val[0:100], input_y: val_label[0:100],
                                                                                label_length: val_label_length[0:100],is_training:False})
                    error_num = 0
                    for i in range(100):
                        e = tf.count_nonzero(o[i] - input_label[i])
                        if i % 10 == 0:
                            print('=====is training=======')
                        if e.eval() != 0:
                            error_num += 1
                    tf.summary.scalar('SVHNmodel_error_num', error_num)
                    valid_eve = error_num
                    
                        
                    valid_acc = 100 - (valid_eve * 100)/100
                    #print(' validation accuracy iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                    writer.add_summary(merge_result, iter_total)
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model11SP/{}'.format(cur_model_name))
    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    
    
    
 