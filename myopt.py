import tensorflow as tf
import myio
import numpy as np
import metric
from sklearn.metrics import accuracy_score
import random
from sklearn.svm import SVC

#batch module
#model module
#training module (load module)
#test module
#save module

def tf_mil_conv_detection(X_train , y_train, Gazes_train, X_test, y_test, Gazes_test,
             nb_iter, train_batch_size,gamma):
    with tf.Graph().as_default(): 

        batch_X_train, batch_y_train, batch_Gazes_train, batch_X_test, batch_y_test, batch_Gazes_test = \
        myio.batch_batch(X_train, y_train, Gazes_train, X_test, y_test, Gazes_test, train_batch_size)    
              
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
    
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        
        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    
        def max_pool(x):
            return tf.nn.max_pool(x, ksize=[1, len(X_train[0]), 1, 1],
                            strides=[1, 1, 1, 1], padding='VALID')
        x = tf.placeholder(tf.float32, shape=[None, len(X_train[0]), 2048, 1],name="input")
        y_ = tf.placeholder(tf.float32, shape=[None, 2],name="gt_output")
        gaze_ = tf.placeholder(tf.float32, shape=[None, len(X_train[0])],name="gaze")
        W_conv1 = weight_variable([1, 2048, 1, 2])
    
        b_conv1 = bias_variable([2])
        with tf.name_scope("conv1"):
            keep_prob = tf.placeholder(tf.float32)
            x_drop = tf.nn.dropout(x, keep_prob)
            h_conv1 = conv2d(x_drop, W_conv1) + b_conv1 
        with tf.name_scope("pool1"):
            h_pool1 = max_pool(h_conv1)
            
        max_instance_index = tf.argmax(h_conv1, dimension=1)
        max_instance_index = tf.reshape(max_instance_index, [-1,2])
        positive_max_instance_index = tf.slice(max_instance_index,[0,1],[train_batch_size,1])
        positive_max_instance_index = tf.reshape(positive_max_instance_index,[-1])
        flatten_positive_max_instance_index = tf.range(0,train_batch_size) * len(X_train[0])+tf.to_int32(positive_max_instance_index)
        max_scored_region_gaze = tf.gather(tf.reshape(gaze_,[-1]), flatten_positive_max_instance_index)
        # can not define a flexible train_batch_size?
        positive_label=tf.reshape(tf.slice(y_,[0,1],[train_batch_size,1]),[-1])
        positive_max_scored_region_gaze = tf.mul(positive_label, max_scored_region_gaze)
    #     max_gaze = tf.reduce_max(gaze_,reduction_indices=1)
    #     gaze_ratio = tf.div(max_region_gaze,max_gaze)
        h_pool1_flat = tf.reshape(h_pool1, [-1, 2])
        with tf.name_scope("logits1"):
            logits = tf.nn.softmax(h_pool1_flat)
        class_weight=[1,1]
        weighted_logits = tf.mul(logits, class_weight)
        with tf.name_scope("loss"):
            train_loss_mean = tf.reduce_mean(
                        -tf.reduce_sum(y_ * tf.log(weighted_logits),
                                       reduction_indices=[1])
                        +tf.mul(tf.constant(gamma),tf.reduce_sum(positive_max_scored_region_gaze))
                        )
            test_loss_mean = tf.reduce_mean(
                        -tf.reduce_sum(y_ * tf.log(weighted_logits),
                                       reduction_indices=[1])
                        )
            tf.scalar_summary("train_loss", train_loss_mean)
            tf.scalar_summary("test_loss", test_loss_mean)
        train_op = tf.train.AdamOptimizer().minimize(train_loss_mean)
        correct_prediction = tf.equal(tf.argmax(logits, 1),tf.argmax(y_, 1) )
        with tf.name_scope("acc1"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary("accuracy", accuracy)
    #     with tf.name_scope("score"):
    #         score = tf.reshape(tf.slice(weighted_logits,[0,1],[len(X_test),1]),[-1])
        with tf.Session() as sess:
            writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph) # for 0.8
            merged = tf.merge_all_summaries()
            tf.initialize_all_variables().run()
            for i in range(nb_iter):
                mean_train_acc=0
                mean_train_lm=0
                for j in range(np.shape(batch_X_train)[0]):
                    ori_shape=np.shape(batch_X_train[j])
                    bxt=np.reshape(batch_X_train[j],(ori_shape[0],ori_shape[1],ori_shape[2],1))
                    _, acc, pmsrg, pl, msrg, tlm= sess.run([
                                        train_op,accuracy,
                                        positive_max_scored_region_gaze,
                                        positive_label,
                                        max_scored_region_gaze,
                                        train_loss_mean
                                        ], feed_dict={
                                                      x:bxt,
                                                      gaze_:batch_Gazes_train[j],
                                                      y_:batch_y_train[j],
                                                      keep_prob: 0.5})
    #                 print acc,mii,pmii,fpmii,mrg,mg,gr,batch_Gazes_train[j]
                    mean_train_lm+=tlm
                    mean_train_acc+=acc
                print "train epoch:%d \t gamma:%f \t mean_acc:%f \t mean_loss:%f"%\
                    (i,gamma,mean_train_acc/np.shape(batch_X_train)[0],mean_train_lm)
                ori_shape=np.shape(batch_X_test[0])
                bxtest=np.reshape(batch_X_test[0],(ori_shape[0],ori_shape[1],ori_shape[2],1))
                summary, acc, lm, wl= sess.run(
                        [merged, accuracy, test_loss_mean, weighted_logits],feed_dict={
                        x:bxtest,
                        gaze_:batch_Gazes_test[0],
                        y_:batch_y_test[0],
                        keep_prob: 1.0}
                        )
                ap=metric.getAP(zip(batch_y_test[0][:,1],wl[:,1]))
                writer.add_summary(summary, i)
                f=open("results/jumping_%f_test_score_%d.txt"%(gamma,i),'w')
                np.savetxt(f,wl)
                f.close()
                print "test \t loss_mean:%f \t acc:%f \t map:%f"%(lm, acc, ap)


def tf_mlp_gaze_reg_universal(X_train , y_train, Gazes_train,
                    X_val, y_val, Gazes_val,
                    X_test, y_test, Gazes_test,
                    SAVE_PATH, MAX_NB_ITER, BATCH_SIZE, trainbool, category):
    
    with tf.Graph().as_default(): 
        batch_X_train, batch_y_train, batch_Gazes_train, batch_X_val, batch_y_val, batch_Gazes_val = \
        myio.batch_batch(X_train, y_train, Gazes_train, X_val, y_val, Gazes_val, BATCH_SIZE)    
        #total number of instances for one example
        instance_number = np.shape(X_test)[1]
        def save_model(saver,sess,save_path):
            path = saver.save(sess, save_path)
            print 'model save in %s'%path
        
        def model(x, w_h, w_o, b_h, b_o):
            h = tf.nn.sigmoid(tf.matmul(x, w_h)+b_h)
            return tf.matmul(h, w_o) + b_o
        # lr is just X*w so this model line is pretty simple
        W1 = tf.Variable(tf.random_normal([2048, 1000], stddev=0.01,dtype=tf.float64))
        W2 = tf.Variable(tf.random_normal([1000, 1],stddev=0.01,dtype=tf.float64))
        b1 = tf.Variable(tf.zeros([1, 1000],dtype=tf.float64))
        b2 = tf.Variable(tf.zeros([1, 1],dtype=tf.float64))
    #     w = tf.get_variable("w1", [28*28, 10])
        x = tf.placeholder(tf.float64, shape=[None, 2048],name="input")
    #     y_ = tf.placeholder(tf.float32, shape=[None, 2],name="gt_output")
        gaze_ = tf.placeholder(tf.float64, shape=[None, 1],name="gaze")
        
        gaze_pred = model(x, W1, W2, b1, b2)
        loss_mean = tf.reduce_mean(tf.square(gaze_pred - gaze_)) # use square error for cost function
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_mean)
        
        best_val_err=100000
        EARLY_STOP_PATIENCE=10
        current_epoch=0
        min_nb_iter=50
        saver = tf.train.Saver()
        init_op = tf.initialize_all_variables()
    
        # Launch the graph in a session
        with tf.Session() as sess:
            
            if trainbool:
                sess.run(init_op)
                
                for i in range(MAX_NB_ITER):
                    mean_train_lm=0
                    batches = range(np.shape(batch_X_train)[0])
                    random.shuffle(batches)
                    for j in batches:
                        _, loss_train = sess.run([train_op, loss_mean], feed_dict={x:batch_X_train[j],
                                                                                   gaze_:batch_Gazes_train[j]})
                        mean_train_lm += loss_train
                    
                    print "epoch:%d, mean_train_loss:%f"%(i, mean_train_lm/np.shape(batch_X_train)[0])
                    loss_val = sess.run(loss_mean, feed_dict={x:batch_X_val[0],
                                                   gaze_:batch_Gazes_val[0]})
                    print "epoch:%d, mean_val_loss:%f"%(i, loss_val)
                    if loss_val < best_val_err:
                        print "save model of epoch:%d"%i
                        current_epoch = i
                        best_val_err = loss_val
                        save_model(saver, sess, SAVE_PATH)
                    elif i > min_nb_iter and (i - current_epoch) >= EARLY_STOP_PATIENCE:
                        print 'early stopping at epoch %d'%i
                        break    
            
            if not trainbool:
                print "load model from %s"%SAVE_PATH
                saver.restore(sess, SAVE_PATH)
            
            def data_formatter(X, y, gaze_pred, sess):
                
                X = np.reshape(X, [-1, 2048])
                gaze_pred = sess.run(gaze_pred, feed_dict={x:X})
                gaze_pred = np.reshape(gaze_pred, [-1, instance_number, 1])
                selector =  np.argmin(gaze_pred, 1).transpose()+np.arange(len(gaze_pred))*instance_number
                X = X[selector, :]
                X = np.reshape(X, [-1,2048])
                y = np.where(y==1)[1]
                X = np.insert(X, 2048, 1, axis=1)
                return X, y
            X_train_clf, y_train_clf = data_formatter(X_train, y_train, gaze_pred, sess)
            X_test_clf, y_test_clf = data_formatter(X_test, y_test, gaze_pred, sess)
            
            from sklearn.svm import SVC
            clf = SVC(C=1000, gamma=0.0001, verbose=False,)
            clf.fit(X_train_clf, y_train_clf)
            
    #         print np.shape(X_train)
    #         print np.shape(np.where(y_train==1)[1])
            train_clf_score = clf.decision_function(X_train_clf)
            test_clf_score = clf.decision_function(X_test_clf)
    
            train_ap=metric.getAP(zip(y_train_clf, train_clf_score))
            print "(best epoch) train ap:%f"%train_ap
            test_ap=metric.getAP(zip(y_test_clf, test_clf_score))
            print "(best epoch) test ap:%f"%test_ap
            f= open("/home/wangxin/code/py_deep_gaze/results/ap_res.txt","a+")
            f.write(" ".join([SAVE_PATH, category, str(train_ap), str(test_ap)])+'\n')
            f.close()

def tf_mlp_multiclass_clf(X_train , y_train, Gazes_train,
                    X_val, y_val, Gazes_val,
                    X_test, y_test, Gazes_test,
                    SAVE_PATH, MAX_NB_ITER, BATCH_SIZE, trainbool, category, category_number):
    """
    regression and classification in the same time, label of the region is the image label
    """
    with tf.Graph().as_default(): 
        batch_X_train, batch_y_train, batch_Gazes_train, batch_X_val, batch_y_val, batch_Gazes_val = \
        myio.batch_batch(X_train, y_train, Gazes_train, X_val, y_val, Gazes_val, BATCH_SIZE)    
        #total number of instances for one example
        instance_number = np.shape(X_test)[1]
        def save_model(saver,sess,save_path):
            path = saver.save(sess, save_path)
            print 'model save in %s'%path
        def model(x, w_h, w_o, b_h, b_o):
            h = tf.nn.relu(tf.matmul(x, w_h)+b_h)
            output = tf.nn.softmax(tf.matmul(h, w_o) + b_o)
            
            return output
        def evaluation(X, y, gaze_pred, sess):
            X = np.reshape(X, [-1, 2048])
            gaze_pred = sess.run(gaze_pred, feed_dict={x:X})
            gaze_pred = np.reshape(gaze_pred, [-1, instance_number*category_number])
#             ap=metric.getAP(zip(y, np.min(gaze_pred,axis=1)))
            label_pred =  np.argmax(gaze_pred,axis=1)%category_number
            y = np.where(y==1)[1]

            if np.shape(y)[0] == np.shape(label_pred)[0] * instance_number:
                """The training dataset needs to reshape label vector,
                ,but the testset (organised as a bag) does not need this."""
                y = y[::instance_number]
            return accuracy_score(label_pred, y)
        # lr is just X*w so this model line is pretty simple
        W1 = tf.Variable(tf.random_normal([2048, 1000], stddev=0.01,dtype=tf.float64))
        W2 = tf.Variable(tf.random_normal([1000, category_number],stddev=0.01,dtype=tf.float64))
        b1 = tf.Variable(tf.zeros([1, 1000],dtype=tf.float64))
        b2 = tf.Variable(tf.zeros([1, category_number],dtype=tf.float64))
    #     w = tf.get_variable("w1", [28*28, 10])
        x = tf.placeholder(tf.float64, shape=[None, 2048],name="input")
        y_ = tf.placeholder(tf.float64, shape=[None, category_number],name="gt_output")
        gaze_ = tf.placeholder(tf.float64, shape=[None, category_number],name="gaze")
        gaze_pred = model(x, W1, W2, b1, b2)
        
#         regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) +
#                   tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
#       cross entropy  
        
        loss_mean = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(gaze_pred), reduction_indices=[1]))
#         L2-loss
#         loss_mean = tf.reduce_mean(tf.pow(gaze_-gaze_pred, 2))
#         loss_mean += regularizers*5e-4
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_mean)

        best_val_acc= -1
        EARLY_STOP_PATIENCE=10
        current_epoch=0
        min_nb_iter=50
        saver = tf.train.Saver()
        init_op = tf.initialize_all_variables()
        # Launch the graph in a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            with tf.device('/cpu:0'):
                if trainbool:
                    sess.run(init_op)
                    print "begin training"
                    for i in range(MAX_NB_ITER):
                        mean_train_lm=0
                        batches = range(np.shape(batch_X_train)[0])
                        random.shuffle(batches)
                        for j in batches:
                            _, loss_train = sess.run([train_op, loss_mean], feed_dict={x:batch_X_train[j],
                                                                                       gaze_:batch_Gazes_train[j],
                                                                                       y_:batch_y_train[j]})
                            mean_train_lm += loss_train
                        
                        print "epoch:%d, mean_train_loss:%f"%(i, mean_train_lm/np.shape(batch_X_train)[0])
                        loss_val = sess.run(loss_mean, feed_dict={x:batch_X_val[0],
                                                       gaze_:batch_Gazes_val[0],
                                                       y_:batch_y_val[0]})
                        print "epoch:%d, mean_val_loss:%f"%(i, loss_val)
                        train_acc = evaluation(X_train, y_train, gaze_pred, sess)
                        print "epoch:%d, acc_train:%f"%(i, train_acc,)
                        val_acc = evaluation(X_val, y_val, gaze_pred, sess)
                        print "epoch:%d, acc_val:%f"%(i, val_acc)
                        test_acc = evaluation(X_test, y_test, gaze_pred, sess)
                        print "epoch:%d, acc_test:%f"%(i, test_acc,)
                        
                        if val_acc > best_val_acc:
                            print "save model of epoch:%d"%i
                            current_epoch = i
                            best_val_acc = val_acc
                            save_model(saver, sess, SAVE_PATH)
                        elif i > min_nb_iter and (i - current_epoch) >= EARLY_STOP_PATIENCE:
                            print 'early stopping at epoch %d'%i
                            break    
                        
                    f= open("/local/wangxin/results/upmc_food/tf_mlp_multiclass_clf/ap_res.txt","a+")
                    f.write(" ".join([SAVE_PATH, category, str(train_acc), str(val_acc), str(test_acc)])+'\n')
                    f.close()
                if not trainbool:
                    print "load model from %s"%SAVE_PATH
                    saver.restore(sess, SAVE_PATH)

def tf_mlp_gaze_regclf_multiclass(X_train , y_train, Gazes_train,
                    X_val, y_val, Gazes_val,
                    X_test, y_test, Gazes_test,
                    SAVE_PATH, MAX_NB_ITER, BATCH_SIZE, trainbool, category, category_number):
    """regression and classification in the same time, label of each region is the gaze density"""

    with tf.Graph().as_default(): 
        batch_X_train, batch_y_train, batch_Gazes_train, batch_X_val, batch_y_val, batch_Gazes_val = \
        myio.batch_batch(X_train, y_train, Gazes_train, X_val, y_val, Gazes_val, BATCH_SIZE)    
        #total number of instances for one example
        instance_number = np.shape(X_test)[1]
        def save_model(saver,sess,save_path):
            path = saver.save(sess, save_path)
            print 'model save in %s'%path
        def model(x, w_h, w_o, b_h, b_o):
            h = tf.nn.relu(tf.matmul(x, w_h)+b_h)
            output = tf.nn.softmax(tf.matmul(h, w_o) + b_o)
#             output = tf.matmul(h, w_o) + b_o
            return output
        def evaluation(X, y, gaze_pred, sess):
            X = np.reshape(X, [-1, 2048])
            gaze_pred = sess.run(gaze_pred, feed_dict={x:X})
            gaze_pred = np.reshape(gaze_pred, [-1, instance_number*category_number])
            label_pred =  np.argmax(gaze_pred,axis=1)%category_number
            y = np.where(y==1)[1]
            return accuracy_score(label_pred, y)
        # lr is just X*w so this model line is pretty simple
        W1 = tf.Variable(tf.random_normal([2048, 1000], stddev=0.01,dtype=tf.float64))
        W2 = tf.Variable(tf.random_normal([1000, category_number],stddev=0.01,dtype=tf.float64))
        b1 = tf.Variable(tf.zeros([1, 1000],dtype=tf.float64))
        b2 = tf.Variable(tf.zeros([1, category_number],dtype=tf.float64))
    #     w = tf.get_variable("w1", [28*28, 10])
        x = tf.placeholder(tf.float64, shape=[None, 2048],name="input")
    #     y_ = tf.placeholder(tf.float32, shape=[None, 2],name="gt_output")
        gaze_ = tf.placeholder(tf.float64, shape=[None, category_number],name="gaze")
        gaze_pred = model(x, W1, W2, b1, b2)
        
#         regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) +
#                   tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
#       cross entropy  
        loss_mean = tf.reduce_mean(-tf.reduce_sum(gaze_ * tf.log(gaze_pred) + tf.pow(gaze_-gaze_pred, 2), reduction_indices=[1]))
#         L2-loss
#         loss_mean = tf.reduce_mean(tf.pow(gaze_-gaze_pred, 2))
#         loss_mean += regularizers*5e-4
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_mean)

        best_val_acc=-1
        EARLY_STOP_PATIENCE=10
        current_epoch=0
        min_nb_iter=50
        saver = tf.train.Saver()
        init_op = tf.initialize_all_variables()
        # Launch the graph in a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if trainbool:
                sess.run(init_op)
                print "begin training"
                for i in range(MAX_NB_ITER):
                    mean_train_lm=0
                    batches = range(np.shape(batch_X_train)[0])
                    random.shuffle(batches)
                    for j in batches:
                        _, loss_train = sess.run([train_op, loss_mean], feed_dict={x:batch_X_train[j],
                                                                                   gaze_:batch_Gazes_train[j]})
                        mean_train_lm += loss_train
                    
                    print "epoch:%d, mean_train_loss:%f"%(i, mean_train_lm/np.shape(batch_X_train)[0])
                    loss_val = sess.run(loss_mean, feed_dict={x:batch_X_val[0],
                                                   gaze_:batch_Gazes_val[0]})
                    print "epoch:%d, mean_val_loss:%f"%(i, loss_val)
                    train_acc = evaluation(X_train, y_train, gaze_pred, sess)
                    print "epoch:%d, acc_train:%f"%(i, train_acc,)
                    val_acc = evaluation(X_val, y_val, gaze_pred, sess)
                    print "epoch:%d, acc_val:%f"%(i, val_acc)
                    test_acc = evaluation(X_test, y_test, gaze_pred, sess)
                    print "epoch:%d, acc_test:%f"%(i, test_acc,)
                    
                    if val_acc > best_val_acc:
                        print "save model of epoch:%d"%i
                        current_epoch = i
                        best_val_acc = val_acc
                        save_model(saver, sess, SAVE_PATH)
                    elif i > min_nb_iter and (i - current_epoch) >= EARLY_STOP_PATIENCE:
                        print 'early stopping at epoch %d'%i
                        break    
                    
            if not trainbool:
                print "load model from %s"%SAVE_PATH
                saver.restore(sess, SAVE_PATH)
            



# multiclass classification for scale 100
# def tf_mlp_multiclass_clf(X_train_100 , y_train_100, Gazes_train,
#                     X_val_100, y_val_100, Gazes_val,
#                     X_test_100, y_test_100, Gazes_test,
#                     SAVE_PATH, MAX_NB_ITER, BATCH_SIZE, trainbool, category, category_number):
#     
#     with tf.Graph().as_default(): 
#         batch_X_train, batch_y_train, batch_Gazes_train, batch_X_val, batch_y_val, batch_Gazes_val = \
#         myio.batch_batch(X_train_100, y_train_100, Gazes_train, X_val_100, y_val_100, Gazes_val, BATCH_SIZE)    
#         #total number of instances for one example
#         instance_number = np.shape(X_test_100)[1]
#         def save_model(saver,sess,save_path):
#             path = saver.save(sess, save_path)
#             print 'model save in %s'%path
#         def model(x, w_h, w_o, b_h, b_o):
#             h = tf.nn.relu(tf.matmul(x, w_h)+b_h)
#             output = tf.nn.softmax(tf.matmul(h, w_o) + b_o)
# #             output = tf.matmul(h, w_o) + b_o
#             return output
#         def evaluation(X, y, sess):
#             X = np.reshape(X, [-1, 2048])
#             y = np.where(y==1)[1]
#             print y.get_shape()
#             print pred_label.get_shape()
#             return accuracy_score(pred_label, y)
#         # lr is just X*w so this model line is pretty simple
#         W1 = tf.Variable(tf.random_normal([2048, 1000], stddev=0.01,dtype=tf.float64))
#         W2 = tf.Variable(tf.random_normal([1000, category_number],stddev=0.01,dtype=tf.float64))
#         b1 = tf.Variable(tf.zeros([1, 1000],dtype=tf.float64))
#         b2 = tf.Variable(tf.zeros([1, category_number],dtype=tf.float64))
#     #     w = tf.get_variable("w1", [28*28, 10])
#         x = tf.placeholder(tf.float64, shape=[None, 2048],name="input")
#     #     y_ = tf.placeholder(tf.float32, shape=[None, 2],name="gt_output")
#         label = tf.placeholder(tf.float64, shape=[None, category_number],name="pred_label")
#         pred_label = model(x, W1, W2, b1, b2)
#         
# #         regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) +
# #                   tf.nn.l2_loss(b1) + tf.nn.l2_loss(b2))
# #       cross entropy  
#         loss_mean = tf.reduce_mean(-tf.reduce_sum(label * tf.log(pred_label), reduction_indices=[0]))
# #         L2-loss
# #         loss_mean = tf.reduce_mean(tf.pow(gaze_-gaze_pred, 2))
# #         loss_mean += regularizers*5e-4
#         train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_mean)
# 
#         best_val_acc=-1
#         EARLY_STOP_PATIENCE=10
#         current_epoch=0
#         min_nb_iter=50
#         saver = tf.train.Saver()
#         init_op = tf.initialize_all_variables()
#         # Launch the graph in a session
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#         with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#             if trainbool:
#                 sess.run(init_op)
#                 print "begin training"
#                 for i in range(MAX_NB_ITER):
#                     mean_train_lm=0
#                     batches = range(np.shape(batch_X_train)[0])
#                     random.shuffle(batches)
#                     for j in batches:
#                         _, loss_train = sess.run([train_op, loss_mean], feed_dict={x:batch_X_train[j],
#                                                                                    label:batch_y_train[j]})
#                         mean_train_lm += loss_train
#                     
#                     print "epoch:%d, mean_train_loss:%f"%(i, mean_train_lm/np.shape(batch_X_train)[0])
# #                     loss_val = sess.run(loss_mean, feed_dict={x:batch_X_val[0],
# #                                                               label:batch_y_train[0]})
# #                     print "epoch:%d, mean_val_loss:%f"%(i, loss_val)
#                     train_acc = evaluation(X_train_100, y_train_100, sess)
#                     print "epoch:%d, acc_train:%f"%(i, train_acc,)
#                     val_acc = evaluation(X_val_100, y_val_100, sess)
#                     print "epoch:%d, acc_val:%f"%(i, val_acc)
#                     test_acc = evaluation(X_test_100, y_test_100, sess)
#                     print "epoch:%d, acc_test:%f"%(i, test_acc,)
#                     
#                     if val_acc > best_val_acc:
#                         print "save model of epoch:%d"%i
#                         current_epoch = i
#                         best_val_acc = val_acc
#                         save_model(saver, sess, SAVE_PATH)
#                     elif i > min_nb_iter and (i - current_epoch) >= EARLY_STOP_PATIENCE:
#                         print 'early stopping at epoch %d'%i
#                         break    
#                     
#             if not trainbool:
#                 print "load model from %s"%SAVE_PATH
#                 saver.restore(sess, SAVE_PATH)
#             
# #             def data_formatter(X, y, gaze_pred, sess):
# #                 X = np.reshape(X, [-1, 2048])
# #                 gaze_pred = sess.run(gaze_pred, feed_dict={x:X})
# #                 gaze_pred = np.reshape(gaze_pred, [-1, instance_number, 1])
# #                 selector =  np.argmin(gaze_pred, 1).transpose()+np.arange(len(gaze_pred))*instance_number
# #                 X = X[selector, :]
# #                 X = np.reshape(X, [-1,2048])
# #                 y = np.where(y==1)[1]
# #                 X = np.insert(X, 2048, 1, axis=1)
# #                 return X, y
# #             X_train_clf, y_train_clf = data_formatter(X_train, y_train, gaze_pred, sess)
# #             X_test_clf, y_test_clf = data_formatter(X_test, y_test, gaze_pred, sess)
# #             
# #             from sklearn.svm import SVC
# #             clf = SVC(C=1000, gamma=0.0001, verbose=False,)
# #             clf.fit(X_train_clf, y_train_clf)
# #             
# #     #         print np.shape(X_train)
# #     #         print np.shape(np.where(y_train==1)[1])
# #             train_clf_score = clf.decision_function(X_train_clf)
# #             test_clf_score = clf.decision_function(X_test_clf)
# #     
# #             train_ap=metric.getAP(zip(y_train_clf, train_clf_score))
# #             print "(best epoch) train ap:%f"%train_ap
# #             test_ap=metric.getAP(zip(y_test_clf, test_clf_score))
# #             print "(best epoch) test ap:%f"%test_ap
# #             f= open("/home/wangxin/code/py_deep_gaze/results/ap_res.txt","a+")
# #             f.write(" ".join([SAVE_PATH, category, str(train_ap), str(test_ap)])+'\n')
# #             f.close()

# regssion only with the positive examples. Apply models learned on all categories for each image, then select the highest scored one.
def tf_mlp_gaze_reg_singleclass(X_train , y_train,Gazes_train,
                    X_val, y_val, Gazes_val,
                    X_test, y_test, Gazes_test,
                    SAVE_PATH, MAX_NB_ITER, BATCH_SIZE, trainbool, category,scale):
    
    with tf.Graph().as_default(): 
        if trainbool:
            batch_X_train, batch_y_train, batch_Gazes_train, batch_X_val, batch_y_val, batch_Gazes_val = \
            myio.batch_batch(X_train, y_train, Gazes_train, X_val, y_val, Gazes_val, BATCH_SIZE)    
        #total number of instances for one example
        instance_number = np.shape(X_test)[1]
        def save_model(saver,sess,save_path):
            path = saver.save(sess, save_path)
            print 'model save in %s'%path
        
        def model(x, w_h, w_o, b_h, b_o):
            h = tf.nn.sigmoid(tf.matmul(x, w_h)+b_h)
            return tf.matmul(h, w_o) + b_o
        # lr is just X*w so this model line is pretty simple
        W1 = tf.Variable(tf.random_normal([2048, 1000], stddev=0.01,dtype=tf.float64))
        W2 = tf.Variable(tf.random_normal([1000, 1],stddev=0.01,dtype=tf.float64))
        b1 = tf.Variable(tf.zeros([1, 1000],dtype=tf.float64))
        b2 = tf.Variable(tf.zeros([1, 1],dtype=tf.float64))
    #     w = tf.get_variable("w1", [28*28, 10])
        x = tf.placeholder(tf.float64, shape=[None, 2048],name="input")
    #     y_ = tf.placeholder(tf.float32, shape=[None, 2],name="gt_output")
        gaze_ = tf.placeholder(tf.float64, shape=[None, 1],name="gaze")
        
        gaze_pred = model(x, W1, W2, b1, b2)
        loss_mean = tf.reduce_mean(tf.square(gaze_pred - gaze_)) # use square error for cost function
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_mean)
        
        best_val_err=100000
        EARLY_STOP_PATIENCE=10
        current_epoch=0
        min_nb_iter=50
        saver = tf.train.Saver()
        init_op = tf.initialize_all_variables()
    
        # Launch the graph in a session
        with tf.Session() as sess:
            if trainbool:
                sess.run(init_op)
                for i in range(MAX_NB_ITER):
                    mean_train_lm=0
                    batches = range(np.shape(batch_X_train)[0])
                    random.shuffle(batches)
                    for j in batches:
                        _, loss_train = sess.run([train_op, loss_mean], feed_dict={x:batch_X_train[j],
                                                                                   gaze_:batch_Gazes_train[j]})
                        mean_train_lm += loss_train
                    
                    print "epoch:%d, mean_train_loss:%f"%(i, mean_train_lm/np.shape(batch_X_train)[0])
                    loss_val = sess.run(loss_mean, feed_dict={x:batch_X_val[0],
                                                   gaze_:batch_Gazes_val[0]})
                    print "epoch:%d, mean_val_loss:%f"%(i, loss_val)
                    if loss_val < best_val_err:
                        print "save model of epoch:%d"%i
                        current_epoch = i
                        best_val_err = loss_val
                        save_model(saver, sess, SAVE_PATH+'_'+category)
                    elif i > min_nb_iter and (i - current_epoch) >= EARLY_STOP_PATIENCE:
                        print 'early stopping at epoch %d'%i
                        break    
            
            if not trainbool:
                import os
                print "load model from %s"%(SAVE_PATH+'_'+category)
                model_root= "/local/wangxin/models/food/tf_mlp_gaze_reg_classwise_clf_binary/"+str(scale)+"/"
                for m in os.listdir(model_root):
                    if m.endswith(category):
                        print (model_root+m)
                        saver.restore(sess, (model_root+m))
                        def data_formatter(X, y, gaze_pred, sess):
                            X = np.reshape(X, [-1, 2048])
                            gaze_pred = sess.run(gaze_pred, feed_dict={x:X})
                            gaze_pred = np.reshape(gaze_pred, [-1, instance_number])
                            gaze_pred = np.max(gaze_pred, axis=1)
                            f=open("/local/wangxin/results/upmc_food/tf_mlp_gaze_reg_classwise_clf_binary/labels"+"/"+category+".txt","w")
                            np.savetxt(f,y)
                        data_formatter(X_test, y_test, gaze_pred, sess)



def tf_linear_gaze_reg(X_train , y_train, Gazes_train, X_test, y_test, Gazes_test,
               learning_rate, nb_iter,train_batch_size,):
    with tf.Graph().as_default(): 
        batch_X_train, batch_y_train, batch_Gazes_train, batch_X_test, batch_y_test, batch_Gazes_test = \
        myio.batch_batch(X_train, y_train, Gazes_train, X_test, y_test, Gazes_test, train_batch_size)    
        
        # train_batch_size is the maximum number of examples in a batch
        train_batch_example_number = len(batch_X_train[0])
        instance_number = len(X_train[0])
        train_X_ph = tf.placeholder(tf.float32, shape=[train_batch_example_number, instance_number, 2048],name="input")
        train_X_ph_instances = tf.reshape(train_X_ph, [train_batch_example_number*instance_number, 2048])
        train_y_ph = tf.placeholder(tf.float32, shape=[train_batch_example_number, 2],name="train_label_output")
        train_gaze_ph = tf.placeholder(tf.float32, shape=[train_batch_example_number, instance_number],name="train_gaze_output")
        train_gaze_ph_instance = tf.reshape(train_gaze_ph, [train_batch_example_number*instance_number])
        
        W = tf.Variable(tf.zeros([2048,1],dtype=tf.float32))
        b = tf.Variable(tf.zeros([1],dtype=tf.float32))
    #     w = tf.get_variable("w1", [28*28, 10])
    
        gaze_pred = tf.add(tf.matmul(train_X_ph_instances, W), b)
        loss_mean = tf.reduce_mean(tf.square(gaze_pred - train_gaze_ph_instance)) # use square error for cost function
    #     train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
        
        train_op = tf.train.AdamOptimizer().minimize(loss_mean)
        
        test_batch_example_number = len(batch_X_test[0])
        test_X_ph = tf.placeholder(tf.float32, shape=[test_batch_example_number, instance_number, 2048],name="test_input") # create symbolic variables
        test_X_ph_instances = tf.reshape(test_X_ph, [test_batch_example_number*instance_number, 2048])
        test_gaze_ph = tf.placeholder(tf.float32, shape=[test_batch_example_number, instance_number],name="test_gaze_output")
        test_gaze_ph_instance = tf.reshape(test_gaze_ph, [test_batch_example_number*instance_number])
        val_eval = tf.reduce_mean(tf.square(tf.add(tf.matmul(test_X_ph_instances, W), b) - test_gaze_ph_instance))
        
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            tf.train.start_queue_runners(sess=sess)
    
            for i in range(nb_iter):
                mean_train_acc=0
                mean_train_lm=0
                for j in range(np.shape(batch_X_train)[0]):
                    _, loss_train = sess.run([train_op, loss_mean], feed_dict={train_X_ph:batch_X_train[j],
                                                                               train_gaze_ph:batch_Gazes_train[j]
                                                                               })
                    mean_train_lm += loss_train
                print "epoch:%d, mean_train_loss:%f"%(i, mean_train_lm/np.shape(batch_X_train)[0])
                print batch_Gazes_test[0],
                val_eval = sess.run(val_eval, feed_dict={
                                                         test_gaze_ph:batch_Gazes_test[0],
                                                         test_X_ph:batch_X_test[0]
                                                         })
                print "epoch:%d, val_eval:%f"%(i, val_eval)
                
    #             f=open("results/linear/jumping_loc_score.txt",'a+')
    #             f.write(str(loss_train)+"\t"+str(loss_test)+"\n")
    #             f.close()    
                
            print np.shape(X_train)
            saver = tf.train.Saver()
            with tf.Session() as sess:
                saver.restore(sess, 'models/'+'test.model')
                sess.run(gaze_pred, feed_dict={train_X_ph:batch_X_train[0]})
            y_train = y_train[:,1]
            clf = SVC(C=100000.0)
            clf.fit(X_train, y_train)
            print "accuracy:"
            print clf.predict(X_test)
            print sum([1 for pred, gt in zip(clf.predict(X_test), y_test[:,1]) if pred==gt])/float(len(y_test[:,1]))


def multiclass_svm(X_train , y_train, Gazes_train,
                  X_val, y_val, Gazes_val, 
                  X_test, y_test, Gazes_test,):
    y_train = np.where(y_train==1)[1]
    for c in [0.01,0.1,1,10,100,1000,10000]:
        clf = SVC(C=c)
        clf.fit(X_train, y_train)
        X_test= np.reshape(X_test, (-1,2048))
        print accuracy_score(clf.predict(X_train), y_train)    
        print accuracy_score(clf.predict(X_val), np.where(y_val==1)[1])    
        print accuracy_score(clf.predict(X_test), np.where(y_test==1)[1])
        #70% max
if __name__=="__main__":
    import console_food_multiclass
    console_food_multiclass.main()
    