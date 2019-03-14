import os
import numpy as np
from tf_cnnvis import *
from keras import backend as K

from colorization import Colorization
from colorization.training_utils import evaluation_pipeline, \
    checkpointing_system, \
    plot_evaluation, training_pipeline, metrics_system, print_log, print_term, \
    load_imgs, plot_conv_output, put_kernels_on_grid

import tensorflow as tf


# PARAMETERS
run_id = 'run1'
epochs = 100  #default 100
val_number_of_images = 10
total_train_images = 2000#65000  #default 130 * 500
batch_size = 100  #default 100
learning_rate = 0.001
batches = total_train_images // batch_size

# START
print_term('Starting session...', run_id)
sess = tf.Session()
K.set_session(sess)
print_term('Started session...', run_id)

# Build the network and the various operations
print_term('Building network...', run_id)
col = Colorization(256)


#X = tf.placeholder(tf.float32, shape = [1, 2])


loaded_imgs = load_imgs(col, batch_size)
opt_operations = training_pipeline(col, learning_rate, batch_size)
evaluations_ops = evaluation_pipeline(col, val_number_of_images)
summary_writer = metrics_system(run_id, sess)
saver, checkpoint_paths, latest_checkpoint = checkpointing_system(run_id)
print_term('Built network', run_id)

with sess.as_default():
    # tf.summary.merge_all()
    # writer = tf.summary.FileWriter('./graphs', sess.graph)

    # Initialize
    print_term('Initializing variables...', run_id)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    print_term('Initialized variables', run_id)

    # Coordinate the loading of image files.
    print_term('Coordinating loaded image files...', run_id)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print_term('Coordinate loaded image files', run_id)

    # Restore
    if latest_checkpoint is not None:
        print_term('Restoring from: {}'.format(latest_checkpoint), run_id)
        saver.restore(sess, latest_checkpoint)
        print_term(' done!', run_id)
    else:
        print_term('No checkpoint found in: {}'.format(checkpoint_paths), run_id)

    # Actual training with epochs as iteration
    for epoch in range(epochs):
        print_term('Starting epoch: {} (total images {})'
                  .format(epoch, total_train_images), run_id)
        # Training step
        for batch in range(batches):
            print_term('Batch: {}/{}'.format(batch, batches), run_id)

            #??????????????????????????????
            '''
            print(sess.graph.get_tensor_by_name("input_1"))
            import sys
            sys.exit(1)
            '''
            '''
            for t in tf.get_default_graph().get_operations():
                if 'placeholder' in t.name:#t.type == "PlaceholderV2":
                    print(t.name)
                    print(type(t))
            import sys
            sys.exit(1)
            '''
            '''
            imgs = sess.run(loaded_imgs)
            #im = np.concatenate([imgs['imgs_l'], imgs['imgs_true_ab']], axis=3) 
            for t in tf.get_default_graph().get_operations():
                if 'input_1' in t.name:
                    print(t.name)
                    input_t = t
            #res = sess.run(opt_operations)
            layers = ['r', 'p', 'c']
            is_success = activation_visualization(sess_graph_path = None, value_feed_dict = {X: imgs['imgs_l']}, 
                                          input_tensor = input_t,
                                        layers=layers, path_logdir=os.path.join("Log","ResVAE"), 
                                          path_outdir=os.path.join("Output","ResVAE"))
            if is_success:
                print('viz success!')
                import sys
                sys.exit(1)
            '''
            #??????????????????????????????

            #for t in tf.get_default_graph().get_operations():

            '''
            res = sess.run(loaded_imgs)
            imgs_l = res['imgs_l']
            input_l = tf.get_collection('input_1')
            print(type(input))
            conv_out, input_l = sess.run(tf.get_collection('conv_output'), feed_dict={input_l: imgs_l})
            for i, c in enumerate(conv_out[0]):
                print(i)
            '''
            #res, conv_out = sess.run([opt_operations, tf.get_collection('conv_output')])
            #for i, c in enumerate(conv_out[0]):
            #    print(i)
            '''
            input_l = tf.get_default_graph().get_tensor_by_name('input_1:0')
            print(input_l)
            res = sess.run(opt_operations)
            conv_out, input_l = sess.run(tf.get_collection('conv_output'), feed_dict={input_l: res['imgs_l']})
            for i, c in enumerate(conv_out[0]):
                print(i)
            '''


            '''
            #????????????????????????????????????????????? Convolution visualization
            input_l = tf.get_default_graph().get_tensor_by_name('input_1:0') #Gets input tensor
            #print(input_l)
            res = sess.run(loaded_imgs)
            imgs_l = res['imgs_l']
            #print(imgs_l.shape)
            imgs_ab = res['imgs_ab']
            imgs = np.concatenate([imgs_l, imgs_ab], axis=3)
            #print(imgs[0,50:70,50:70,0])
            #print(imgs.shape)
            #x = np.arange(4).reshape(2,2)
            #print(x)
            #print(X)
            #print(x.shape)
            layers = ['r', 'p', 'c']
            
            
            #is_success = activation_visualization(sess_graph_path = None, value_feed_dict = {input_l: imgs_l}, #X: x}, 
            #    input_tensor = None, layers=layers, path_logdir=os.path.join("Log","ResVAE"), 
            #    path_outdir=os.path.join("Output","ResVAE"))
            
            input_2 = tf.get_default_graph().get_tensor_by_name('input_2:0')
            opts = tf.get_default_graph().get_operations()
            for o in opts:
                if 'model_1/conv2d_9/Relu' in o.name:
                    conv2d_9_tensor = o.values()
                    print(conv2d_9_tensor)
            #import sys
            #sys.exit(1)
            #conv2d_9_tensor = tf.get_default_graph().get_tensor_by_name('conv2d_9:0')
            inp2 = sess.run(conv2d_9_tensor)
            print(inp2[0].shape)
            #import sys
            #sys.exit(1)
            is_success = activation_visualization(sess_graph_path = None, value_feed_dict = {input_l: imgs_l, input_2: inp2[0]}, #X: x}, 
                input_tensor = input_l, layers=layers, path_logdir=os.path.join("Log","ResVAE"), 
                path_outdir=os.path.join("Output","ResVAE"))

            if is_success:
                print('viz success!')
                import sys
                sys.exit(1)
            #conv_out = tf.get_collection('conv_output')
            #for i, c in enumerate(conv_out[0]):
            #    print(i)
            #????????????????????????????????????????????? Convolution visualization
            '''

            #>>>>>>>>>>>>>>>>>>>> activations viz separate attempt
            #conv_output = tf.get_collections('')
            #>>>>>>>>>>>>>>>>>>>>

            res = sess.run(opt_operations)
            global_step = res['global_step']
            print_term('Cost: {} Global step: {}'
                      .format(res['cost'], global_step), run_id, res['cost'])
            summary_writer.add_summary(res['summary'], global_step)

        # Save the variables to disk
        save_path = saver.save(sess, checkpoint_paths, global_step)
        print_term("Model saved in: %s" % save_path, run_id)

        # Evaluation step on validation
        res = sess.run(evaluations_ops)
        summary_writer.add_summary(res['summary'], global_step)
        plot_evaluation(res, run_id, epoch)

        # get output of all convolutional layers
        # here we need to provide an input image

        #input = sess.run([tf.get_collection('')])
        '''
        conv_out = sess.run([tf.get_collection('conv_output')])#, feed_dict={x: mnist.test.images[:1]})
        for i, c in enumerate(conv_out[0]):
            print(i)
            #plot_conv_output(c, 'conv{}'.format(i))
        '''
        #weights = []

        for t_var in tf.trainable_variables():
            if t_var.name.endswith('kernel:0'):
                #weights.append(t_var)
                print(t_var)
                grid = put_kernels_on_grid (t_var)
                summary = tf.summary.image(t_var.name, grid, max_outputs=1)
                summary = sess.run(summary)
                summary_writer.add_summary(summary, global_step)
                break

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
