from keras import backend as K

import pickle
import time
from os.path import join

import matplotlib
import numpy as np
import cv2
from skimage import color
from PIL import Image, ImageChops

from dataset.shared import dir_tfrecord, dir_metrics, dir_checkpoints, dir_root, \
    maybe_create_folder
from dataset.tfrecords import LabImageRecordReader

# import datetime for clocking training speed per epoch
from datetime import datetime
prev_time = "00:00:00.000000"

# for loading the ImageNet data
import glob
from skimage import img_as_float

matplotlib.use('Agg')
matplotlib.rcParams['figure.figsize'] = (10.0, 4.0)
import matplotlib.pyplot as plt
import tensorflow as tf


labels_to_categories = pickle.load(
    open(join(dir_root, 'imagenet1000_clsid_to_human.pkl'), 'rb'))


def show_images(images):
    images = np.reshape(images, [images.shape[0], -1,3])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg,3]))
    plt.show()
    return


def get_weight():
    image_list = []
    for filename in glob.glob('~/imagenet/resized/*.jpeg'):
        im = Image.open(filename)
        img = np.array(im)
        image_list.append(img)
        im.close()
    train = np.asarray(image_list, dtype=np.float64)
    train = img_as_float(train.astype('uint8'))
    train = train.astype('float32')
    show_images(train[:16])


def loss_with_metrics(img_ab_out, img_ab_true, name=''):
    # Loss is mean square erros
    cost = tf.reduce_mean(
        tf.squared_difference(img_ab_out, img_ab_true), name="mse")
    # Metrics for tensorboard
    summary = tf.summary.scalar('cost ' + name, cost)
    return cost, summary


def categorical_crossentropy(y_pred, y_l, y_true_ab, wgt, name=''):

    n, h, w, q = y_pred.shape
    print(n, h, w, q)
    YY = tf.placeholder(tf.float32, [None, h, w, q])
    #tf.assign(YY, prob_dist_batch)
    #YY = y_true
    n1, h1, w1, q1 = YY.shape
    print(n1, h1, w1, q1)
    q2 = wgt.shape
    print(q2)
    n3, h3, w3, q3 = y_true.shape
    print(n3, h3, w3, q3)
    reshaped_Y = tf.reshape(YY, shape = [-1, q])
    reshaped_y_pred = tf.reshape(y_pred, shape = [-1, q])
    loss_temp = tf.nn.softmax_cross_entropy_with_logits_v2(logits=reshaped_y_pred,labels=reshaped_Y)
    weight_mat = tf.constant(wgt, shape=[q,])
    
    max_y = tf.argmax(YY, axis=3)
    print(max_y)
    y_reshaped = tf.reshape(max_y, shape=[tf.size(max_y)])
    print(y_reshaped)
    weight = tf.gather(weight_mat, y_reshaped)
    print(weight)

    weight = tf.reshape(weight, shape=[-1, h, w])
    loss_reshaped = tf.reshape(loss_temp, [-1, h, w])
    cost = tf.reduce_mean(weight * tf.cast(loss_reshaped, tf.float64))
    # Metrics for tensorboard
    summary = tf.summary.scalar('cost ' + name, cost)
    return cost, summary


def categorical_crossentropy_color(y_pred, y_true, name=''):

    # Flatten
    #print(type(y_pred))
    #print(type(y_true))
    n, h, w, q = y_true.shape#n, h, w, q = y_true.shape
    y_true = K.reshape(y_true, (n * h * w, q))
    y_pred = K.reshape(y_pred, (n * h * w, q))

    weights = y_true[:, 400:]  # extract weight from y_true
    weights = K.concatenate([weights] * 400, axis=1)
    y_true = y_true[:, :-1]  # remove last column
    y_pred = y_pred[:, :-1]  # remove last column

    # multiply y_true by weights
    y_true = y_true * weights

    cross_ent = K.categorical_crossentropy(y_pred, y_true)
    cross_ent = tf.reduce_mean(cross_ent, name="crossentropy")
    #cross_ent = K.mean(cross_ent, axis=-1)

    # Metrics for tensorboard
    summary = tf.summary.scalar('cost ' + name, cross_ent)

    return cross_ent, summary


def tensor_to_nparray(x):
    return x.eval()


def training_pipeline(col, learning_rate, batch_size, wgt):
    # Set up training (input queues, graph, optimizer)
    irr = LabImageRecordReader('lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(batch_size, shuffle=True)
    # read_batched_examples = irr.read_one()
    
    with tf.Session() as sess:
        f = read_batched_examples['image_name']
        sess.run(tf.Print(f.eval()))
    im = Image.open(f)
    img = np.array(im)
    #img_list.append(img)
    print(img)
    im.close()
    #prob_dist = get_prob_dist(img_list)
    #print(prob_dist)
    import sys
    sys.exit(1)
    
    imgs_l = read_batched_examples['image_l']
    imgs_true_ab = read_batched_examples['image_ab']
    imgs_emb = read_batched_examples['image_embedding']
    imgs_ab = col.build(imgs_l, imgs_emb)
    
    # merge imgs_l and imgs_true_ab as numpy array
    # get prob dist
    #tfe.py_func(my_py_func, [x], tf.float32)
    print(type(imgs_l))
    print(type(imgs_true_ab))
    y_l = tf.py_func(tensor_to_nparray, [imgs_l], tf.float32)
    #y_l = imgs_l.numpy()#tf.Session().run(tf.constant(imgs_l))
    print('l after conversion ')
    print(type(y_l))
    y_true_ab = tf.py_func(tensor_to_nparray, [imgs_true_ab], tf.float32)
    #y_true_ab = imgs_true_ab.numpy()#tf.Session().run(tf.constant(imgs_true_ab))
    print('ab after conversion ')
    print(type(y_true_ab))

    '''
    print(type(y_l))
    #y_l = tf.Session().run(tf.constant(y_l))
    y_l = y_l.numpy()
    print(type(y_l))
    y_true_ab = tf.Session().run(tf.constant(y_true_ab))
    y_true_ab = y_true_ab.numpy()
    print(type(y_true_ab))
    y_true = K.concatenate([y_l, y_true_ab], axis=2)
    samp = y_true
    print(type(samp))
    print(dtype(samp))
    '''

    '''
    samp_float = (samp)
    samp_YUV = samp_float.dot(yuv_converter)
    prob_dist_batch = Prob_dist(samp_YUV)
    #feed_dict = {Y: }
    '''

    cost, summary = categorical_crossentropy(imgs_ab, imgs_l, imgs_true_ab, wgt, 'training')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        cost, global_step=global_step)
    return {
        'global_step': global_step,
        'optimizer': optimizer,
        'cost': cost,
        'summary': summary
    }#, irr, read_batched_examples


def evaluation_pipeline(col, number_of_images, wgt):
    # Set up validation (input queues, graph)
    irr = LabImageRecordReader('val_lab_images_*.tfrecord', dir_tfrecord)
    read_batched_examples = irr.read_batch(number_of_images, shuffle=False)
    imgs_l_val = read_batched_examples['image_l']
    imgs_true_ab_val = read_batched_examples['image_ab']
    imgs_emb_val = read_batched_examples['image_embedding']
    imgs_ab_val = col.build(imgs_l_val, imgs_emb_val)
    cost, summary = categorical_crossentropy(imgs_ab_val, imgs_l_val, imgs_true_ab_val,
                                      wgt, 'validation')
    return {
        'imgs_l': imgs_l_val,
        'imgs_ab': imgs_ab_val,
        'imgs_true_ab': imgs_true_ab_val,
        'imgs_emb': imgs_emb_val,
        'cost': cost,
        'summary': summary
    }


def print_log(content, run_id):
    with open('output_{}.txt'.format(run_id), mode='a') as f:
        f.write('[{}] {}\n'.format(time.strftime("%c"), content))


def print_term(content, run_id, cost=None):
    global prev_time
    curr_time = datetime.now().strftime("%H:%M:%S.%f")
    FMT = '%H:%M:%S.%f'
    time_diff = datetime.strptime(curr_time, FMT) - datetime.strptime(prev_time, FMT) if "Global step" in content else ""
    # print('{}[{}][{}] {}\n'.format(run_id, time.strftime("%c"), time_diff, content))
    print_log(content, run_id)
    # write on the output_train_time_per_batch_*.txt file the train_time_time_per_batch or time_diff 
    if time_diff:
        # tf.summary.scalar('time_diff', time_diff)
        with open('output_train_time_per_batch_{}.txt'.format(run_id), mode='a') as f:
            f.write('{}\n'.format(time_diff))
    # if cost:
    #     with open('output_cost_{}.txt'.format(run_id), mode='a') as f:
    #         f.write('{}\n'.format(cost))
    prev_time = curr_time


def metrics_system(run_id, sess):
    # Merge all the summaries and set up the writers
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(join(dir_metrics, run_id), sess.graph)
    return train_writer


def checkpointing_system(run_id):
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    checkpoint_paths = join(dir_checkpoints, run_id)
    latest_checkpoint = tf.train.latest_checkpoint(dir_checkpoints)
    return saver, checkpoint_paths, latest_checkpoint


def image_colorfulness(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rgMean, rgStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rgStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rgMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)


def trim(im):
    im = im * 255
    im = Image.fromarray(im.astype('uint8'), 'RGB')
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def plot_evaluation(res, run_id, epoch, is_eval=False):
    maybe_create_folder(join(dir_root, 'images', run_id))
    for k in range(len(res['imgs_l'])):
        img_gray = l_to_rgb(res['imgs_l'][k][:, :, 0])
        img_output = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                                res['imgs_ab'][k])
        
        # save simple single image output
        if is_eval:
            im = trim(img_output)
            im.save(join(dir_root, 'images', run_id, '{}.png'.format(k)), "PNG")

        # display the colorfulness score on the image
        C_output = image_colorfulness(img_output)
        img_true = lab_to_rgb(res['imgs_l'][k][:, :, 0],
                              res['imgs_true_ab'][k])
        C_true = image_colorfulness(img_true)
        # display the cost function(MSE) output of the image
        cost = res['cost']

        plt.subplot(1, 3, 1)
        plt.imshow(img_gray)
        plt.title('Input (grayscale)')
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(img_output)
        plt.title('Network output\n' + ("{:.4f}".format(C_output)))
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(img_true)
        plt.title('Target (original)\n' + ("{:.4f}".format(C_true)))
        plt.axis('off')
        plt.suptitle('Cost(MSE): ' + str(cost), fontsize=7)

        plt.savefig(join(
            dir_root, 'images', run_id, '{}_{}.png'.format(epoch, k)))
        plt.clf()
        plt.close()

        # write on the output_colorfulness_*.txt file the colorfulness of the output image and the groundtruth image
        with open('output_colorfulness_{}.txt'.format(run_id), mode='a') as f:
            f.write('{},{}\n'.format(C_output, C_true))


def l_to_rgb(img_l):
    """
    Convert a numpy array (l channel) into an rgb image
    :param img_l:
    :return:
    """
    lab = np.squeeze(255 * (img_l + 1) / 2)
    return color.gray2rgb(lab) / 255


def lab_to_rgb(img_l, img_ab):
    """
    Convert a pair of numpy arrays (l channel and ab channels) into an rgb image
    :param img_l:
    :return:
    """
    lab = np.empty([*img_l.shape[0:2], 3])
    lab[:, :, 0] = np.squeeze(((img_l + 1) * 50))
    lab[:, :, 1:] = img_ab * 127
    return color.lab2rgb(lab)
