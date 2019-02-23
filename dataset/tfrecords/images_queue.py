import multiprocessing
from os.path import join, expanduser

from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from skimage import img_as_float

import tensorflow as tf


def get_prob_dist(img_list):
    yuv_converter = np.array([[0.299,0.587,0.114],[-0.14713,-0.2888,0.436],
                                  [0.615,-0.514999,-0.10001]])
    train = np.asarray(img_list, dtype=np.float64)
    train = img_as_float(train.astype('uint8'))
    train = train.astype('float32')
    img_YUV = img_float.dot(yuv_converter)
    prob_dist_batch = Prob_dist(img_YUV)
    return prob_dist_batch


def queue_single_images_from_folder(folder):
    # Normalize the path
    folder = expanduser(folder)

    # This queue will yield a filename every time it is polled
    file_matcher = tf.train.match_filenames_once(join(folder, '*.jpeg'))
    '''
    with tf.Session() as sess:
        # Run the variable initializer.
        sess.run(file_matcher.initializer)
        filenames = file_matcher.read_value().eval()
        print(len(filenames))
        print(filenames)
    img_list = []
    for f in filenames:
        im = Image.open(f)
        img = np.array(im)
        img_list.append(img)
        im.close()
    prob_dist = get_prob_dist(img_list)
    print(prob_dist)
    import sys
    sys.exit(1)
    '''


    # NOTE: if num_epochs is set to something different than None, then we
    # need to run tf.local_variables_initializer when launching the session!!
    # https://www.tensorflow.org/api_docs/python/tf/train/string_input_producer
    filename_queue = tf.train.string_input_producer(
        file_matcher, shuffle=False, num_epochs=1)

    # This is the reader we'll use to read each image given the file name
    image_reader = tf.WholeFileReader()

    # This operation polls the queue and reads the image
    #print(filename_queue.names)
    image_key, image_file = image_reader.read(filename_queue) 



    # get numpy array of image file
    #image_arr = img_to_array(image_file)
    #print(type(image_file))
    #print(type(tf.Session().run(tf.constant(tf.stack([image_file])))))


    # The file needs to be decoded as image and we also need its dimensions
    image_tensor = tf.image.decode_jpeg(image_file)
    image_shape = tf.shape(image_tensor)
    #print(type(tf.Session().run(tf.constant(image_tensor, shape=[3,], dtype='int32'))))
    #print(image_shape)



    # Note: nothing has happened yet, we've only defined operations,
    # what we return are tensors
    return image_key, image_tensor, image_shape


def batch_operations(operations, batch_size):
    """
    Once you have created the operation(s) with the other methods of this class,
    use this method to batch it(them).

    :Note:

        If a single queue operation is `[a, b, c]`,
        the batched queue_operation will be `[[a1, a2], [b1,b2], [c1, c2]]`
        and not `[[a1, b1, c1], [a2, b2, c3]]`

    :param operations: can be a tensor or a list of tensors
    :param batch_size: the batch
    :return:
    """
    # Recommended configuration for these parameters (found online)
    num_threads = multiprocessing.cpu_count()
    min_after_dequeue = 3 * batch_size
    capacity = min_after_dequeue + (num_threads + 1) * batch_size
    return tf.train.batch(
        operations,
        batch_size,
        num_threads,
        capacity,
        dynamic_pad=True,
        allow_smaller_final_batch=True,
    )
