#for loading the ImageNet data
import numpy as np
from PIL import Image
import glob
import os
from os.path import expanduser, join
from skimage import img_as_float
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pathlib import Path
import scipy


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


def get_weight(filename: str):
    # Check if filename already extracted with weights
    f = open('weights_filenames.txt', 'a+')
    f.close()
    if filename not in open('weights_filenames.txt').read():
        print('{} not in weights_filenames.txt'.format(filename))
        # print filename to weights_filenames.txt
        f = open('weights_filenames.txt', 'a+')
        f.write(filename + '\n')
        f.close()

        image_list = []
        directory = join(expanduser('~'), 'imagenet/resized/')
        pathlist = Path(directory).glob(filename)
        for file in pathlist:
        #for filename in os.listdir(directory):
            #print(directory)
            #print(filename)
            #file = os.path.join(directory, filename)
            print(file)
            im = Image.open(file)
            img = np.array(im)
            image_list.append(img)
            im.close()
        #print(len(image_list))
        train = np.asarray(image_list, dtype=np.float64)
        train = img_as_float(train.astype('uint8'))
        train = train.astype('float32')
        if os.path.isfile('./weights.npy'):
            image_ndarr = np.load('weights.npy')
            train = np.concatenate((image_ndarr, train), axis=0)
        print(train.shape)
        print(type(train))
        #show_images(train[:16])
        np.save('weights.npy', train)
        print('Weights saved to weights.npy')


def NN_ab(y):
    # y is [N, H, W, 3]
    NN_ab_x = np.round((y[:,:,:,1]+0.6)*19/1.2)
    NN_ab_y = np.round((y[:,:,:,2]+0.6)*19/1.2)
    NN_ab = NN_ab_x*20+NN_ab_y
    return NN_ab.astype(int)


def assign_bin(y):
    # Returns the ab bin value for a given training batch
    # y is [N, H, W, 3] dim
    # NN is [N, H, W] dim
    NN = NN_ab(y)
    return NN


def get_weighting_factor():
    X_train = np.load('weights.npy')
    yuv_converter = np.array([[0.299,0.587,0.114],[-0.14713,-0.2888,0.436],
                                          [0.615,-0.514999,-0.10001]])
    dist = np.zeros([313])
    X_YUV = X_train.dot(yuv_converter)
    ab = assign_bin(X_YUV)
    np.add.at(dist, ab, 1)
    dist = dist/np.sum(dist)
    normalized = scipy.ndimage.filters.gaussian_filter(dist, 5, order=0)
    print(np.sum(normalized))
    lamda = 0.5
    inv_wgt = (1-lamda)*normalized + lamda/313
    wgt = 1/inv_wgt
    np.save('weighting_factor.npy', wgt)
    plt.imshow(wgt.reshape([20,20]))
    plt.show()


# Run from the top folder as:
# python3 -m dataset.weights <args>
if __name__ == '__main__':
    import argparse
    from dataset.shared import dir_originals, dir_resized

    # Argparse setup
    filename = '*.jpeg'
    parser = argparse.ArgumentParser(
        description='get weights of resized images from a folder of 299x299 image size')
    parser.add_argument('-f', '--file',
                        default=filename,
                        type=str,
                        metavar='FILE',
                        dest='source',
                        help='use FILE to determine which files to get weights (default: {})'
                        .format(filename))
    '''
    parser.add_argument('-o', '--output-folder',
                        default=dir_resized,
                        type=str,
                        metavar='FOLDER',
                        dest='output',
                        help='use FOLDER as destination (default: {})'
                        .format(dir_resized))
    '''

    args = parser.parse_args()
    #get_weight(args.source)
    get_weighting_factor()
