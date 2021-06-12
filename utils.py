"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import csv
import json
import random
import pprint
import scipy.misc
import numpy as np
from glob import glob
import os
#import matplotlib.pyplot as plt
from time import gmtime, strftime
from config import FACE_ALIGNMENT_RECROP_DIR, FACEWAREHOUSE_RECROP_DIR, DATA_DIR, VGG2_DATA_DIR


pp = pprint.PrettyPrinter()


def get_stddev(x, k_h, k_w): return 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def get_image(image_path, image_size, is_crop=True, is_random_crop=False, resize_w=64, is_grayscale=False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, is_random_crop, resize_w)


def save_images(images, size, image_path, inverse=True):
    if len(size) == 1:
        size = [size, -1]
    if size[1] == -1:
        size[1] = int(math.ceil(images.shape[0]/size[0]))
    if size[0] == -1:
        size[0] = int(math.ceil(images.shape[0]/size[1]))
    if (inverse):
        images = inverse_transform(images)

    return imsave(images, size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    nn = images.shape[0]

    if size[1] < 0:
        size[1] = int(math.ceil(nn/size[0]))
    if size[0] < 0:
        size[0] = int(math.ceil(nn/size[1]))

    if (images.ndim == 4):
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
    else:
        img = images

    return img


def imresize(img, sz):
    return scipy.misc.imresize(img, sz)


def imsave(images, size, path):
    img = merge(images, size)

    # plt.imshow(img)
    # plt.show()

    return scipy.misc.imsave(path, img)


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])


def random_crop(x, crop_h, crop_w=None, with_crop_size=None):
    if crop_w is None:
        crop_w = crop_h
    if with_crop_size is None:
        with_crop_size = False
    h, w = x.shape[:2]

    j = random.randint(0, h - crop_h)
    i = random.randint(0, w - crop_w)

    if with_crop_size:
        return x[j:j+crop_h, i:i+crop_w, :], j, i
    else:
        return x[j:j+crop_h, i:i+crop_w, :]


def crop(x, crop_h, crop_w, j, i):
    if crop_w is None:
        crop_w = crop_h

    return x[j:j+crop_h, i:i+crop_w]

    # return scipy.misc.imresize(x, [96, 96] )


def transform(image, npx=64, is_crop=True, is_random_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        if is_random_crop:
            cropped_image = random_crop(image, npx)
        else:
            cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth,
                      "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": [
                    '%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": [
                    '%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": [
                              '%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc", 
                        "sy": 1, "sx": 1, 
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": [
                              '%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv", 
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5,
                                     size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [8, 8], './samples/test_%s.png' %
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, [8, 8],
                        './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in [random.randint(0, 99) for _ in xrange(100)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 3:
        values = np.arange(0, 1, 1./config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1./config.batch_size)

        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            image_set.append(
                sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10])
                         for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def load_FaceAlignment_datasets_good():
    fd = open(FACE_ALIGNMENT_RECROP_DIR + 'good_images.dat')
    all_images = np.fromfile(file=fd, dtype=np.uint8)
    fd.close()
    all_images = all_images.astype(np.bool_)

    return all_images


def load_FaceAlignment_dataset(dataset):
    print('Loading ' + dataset + ' ...')

    fd = open('../data/FaceAlignment/'+dataset+'.dat')
    all_images = np.fromfile(file=fd, dtype=np.uint8)
    fd.close()
    all_images = all_images.reshape((-1, 100, 100, 3)).astype(np.uint8)

    fd = open('../data/FaceAlignment/'+dataset+'_param.dat')
    all_paras = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    mDim = 8
    poseDim = mDim + 7
    shapeDim = poseDim + 199
    expDim = shapeDim + 29
    texDim = expDim + 40

    all_paras = all_paras.reshape((-1, texDim)).astype(np.float32)
    m = all_paras[:, 0:mDim]
    pose = all_paras[:, mDim:poseDim]
    shape = all_paras[:, poseDim:shapeDim]
    exp = all_paras[:, shapeDim:expDim]
    tex = all_paras[:, expDim:texDim]

    assert (all_images.shape[0] == all_paras.shape[0]
            ), "Number of samples must be the same"
    # print all_images.shape[0]
    print('    DONE. Finish loading ' + dataset +
          ' with ' + str(all_images.shape[0]) + ' images')
    return all_images, m, pose, shape, exp, tex


def image2texture_fn(image_fn):

    last = image_fn[-7:].find('_')

    if (last < 0):
        return image_fn
    else:
        return image_fn[:-7 + last] + '_0.png'


def load_FaceAlignment_dataset_recrop_sz224(dataset, with_sh=False):
    print('Loading ' + dataset + ' ...')

    #dataset = 'IBUG'

    fd = open(FACE_ALIGNMENT_RECROP_DIR+dataset+'_filelist.txt', 'r')
    all_images = []
    for line in fd:
        all_images.append(line.strip())
    fd.close()
    print('    DONE. Finish loading ' + dataset +
          ' with ' + str(len(all_images)) + ' images')

    #all_images = np.fromfile(file=fd, dtype=np.uint8)
    # print(all_images.shape)
    # fd.close()
    #all_images = all_images.reshape((-1,256,256,3)).astype(np.uint8)

    fd = open(FACE_ALIGNMENT_RECROP_DIR+dataset+'_param.dat')
    all_paras = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    idDim = 1
    mDim = idDim + 8
    poseDim = mDim + 7
    shapeDim = poseDim + 199
    expDim = shapeDim + 29
    texDim = expDim + 40
    ilDim = texDim + 10
    #colorDim  = ilDim + 7

    all_paras = all_paras.reshape((-1, ilDim)).astype(np.float32)
    pid = all_paras[:, 0:idDim]
    m = all_paras[:, idDim:mDim]
    pose = all_paras[:, mDim:poseDim]
    shape = all_paras[:, poseDim:shapeDim]
    exp = all_paras[:, shapeDim:expDim]
    tex = all_paras[:, expDim:texDim]
    il = all_paras[:, texDim:ilDim]
    #color = all_paras[:,ilDim:colorDim]

    assert (len(all_images) ==
            all_paras.shape[0]), "Number of samples must be the same between images and paras"

    #all_textures = load_FaceAlignment_dataset_texture_sz224(dataset)
    #assert (all_images.shape[0] == all_textures.shape[0]),"Number of samples must be the same between images and texture"

    # if dataset != 'AFLW2000':
    #    pid = load_FaceAlignment_id_recrop(dataset)
    #    assert (all_images.shape[0] == pid.shape[0]),"Number of samples must be the same between images and ids"

    if with_sh:
        sh = load_FaceAlignment_il_recrop(dataset)
        assert (all_images.shape[0] == sh.shape[0]
                ), "Number of samples must be the same between images and sh coefficients"

        return all_images, pid, m, pose, shape, exp, tex, il, color, all_textures, sh
    return all_images, pid, m, pose, shape, exp, tex, il


def load_FaceAlignment_id_recrop(dataset):
    print('Loading ' + dataset + ' ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + ''+dataset+'_id.dat')
    all_paras = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    all_paras = all_paras.reshape((-1, 1)).astype(np.float32)

    return all_paras


VERTEX_NUM = 53215
TRI_NUM = 105840
N = VERTEX_NUM * 3

INDEXES = range(0, 53000, 25)


def load_FaceAlignment_basic(element, is_reduce=False):
    fn = FACE_ALIGNMENT_RECROP_DIR + '3DMM_'+element+'_basis.dat'
    print('Loading ' + fn + ' ...')

    fd = open(fn)
    all_paras = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    all_paras = np.transpose(all_paras.reshape((-1, N)).astype(np.float32))

    if is_reduce:
        _, sub_idxes_ = load_3DMM_sub_idxes_reduce()
        mu = all_paras[sub_idxes_, 0]
        w = all_paras[sub_idxes_, 1:]
    else:
        mu = all_paras[:, 0]
        w = all_paras[:, 1:]

    print('    DONE')

    return mu, w


def load_FaceAlignment_2dbasic(element):
    print('Loading ' + element + ' basis ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_'+element+'2d_basis.dat')
    all_paras = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    all_paras = all_paras.reshape((-1, 192*224*3)).astype(np.float32)
    # print(all_paras.shape)
    mu = all_paras[0:1]  # [indexes,0]
    w = all_paras[1:]  # [indexes,1:]

    print('    DONE!')
    print(w.shape)

    return mu, w


def load_FaceAlignment_vt2pixel():

    fd = open(FACE_ALIGNMENT_RECROP_DIR + 'vertices_2d_u.dat')
    vt2pixel_u = np.fromfile(file=fd, dtype=np.float32)
    vt2pixel_u = np.append(vt2pixel_u - 1, 0)
    fd.close()

    fd = open(FACE_ALIGNMENT_RECROP_DIR + 'vertices_2d_v.dat')
    vt2pixel_v = np.fromfile(file=fd, dtype=np.float32)
    vt2pixel_v = np.append(vt2pixel_v - 1, 0)  # vt2pixel_v[VERTEX_NUM] = 0
    fd.close()

    return vt2pixel_u, vt2pixel_v


def load_FaceAlignment_tri_2d(with_mask=False):
    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_tri_2d.dat')
    tri_2d = np.fromfile(file=fd, dtype=np.int32)
    fd.close()

    tri_2d = tri_2d.reshape(192, 224)

    tri_mask = tri_2d != 0

    tri_2d[tri_2d == 0] = TRI_NUM+1  # VERTEX_NUM + 1
    tri_2d = tri_2d - 1

    if with_mask:
        return tri_2d, tri_mask

    return tri_2d


def load_Normalized_2D_lnmk():

    print('Loading 2D landmarks ...')

    fd = open('data/FaceAlignment/3DMM_normalized_2D_lnmk_128_128.dat')
    lnmks = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    lnmks = lnmks.reshape((2, -1)).astype(np.float32)
    print('    DONE')
    return lnmks  # [:,INDEXES]   #Shape 2*N


def load_FaceAlignment_texture_mask():
    print('Loading texture mask ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + 'texture_mask.dat')
    texture_mask = np.fromfile(file=fd, dtype=np.uint8)
    fd.close()

    texture_mask = texture_mask.reshape((1, 64, 64, 1)).astype(np.float32)

    print('    DONE!')

    return texture_mask


def load_3DMM_tri():

    print('Loading 3DMM tri ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_tri.dat')
    tri = np.fromfile(file=fd, dtype=np.int32)
    fd.close()
    # print tri

    tri = tri.reshape((3, -1)).astype(np.int32)
    tri = tri - 1
    tri = np.append(tri, [[VERTEX_NUM], [VERTEX_NUM], [VERTEX_NUM]], axis=1)

    print('    DONE')
    return tri


def load_3DMM_vertex_tri():

    print('Loading 3DMM vertex tri ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_vertex_tri.dat')
    vertex_tri = np.fromfile(file=fd, dtype=np.int32)
    fd.close()
    # print tri

    vertex_tri = vertex_tri.reshape((8, -1)).astype(np.int32)
    #vertex_tri = np.append(vertex_tri, np.zeros([8,1]), 1)
    vertex_tri[vertex_tri == 0] = TRI_NUM + 1
    vertex_tri = vertex_tri - 1

    print('    DONE')
    return vertex_tri


def load_3DMM_kpts():

    print('Loading 3DMM keypoints ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_keypoints.dat')
    kpts = np.fromfile(file=fd, dtype=np.int32)
    kpts = kpts.reshape((-1, 1))
    fd.close()

    return kpts - 1

# Remesh


VERTEX_NUM_REDUCE = 39111
TRI_NUM_REDUCE = 77572
N_REDUCE = VERTEX_NUM_REDUCE * 3


def load_FaceAlignment_basic_remesh6k(element):
    print('Loading ' + element + ' basis ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_' +
              element+'_basis_remesh6k.dat')
    all_paras = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    all_paras = np.transpose(all_paras.reshape(
        (-1, N_remesh6k)).astype(np.float32))
    mu = all_paras[:, 0]
    w = all_paras[:, 1:]

    print(all_paras.shape)

    print('    DONE')
    return mu, w


def load_FaceAlignment_vt2pixel_reduce():
    fd = open(FACE_ALIGNMENT_RECROP_DIR + 'vertices_2d_u_reduce.dat')
    vt2pixel_u = np.fromfile(file=fd, dtype=np.float32)
    vt2pixel_u = np.append(vt2pixel_u, 0)
    fd.close()

    fd = open(FACE_ALIGNMENT_RECROP_DIR + 'vertices_2d_v_reduce.dat')
    vt2pixel_v = np.fromfile(file=fd, dtype=np.float32)
    vt2pixel_v = np.append(vt2pixel_v, 0)
    fd.close()
    return vt2pixel_u, vt2pixel_v


def load_3DMM_tri_reduce():
    print('Loading 3DMM tri ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_tri_reduce.dat')
    tri = np.fromfile(file=fd, dtype=np.int32)
    fd.close()
    # print tri

    tri = tri.reshape((3, -1)).astype(np.int32)
    tri = tri - 1
    tri = np.append(tri, [[VERTEX_NUM_REDUCE], [
                    VERTEX_NUM_REDUCE], [VERTEX_NUM_REDUCE]], axis=1)
    print('    DONE')
    return tri


def load_3DMM_vertex_tri_reduce():
    print('Loading 3DMM vertex tri ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_vertex_tri_reduce.dat')
    vertex_tri = np.fromfile(file=fd, dtype=np.int32)
    fd.close()

    vertex_tri = vertex_tri.reshape((8, -1)).astype(np.int32)
    vertex_tri[vertex_tri == 0] = TRI_NUM_REDUCE + 1
    vertex_tri = vertex_tri - 1

    print('    DONE')
    return vertex_tri


def load_FaceAlignment_tri_2d_reduce(with_mask=False):
    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_tri_2d_reduce.dat')
    tri_2d = np.fromfile(file=fd, dtype=np.int32)
    fd.close()

    tri_2d = tri_2d.reshape(192, 224)
    tri_mask = tri_2d != 0

    tri_2d[tri_2d == 0] = TRI_NUM_REDUCE+1  # VERTEX_NUM + 1
    tri_2d = tri_2d - 1

    if with_mask:
        return tri_2d, tri_mask

    return tri_2d


def load_FaceAlignment_tri_2d_barycoord_reduce():
    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_tri_2d_barycoord_reduce.dat')
    tri_2d_barycoord = np.fromfile(file=fd, dtype=np.float32)
    fd.close()

    tri_2d_barycoord = tri_2d_barycoord.reshape(192, 224, 3)

    return tri_2d_barycoord


def load_3DMM_sub_idxes_reduce():
    print('Loading 3DMM sub_idxes ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_sub_idxes_reduce.dat')
    sub_idxes = np.fromfile(file=fd, dtype=np.int32)
    fd.close()
    # print tri

    sub_idxes = sub_idxes - 1
    # print(sub_idxes[0:6])

    sub_idxes_ = np.reshape(sub_idxes*3, newshape=[1, -1])

    sub_idxes_ = np.concatenate([sub_idxes_, sub_idxes_ + 1, sub_idxes_ + 2])
    # print(sub_idxes_.shape)
    sub_idxes_ = np.reshape(np.transpose(sub_idxes_), newshape=[-1])
    # print(sub_idxes_.shape)
    # print(sub_idxes_[0:6])

    #tri = np.append(tri, [[ VERTEX_NUM_REDUCE], [VERTEX_NUM_REDUCE], [VERTEX_NUM_REDUCE]], axis = 1 )
    # print '    DONE'
    return sub_idxes, sub_idxes_


def load_3DMM_kpts_reduce():
    print('Loading 3DMM keypoints ...')

    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_keypoints_reduce.dat')
    kpts = np.fromfile(file=fd, dtype=np.int32)
    kpts = kpts.reshape((-1, 1))
    fd.close()
    return kpts - 1


def load_const_alb_mask():
    fd = open(FACE_ALIGNMENT_RECROP_DIR + '3DMM_const_alb_mask.dat')
    const_alb_mask = np.fromfile(file=fd, dtype=np.uint8)
    fd.close()
    const_alb_mask = const_alb_mask - 1
    const_alb_mask = const_alb_mask.reshape((-1, 2)).astype(np.uint8)

    return const_alb_mask


def RotationMatrix(phi, gamma, theta):
    # get rotation matrix by rotate angle
    R_x = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)],
                   [0, -np.sin(phi), np.cos(phi)]])
    R_y = np.array([[np.cos(gamma), 0, -np.sin(gamma)],
                   [0, 1, 0], [np.sin(gamma), 0, np.cos(gamma)]])
    R_z = np.array([[np.cos(theta), np.sin(theta), 0],
                   [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return np.matmul(np.matmul(R_x, R_y), R_z)


def pose2m(Pose_Para):
    phi = Pose_Para[0]
    gamma = Pose_Para[1]
    theta = Pose_Para[2]

    M = RotationMatrix(phi, gamma, theta)

    if len(Pose_Para) > 3:
        tx = Pose_Para[3]
        ty = Pose_Para[4]
        tz = Pose_Para[5]

        s = Pose_Para[6]

        M = np.concatenate((s*M, np.array([[tx], [ty], [tz]])), axis=1)

    return M


def crossp(x, y):

    z = np.zeros(x.shape)

    z[0, :] = x[1, :]*y[2, :] - x[2, :]*y[1, :]
    z[1, :] = x[2, :]*y[0, :] - x[0, :]*y[2, :]
    z[2, :] = x[0, :]*y[1, :] - x[1, :]*y[0, :]

    return z


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def compute_normal_np(vertex, face):
    # Unit normals to the faces
    normalf = crossp(vertex[:, face[1, :]]-vertex[:, face[0, :]],
                     vertex[:, face[2, :]]-vertex[:, face[0, :]])
    normalf = normalized(normalf, 0)

    nface = face.shape[1]
    nvert = vertex.shape[1]
    # Unit normals to the vetices
    normal = np.zeros((3, nvert), np.float32)
    count = 0
    for i in range(nface):
        f = face[:, i]
        for j in range(3):
            count += 1
            normal[:, face[j, i]] += normalf[:, i]
    normal = normalized(normal, 0)

    # enforce that the normal are outward
    v = vertex - np.mean(vertex, 0)
    s = np.sum(v*normal, 1)

    if np.sum(s > 0) < np.sum(s < 0):
        normal = -normal
        normalf = -normalf

    return normal, normalf


def get_FaceWarehouse_path(pid, pose, isMask=False):
    if isMask:
        part = 'masks'
    else:
        part = 'images'

    return '%s/%s/Tester_%d/TrainingPose/pose_%d.png' % (FACEWAREHOUSE_RECROP_DIR, part, pid, pose)


def get_FaceWarehouse_landmark(pid, pose):
    part = 'images'

    return '%s/%s/Tester_%d/TrainingPose/pose_%d.landmk' % (FACEWAREHOUSE_RECROP_DIR, part, pid, pose)


def load_CelebA_recrop_test():

    print('Loading CelebA recrop...')
    fd = open(FACE_ALIGNMENT_RECROP_DIR +
              '../CelebA/images/list_attr_celeba.txt')
    N = int(fd.readline().strip())
    attribute_names = fd.readline().strip().split()

    filenames = []
    attribute_values = []

    for line in fd:
        line = line.strip().split()
        filenames.append(line[0])
        attribute_values.append(line[1:])

    attribute_values = np.asarray(attribute_values).astype(np.int8)
    attribute_values = attribute_values > 0

    fd.close()
    return filenames, attribute_values, attribute_names


def load_database_by_list(txt, initial_path='', initial_id=0, threshold=None, fa_threshold=None):
    paths = []
    labels = []
    print("Opening " + txt + " ...")
    f = open(txt, "r")
    lines = f.readlines()
    for line in lines:
        line = line.split(';')
        is_good = True
        if threshold is not None:
            coeff = float(line[2])
            if coeff < threshold:
                is_good = False

        if fa_threshold is not None:
            fa_error = float(line[3])
            if fa_error > fa_threshold:
                is_good = False

        if is_good:
            paths.append(initial_path + line[0])
            labels.append(int(line[1]))

    f.close()

    print("Opened %d images" % len(paths))

    return paths, labels


def load_VGG2_by_list(txt=None, initial_path='./', initial_id=0, threshold=None, fa_threshold=None):
    txt = VGG2_DATA_DIR + 'VGG2_filelist_coeff_error.txt'
    return load_database_by_list(txt, initial_path, initial_id, threshold, fa_threshold)
