from __future__ import division
from natsort import natsorted, ns
from utils import *
from ops import *
from rendering_ops import *
from six.moves import xrange
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from math import floor
from random import randint
import random
import csv
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from progress.bar import Bar


TRI_NUM = 105840
VERTEX_NUM = 53215

VERTEX_NUM_REDUCE = 39111


WARPING = True

CONST_PIXELS_NUM = 10


class DCGAN(object):
    def __init__(self, sess, config, devices=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        #self.is_crop = is_crop
        self.c_dim = config.c_dim

        self.is_partbase_albedo = config.is_partbase_albedo

        self.is_2d_normalize = True

        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.sample_size = config.batch_size  # sample_size
        self.output_size = 224  # output_size
        self.input_size = 224  # output_size
        self.texture_size = [192, 224]
        self.z_dim = config.z_dim
        self.gf_dim = config.gf_dim
        self.df_dim = config.df_dim
        self.gfc_dim = config.gfc_dim
        self.dfc_dim = config.dfc_dim

        self.shape_loss = config.shape_loss if hasattr(
            config, 'shape_loss') else "l2"
        self.tex_loss = config.tex_loss if hasattr(
            config, 'tex_loss') else "l1"

        self.mDim = 8
        self.poseDim = 7
        self.shapeDim = 199
        self.expDim = 29
        self.texDim = 40
        self.ilDim = 9 * 3
        self.is_reduce = config.is_reduce

        if self.is_reduce:
            self.vertexNum = VERTEX_NUM_REDUCE
        else:
            self.vertexNum = VERTEX_NUM

        self.landmark_num = 68

        # batch normalization : deals with poor initialization helps gradient flow
        self.bns = {}

        self.g2_bn0_0 = batch_norm(name='g_h2_bn0_0')
        self.g2_bn0_1 = batch_norm(name='g_h2_bn0_1')
        self.g2_bn0_2 = batch_norm(name='g_h2_bn0_2')

        self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
        self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
        self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')
        self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
        self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
        self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
        self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
        self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
        self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
        self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
        self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
        self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
        self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
        self.g1_bn4 = batch_norm(name='g_h_bn4')
        self.g1_bn5 = batch_norm(name='g_h_bn5')

        self.g1_bn6 = batch_norm(name='g_s_bn6')
        self.dataset_name = config.dataset
        self.checkpoint_dir = config.checkpoint_dir

        self.devices = devices
        self.setupParaStat()
        self.setupValData()
        self.build_model()

    def build_model(self):

        self.m_labels = tf.placeholder(
            tf.float32, [self.batch_size, self.mDim], name='m_labels')
        self.shape_labels = tf.placeholder(
            tf.float32, [self.batch_size, self.vertexNum * 3], name='shape_labels')
        self.texture_labels = tf.placeholder(tf.float32, [
                                             self.batch_size, self.texture_size[0], self.texture_size[1], self.c_dim], name='tex_labels')
        self.exp_labels = tf.placeholder(
            tf.float32, [self.batch_size, self.expDim], name='exp_labels')
        self.il_labels = tf.placeholder(
            tf.float32, [self.batch_size, self.ilDim], name='lighting_labels')

        def filename2image(input_filenames, offset_height=None, offset_width=None, target_height=None, target_width=None):
            print('filename2image info', offset_height, offset_width, target_height, target_width)
            batch_size = len(input_filenames)
            if offset_height != None:
                offset_height = tf.split(offset_height, batch_size)
                offset_width = tf.split(offset_width, batch_size)

            images = []
            for i in range(batch_size):
                file_contents = tf.read_file(input_filenames[i])
                image = tf.image.decode_image(file_contents, channels=3)
                image.set_shape((256, 256, 3))
                if offset_height != None:
                    image = tf.image.crop_to_bounding_box(image, tf.reshape(
                        offset_height[i], []), tf.reshape(offset_width[i], []), target_height, target_width)

                images.append(image)
            return tf.cast(tf.stack(images), tf.float32)

        self.input_offset_height = tf.placeholder(
            tf.int32, [self.batch_size], name='input_offset_height')
        self.input_offset_width = tf.placeholder(
            tf.int32, [self.batch_size], name='input_offset_width')
        self.input_images_fn = [tf.placeholder(
            dtype=tf.string) for _ in range(self.batch_size)]
        self.input_masks_fn = [tf.placeholder(
            dtype=tf.string) for _ in range(self.batch_size)]
        self.input_images = filename2image(self.input_images_fn, offset_height=self.input_offset_height,
                                           offset_width=self.input_offset_width, target_height=self.output_size, target_width=self.output_size)
        self.input_images = self.input_images / 127.5 - 1
        self.input_masks = filename2image(self.input_masks_fn,  offset_height=self.input_offset_height,
                                          offset_width=self.input_offset_width, target_height=self.output_size, target_width=self.output_size)
        self.input_masks = self.input_masks / 255.0

        self.texture_labels_fn = [tf.placeholder(
            dtype=tf.string) for _ in range(self.batch_size)]
        self.texture_labels = filename2image(self.texture_labels_fn)
        self.texture_labels = self.texture_labels / 127.5 - 1

        self.texture_masks_fn = [tf.placeholder(
            dtype=tf.string) for _ in range(self.batch_size)]
        self.texture_mask_labels = filename2image(self.texture_masks_fn)
        self.texture_mask_labels = self.texture_mask_labels / 255.0

        # Networks
        self.input_images_ph = tf.placeholder(
            tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim])
        self.input_masks_ph = tf.placeholder(
            tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim])
        self.shape_fx, self.alb_fx, self.m, self.il = self.generator_encoder(
            self.input_images_ph, is_reuse=False, is_training=False)
        self.shape_fx_id = tf.slice(
            self.shape_fx, begin=[0, 0], size=[-1, 130])
        self.shape_fx_exp = tf.slice(
            self.shape_fx, begin=[0, 130], size=[-1, -1])

        #self.shape_fx = tf.concat([self.shape_fx_id, tf.zeros_like(self.shape_fx_exp)], axis = 1)

        self.shape_base, self.shape = self.generator_decoder_shape(
            self.shape_fx, is_reuse=False, is_training=False)
        self.albedo_base, self.albedo = self.generator_decoder_albedo(
            self.alb_fx, is_reuse=False, is_training=False)

        self.m_full = self.m * self.std_m_tf + self.mean_m_tf
        self.shape_full = self.shape * self.std_shape_tf + self.mean_shape_tf
        self.shape_base_full = self.shape_base * self.std_shape_tf + self.mean_shape_tf

        self.shape_labels_full = self.shape_labels * \
            self.std_shape_tf + self.mean_shape_tf
        self.m_labels_full = self.m_labels * self.std_m_tf + self.mean_m_tf

        # self.m_full = self.m_labels_full ##### AFLW2000 only

        # Rendering
        self.shade_base = generate_shade(
            self.il, self.m_full, self.shape_base_full, self.texture_size, is_reduce=self.is_reduce)
        self.shade = generate_shade(
            self.il, self.m_full, self.shape_full, self.texture_size, is_reduce=self.is_reduce)
        self.rrotated_shape = rotate_shape(
            self.m_full, self.shape_full, output_size=self.output_size)
        self.rrotated_shape_base = rotate_shape(
            self.m_full, self.shape_base_full, output_size=self.output_size)

        self.texture = 2.0*tf.multiply((self.albedo + 1.0)/2.0, self.shade) - 1
        self.texture = tf.clip_by_value(self.texture, -1, 1)

        self.texture_base = 2.0 * \
            tf.multiply((self.albedo_base + 1.0)/2.0, self.shade_base) - 1
        self.texture_base = tf.clip_by_value(self.texture_base, -1, 1)

        #self.G_texture_images, self.G_images_mask = warp_texture(self.texture, self.m_full, self.shape_full, output_size=self.output_size, is_reduce = self.is_reduce)

        # base
        pixel_u, pixel_v, self.G_images_base_mask = warping_flow(
            self.m_full, self.shape_base_full, output_size=self.output_size, is_reduce=self.is_reduce)
        self.G_images_base_mask = tf.expand_dims(self.G_images_base_mask, -1)

        self.G_texture_base_images = bilinear_sampler(
            self.texture, pixel_v, pixel_u)
        self.G_texture_base_images = tf.multiply(
            self.G_texture_base_images, self.G_images_base_mask) + (1 - self.G_images_base_mask)

        self.G_shade_base_images = bilinear_sampler(
            self.shade_base, pixel_v, pixel_u)
        self.G_shade_base_images = tf.multiply(
            self.G_shade_base_images, self.G_images_base_mask) + tf.multiply(self.input_images_ph, 1 - self.G_images_base_mask)

        self.G_albedo_base_images = bilinear_sampler(
            self.albedo_base, pixel_v, pixel_u)
        self.G_albedo_base_images = tf.multiply(
            self.G_albedo_base_images, self.G_images_base_mask) + (1 - self.G_images_base_mask)

        self.G_images_base_nc = tf.multiply(self.G_texture_base_images, self.G_images_base_mask) + \
            tf.multiply(self.input_images_ph, 1 - self.G_images_base_mask)

        self.G_images_base_mask = tf.multiply(
            self.input_masks_ph, self.G_images_base_mask)
        self.G_images_base = tf.multiply(self.G_texture_base_images, self.G_images_base_mask) + \
            tf.multiply(self.input_images_ph, 1 - self.G_images_base_mask)

        # Full

        pixel_u, pixel_v, self.G_images_mask = warping_flow(
            self.m_full, self.shape_full, output_size=self.output_size, is_reduce=self.is_reduce)
        self.G_images_mask = tf.expand_dims(self.G_images_mask, -1)

        self.G_texture_images = bilinear_sampler(
            self.texture, pixel_v, pixel_u)
        self.G_texture_images = tf.multiply(
            self.G_texture_images, self.G_images_mask) + (1 - self.G_images_mask)

        self.G_shade_images = bilinear_sampler(self.shade, pixel_v, pixel_u)
        self.G_shade_images = tf.multiply(
            self.G_shade_images, self.G_images_mask) + tf.multiply(self.input_images_ph, 1 - self.G_images_mask)

        self.G_albedo_images = bilinear_sampler(self.albedo, pixel_v, pixel_u)
        self.G_albedo_images = tf.multiply(
            self.G_albedo_images, self.G_images_mask) + (1 - self.G_images_mask)

        self.G_images_nc = tf.multiply(self.G_texture_images, self.G_images_mask) + \
            tf.multiply(self.input_images_ph, 1 - self.G_images_mask)

        self.G_images_mask = tf.multiply(
            self.input_masks_ph, self.G_images_mask)
        self.G_images = tf.multiply(self.G_texture_images, self.G_images_mask) + \
            tf.multiply(self.input_images_ph, 1 - self.G_images_mask)
        #self.G_shade_images    = tf.multiply(self.G_shade_images, self.G_images_mask) + (1 - self.G_images_mask) ##

        #self.landmark_u, self.landmark_v = compute_landmarks(self.m_full, self.shape_full, output_size=self.output_size, is_reduce = self.is_reduce)
        #self.landmark_u_labels, self.landmark_v_labels = compute_landmarks(self.m_labels_full, self.shape_labels_full, output_size=self.output_size, is_reduce = self.is_reduce)

        # New exp
        self.unwarped_texture, self.test_masks = unwarp_texture(
            self.input_images_ph, self.m_full, self.shape_full, output_size=self.output_size, is_reduce=self.is_reduce)

        # Sampler

        # var
        self.img_var = tf.get_variable("img_var__", [
                                       1, self.output_size, self.output_size, self.c_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.img_mask_var = tf.get_variable("img_mask_var__", [
                                            1, self.output_size, self.output_size, self.c_dim], dtype=tf.float32, initializer=tf.zeros_initializer())

        self.texture_mask_var = tf.get_variable("tex_mask_var__", [
                                                1, self.texture_size[0], self.texture_size[1], self.c_dim], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.texture_var = tf.get_variable("tex_var__", [
                                           1, self.texture_size[0], self.texture_size[1], self.c_dim], dtype=tf.float32, initializer=tf.zeros_initializer())

        self.alb_feature_var = tf.get_variable("alb_fx_var__", [1, int(
            self.gfc_dim/2)], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.shape_var = tf.get_variable("shape_var__", [
                                         1, self.vertexNum*3], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.il_var = tf.get_variable(
            "il_var__", [1, self.ilDim], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.m_var = tf.get_variable(
            "m_var__", [1, self.mDim], dtype=tf.float32, initializer=tf.zeros_initializer())

        self.pred_albedo_base, self.pred_albedo = self.generator_decoder_albedo(
            (self.alb_feature_var+1e-6),  is_reuse=True, is_training=False)

        self.shape_feature_var = tf.get_variable("shape_fx_var__", [1, int(
            self.gfc_dim/2)], dtype=tf.float32, initializer=tf.zeros_initializer())
        self.pred_shape_base, self.pred_shape = self.generator_decoder_shape(
            self.shape_feature_var, is_reuse=True, is_training=False)

        self.pred_shape_final = self.pred_shape * \
            self.std_shape_tf + self.mean_shape_tf
        self.pred_shape_base_final = self.pred_shape_base * \
            self.std_shape_tf + self.mean_shape_tf

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.g_en_vars = [var for var in t_vars if 'g_k' in var.name]
        self.g_tex_de_vars = [var for var in t_vars if 'g_h' in var.name]
        self.g_shape_de_vars = [var for var in t_vars if 'g_s' in var.name]

        self.drgan_vars = [var for var in t_vars if (
            (var in self.g_en_vars and 'g_k42' not in var.name and 'g_k5' not in var.name and 'g_k6' not in var.name and 'g_k_bn5_' not in var.name))]  # \
        #                                        or   (var in self.g_tex_de_vars and 'g_h5'  not in var.name and 'g_h4' not in var.name                                                           )) ]

        self.saver = tf.train.Saver([var for var in tf.global_variables(
        ) if 'var__' not in var.name], keep_checkpoint_every_n_hours=1, max_to_keep=10)

        self.drgan_saver = tf.train.Saver(
            self.drgan_vars, keep_checkpoint_every_n_hours=1, max_to_keep=0)

        # def name_in_checkpoint(var):
        #    print(var.op.name)
        #    print('d' + var.op.name[1:])
        #    return 'd' + var.op.name[1:]

        #variables_to_restore = {name_in_checkpoint(var):var for var in self.g_en_vars}
        #self.g_en_saver = tf.train.Saver(variables_to_restore, keep_checkpoint_every_n_hours=1, max_to_keep = 0)
        #self.g_de_saver = tf.train.Saver(self.g_de_vars, keep_checkpoint_every_n_hours=1, max_to_keep = 0)

    def setupParaStat(self):

        if self.is_reduce:
            self.tri = load_3DMM_tri_reduce()
            self.vertex_tri = load_3DMM_vertex_tri_reduce()
            self.vt2pixel_u, self.vt2pixel_v = load_FaceAlignment_vt2pixel_reduce()
            self.uv_tri, self.uv_mask = load_FaceAlignment_tri_2d_reduce(
                with_mask=True)
        else:
            self.tri = load_3DMM_tri()
            self.vertex_tri = load_3DMM_vertex_tri()
            self.vt2pixel_u, self.vt2pixel_v = load_FaceAlignment_vt2pixel()
            self.uv_tri, self.uv_mask = load_FaceAlignment_tri_2d(
                with_mask=True)

        self.mean_m = np.load('mean_m.npy')
        self.std_m = np.load('std_m.npy')

        # Basis
        mu_shape, w_shape = load_FaceAlignment_basic(
            'shape', is_reduce=self.is_reduce)
        mu_exp, w_exp = load_FaceAlignment_basic(
            'exp', is_reduce=self.is_reduce)

        self.mean_shape = mu_shape + mu_exp
        if self.is_2d_normalize:
            #self.mean_shape = np.tile(np.array([0, 0, 6e4]), VERTEX_NUM)
            self.std_shape = np.tile(np.array([1e4, 1e4, 1e4]), self.vertexNum)
        else:
            #self.mean_shape = np.load('mean_shape.npy')
            self.std_shape = np.load('std_shape.npy')

        self.mean_shape_tf = tf.constant(self.mean_shape, tf.float32)
        self.std_shape_tf = tf.constant(self.std_shape, tf.float32)

        self.mean_m_tf = tf.constant(self.mean_m, tf.float32)
        self.std_m_tf = tf.constant(self.std_m, tf.float32)

        self.w_shape = w_shape
        self.w_exp = w_exp

    def setupValData(self):
        # Samples data - AFLW200
        self.images_AFLW2000, pid_AFLW2000, m_AFLW2000, pose_AFLW2000, shape_AFLW2000, exp_AFLW2000, _, _,  = load_FaceAlignment_dataset_recrop_sz224(
            'AFLW2000', False)
        self.AFLW2000_m = np.divide(np.subtract(
            m_AFLW2000, self.mean_m), self.std_m)
        self.AFLW2000_shape_para = shape_AFLW2000
        self.AFLW2000_exp_para = exp_AFLW2000

    def evaluation_LWF_recursive(self, folder='./', data_folder='/home/luan/Documents/data/300VW/recropped/004/', output_folder='./images/300VW_004/'):

        tf.global_variables_initializer().run()
        self.load_checkpoint(self.checkpoint_dir)

        #data_folder = '/scratch/tranluan/Repos/Nonlinear_3DMM_sz224/data_MoFA/'
        #data_folder = '../data_MoFA3/'
        #data_folder = '../CelebA_161k/'
        #data_folder = '../3dMDLab_real/'

        def process_image(img_path, output_path):
            if os.path.isfile(output_path):
                return
            extension_list = ('.png', '.jpg', '.jpeg', '.JPG',
                              '.JPEG', '.PNG', '.bmp', '.BMP')
            if not img_path.endswith(extension_list):
                return

            tx = 16*np.ones(self.sample_size, dtype=np.int)
            ty = 16*np.ones(self.sample_size, dtype=np.int)

            ffeed_dict = {self.input_offset_height: tx,
                          self.input_offset_width: ty}

            ffeed_dict[self.input_images_fn[0]] = img_path
            sample_images = self.sess.run(
                self.input_images, feed_dict=ffeed_dict)

            s_shape, s_texture, s_albedo, s_m, s_img_nc, s_albedo_image, s_shade_image, s_shape_fx, s_alb_fx, s_il, \
                s_shape_base, s_texture_base, s_albedo_base, s_img_base_nc, s_albedo_base_image, s_shade_base_image, s_test_mask = \
                self.sess.run([self.rrotated_shape, self.G_texture_images, self.albedo, self.m, self.G_images_nc, self.G_albedo_images, self.G_shade_images,  self.shape_fx, self.alb_fx, self.il,
                               self.rrotated_shape_base, self.G_texture_base_images, self.albedo_base, self.G_images_base_nc, self.G_albedo_base_images, self.G_shade_base_images, self.test_masks],
                              feed_dict={self.input_images_ph: sample_images})

            save_img = True
            idx = 0

            if save_img:

                # save_images(
                    # sample_images, [-1, 1], '%s_pred_img_%02d_in.png' % (output_path, idx))
                #save_images(s_img, [8, 8], '%s/pred_img_%02d_img.png' % (path, idx))
                ##save_images(s_img_nc, [-1, 1], '%s_pred_img_%02d_img_nc.png' % (output_path, idx))
                ##save_images(s_texture, [-1, 1], '%s_pred_img_%02d_tex.png' % (output_path, idx))
                ##save_images(s_albedo_image, [-1, 1], '%s_pred_img_%02d_alb.png' % (output_path, idx))
                # save_images(
                #     s_shade_image, [-1, 1], '%s_pred_img_%02d_shade.png' % (output_path, idx))
                mask_path = '%s_pred_img_%02d_mask.png' % (output_path, idx)
                scipy.misc.imsave(mask_path, s_test_mask[0])
                mask = (s_shade_image - sample_images) != 0

                masked_shading = mask * s_shade_image
                
                save_images(
                    masked_shading, [-1, 1], '%s_pred_img_%02d_masked_shading.png' % (output_path, idx))
                # save_images(
                #     mask, [-1, 1], '%s_pred_img_%02d_mask.png' % (output_path, idx))

                ##save_images(s_img_base_nc, [-1, 1], '%s_pred_img_%02d_img_nc_base.png' % (output_path, idx))

                ##save_images(s_texture_base, [-1, 1], '%s_pred_img_%02d_tex_base.png' % (output_path, idx))
                # save_images(
                #     s_shade_base_image, [-1, 1], '%s_pred_img_%02d_shade_base.png' % (output_path, idx))

            # Shape

            # np.savetxt('%s_pred_shape.txt' % (output_path),
            #            np.reshape(s_shape[0], [1, -1]))
            # np.savetxt('%s_pred_shape_base.txt' % (output_path),
            #            np.reshape(s_shape_base[0], [1, -1]))

        def process_path(input_path, output_path):
            if os.path.isfile(input_path):
                try:
                    process_image(input_path, output_path)
                except Exception as e:
                    print(e)

            elif os.path.isdir(input_path):
                print("Processing folder %s" % (input_path))
                # Make outputpath if necessary
                if not os.path.isdir(output_path):
                    os.mkdir(output_path)
                # Read all images in the folder
                f_list = os.listdir(input_path)
                random.shuffle(f_list)
                # f_list.sort(reverse=True)
                for f in tqdm(f_list):
                    input_fn = os.path.join(input_path, f)
                    output_fn = os.path.join(output_path, f)
                    process_path(input_fn, output_fn)
            return

        process_path(data_folder, output_folder + folder)

    def generator_encoder(self, image,  is_reuse=False, is_training=True):
        if not is_reuse:
            self.g_bn0_0 = batch_norm(name='g_k_bn0_0')
            self.g_bn0_1 = batch_norm(name='g_k_bn0_1')
            self.g_bn0_2 = batch_norm(name='g_k_bn0_2')
            self.g_bn0_3 = batch_norm(name='g_k_bn0_3')
            self.g_bn1_0 = batch_norm(name='g_k_bn1_0')
            self.g_bn1_1 = batch_norm(name='g_k_bn1_1')
            self.g_bn1_2 = batch_norm(name='g_k_bn1_2')
            self.g_bn1_3 = batch_norm(name='g_k_bn1_3')
            self.g_bn2_0 = batch_norm(name='g_k_bn2_0')
            self.g_bn2_1 = batch_norm(name='g_k_bn2_1')
            self.g_bn2_2 = batch_norm(name='g_k_bn2_2')
            self.g_bn2_3 = batch_norm(name='g_k_bn2_3')
            self.g_bn3_0 = batch_norm(name='g_k_bn3_0')
            self.g_bn3_1 = batch_norm(name='g_k_bn3_1')
            self.g_bn3_2 = batch_norm(name='g_k_bn3_2')
            self.g_bn3_3 = batch_norm(name='g_k_bn3_3')
            self.g_bn4_0 = batch_norm(name='g_k_bn4_0')
            self.g_bn4_1 = batch_norm(name='g_k_bn4_1')
            self.g_bn4_2 = batch_norm(name='g_k_bn4_2')
            self.g_bn4_c = batch_norm(name='g_h_bn4_c')
            self.g_bn5 = batch_norm(name='g_k_bn5')
            self.g_bn5_m = batch_norm(name='g_k_bn5_m')
            self.g_bn5_il = batch_norm(name='g_k_bn5_il')
            self.g_bn5_shape = batch_norm(name='g_k_bn5_shape')
            self.g_bn5_shape_linear = batch_norm(name='g_k_bn5_shape_linear')
            self.g_bn5_tex = batch_norm(name='g_k_bn5_tex')

        k0_1 = elu(self.g_bn0_1(conv2d(image, self.gf_dim*1, k_h=7, k_w=7, d_h=2, d_w=2,
                   use_bias=False, name='g_k01_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))
        k0_2 = elu(self.g_bn0_2(conv2d(k0_1, self.gf_dim*2, d_h=1, d_w=1, use_bias=False,
                   name='g_k02_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))

        k1_0 = elu(self.g_bn1_0(conv2d(k0_2, self.gf_dim*2, d_h=2, d_w=2, use_bias=False,
                   name='g_k10_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))
        k1_1 = elu(self.g_bn1_1(conv2d(k1_0, self.gf_dim*2, d_h=1, d_w=1, use_bias=False,
                   name='g_k11_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))
        k1_2 = elu(self.g_bn1_2(conv2d(k1_1, self.gf_dim*4, d_h=1, d_w=1, use_bias=False,
                   name='g_k12_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))

        k2_0 = elu(self.g_bn2_0(conv2d(k1_2, self.gf_dim*4, d_h=2, d_w=2, use_bias=False,
                   name='g_k20_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))
        k2_1 = elu(self.g_bn2_1(conv2d(k2_0, self.gf_dim*3, d_h=1, d_w=1, use_bias=False,
                   name='g_k21_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))
        k2_2 = elu(self.g_bn2_2(conv2d(k2_1, self.gf_dim*6, d_h=1, d_w=1, use_bias=False,
                   name='g_k22_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))

        k3_0 = elu(self.g_bn3_0(conv2d(k2_2, self.gf_dim*6, d_h=2, d_w=2, use_bias=False,
                   name='g_k30_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))
        k3_1 = elu(self.g_bn3_1(conv2d(k3_0, self.gf_dim*4, d_h=1, d_w=1, use_bias=False,
                   name='g_k31_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))
        k3_2 = elu(self.g_bn3_2(conv2d(k3_1, self.gf_dim*8, d_h=1, d_w=1, use_bias=False,
                   name='g_k32_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))

        k4_0 = elu(self.g_bn4_0(conv2d(k3_2, self.gf_dim*8, d_h=2, d_w=2, use_bias=False,
                   name='g_k40_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))
        k4_1 = elu(self.g_bn4_1(conv2d(k4_0, self.gf_dim*5, d_h=1, d_w=1, use_bias=False,
                   name='g_k41_conv', reuse=is_reuse), train=is_training, reuse=is_reuse))

        # M
        k51_m = self.g_bn5_m(conv2d(k4_1, int(self.gfc_dim/5),  d_h=1, d_w=1,
                             name='g_k5_m_conv', reuse=is_reuse), train=is_training, reuse=is_reuse)
        k51_shape_ = get_shape(k51_m)
        k52_m = tf.nn.avg_pool(k51_m, ksize=[1, k51_shape_[1], k51_shape_[
                               2], 1], strides=[1, 1, 1, 1], padding='VALID')
        k52_m = tf.reshape(k52_m, [-1, int(self.gfc_dim/5)])
        # if (is_training):
        #    k52_m = tf.nn.dropout(k52_m, keep_prob = 0.6)
        k6_m = linear(k52_m, self.mDim, 'g_k6_m_lin', reuse=is_reuse)

        # Il
        k51_il = self.g_bn5_il(conv2d(k4_1, int(self.gfc_dim/5),  d_h=1, d_w=1,
                               name='g_k5_il_conv', reuse=is_reuse), train=is_training, reuse=is_reuse)
        k52_il = tf.nn.avg_pool(k51_il, ksize=[1, k51_shape_[1], k51_shape_[
                                2], 1], strides=[1, 1, 1, 1], padding='VALID')
        k52_il = tf.reshape(k52_il, [-1, int(self.gfc_dim/5)])
        # if (is_training):
        #    k52_il = tf.nn.dropout(k52_il, keep_prob = 0.6)
        k6_il = linear(k52_il, self.ilDim, 'g_k6_il_lin', reuse=is_reuse)

        # Shape
        k51_shape = self.g_bn5_shape(conv2d(k4_1, self.gfc_dim/2,  d_h=1, d_w=1,
                                     name='g_k5_shape_conv', reuse=is_reuse), train=is_training, reuse=is_reuse)
        k52_shape = tf.nn.avg_pool(k51_shape, ksize=[1, k51_shape_[1], k51_shape_[
                                   2], 1], strides=[1, 1, 1, 1], padding='VALID')
        k52_shape = tf.reshape(k52_shape, [-1, int(self.gfc_dim/2)])
        # if (is_training):
        #    k52_shape = tf.nn.dropout(k52_shape, keep_prob = 0.6)

        k51_tex = self.g_bn5_tex(conv2d(k4_1, self.gfc_dim/2,  d_h=1, d_w=1,
                                 name='g_k5_tex_conv', reuse=is_reuse), train=is_training, reuse=is_reuse)
        k52_tex = tf.nn.avg_pool(k51_tex, ksize=[1, k51_shape_[1], k51_shape_[
                                 2], 1], strides=[1, 1, 1, 1], padding='VALID')
        k52_tex = tf.reshape(k52_tex, [-1, int(self.gfc_dim/2)])
        # if (is_training):
        #    k52_tex = tf.nn.dropout(k52_tex, keep_prob = 0.6)

        '''
        if self.is_using_linear:
            k51_shape_linear = self.g_bn5_shape_linear(conv2d(k4_1, int(self.gfc_dim/2),  d_h=1, d_w =1, name='g_k5_shape_linear_conv', reuse = is_reuse), train=is_training, reuse = is_reuse)
            k52_shape_linear = tf.nn.avg_pool(k51_shape_linear, ksize = [1, k51_shape_[1], k51_shape_[2], 1], strides = [1,1,1,1],padding = 'VALID')
            k52_shape_linear = tf.reshape(k52_shape_linear, [-1, int(self.gfc_dim/2)])

            k6_shape_linear = linear(k52_shape_linear, self.shapeDim + self.expDim, 'g_k6_shape_linear_lin', reuse = is_reuse)
        else:
            k6_shape_linear = 0
        '''

        return k52_shape, k52_tex, k6_m, k6_il  # , k6_shape_linear

    def generator_decoder_albedo(self, k52_tex, is_reuse=False, is_training=True):
        # return tf.zeros(shape = [self.batch_size, self.texture_size[0], self.texture_size[1], 3])

        if self.is_partbase_albedo:
            return self.generator_decoder_albedo_part_based_v2_relu(k52_tex, is_reuse, is_training)
        else:
            return self.generator_decoder_albedo_v1(k52_tex, is_reuse, is_training)

    def generator_decoder_shape(self, k52_shape, is_reuse=False, is_training=True, is_remesh=False):
        if False:
            return self.generator_decoder_shape_1d(k52_shape, is_reuse, is_training)
        else:

            n_size = get_shape(k52_shape)
            n_size = n_size[0]

            if self.is_reduce:
                #tri = load_3DMM_tri_remesh6k()
                vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel_reduce()
            else:
                #tri = load_3DMM_tri()
                vt2pixel_u, vt2pixel_v = load_FaceAlignment_vt2pixel()

            #tri2vt1_const = tf.constant(tri[0,:], tf.int32)
            #tri2vt2_const = tf.constant(tri[1,:], tf.int32)
            #tri2vt3_const = tf.constant(tri[2,:], tf.int32)

            # Vt2pix
            vt2pixel_u_const = tf.constant(vt2pixel_u[:-1], tf.float32)
            vt2pixel_v_const = tf.constant(vt2pixel_v[:-1], tf.float32)

            if self.is_partbase_albedo:
                shape_2d, shape_2d_res = self.generator_decoder_shape_2d_partbase_v2_relu(
                    k52_shape, is_reuse, is_training)
                print('get_shape(shape_2d)')
                print(get_shape(shape_2d))
            else:
                shape_2d = self.generator_decoder_shape_2d_v1(
                    k52_shape, is_reuse, is_training)

            vt2pixel_v_const_ = tf.tile(tf.reshape(
                vt2pixel_v_const, shape=[1, 1, -1]), [n_size, 1, 1])
            vt2pixel_u_const_ = tf.tile(tf.reshape(
                vt2pixel_u_const, shape=[1, 1, -1]), [n_size, 1, 1])

            shape_1d = tf.reshape(bilinear_sampler(
                shape_2d, vt2pixel_v_const_, vt2pixel_u_const_), shape=[n_size, -1])

            shape_1d_res = tf.reshape(bilinear_sampler(
                shape_2d_res, vt2pixel_v_const_, vt2pixel_u_const_), shape=[n_size, -1])

            return shape_1d, shape_1d_res  # shape_1d, shape_2d, shape_1d_res, shape_2d_res

    def generator_decoder_shape_1d(self, k52_shape, is_reuse=False, is_training=True):
        s6 = elu(self.g1_bn6(linear(k52_shape, 1000, scope='g_s6_lin',
                 reuse=is_reuse), train=is_training, reuse=is_reuse), name="g_s6_prelu")
        s7 = linear(s6, self.vertexNum*3, scope='g_s7_lin', reuse=is_reuse)

        return s7

    def generator_decoder_shape_2d_v1(self, k52_tex, is_reuse=False, is_training=True):
        if not is_reuse:
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4 = batch_norm(name='g_l_bn4')
            self.g2_bn5 = batch_norm(name='g_l_bn5')

        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(k52_tex, self.gfc_dim*s32_h*s32_w,
                    scope='g_l5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, self.gfc_dim])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse=is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(h4_1, self.gf_dim*8,
                        strides=[1, 1], name='g_l40', reuse=is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(h4_0, self.gf_dim*8,
                        strides=[2, 2], name='g_l32', reuse=is_reuse)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(h3_2, self.gf_dim*4,
                        strides=[1, 1], name='g_l31', reuse=is_reuse)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(h3_1, self.gf_dim*6,
                        strides=[1, 1], name='g_l30', reuse=is_reuse)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(h3_0, self.gf_dim*6,
                        strides=[2, 2], name='g_l22', reuse=is_reuse)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(h2_2, self.gf_dim*3,
                        strides=[1, 1], name='g_l21', reuse=is_reuse)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(h2_1, self.gf_dim*4,
                        strides=[1, 1], name='g_l20', reuse=is_reuse)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(h2_0, self.gf_dim*4,
                        strides=[2, 2], name='g_l12', reuse=is_reuse)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(h1_2, self.gf_dim*2,
                        strides=[1, 1], name='g_l11', reuse=is_reuse)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(h1_1, self.gf_dim*2,
                        strides=[1, 1], name='g_l10', reuse=is_reuse)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(h1_0, self.gf_dim*2,
                        strides=[2, 2], name='g_l02', reuse=is_reuse)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, self.gf_dim, strides=[
                        1, 1], name='g_l01', reuse=is_reuse)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_1, self.c_dim,
                          strides=[1, 1], name='g_l0', reuse=is_reuse))

        return h0

    def generator_decoder_shape_2d_partbase(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1",
                                    "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w = int(s_w/8)
                s8_h = int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_l4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_l_bn4"]
                         (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse)
                h3_1 = elu(self.bns[name + "/" + "g_l_bn3_1"]
                           (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse)
                h3_0 = elu(self.bns[name + "/" + "g_l_bn3_0"]
                           (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse)
                h2_2 = elu(self.bns[name + "/" + "g_l_bn2_2"]
                           (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse)
                h2_1 = elu(self.bns[name + "/" "g_l_bn2_1"]
                           (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse)
                h2_0 = elu(self.bns[name + "/" + "g_l_bn2_0"]
                           (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse)
                h1_2 = elu(self.bns[name + "/" + "g_l_bn1_2"]
                           (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse)
                h1_1 = elu(self.bns[name + "/" + "g_l_bn1_1"]
                           (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse)
                h1_0 = elu(self.bns[name + "/" + "g_l_bn1_0"]
                           (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse)
                h0_2 = elu(self.bns[name + "/" + "g_l_bn0_2"]
                           (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_l01', reuse=is_reuse)
                h0_1 = elu(self.bns[name + "/" + "g_l_bn0_1"]
                           (h0_1, train=is_training, reuse=is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48]  # nose
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        if not is_reuse:
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4 = batch_norm(name='g_l_bn4')
            self.g2_bn5 = batch_norm(name='g_l_bn5')

        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w,
                    scope='g_l5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, df*10, name='g_l4', reuse=is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(
            h4_1, df*8, strides=[1, 1], name='g_l40', reuse=is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_l32', reuse=is_reuse)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1, 1], name='g_l01', reuse=is_reuse)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00', reuse=is_reuse)
        h0_0 = elu(self.g2_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim,
                          strides=[1, 1], name='g_l0', reuse=is_reuse))

        return h0

    def generator_decoder_shape_2d_partbase_v2_elu(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1",
                                    "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w = int(s_w/8)
                s8_h = int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_l4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_l_bn4"]
                         (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse)
                h3_1 = elu(self.bns[name + "/" + "g_l_bn3_1"]
                           (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse)
                h3_0 = elu(self.bns[name + "/" + "g_l_bn3_0"]
                           (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse)
                h2_2 = elu(self.bns[name + "/" + "g_l_bn2_2"]
                           (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse)
                h2_1 = elu(self.bns[name + "/" "g_l_bn2_1"]
                           (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse)
                h2_0 = elu(self.bns[name + "/" + "g_l_bn2_0"]
                           (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse)
                h1_2 = elu(self.bns[name + "/" + "g_l_bn1_2"]
                           (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse)
                h1_1 = elu(self.bns[name + "/" + "g_l_bn1_1"]
                           (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse)
                h1_0 = elu(self.bns[name + "/" + "g_l_bn1_0"]
                           (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse)
                h0_2 = elu(self.bns[name + "/" + "g_l_bn0_2"]
                           (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_l01', reuse=is_reuse)
                h0_1 = elu(self.bns[name + "/" + "g_l_bn0_1"]
                           (h0_1, train=is_training, reuse=is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48]  # nose
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4 = batch_norm(name='g_l_bn4')
            self.g2_bn5 = batch_norm(name='g_l_bn5')

        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w,
                    scope='g_l5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse=is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(
            h4_1, df*8, strides=[1, 1], name='g_l40', reuse=is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_l32', reuse=is_reuse)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1, 1], name='g_l01', reuse=is_reuse)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00', reuse=is_reuse)
        h0_0 = elu(self.g2_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim,
                          strides=[1, 1], name='g_l0', reuse=is_reuse))

        # Final res
        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00_res', reuse=is_reuse)
        h0_0_res = elu(self.g2_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim,
                              strides=[1, 1], name='g_l0_res', reuse=is_reuse))

        return h0, h0 + h0_res

    def generator_decoder_shape_2d_partbase_v2_relu(self, input_feature, is_reuse=False, is_training=True):
        activ = relu

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1",
                                    "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w = int(s_w/8)
                s8_h = int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_l4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_l_bn4"]
                           (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_l_bn3_1"]
                             (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_l_bn3_0"]
                             (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_l_bn2_2"]
                             (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_l_bn2_1"]
                             (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_l_bn2_0"]
                             (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_l_bn1_2"]
                             (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_l_bn1_1"]
                             (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_l_bn1_0"]
                             (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_l_bn0_2"]
                             (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_l01', reuse=is_reuse)
                h0_1 = activ(self.bns[name + "/" + "g_l_bn0_1"]
                             (h0_1, train=is_training, reuse=is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48]  # nose
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4 = batch_norm(name='g_l_bn4')
            self.g2_bn5 = batch_norm(name='g_l_bn5')

        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w,
                    scope='g_l5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g2_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse=is_reuse)
        h4_1 = activ(self.g2_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(
            h4_1, df*8, strides=[1, 1], name='g_l40', reuse=is_reuse)
        h4_0 = activ(self.g2_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_l32', reuse=is_reuse)
        h3_2 = activ(self.g2_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse)
        h3_1 = activ(self.g2_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse)
        h3_0 = activ(self.g2_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse)
        h2_2 = activ(self.g2_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse)
        h2_1 = activ(self.g2_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse)
        h2_0 = activ(self.g2_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse)
        h1_2 = activ(self.g2_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse)
        h1_1 = activ(self.g2_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse)
        h1_0 = activ(self.g2_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse)
        h0_2 = activ(self.g2_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1, 1], name='g_l01', reuse=is_reuse)
        h0_1 = activ(self.g2_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00', reuse=is_reuse)
        h0_0 = activ(self.g2_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim,
                          strides=[1, 1], name='g_l0', reuse=is_reuse))

        # Final res
        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00_res', reuse=is_reuse)
        h0_0_res = activ(self.g2_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim,
                              strides=[1, 1], name='g_l0_res', reuse=is_reuse))

        return h0, h0_res  # h0, h0_res

    def generator_decoder_shape_2d_partbase_v3(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1",
                                    "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w = int(s_w/8)
                s8_h = int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_l4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_l_bn4"]
                         (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse, use_bias=False)
                h3_1 = elu(self.bns[name + "/" + "g_l_bn3_1"]
                           (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse, use_bias=False)
                h3_0 = elu(self.bns[name + "/" + "g_l_bn3_0"]
                           (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse, use_bias=False)
                h2_2 = elu(self.bns[name + "/" + "g_l_bn2_2"]
                           (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse, use_bias=False)
                h2_1 = elu(self.bns[name + "/" "g_l_bn2_1"]
                           (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse, use_bias=False)
                h2_0 = elu(self.bns[name + "/" + "g_l_bn2_0"]
                           (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse, use_bias=False)
                h1_2 = elu(self.bns[name + "/" + "g_l_bn1_2"]
                           (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse, use_bias=False)
                h1_1 = elu(self.bns[name + "/" + "g_l_bn1_1"]
                           (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse, use_bias=False)
                h1_0 = elu(self.bns[name + "/" + "g_l_bn1_0"]
                           (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse, use_bias=False)
                h0_2 = elu(self.bns[name + "/" + "g_l_bn0_2"]
                           (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_l01', reuse=is_reuse, use_bias=False)
                h0_1 = elu(self.bns[name + "/" + "g_l_bn0_1"]
                           (h0_1, train=is_training, reuse=is_reuse))

            return h0_1

         # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48]  # nose
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4 = batch_norm(name='g_l_bn4')
            self.g2_bn5 = batch_norm(name='g_l_bn5')

        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s16_h = int(s_h/16)
        s16_w = int(s_w/16)

        # project `z` and reshape

        '''
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_l5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g2_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df_dim*5, strides=[2,2], name='g_l4', reuse = is_reuse)
        h4_1 = elu(self.g2_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_l40', reuse = is_reuse)
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse = is_reuse))
        '''

        h4_0 = linear(input_feature, df*4*s16_h*s16_w,
                      scope='g_l40_lin', reuse=is_reuse)
        h4_0 = tf.reshape(h4_0, [-1, s16_h, s16_w, df*4])
        h4_0 = elu(self.g2_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_l32', reuse=is_reuse, use_bias=False)
        h3_2 = elu(self.g2_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse, use_bias=False)
        h3_1 = elu(self.g2_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse, use_bias=False)
        h3_0 = elu(self.g2_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse, use_bias=False)
        h2_2 = elu(self.g2_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse, use_bias=False)
        h2_1 = elu(self.g2_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse, use_bias=False)
        h2_0 = elu(self.g2_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse, use_bias=False)
        h1_2 = elu(self.g2_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse, use_bias=False)
        h1_1 = elu(self.g2_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse, use_bias=False)
        h1_0 = elu(self.g2_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse, use_bias=False)
        h0_2 = elu(self.g2_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[
                        1, 1], name='g_l01', reuse=is_reuse, use_bias=False)
        h0_1 = elu(self.g2_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00', reuse=is_reuse, use_bias=False)
        h0_0 = elu(self.g2_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim,
                          strides=[1, 1], name='g_l0', reuse=is_reuse))

        # Final res
        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00_res', reuse=is_reuse, use_bias=False)
        h0_0_res = elu(self.g2_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim,
                              strides=[1, 1], name='g_l0_res', reuse=is_reuse))

        return h0, h0_res

    def generator_decoder_shape_2d_partbase_v6_elu(self, input_feature, is_reuse=False, is_training=True):
        # v2_elu_nose_bg
        activ = elu

        def decoder_part_shape(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_l_bn4", "g_l_bn3_1", "g_l_bn3_0", "g_l_bn2_2", "g_l_bn2_1",
                                    "g_l_bn2_0",  "g_l_bn1_2", "g_l_bn1_1", "g_l_bn1_0",  "g_l_bn0_2", "g_l_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s8_w = int(s_w/8)
                s8_h = int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_l4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_l_bn4"]
                           (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_l_bn3_1"]
                             (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_l_bn3_0"]
                             (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_l_bn2_2"]
                             (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_l_bn2_1"]
                             (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_l_bn2_0"]
                             (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_l_bn1_2"]
                             (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_l_bn1_1"]
                             (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_l_bn1_0"]
                             (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_l_bn0_2"]
                             (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_l01', reuse=is_reuse)
                h0_1 = self.bns[name + "/" +
                                "g_l_bn0_1"](h0_1, train=is_training, reuse=is_reuse)

            return h0_1

        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48]  # nose
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_shape(self, input_feature, [
                                  bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        if not is_reuse:
            self.g2_bn0_0_res = batch_norm(name='g_l_bn0_0_res')
            self.g2_bn0_0 = batch_norm(name='g_l_bn0_0')
            self.g2_bn0_1 = batch_norm(name='g_l_bn0_1')
            self.g2_bn0_1_local = batch_norm(name='g_l_bn0_1_local_v6')
            self.g2_bn0_2 = batch_norm(name='g_l_bn0_2')
            self.g2_bn1_0 = batch_norm(name='g_l_bn1_0')
            self.g2_bn1_1 = batch_norm(name='g_l_bn1_1')
            self.g2_bn1_2 = batch_norm(name='g_l_bn1_2')
            self.g2_bn2_0 = batch_norm(name='g_l_bn2_0')
            self.g2_bn2_1 = batch_norm(name='g_l_bn2_1')
            self.g2_bn2_2 = batch_norm(name='g_l_bn2_2')
            self.g2_bn3_0 = batch_norm(name='g_l_bn3_0')
            self.g2_bn3_1 = batch_norm(name='g_l_bn3_1')
            self.g2_bn3_2 = batch_norm(name='g_l_bn3_2')
            self.g2_bn4_0 = batch_norm(name='g_l_bn4_0')
            self.g2_bn4 = batch_norm(name='g_l_bn4')
            self.g2_bn5 = batch_norm(name='g_l_bn5')

        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w,
                    scope='g_l5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g2_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, self.gf_dim*5, name='g_l4', reuse=is_reuse)
        h4_1 = activ(self.g2_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(
            h4_1, df*8, strides=[1, 1], name='g_l40', reuse=is_reuse)
        h4_0 = activ(self.g2_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_l32', reuse=is_reuse)
        h3_2 = activ(self.g2_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_l31', reuse=is_reuse)
        h3_1 = activ(self.g2_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_l30', reuse=is_reuse)
        h3_0 = activ(self.g2_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_l22', reuse=is_reuse)
        h2_2 = activ(self.g2_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_l21', reuse=is_reuse)
        h2_1 = activ(self.g2_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_l20', reuse=is_reuse)
        h2_0 = activ(self.g2_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_l12', reuse=is_reuse)
        h1_2 = activ(self.g2_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_l11', reuse=is_reuse)
        h1_1 = activ(self.g2_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_l10', reuse=is_reuse)
        h1_0 = activ(self.g2_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_l02', reuse=is_reuse)
        h0_2 = activ(self.g2_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1, 1], name='g_l01', reuse=is_reuse)
        h0_1 = activ(self.g2_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        local_bg = deconv2d(h0_2, df, strides=[
                            1, 1], name='g_l01_local_v6', reuse=is_reuse, use_bias=False)
        local_bg = self.g2_bn0_1_local(
            local_bg, train=is_training, reuse=is_reuse)
        local = activ(tf.maximum(local, local_bg))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00', reuse=is_reuse)
        h0_0 = activ(self.g2_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = 2*tf.nn.tanh(deconv2d(h0_0, self.c_dim,
                          strides=[1, 1], name='g_l0', reuse=is_reuse))

        # Final res
        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_l00_res', reuse=is_reuse)
        h0_0_res = activ(self.g2_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = 2*tf.nn.tanh(deconv2d(h0_0_res, self.c_dim,
                              strides=[1, 1], name='g_l0_res', reuse=is_reuse))

        return h0, h0_res

    '''
    def generator_decoder_albedo_v1(self, k52_tex, is_reuse=False, is_training=True):
        if not is_reuse:
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')        
            self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
            self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
            self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
            self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
            self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
            self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
            self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
            self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
            self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
            self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
            self.g1_bn4   = batch_norm(name='g_h_bn4')
            self.g1_bn5   = batch_norm(name='g_h_bn5')
            #self.g1_bn6   = batch_norm(name='g_s_bn6')
        
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)

        df = int(self.gf_dim)
                    
        # project `z` and reshape
        h5 = linear(k52_tex, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = elu(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_h32', reuse = is_reuse)
        h3_2 = elu(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
        h3_1 = elu(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
        h3_0 = elu(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
        h2_2 = elu(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
        h2_1 = elu(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
        h2_0 = elu(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
        h1_2 = elu(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
        h1_1 = elu(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
        h1_0 = elu(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
        h0_2 = elu(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
        h0_1 = elu(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse))
           
        h0 = tf.nn.tanh(deconv2d(h0_1, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))
            
        return h0



    def generator_decoder_albedo_part_based(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1", "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" + bn_name]   = batch_norm(name=name + "/" + bn_name)


            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w= int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h= int(s_h/2), int(s_h/4), int(s_h/8)

                
                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 *s8_w*s8_h, scope= 'g_h4_lin', reuse = is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_h_bn4"](h4, train=is_training, reuse = is_reuse))

                h3_1 = deconv2d(h4, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
                h3_1 = elu(self.bns[name + "/" + "g_h_bn3_1"](h3_1, train=is_training, reuse = is_reuse))
                h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
                h3_0 = elu(self.bns[name + "/" + "g_h_bn3_0"](h3_0, train=is_training, reuse = is_reuse))

                h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
                h2_2 = elu(self.bns[name + "/" + "g_h_bn2_2"](h2_2, train=is_training, reuse = is_reuse))
                h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
                h2_1 = elu(self.bns[name + "/" "g_h_bn2_1"](h2_1, train=is_training, reuse = is_reuse))
                h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
                h2_0 = elu(self.bns[name + "/" + "g_h_bn2_0"](h2_0, train=is_training, reuse = is_reuse))

                h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
                h1_2 = elu(self.bns[name + "/" + "g_h_bn1_2"](h1_2, train=is_training, reuse = is_reuse))
                h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
                h1_1 = elu(self.bns[name + "/" + "g_h_bn1_1"](h1_1, train=is_training, reuse = is_reuse))
                h1_0 = deconv2d(h1_1, df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
                h1_0 = elu(self.bns[name + "/" + "g_h_bn1_0"](h1_0, train=is_training, reuse = is_reuse))


                h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
                h0_2 = elu(self.bns[name + "/" + "g_h_bn0_2"](h0_2, train=is_training, reuse = is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
                h0_1 = elu(self.bns[name + "/" + "g_h_bn0_1"](h0_1, train=is_training, reuse = is_reuse))


            return h0_1

        if not is_reuse:
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')        
            self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
            self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
            self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
            self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
            self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
            self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
            self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
            self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
            self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
            self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
            self.g1_bn4   = batch_norm(name='g_h_bn4')
            self.g1_bn5   = batch_norm(name='g_h_bn5')


        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48] # left eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')
        
        bbox = [38, 60, 40 , 48] # right eye
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  #mouth
        part = decoder_part_albedo(self, input_feature, [bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0,0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0,0]], mode='CONSTANT')

        local = leye + reye + mouth
        


        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h= int(s_h/32)
        s32_w= int(s_w/32)
                    
        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = elu(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))

        h3_2 = deconv2d(h4_0, df*8, strides=[2,2], name='g_h32', reuse = is_reuse)
        h3_2 = elu(self.g1_bn3_2(h3_2, train=is_training, reuse = is_reuse))
        h3_1 = deconv2d(h3_2, df*4, strides=[1,1], name='g_h31', reuse = is_reuse)
        h3_1 = elu(self.g1_bn3_1(h3_1, train=is_training, reuse = is_reuse))
        h3_0 = deconv2d(h3_1, df*6, strides=[1,1], name='g_h30', reuse = is_reuse)
        h3_0 = elu(self.g1_bn3_0(h3_0, train=is_training, reuse = is_reuse))

        h2_2 = deconv2d(h3_0, df*6, strides=[2,2], name='g_h22', reuse = is_reuse)
        h2_2 = elu(self.g1_bn2_2(h2_2, train=is_training, reuse = is_reuse))
        h2_1 = deconv2d(h2_2, df*3, strides=[1,1], name='g_h21', reuse = is_reuse)
        h2_1 = elu(self.g1_bn2_1(h2_1, train=is_training, reuse = is_reuse))
        h2_0 = deconv2d(h2_1, df*4, strides=[1,1], name='g_h20', reuse = is_reuse)
        h2_0 = elu(self.g1_bn2_0(h2_0, train=is_training, reuse = is_reuse))

        h1_2 = deconv2d(h2_0, df*4, strides=[2,2], name='g_h12', reuse = is_reuse)
        h1_2 = elu(self.g1_bn1_2(h1_2, train=is_training, reuse = is_reuse))
        h1_1 = deconv2d(h1_2, df*2, strides=[1,1], name='g_h11', reuse = is_reuse)
        h1_1 = elu(self.g1_bn1_1(h1_1, train=is_training, reuse = is_reuse))
        h1_0 = deconv2d(h1_1,df*2, strides=[1,1], name='g_h10', reuse = is_reuse)
        h1_0 = elu(self.g1_bn1_0(h1_0, train=is_training, reuse = is_reuse))

        h0_2 = deconv2d(h1_0, df*2, strides=[2,2], name='g_h02', reuse = is_reuse)
        h0_2 = elu(self.g1_bn0_2(h0_2, train=is_training, reuse = is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1,1], name='g_h01', reuse = is_reuse)
        h0_1 = elu(self.g1_bn0_1(h0_1, train=is_training, reuse = is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        #Final
        h0_0 = deconv2d(h0_1_all, df*2, strides=[1,1], name='g_h00', reuse = is_reuse)
        h0_0 = elu(self.g1_bn0_0(h0_0, train=is_training, reuse = is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[1,1], name='g_h0', reuse = is_reuse))
            
        return h0
    '''

    def generator_decoder_albedo_part_based_v2_elu(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1",
                                    "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w = int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h = int(s_h/2), int(s_h/4), int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_h4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_h_bn4"]
                         (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse)
                h3_1 = elu(self.bns[name + "/" + "g_h_bn3_1"]
                           (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse)
                h3_0 = elu(self.bns[name + "/" + "g_h_bn3_0"]
                           (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse)
                h2_2 = elu(self.bns[name + "/" + "g_h_bn2_2"]
                           (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse)
                h2_1 = elu(self.bns[name + "/" "g_h_bn2_1"]
                           (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse)
                h2_0 = elu(self.bns[name + "/" + "g_h_bn2_0"]
                           (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse)
                h1_2 = elu(self.bns[name + "/" + "g_h_bn1_2"]
                           (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse)
                h1_1 = elu(self.bns[name + "/" + "g_h_bn1_1"]
                           (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse)
                h1_0 = elu(self.bns[name + "/" + "g_h_bn1_0"]
                           (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse)
                h0_2 = elu(self.bns[name + "/" + "g_h_bn0_2"]
                           (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_h01', reuse=is_reuse)
                h0_1 = elu(self.bns[name + "/" + "g_h_bn0_1"]
                           (h0_1, train=is_training, reuse=is_reuse))

            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')
            self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
            self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
            self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
            self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
            self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
            self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
            self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
            self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
            self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
            self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
            self.g1_bn4 = batch_norm(name='g_h_bn4')
            self.g1_bn5 = batch_norm(name='g_h_bn5')

        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth

        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w,
                    scope='g_h5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g1_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse=is_reuse)
        h4_1 = elu(self.g1_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(
            h4_1, df*8, strides=[1, 1], name='g_h40', reuse=is_reuse)
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_h32', reuse=is_reuse)
        h3_2 = elu(self.g1_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse)
        h3_1 = elu(self.g1_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse)
        h3_0 = elu(self.g1_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse)
        h2_2 = elu(self.g1_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse)
        h2_1 = elu(self.g1_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse)
        h2_0 = elu(self.g1_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse)
        h1_2 = elu(self.g1_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse)
        h1_1 = elu(self.g1_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse)
        h1_0 = elu(self.g1_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse)
        h0_2 = elu(self.g1_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1, 1], name='g_h01', reuse=is_reuse)
        h0_1 = elu(self.g1_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00', reuse=is_reuse)
        h0_0 = elu(self.g1_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[
                        1, 1], name='g_h0', reuse=is_reuse))

        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00_res', reuse=is_reuse)
        h0_0_res = elu(self.g1_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[
                            1, 1], name='g_h0_res', reuse=is_reuse))

        return h0, h0 + h0_res

    def generator_decoder_albedo_part_based_v3(self, input_feature, is_reuse=False, is_training=True):

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1",
                                    "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w = int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h = int(s_h/2), int(s_h/4), int(s_h/8)

                #s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_h4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = elu(self.bns[name + "/" + "g_h_bn4"]
                         (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse, use_bias=False)
                h3_1 = elu(self.bns[name + "/" + "g_h_bn3_1"]
                           (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse, use_bias=False)
                h3_0 = elu(self.bns[name + "/" + "g_h_bn3_0"]
                           (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse, use_bias=False)
                h2_2 = elu(self.bns[name + "/" + "g_h_bn2_2"]
                           (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse, use_bias=False)
                h2_1 = elu(self.bns[name + "/" "g_h_bn2_1"]
                           (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse, use_bias=False)
                h2_0 = elu(self.bns[name + "/" + "g_h_bn2_0"]
                           (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse, use_bias=False)
                h1_2 = elu(self.bns[name + "/" + "g_h_bn1_2"]
                           (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse, use_bias=False)
                h1_1 = elu(self.bns[name + "/" + "g_h_bn1_1"]
                           (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse, use_bias=False)
                h1_0 = elu(self.bns[name + "/" + "g_h_bn1_0"]
                           (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse, use_bias=False)
                h0_2 = elu(self.bns[name + "/" + "g_h_bn0_2"]
                           (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_h01', reuse=is_reuse, use_bias=False)
                h0_1 = elu(self.bns[name + "/" + "g_h_bn0_1"]
                           (h0_1, train=is_training, reuse=is_reuse))

            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')
            self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
            self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
            self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
            self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
            self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
            self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
            self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
            self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
            self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
            self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
            self.g1_bn4 = batch_norm(name='g_h_bn4')
            self.g1_bn5 = batch_norm(name='g_h_bn5')

        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48]  # nose
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s16_h = int(s_h/16)
        s16_w = int(s_w/16)

        # project `z` and reshape
        '''
        h5 = linear(input_feature, df*10*s32_h*s32_w, scope= 'g_h5_lin', reuse = is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = elu(self.g1_bn5(h5, train=is_training, reuse = is_reuse))
        
        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse = is_reuse)
        h4_1 = elu(self.g1_bn4(h4_1, train=is_training, reuse = is_reuse))
        h4_0 = deconv2d(h4_1, df*8, strides=[1,1], name='g_h40', reuse = is_reuse)
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse = is_reuse))
        '''

        h4_0 = linear(input_feature, df*4*s16_h*s16_w,
                      scope='g_h40_lin', reuse=is_reuse)
        h4_0 = tf.reshape(h4_0, [-1, s16_h, s16_w, df*4])
        h4_0 = elu(self.g1_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_h32', reuse=is_reuse, use_bias=False)
        h3_2 = elu(self.g1_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse, use_bias=False)
        h3_1 = elu(self.g1_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse, use_bias=False)
        h3_0 = elu(self.g1_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse, use_bias=False)
        h2_2 = elu(self.g1_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse, use_bias=False)
        h2_1 = elu(self.g1_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse, use_bias=False)
        h2_0 = elu(self.g1_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse, use_bias=False)
        h1_2 = elu(self.g1_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse, use_bias=False)
        h1_1 = elu(self.g1_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse, use_bias=False)
        h1_0 = elu(self.g1_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse, use_bias=False)
        h0_2 = elu(self.g1_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[
                        1, 1], name='g_h01', reuse=is_reuse, use_bias=False)
        h0_1 = elu(self.g1_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00', reuse=is_reuse, use_bias=False)
        h0_0 = elu(self.g1_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[
                        1, 1], name='g_h0', reuse=is_reuse))

        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00_res', reuse=is_reuse, use_bias=False)
        h0_0_res = elu(self.g1_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[
                            1, 1], name='g_h0_res', reuse=is_reuse))

        return h0, h0_res

    def generator_decoder_albedo_part_based_v2_relu(self, input_feature, is_reuse=False, is_training=True):
        #v2 + RELU

        activ = relu

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1",
                                    "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w = int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h = int(s_h/2), int(s_h/4), int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_h4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_h_bn4"]
                           (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_h_bn3_1"]
                             (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_h_bn3_0"]
                             (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_h_bn2_2"]
                             (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_h_bn2_1"]
                             (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_h_bn2_0"]
                             (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_h_bn1_2"]
                             (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_h_bn1_1"]
                             (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_h_bn1_0"]
                             (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_h_bn0_2"]
                             (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_h01', reuse=is_reuse)
                h0_1 = activ(self.bns[name + "/" + "g_h_bn0_1"]
                             (h0_1, train=is_training, reuse=is_reuse))

            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')
            self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
            self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
            self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
            self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
            self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
            self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
            self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
            self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
            self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
            self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
            self.g1_bn4 = batch_norm(name='g_h_bn4')
            self.g1_bn5 = batch_norm(name='g_h_bn5')

        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth

        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w,
                    scope='g_h5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g1_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse=is_reuse)
        h4_1 = activ(self.g1_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(
            h4_1, df*8, strides=[1, 1], name='g_h40', reuse=is_reuse)
        h4_0 = activ(self.g1_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_h32', reuse=is_reuse)
        h3_2 = activ(self.g1_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse)
        h3_1 = activ(self.g1_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse)
        h3_0 = activ(self.g1_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse)
        h2_2 = activ(self.g1_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse)
        h2_1 = activ(self.g1_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse)
        h2_0 = activ(self.g1_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse)
        h1_2 = activ(self.g1_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse)
        h1_1 = activ(self.g1_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse)
        h1_0 = activ(self.g1_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse)
        h0_2 = activ(self.g1_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1, 1], name='g_h01', reuse=is_reuse)
        h0_1 = activ(self.g1_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00', reuse=is_reuse)
        h0_0 = activ(self.g1_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[
                        1, 1], name='g_h0', reuse=is_reuse))

        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00_res', reuse=is_reuse)
        h0_0_res = activ(self.g1_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[
                            1, 1], name='g_h0_res', reuse=is_reuse))

        print('Return h0, h0_res')

        return h0, h0_res

    def generator_decoder_albedo_part_based_v4_relu(self, input_feature, is_reuse=False, is_training=True):
        #v2 + RELU

        activ = relu

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1",
                                    "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w = int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h = int(s_h/2), int(s_h/4), int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_h4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_h_bn4"]
                           (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_h_bn3_1"]
                             (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_h_bn3_0"]
                             (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_h_bn2_2"]
                             (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_h_bn2_1"]
                             (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_h_bn2_0"]
                             (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_h_bn1_2"]
                             (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_h_bn1_1"]
                             (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_h_bn1_0"]
                             (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_h_bn0_2"]
                             (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_h01', reuse=is_reuse)
                h0_1 = activ(self.bns[name + "/" + "g_h_bn0_1"]
                             (h0_1, train=is_training, reuse=is_reuse))

            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')
            self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
            self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
            self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
            self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
            self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
            self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
            self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
            self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
            self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
            self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
            self.g1_bn4 = batch_norm(name='g_h_bn4')
            self.g1_bn5 = batch_norm(name='g_h_bn5')

        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48]  # nose
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='nose', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w,
                    scope='g_h5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g1_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse=is_reuse)
        h4_1 = activ(self.g1_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(
            h4_1, df*8, strides=[1, 1], name='g_h40', reuse=is_reuse)
        h4_0 = activ(self.g1_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_h32', reuse=is_reuse)
        h3_2 = activ(self.g1_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse)
        h3_1 = activ(self.g1_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse)
        h3_0 = activ(self.g1_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse)
        h2_2 = activ(self.g1_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse)
        h2_1 = activ(self.g1_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse)
        h2_0 = activ(self.g1_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse)
        h1_2 = activ(self.g1_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse)
        h1_1 = activ(self.g1_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse)
        h1_0 = activ(self.g1_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse)
        h0_2 = activ(self.g1_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1, 1], name='g_h01', reuse=is_reuse)
        h0_1 = activ(self.g1_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00', reuse=is_reuse)
        h0_0 = activ(self.g1_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[
                        1, 1], name='g_h0', reuse=is_reuse))

        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00_res', reuse=is_reuse)
        h0_0_res = activ(self.g1_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[
                            1, 1], name='g_h0_res', reuse=is_reuse))

        return h0, h0_res

    def generator_decoder_albedo_part_based_v6_elu(self, input_feature, is_reuse=False, is_training=True):
        #v2 + ELU + nose + local_bg
        activ = elu

        def decoder_part_albedo(self, input_feature, output_shape, name, is_reuse=False, is_training=True):

            if not is_reuse:
                batch_norm_names = ["g_h_bn4", "g_h_bn3_1", "g_h_bn3_0", "g_h_bn2_2", "g_h_bn2_1",
                                    "g_h_bn2_0",  "g_h_bn1_2", "g_h_bn1_1", "g_h_bn1_0",  "g_h_bn0_2", "g_h_bn0_1"]
                for bn_name in batch_norm_names:
                    self.bns[name + "/" +
                             bn_name] = batch_norm(name=name + "/" + bn_name)

            #print("--------- Part: " + name)
            # print(self.bns.keys())
            # print("----------------")

            with tf.variable_scope(name, reuse=is_reuse):
                s_w = int(output_shape[0])
                s_h = int(output_shape[1])
                s2_w, s4_w, s8_w = int(s_w/2), int(s_w/4), int(s_w/8)
                s2_h, s4_h, s8_h = int(s_h/2), int(s_h/4), int(s_h/8)

                s8_h = int(output_shape[1]/8)

                df = output_shape[2]

                h4 = linear(input_feature, df * 8 * s8_w*s8_h,
                            scope='g_h4_lin', reuse=is_reuse)
                h4 = tf.reshape(h4, [-1, s8_w, s8_h, df * 8])
                h4 = activ(self.bns[name + "/" + "g_h_bn4"]
                           (h4, train=is_training, reuse=is_reuse))

                h3_1 = deconv2d(
                    h4, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse)
                h3_1 = activ(self.bns[name + "/" + "g_h_bn3_1"]
                             (h3_1, train=is_training, reuse=is_reuse))
                h3_0 = deconv2d(
                    h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse)
                h3_0 = activ(self.bns[name + "/" + "g_h_bn3_0"]
                             (h3_0, train=is_training, reuse=is_reuse))

                h2_2 = deconv2d(
                    h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse)
                h2_2 = activ(self.bns[name + "/" + "g_h_bn2_2"]
                             (h2_2, train=is_training, reuse=is_reuse))
                h2_1 = deconv2d(
                    h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse)
                h2_1 = activ(self.bns[name + "/" "g_h_bn2_1"]
                             (h2_1, train=is_training, reuse=is_reuse))
                h2_0 = deconv2d(
                    h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse)
                h2_0 = activ(self.bns[name + "/" + "g_h_bn2_0"]
                             (h2_0, train=is_training, reuse=is_reuse))

                h1_2 = deconv2d(
                    h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse)
                h1_2 = activ(self.bns[name + "/" + "g_h_bn1_2"]
                             (h1_2, train=is_training, reuse=is_reuse))
                h1_1 = deconv2d(
                    h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse)
                h1_1 = activ(self.bns[name + "/" + "g_h_bn1_1"]
                             (h1_1, train=is_training, reuse=is_reuse))
                h1_0 = deconv2d(
                    h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse)
                h1_0 = activ(self.bns[name + "/" + "g_h_bn1_0"]
                             (h1_0, train=is_training, reuse=is_reuse))

                h0_2 = deconv2d(
                    h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse)
                h0_2 = activ(self.bns[name + "/" + "g_h_bn0_2"]
                             (h0_2, train=is_training, reuse=is_reuse))
                h0_1 = deconv2d(h0_2, df, strides=[
                                1, 1], name='g_h01', reuse=is_reuse)
                h0_1 = self.bns[name + "/" +
                                "g_h_bn0_1"](h0_1, train=is_training, reuse=is_reuse)

            return h0_1

        if not is_reuse:
            self.g1_bn0_0_res = batch_norm(name='g_h_bn0_0_res')
            self.g1_bn0_0 = batch_norm(name='g_h_bn0_0')
            self.g1_bn0_1 = batch_norm(name='g_h_bn0_1')
            self.g1_bn0_1_local = batch_norm(name='g_h_bn0_1_local_v6')
            self.g1_bn0_2 = batch_norm(name='g_h_bn0_2')
            self.g1_bn1_0 = batch_norm(name='g_h_bn1_0')
            self.g1_bn1_1 = batch_norm(name='g_h_bn1_1')
            self.g1_bn1_2 = batch_norm(name='g_h_bn1_2')
            self.g1_bn2_0 = batch_norm(name='g_h_bn2_0')
            self.g1_bn2_1 = batch_norm(name='g_h_bn2_1')
            self.g1_bn2_2 = batch_norm(name='g_h_bn2_2')
            self.g1_bn3_0 = batch_norm(name='g_h_bn3_0')
            self.g1_bn3_1 = batch_norm(name='g_h_bn3_1')
            self.g1_bn3_2 = batch_norm(name='g_h_bn3_2')
            self.g1_bn4_0 = batch_norm(name='g_h_bn4_0')
            self.g1_bn4 = batch_norm(name='g_h_bn4')
            self.g1_bn5 = batch_norm(name='g_h_bn5')

        # Local
        df = int(self.gf_dim/2)

        bbox = [38, 116, 40, 48]  # left eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='leye', is_reuse=is_reuse, is_training=is_training)
        leye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [38, 60, 40, 48]  # right eye
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='reye', is_reuse=is_reuse, is_training=is_training)
        reye = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [96, 63, 64, 96]  # mouth
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], 16], name='mouth', is_reuse=is_reuse, is_training=is_training)
        mouth = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                       bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        bbox = [46, 88, 56, 48]  # nose
        part = decoder_part_albedo(self, input_feature, [
                                   bbox[2], bbox[3], df], name='nose_v6', is_reuse=is_reuse, is_training=is_training)
        nose = tf.pad(part, [[0, 0], [bbox[0], self.texture_size[0]-bbox[0] - bbox[2]], [
                      bbox[1], self.texture_size[1]-bbox[1] - bbox[3]], [0, 0]], mode='CONSTANT')

        local = leye + reye + mouth
        local = tf.maximum(local, nose)

        # Global
        s_h = int(self.texture_size[0])
        s_w = int(self.texture_size[1])
        s32_h = int(s_h/32)
        s32_w = int(s_w/32)

        # project `z` and reshape
        h5 = linear(input_feature, df*10*s32_h*s32_w,
                    scope='g_h5_lin', reuse=is_reuse)
        h5 = tf.reshape(h5, [-1, s32_h, s32_w, df*10])
        h5 = activ(self.g1_bn5(h5, train=is_training, reuse=is_reuse))

        h4_1 = deconv2d(h5, df*5, name='g_h4', reuse=is_reuse)
        h4_1 = activ(self.g1_bn4(h4_1, train=is_training, reuse=is_reuse))
        h4_0 = deconv2d(
            h4_1, df*8, strides=[1, 1], name='g_h40', reuse=is_reuse)
        h4_0 = activ(self.g1_bn4_0(h4_0, train=is_training, reuse=is_reuse))

        h3_2 = deconv2d(
            h4_0, df*8, strides=[2, 2], name='g_h32', reuse=is_reuse)
        h3_2 = activ(self.g1_bn3_2(h3_2, train=is_training, reuse=is_reuse))
        h3_1 = deconv2d(
            h3_2, df*4, strides=[1, 1], name='g_h31', reuse=is_reuse)
        h3_1 = activ(self.g1_bn3_1(h3_1, train=is_training, reuse=is_reuse))
        h3_0 = deconv2d(
            h3_1, df*6, strides=[1, 1], name='g_h30', reuse=is_reuse)
        h3_0 = activ(self.g1_bn3_0(h3_0, train=is_training, reuse=is_reuse))

        h2_2 = deconv2d(
            h3_0, df*6, strides=[2, 2], name='g_h22', reuse=is_reuse)
        h2_2 = activ(self.g1_bn2_2(h2_2, train=is_training, reuse=is_reuse))
        h2_1 = deconv2d(
            h2_2, df*3, strides=[1, 1], name='g_h21', reuse=is_reuse)
        h2_1 = activ(self.g1_bn2_1(h2_1, train=is_training, reuse=is_reuse))
        h2_0 = deconv2d(
            h2_1, df*4, strides=[1, 1], name='g_h20', reuse=is_reuse)
        h2_0 = activ(self.g1_bn2_0(h2_0, train=is_training, reuse=is_reuse))

        h1_2 = deconv2d(
            h2_0, df*4, strides=[2, 2], name='g_h12', reuse=is_reuse)
        h1_2 = activ(self.g1_bn1_2(h1_2, train=is_training, reuse=is_reuse))
        h1_1 = deconv2d(
            h1_2, df*2, strides=[1, 1], name='g_h11', reuse=is_reuse)
        h1_1 = activ(self.g1_bn1_1(h1_1, train=is_training, reuse=is_reuse))
        h1_0 = deconv2d(
            h1_1, df*2, strides=[1, 1], name='g_h10', reuse=is_reuse)
        h1_0 = activ(self.g1_bn1_0(h1_0, train=is_training, reuse=is_reuse))

        h0_2 = deconv2d(
            h1_0, df*2, strides=[2, 2], name='g_h02', reuse=is_reuse)
        h0_2 = activ(self.g1_bn0_2(h0_2, train=is_training, reuse=is_reuse))
        h0_1 = deconv2d(h0_2, df, strides=[1, 1], name='g_h01', reuse=is_reuse)
        h0_1 = activ(self.g1_bn0_1(h0_1, train=is_training, reuse=is_reuse))

        local_bg = deconv2d(h0_2, df, strides=[
                            1, 1], name='g_h01_local_v6', reuse=is_reuse, use_bias=False)
        local_bg = self.g1_bn0_1_local(
            local_bg, train=is_training, reuse=is_reuse)
        local = activ(tf.maximum(local, local_bg))

        h0_1_all = tf.concat([local, h0_1], axis=3)

        # Final
        h0_0 = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00', reuse=is_reuse)
        h0_0 = activ(self.g1_bn0_0(h0_0, train=is_training, reuse=is_reuse))

        h0 = tf.nn.tanh(deconv2d(h0_0, self.c_dim, strides=[
                        1, 1], name='g_h0', reuse=is_reuse))

        h0_0_res = deconv2d(
            h0_1_all, df*2, strides=[1, 1], name='g_h00_res', reuse=is_reuse)
        h0_0_res = activ(self.g1_bn0_0_res(
            h0_0_res, train=is_training, reuse=is_reuse))

        h0_res = tf.nn.tanh(deconv2d(h0_0_res, self.c_dim, strides=[
                            1, 1], name='g_h0_res', reuse=is_reuse))

        return h0, h0 + h0_res

    def sampler(self, input_images, with_landmark=True):

        shape_fx, tex_fx, m, il = self.generator_encoder(
            input_images, is_reuse=True, is_training=False)
        shape = self.generator_decoder_shape(
            shape_fx, is_reuse=True, is_training=False)
        albedo = self.generator_decoder_albedo(
            tex_fx, is_reuse=True, is_training=False)

        shape_full = shape * \
            tf.constant(self.std_shape) + tf.constant(self.mean_shape)
        m_full = m * tf.constant(self.std_m) + tf.constant(self.mean_m)

        shade = generate_shade(il, m_full, shape_full)
        texture = 2.0*tf.multiply((albedo + 1.0)/2.0, shade) - 1
        texture = tf.clip_by_value(texture, -1, 1)
        warped_img, mask = warp_texture(texture, m_full, shape_full)

        mask = tf.expand_dims(mask, -1)

        overlay_img = tf.multiply(warped_img, mask) + \
            tf.multiply(input_images, 1 - mask)

        return shape_full, shade, texture, albedo, m, warped_img, overlay_img

    @property
    def model_dir(self):
        # "%s_%s_%s_%s_%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size, self.gf_dim, self.gfc_dim, self.df_dim, self.dfc_dim)
        return ""

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, 0o755)

        self.saver.save(self.sess, os.path.join(
            checkpoint_dir, model_name), global_step=step)
        print(" Saved checkpoint %s-%d" %
              (os.path.join(checkpoint_dir, model_name), step))

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            #self.d_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            #self.g_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))

            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")

            return False, 0

    def load_checkpoint(self, ckpt_file):
        if os.path.isfile(ckpt_file):
            self.saver.restore(self.sess, ckpt_file)
            print(" [*] Success to read {}".format(ckpt_file))
        else:
            self.load(ckpt_file)
