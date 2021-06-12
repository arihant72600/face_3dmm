import os
import scipy.misc
import numpy as np

from model_two_res_3d_recon_with_base import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002,
                   "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 1, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 1, "The size of batch samples images [64]")
flags.DEFINE_integer(
    "image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer(
    "output_size", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_boolean("is_with_y", True, "True for with lable")
flags.DEFINE_string("dataset", "celebA",
                    "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint",
                    "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("samples_dir", "samples",
                    "Directory name to save the image samples [samples]")
flags.DEFINE_boolean(
    "is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_reduce", False,
                     "True for 6k verteices, False for 50k vertices")
flags.DEFINE_boolean(
    "is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False,
                     "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("gf_dim", 32, "")
flags.DEFINE_integer("gfc_dim", 512, "")
flags.DEFINE_integer("df_dim", 32, "")
flags.DEFINE_integer("dfc_dim", 512, "")
flags.DEFINE_integer("z_dim", 50, "")
flags.DEFINE_string("gpu", "-1", "GPU to use [0]")

flags.DEFINE_boolean("is_partbase_albedo", False,
                     "Using part based albedo decoder [False]")

FLAGS = flags.FLAGS
# tf.app.flags.FLAGS.flag_values_dict()


def main(_):
    # pp.pprint(flags.FLAGS.__flags)
    pp.pprint(tf.app.flags.FLAGS.flag_values_dict())

    gpu_options = tf.GPUOptions(visible_device_list=FLAGS.gpu,
                                per_process_gpu_memory_fraction=0.99, allow_growth=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:
        dcgan = DCGAN(sess, FLAGS)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:

            save_folder = FLAGS.checkpoint_dir.split('/')[-1]
            dcgan.load_checkpoint(FLAGS.checkpoint_dir)
            # dcgan.evaluation_AFLW2000(folder=save_folder)
            # dcgan.evaluation_CelebA(folder=save_folder)

            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/data/300VW/recropped/031/', output_folder = './images/300VW_031/' )

            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/data/300VW/recropped/041/', output_folder = './images/300VW_041_final_2/' )

            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/data/300VW/recropped/004/', output_folder = './images/300VW_004_final/' )

            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/Repos/Nonlinear_3DMM_sz224/data_MoFA_small/', output_folder = './images/MoFA_small/' )
            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/data/patient/cropped_aligned/', output_folder = '/home/luan/Documents/data/patient/3d_aligned/' )
            # dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/data/patient/cropped_aligned2/', output_folder = '/home/luan/Documents/data/patient/3d_aligned_mask_4/')#, mask_folder='/home/luan/Documents/data/patient/cropped_aligned_mask/' )
            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/data/patient/cropped_aligned2/', output_folder = '/home/luan/Documents/data/patient/3d_aligned_mask_5/', mask_folder='/home/luan/Documents/data/patient/masks2/' )
            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/Repos/Nonlinear_3DMM_sz224/images/florence/', output_folder = './images/florence/')
            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan//Documents/data/Masa_Clip/cropped_frames_small/', output_folder = './images/Masa_clip_small/')
            #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan//Documents/data//0lSLIhXrs1Y_frames/cropped/clip1/', output_folder = './images/Obama_clip1/')
            # for i in [4,136,138,145,146,147,148,149,150]:
            #    #dcgan.evaluation_LWF(folder=save_folder, data_folder='/home/luan/Documents/data/FaceWarehouse_recrop/images/Tester_%d/TrainingPose/' % i, output_folder = './images/FaceWarehouse_2/Tester_%d/'%i )
            #    dcgan.evaluation_LWF_wmask(folder=save_folder, data_folder='/home/luan/Documents/data/FaceWarehouse_recrop/images/Tester_%d/TrainingPose/' % i, output_folder = './images/FaceWarehouse_2/Tester_%d/'%i, mask_folder='/home/luan/Documents/data/FaceWarehouse_recrop/masks/Tester_%d/TrainingPose/' % i)

            #dcgan.evaluation_LWF_recursive(folder=save_folder, data_folder='/home/luan/Documents/data/NoW_Dataset/final_release_version/iphone_pictures_cropped/', output_folder = '/home/luan/Documents/data/NoW_Dataset/final_release_version/iphone_pictures_recon/' )
            dcgan.evaluation_LWF_recursive(
                folder=save_folder, data_folder='../../img_align_celeba', output_folder='../../img_align_celeba-out')


if __name__ == '__main__':
    tf.app.run()
