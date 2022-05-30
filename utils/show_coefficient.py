import sys
sys.path.append("../")
import matplotlib.pyplot as plt 
from tensorflow.keras.backend import set_session
import glob
import os
import evaluation as evl
import argparse
import json
import numpy as np
import tensorflow as tf
import time
import dataset as dataset
import nnmodel as nn
import misc as misc
import evaluation as evl
from progressbar import progressbar
from scipy.stats import pearsonr
from scipy.stats import spearmanr
tf.enable_eager_execution()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--batch_size", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=64)
parser.add_argument("--n_sample", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=100)
parser.add_argument("--save_path", type=str,
                    help="Maximum number of items that can be displayed.",
                    default="../loal_output")
parser.add_argument("--model_dir", type=str,
                    help="model_dir",
                    default="../loal_output")
args = parser.parse_args()
model_dir=args.model_dir
gpu_id=str(np.random.choice(8,1)[0]) 
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

data_setting={"dataset_info_path":"../local_dataset_info.txt",
         "dataset":args.dataset,
         "fold_id":1 }
data=dataset.get_data(**data_setting)
data_split=data.train
# model_dir="/home/ec2-user/documents/uncertainty/2021wsdm-unifying-LTR/local_output_point_wise_good_initial_model/"+args.dataset+"/plackettluce/n_updates_10/trial_0/"
steps=[372, 1389, 5179, 19306, 37275, 138949, 517947, 1930697, 7196856, 26826957,10**8]

model_dir_list=glob.glob(model_dir+"*.h5")
length=min(len(model_dir_list),len(steps))
evl.plot_coefficient(model_dir_list[:length],data_split,steps[:length],n_samples=args.n_sample,batch_size=args.batch_size,save_path=args.save_path)