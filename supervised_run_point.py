# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import pickle
import random
import tensorflow as tf
import json
import utils.dataset as dataset
import utils.clicks as clk
import utils.nnmodel as nn
import utils.optimization as opt
import utils.evaluation as evl
import os
from str2bool import str2bool
from tensorflow.keras.backend import set_session

import logging
evl.configure_logging(logging)
parser = argparse.ArgumentParser()
# parser.add_argument("model_file", type=str,
#                     help="Model file output from pretrained model.")
parser.add_argument("--output_path", type=str,
                    help="Path to output model.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--click_model", type=str,
                    help="Name of click model to use.",
                    default='default')
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="local_dataset_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--cutoff", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
parser.add_argument("--n_train_queries_ratio", type=float,
                    help="ratio of randomly selected training queries used for training.",
                    default=1.0)
parser.add_argument("--n_vali_queries_ratio", type=float,
                    help="ratio of randomly selected training queries used for early stopping.",
                    default=1.0)
parser.add_argument("--epochs", type=int,
                    default=100,
                    help="number of epoches during training")
parser.add_argument("--use_GPU",  type=str2bool, nargs='?',
                        const=True, default=True,
                    help="use_GPU or not. Default false.")
args = parser.parse_args()
gpu_id=str(np.random.choice(4,1)[0]) if args.use_GPU else str(-1)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

click_model_name = args.click_model
# cutoff = args.cutoff
epochs= args.epochs

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                  shared_resource = False,
                )
fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]


start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

true_train_doc_weights = data.train.label_vector*0.25
true_vali_doc_weights = data.validation.label_vector*0.25
true_test_doc_weights = data.test.label_vector*0.25

n_train_queries = int(args.n_train_queries_ratio*data.train.num_queries())
n_vali_queries = int(args.n_vali_queries_ratio*data.validation.num_queries())

train_selection = np.random.choice(data.train.num_queries(),
                                  replace=False,
                                  size=n_train_queries)
vali_selection = np.random.choice(data.validation.num_queries(),
                                  replace=False,
                                  size=n_vali_queries)

model_params = {'hidden units': [32, 32],}
model = nn.init_model_dropout(model_params)
model.build(input_shape=data.train.feature_matrix.shape)

# optimizer = tf.keras.optimizers.Adam()
train_selection = np.random.choice(data.train.num_queries(),
                                  replace=False,
                                  size=n_train_queries)
vali_selection = np.random.choice(data.validation.num_queries(),
                                  replace=False,
                                  size=n_vali_queries)


queue_train_ids=data.train.get_subset_doc_ids(train_selection)
queue_vali_ids=data.validation.get_subset_doc_ids(vali_selection)
training_generator=dataset.DataGenerator(
                  queue_train_ids,
                  data.train.feature_matrix,
                  true_train_doc_weights,
                  # batch_size=256,
                  batch_size=1024,
                  )
validation_generator=dataset.DataGenerator(
                  queue_vali_ids,
                  data.validation.feature_matrix,
                  true_vali_doc_weights,
                  batch_size=1024,
                  # batch_size=1024,
                  )


# model.set_weights(init_weights)
model_path=os.path.join(args.output_path,"model.h5")
if os.path.exists(model_path) and False:
# if False:
  print(model_path)
  logging.info("found existing model, no training.")
  model.load_weights(model_path)
else:
  # optimizer =tf.keras.optimizers.Adagrad(learning_rate=0.01)
  # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
  model.compile(optimizer=optimizer,
                loss="mse", # Call the loss function with the selected layer
                metrics=["mse"])
  callback=tf.keras.callbacks.EarlyStopping(
    monitor="val_mean_squared_error",
    # monitor="val_loss_sigmoid",
    min_delta=0,
    patience=20,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
  )
  logging.info("begin fit")
  history=model.fit_generator(generator=training_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=[callback],
                verbose=2,
                workers=2,
                use_multiprocessing=True,
                max_queue_size=100,
  )
  model.save(model_path)
output={
  'iterations': [1],
}
cutoffs=np.arange(1,20)
supervised_model_score=nn.get_score_matrix_by_dropout_batch_query(data.test,model,1,\
                                                              use_GPU=args.use_GPU,training=False,batch_size=2**10)[:,0]
for cutoff in cutoffs:
  cur_test_ndcg= evl.test_ndcg(
                                  supervised_model_score,
                                  data.test,
                                  true_test_doc_weights,
                                  cutoff
                                )
  output["ndcg@"+str(cutoff)]=[cur_test_ndcg]
results_path=os.path.join(args.output_path,"result.jjson")
print('Writing results to %s' % results_path)
with open(results_path, 'w') as f:
  json.dump(output, f)

