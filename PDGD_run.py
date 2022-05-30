# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import json
import numpy as np
import tensorflow as tf
import time
from progressbar import progressbar
import utils.click_generation as clkgn
import utils.clicks as clk
import utils.dataset as dataset
import utils.estimators as est
import utils.evaluation as evl
import utils.nnmodel as nn
import utils.misc as misc
import utils.PDGD as PDGD
from collections import defaultdict
from tensorflow.keras.backend import set_session
import os
tf.enable_eager_execution()

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
parser.add_argument("--pretrained_model", type=str,
                    default=None,
                    help="Path to pretrianed model file.")
parser.add_argument("--n_iteration", type=int,
                    default=10**8,
                    help="how many iteractions totally")
parser.add_argument("--intervene_strategy", type=str,
                    default="gumbel",
                    help="Path to pretrianed model file.")
parser.add_argument("--n_eval", type=int,
                    default=50,
                    help="number of evalutaions")
parser.add_argument("--dropout", type=bool,
                    help="use dropout or not in model, default False",
                    default=False)
parser.add_argument("--random_seed", type=int,
                    default=0,
                    help="random seed")
args = parser.parse_args()

gpu_id=str(np.random.choice(4,1)[0])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
random_seed=args.random_seed
tf.set_random_seed(random_seed)
dropout = args.dropout
n_iteration=args.n_iteration
intervene_strategy=args.intervene_strategy
click_model_name = args.click_model
cutoff = args.cutoff
n_eval = args.n_eval
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

max_ranking_size = np.min((cutoff, data.max_query_size()))

click_model = clk.get_click_model(click_model_name)

alpha, beta = click_model(np.arange(max_ranking_size))

true_train_doc_weights = data.train.label_vector*0.25
true_vali_doc_weights = data.validation.label_vector*0.25
true_test_doc_weights = data.test.label_vector*0.25

data.train.filtered_query_sizes(max_ranking_size)
data.validation.filtered_query_sizes(max_ranking_size)
data.test.filtered_query_sizes(max_ranking_size)
output=defaultdict(list)
workspace=os.path.dirname(args.output_path)
results_path=workspace+"/result.jjson"
results = []
model_params = {'hidden units': [32, 32],}
if dropout:
  # print("using dropout model")
  model = nn.init_model_dropout(model_params)
else:
  model = nn.init_model(model_params)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.build(input_shape=data.train.feature_matrix.shape)
if args.pretrained_model:
  model.load_weights(args.pretrained_model)

n_sampled = 0

# eval_points = np.logspace(
#                           2, 7, n_eval+2,
#                           endpoint=True,
#                           dtype=np.int32)
# eval_points = np.geomspace(10**2, n_iteration, n_eval+2,
#                           endpoint=True,
#                           dtype=np.int32)
eval_points,_= misc.efficient_spacing(n_eval, n_eval,  np.log10(n_iteration), np.log10(n_iteration),True)

test_scores=nn.get_score_matrix_by_dropout_batch_query(data.test,model,1,\
                                                     use_GPU=True,training=False,batch_size=2**12)[:,0]
initial_model_metrics= evl.test_ndcg(
                              test_scores,
                              data.test,
                              true_test_doc_weights,
                              cutoff
                            )
output["initial_test_ndcg"]=[initial_model_metrics]*len(eval_points)
print("initial logging model ndcg",initial_model_metrics)
# output = {
#   'dataset': args.dataset,
#   'fold number': args.fold_id,
#   'click model': click_model_name,
#   'initial model': 'random initialization',
#   'run name': 'PDGD',
#   'number of evaluation points': n_eval,
#   'evaluation iterations': [int(x) for x in eval_points],
#   'model hyperparameters': model_params,
#   'results': results,
# }
# if args.pretrained_model:
#   output['initial model'] = args.pretrained_model

n_train_queries = data.train.num_queries()
n_vali_queries = data.validation.num_queries()
train_ratio = n_train_queries/float(n_train_queries + n_vali_queries)
result_logging=defaultdict(list)
for i in progressbar(range(np.amax(eval_points))):
  train_or_vali = np.random.choice(2, p=[1.-train_ratio, train_ratio])

  if train_or_vali:
    data_split = data.train
    qid = np.random.choice(n_train_queries)
    doc_weights = true_train_doc_weights
  else:
    data_split = data.validation
    qid = np.random.choice(n_vali_queries)
    doc_weights = true_vali_doc_weights

  (ranking, clicks, scores) = clkgn.single_ranking_generation(
                    qid,
                    data_split,
                    doc_weights,
                    alpha,
                    beta,
                    model=model,
                    return_scores=True,intervene_strategy=intervene_strategy,result_logging=result_logging)
 
  PDGD.update(model,
                optimizer,
                qid,
                data_split,
                ranking,
                clicks,
                scores)
  
  n_queries_sampled = i + 1
  if n_queries_sampled in eval_points:
      output["iterations"].append(int(n_queries_sampled))
      test_scores=nn.get_score_matrix_by_dropout_batch_query(data.test,model,1,\
                                                             use_GPU=True,training=False,batch_size=2**12)[:,0]

      cur_test_ndcg= evl.test_ndcg(
                                      test_scores,
                                      data.test,
                                      true_test_doc_weights,
                                      cutoff
                                    )
      output["test_ndcg"].append(cur_test_ndcg)


      print('No. query %09d,  NDCG %0.5f ' % (
        n_queries_sampled, cur_test_ndcg), flush=True)
logging_metrics=result_logging
for metrics_key in logging_metrics.keys():
    print(np.concatenate(logging_metrics[metrics_key]).shape,"logging_metrics[metrics_key]")
    step_metrics=np.concatenate(logging_metrics[metrics_key])
    iterations=np.array(output["iterations"]).astype(int)-1
    output["discounted_sum_"+metrics_key]=evl.dicounted_metrics(step_metrics)[iterations].tolist()
    output["overall_"+metrics_key]=[np.mean(step_metrics)]*len(output["iterations"])
print(result_logging,"result_logging")
print(output)
with open(results_path, 'w') as f:
  json.dump(output, f)
print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)

