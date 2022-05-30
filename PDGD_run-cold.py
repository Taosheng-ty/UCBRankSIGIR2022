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
from str2bool import str2bool
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
parser.add_argument("--query_least_size", type=int,
                    default=5,
                    help="query_least_size")
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
parser.add_argument("--shown_prob", type=float,
                    default=1.0,
                    help="shown_prob")
parser.add_argument("--tradeoff_param", type=float,
                    default=None,
                    help="tradeoff_param")
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
parser.add_argument("--add_behavior",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="add_behavior, default False")
parser.add_argument("--add_ips",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="add_ips, default False")
parser.add_argument("--add_one_dim",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="add_one_dim, default False")
parser.add_argument("--ips_initialize",  type=float,
                    default=-2.0,
                    help="ips_initialize")
parser.add_argument("--linspace_sampling",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="use_ linspace_sampling, default False")
args = parser.parse_args()
ips_initialize=args.ips_initialize
add_behavior=args.add_behavior
add_one_dim=args.add_one_dim
add_ips=args.add_ips
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
shown_prob=args.shown_prob
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
query_least_size=max(args.query_least_size,cutoff)
click_model = clk.get_click_model(click_model_name)

alpha, beta = click_model(np.arange(max_ranking_size))

max_label=np.max(data.train.label_vector)
true_train_doc_weights = data.train.label_vector/max_label
true_vali_doc_weights = data.validation.label_vector/max_label
true_test_doc_weights = data.test.label_vector/max_label

data.train.filtered_query_sizes(query_least_size)
data.validation.filtered_query_sizes(query_least_size)
data.test.filtered_query_sizes(query_least_size)

train_clicks = np.zeros(data.train.num_docs())
vali_clicks = np.zeros(data.validation.num_docs())
test_clicks = np.zeros(data.test.num_docs())

train_exp_alpha = np.zeros(data.train.num_docs())
train_exp_beta = np.zeros(data.train.num_docs())

vali_exp_alpha = np.zeros(data.validation.num_docs())
vali_exp_beta = np.zeros(data.validation.num_docs())

test_exp_alpha = np.zeros(data.test.num_docs())
test_exp_beta = np.zeros(data.test.num_docs())

# if add_behavior or add_one_dim:
#     test_clicks = np.zeros(data.test.num_docs())
#     data.train.feature_matrix=np.concatenate([data.train.feature_matrix,train_clicks[:,None]],axis=1)
#     data.validation.feature_matrix=np.concatenate([data.validation.feature_matrix\
#                                                        ,vali_clicks[:,None]],axis=1)
#     data.test.feature_matrix=np.concatenate([data.test.feature_matrix,test_clicks[:,None]],axis=1)
    
#     train_last_dim=data.train.feature_matrix[:,-1]
#     vali_last_dim=data.validation.feature_matrix[:,-1]
#     test_last_dim=data.test.feature_matrix[:,-1]

if add_behavior or add_one_dim:
    if add_ips:
        train_q_weights=np.ones(data.train.num_docs())*ips_initialize
        vali_q_weights=np.ones(data.validation.num_docs())*ips_initialize
        test_q_weights = np.ones(data.test.num_docs())*ips_initialize
        data.train.feature_matrix=np.concatenate([data.train.feature_matrix,train_q_weights[:,None]],axis=1)
        data.validation.feature_matrix=np.concatenate([data.validation.feature_matrix,vali_q_weights[:,None]],axis=1)
        data.test.feature_matrix=np.concatenate([data.test.feature_matrix,test_q_weights[:,None]],axis=1) 
    else:
        data.train.feature_matrix=np.concatenate([data.train.feature_matrix,train_clicks[:,None]],axis=1)
        data.validation.feature_matrix=np.concatenate([data.validation.feature_matrix,vali_clicks[:,None]],axis=1)
        data.test.feature_matrix=np.concatenate([data.test.feature_matrix,test_clicks[:,None]],axis=1)
    train_last_dim=data.train.feature_matrix[:,-1]
    vali_last_dim=data.validation.feature_matrix[:,-1]
    test_last_dim=data.test.feature_matrix[:,-1]
    
test_feature_orig=np.copy(data.test.feature_matrix)    
    
    
    
output=defaultdict(list)
workspace=os.path.dirname(args.output_path)
results_path=workspace+"/result.jjson"
if os.path.exists(results_path):
  print("results already here")
  sys.exit()

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
print(eval_points)
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

n_train_queries =len(data.train.get_filtered_queries())
n_vali_queries = len(data.validation.get_filtered_queries())
n_test_queries = len(data.test.get_filtered_queries())
total_n_query=n_train_queries+n_vali_queries+n_test_queries
query_ratio = np.array([n_train_queries,n_vali_queries,n_test_queries])/total_n_query
# train_ratio = n_train_queries/float(n_train_queries + n_vali_queries)
result_logging=defaultdict(list)

query_rng=np.random.default_rng(random_seed)
cold_rng=np.random.default_rng(random_seed)
train_ranking_mask=dataset.get_mask(data.train,cold_rng)
train_ranking_cold_rng=cold_rng
vali_ranking_mask=dataset.get_mask(data.validation,cold_rng)
vali_ranking_cold_rng=cold_rng
test_ranking_mask=dataset.get_mask(data.test,cold_rng)
test_ranking_cold_rng=cold_rng
print(eval_points,"eval_points",flush=True)
prefix={0:"train_",1:"vali_",2:"test_"}
for i in range(np.amax(eval_points)):
  train_vali_or_test = query_rng.choice(3, p=query_ratio)

  if train_vali_or_test==0:
    data_split = data.train
    qid = np.random.choice(data_split.get_filtered_queries())
    doc_weights = true_train_doc_weights
    mask=data.train.query_values_from_vector(qid,train_ranking_mask)
    if add_behavior or add_one_dim:
        last_dim=data.train.query_values_from_vector(qid,train_last_dim)
    exp_clicks= data.train.query_values_from_vector(qid,train_clicks)
    exp_alpha = data.train.query_values_from_vector(qid,train_exp_alpha)
    exp_beta = data.train.query_values_from_vector(qid,train_exp_beta)
  elif train_vali_or_test==1:
    data_split = data.validation
    qid = np.random.choice(data_split.get_filtered_queries())
    doc_weights = true_vali_doc_weights
    mask=data.validation.query_values_from_vector(qid,vali_ranking_mask)
    if add_behavior or add_one_dim:
        last_dim=data.validation.query_values_from_vector(qid,vali_last_dim)
    exp_clicks= data.validation.query_values_from_vector(qid,vali_clicks)
    exp_alpha = data.validation.query_values_from_vector(qid,vali_exp_alpha)
    exp_beta = data.validation.query_values_from_vector(qid,vali_exp_beta)
  elif train_vali_or_test==2:
    data_split = data.test
    qid = np.random.choice(data_split.get_filtered_queries())
    doc_weights = true_test_doc_weights
    mask=data.test.query_values_from_vector(qid,test_ranking_mask)
    if add_behavior or add_one_dim:
        last_dim=data.test.query_values_from_vector(qid,test_last_dim)
    exp_clicks= data.test.query_values_from_vector(qid,test_clicks)
    exp_alpha = data.test.query_values_from_vector(qid,test_exp_alpha)
    exp_beta = data.test.query_values_from_vector(qid,test_exp_beta)
  else:
    raise 
#   print(qid,"qid")
  (ranking, clicks, scores) = clkgn.single_ranking_generation_cold(
                    qid,
                    data_split,
                    doc_weights,
                    alpha,
                    beta,
                    model=model,
                    return_scores=True,result_logging=result_logging,\
                    mask=mask,mask_rng=cold_rng,prob=shown_prob,prefix=prefix[train_vali_or_test])
#   print(exp_alpha.shape,ranking.shape,alpha.shape,"exp_alpha.shape,ranking.shape,alpha.shape")
  np.add.at(exp_alpha,ranking,alpha)
  np.add.at(exp_beta,ranking,beta)
  np.add.at(exp_clicks,ranking,clicks)
  if add_behavior:
        if add_ips:
            shown_id=exp_alpha>0
            corrected_cur=(exp_clicks-exp_beta)/(exp_alpha+1e-5)
            last_dim[shown_id]=corrected_cur[shown_id]
            
        else:
#             print(last_dim.shape,exp_clicks.shape,"last_dim.shape")
            last_dim[:]=exp_clicks
#             vali_last_dim[:]=vali_clicks
#             train_last_dim[:]=train_clicks
#             print("add to last dim")
#             np.add.at(last_dim,ranking,clicks)
  if train_vali_or_test==0:
      PDGD.update(model,
                    optimizer,
                    qid,
                    data_split,
                    ranking,
                    clicks,
                    scores)
  
  n_queries_sampled = i + 1
  if n_queries_sampled in eval_points:
#         print(last_dim[:50])
#         print(train_last_dim.shape)
#         print(np.sum(train_last_dim[:100]),"np sum")
#         print(np.sum(train_clicks[:100]),"np sum")
#         print(np.sum(data.train.feature_matrix[:,-1][:100]),"np sum")
        output["num_shown_train_ids_uniq"].append(int(np.sum(train_ranking_mask==1)))
        output["num_shown_vali_ids_uniq"].append(int(np.sum(vali_ranking_mask==1)))
        output["num_shown_test_ids_uniq"].append(int(np.sum(test_ranking_mask==1)))
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
        model_test_scores_ignore_behav=nn.get_score_matrix_by_dropout_batch_query(data.test,model,1,\
                                                             use_GPU=True,training=False,\
                                                               batch_size=2**12,feature=test_feature_orig)[:,0]
        test_mask=test_ranking_mask
        test_scores_masked=dataset.mask_items(test_mask,test_scores,flag=1)
        test_dcg= evl.test_dcg(
                          test_scores,
                          data.test,
                          true_test_doc_weights,
                          cutoff
                        )
        output["test_warm_dcg"].append(test_dcg)
        cur_test_ndcg= evl.test_ndcg(
                                      test_scores,
                                      data.test,
                                      true_test_doc_weights,
                                      cutoff
                                    )
        output["test_warm_ndcg_unmasked"].append(cur_test_ndcg)
 
        cur_test_ndcg_masked= evl.test_ndcg(
                                      test_scores_masked,
                                      data.test,
                                      true_test_doc_weights,
                                      cutoff
                                    )
        output["test_warm_ndcg_masked"].append(cur_test_ndcg_masked)
        model_test_scores_ignore_behav_masked=dataset.mask_items(test_mask,model_test_scores_ignore_behav,flag=1)
        cur_test_ndcg_cold= evl.test_ndcg(
                                      model_test_scores_ignore_behav,
                                      data.test,
                                      true_test_doc_weights,
                                      cutoff
                                    )
        output["test_cold_ndcg_unmasked"].append(cur_test_ndcg_cold)  

        cur_test_ndcg_cold_masked= evl.test_ndcg(
                                      model_test_scores_ignore_behav_masked,
                                      data.test,
                                      true_test_doc_weights,
                                      cutoff
                                    )
        output["test_cold_ndcg_masked"].append(cur_test_ndcg_cold_masked) 
        
        test_unfairness= evl.test_unfairness(
                                      test_exp_alpha+test_exp_beta,
                                      data.test,
                                      true_test_doc_weights,
                                      test_mask
                                    )
        output["test_unfairness"].append(test_unfairness)     
        train_unfairness= evl.test_unfairness(
                                      train_exp_alpha+train_exp_beta,
                                      data.train,
                                      true_train_doc_weights,
                                      train_ranking_mask
                                    )
        output["train_unfairness"].append(train_unfairness)
        output["shown_test_item_ratio"].append(np.sum(test_mask==1)/test_mask.shape[0])
        print(output["shown_test_item_ratio"][-1],"test shown ratio")
        logging_metrics=result_logging
        for metrics_key in logging_metrics.keys():
            print(np.concatenate(logging_metrics[metrics_key]).shape,"logging_metrics[metrics_key]")
            step_metrics=np.concatenate(logging_metrics[metrics_key])
            output["discounted_sum_"+metrics_key].append(evl.dicounted_metrics(step_metrics)[-1])
            output["overall_"+metrics_key].append(np.mean(step_metrics))
# print(result_logging,"result_logging")
print(output)
with open(results_path, 'w') as f:
  json.dump(output, f)
print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)

