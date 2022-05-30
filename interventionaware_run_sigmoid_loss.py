import argparse
import json
import numpy as np
import tensorflow as tf
import time
import os
import gc
from collections import defaultdict
import utils.PDGD as PDGD
from str2bool import str2bool
import utils.click_generation as clkgn
import utils.clicks as clk
import utils.dataset as dataset
import utils.estimators as est
import utils.evaluation as evl
import utils.nnmodel as nn
import utils.optimization as opt
import utils.misc as misc
from progressbar import progressbar
from collections import deque
import sys
tf.enable_eager_execution()
from tensorflow.keras.backend import set_session
from  tensorflow.keras.utils import OrderedEnqueuer
import logging
evl.configure_logging(logging)
import os, psutil
process = psutil.Process(os.getpid())
# gc.set_threshold(1, 1, 1)
class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
#         clear_session()

def return_batchid(length,ceilling_base):
    batch_size_candi=2**np.arange(4,ceilling_base)
    batch_size_id=np.sum(length>batch_size_candi).astype(np.int)-1
    return batch_size_candi[batch_size_id]
# def mean_std_re
# logging.basicConfig(
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     level=logging.INFO,
#     datefmt='%Y-%m-%d %H:%M:%S')
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
parser.add_argument("--n_updates", type=int,
                    help="Number of updates during run.",
                    default=0)
parser.add_argument("--pretrained_model", type=str,
                    default=None,
                    help="Path to pretrianed model file.")
parser.add_argument("--supervised_model", type=str,
                    default=None,
                    help="Path to supervised_model model file.")
parser.add_argument("--intervene_strategy", type=str,
                    default="gumbel",
                    help="Path to pretrianed model file.")
parser.add_argument("--n_iteration", type=int,
                    default=10**8,
                    help="how many iteractions totally")
parser.add_argument("--n_eval", type=int,
                    default=20,
                    help="number of evalutaions")
parser.add_argument("--session_aware",  type=str2bool, nargs='?',
                        const=True, default=True,
                    help="session_aware training or not. Default false.")
parser.add_argument("--query_least_size", type=int,
                    default=10,
                    help="query_least_size")
parser.add_argument("--epochs", type=int,
                    default=100,
                    help="number of epoches during training")
parser.add_argument("--ucb_param_c", type=float,
                    default=None,
                    help="number of epoches during training")
parser.add_argument("--random_param_c", type=float,
                    default=None,
                    help="number of epoches during training")
parser.add_argument("--n_dropout", type=int,
                    default=10,
                    help="number of dropout")
parser.add_argument("--risk_averse_param", type=float,
                    default=None,
                    help="risk averse param")
parser.add_argument("--risk_averse_param_v1", type=float,
                    default=None,
                    help="risk averse param")
parser.add_argument("--tradeoff_param", type=float,
                    default=None,
                    help="tradeoff_param")
parser.add_argument("--optimizer", type=str,
                    default="Adam",
                    help="optimizer")
parser.add_argument("--single_training",  type=str2bool, nargs='?',
                        const=True, default=False,
                    help="single training or not. Default false.")
parser.add_argument("--use_GPU",  type=str2bool, nargs='?',
                        const=True, default=True,
                    help="use_GPU or not. Default True.")
parser.add_argument("--cold_start",  type=str2bool, nargs='?',
                        const=True, default=True,
                    help="use_GPU or not. Default True.")
parser.add_argument("--early_stop_patience", type=int,
                    default=50,
                    help="number of epoch for early stop")
parser.add_argument("--batch_size", type=int,
                    default=2**15,
                    help="number of epoch for early stop")
parser.add_argument("--shown_prob", type=float,
                    default=0.2,
                    help="shown_prob")
parser.add_argument("--random_seed", type=int,
                    default=0,
                    help="random seed")
parser.add_argument("--linspace_sampling",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="use_ linspace_sampling, default False")
parser.add_argument("--add_behavior",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="add_behavior, default False")
parser.add_argument("--add_ips",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="add_ips, default False")
parser.add_argument("--add_ctr",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="add_ctr, default False")
parser.add_argument("--add_one_dim",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="add_one_dim, default False")

parser.add_argument("--pdgd_update",  type=str2bool, nargs='?',
                        const=False, default=False,
                    help="add_one_dim, default False")
parser.add_argument("--ips_initialize",  type=float,
                    default=-2.0,
                    help="ips_initialize")
args = parser.parse_args()
gpu_id=str(np.random.choice([0,2,3],1)[0]) if args.use_GPU else str(-1)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
early_stop_patience=args.early_stop_patience
single_training=args.single_training
ucb_param_c=args.ucb_param_c
tradeoff_param=args.tradeoff_param
optimizer=args.optimizer
ips_initialize=args.ips_initialize
risk_averse_param=args.risk_averse_param
random_param_c=args.random_param_c
risk_averse_param_v1=args.risk_averse_param_v1
n_dropout=args.n_dropout
batch_size=args.batch_size
cold_start=args.cold_start
shown_prob=args.shown_prob
add_behavior=args.add_behavior
add_one_dim=args.add_one_dim
add_ips=args.add_ips
add_ctr=args.add_ctr
pdgd_update=args.pdgd_update
linspace_sampling=args.linspace_sampling
random_seed=args.random_seed
query_rng=np.random.default_rng(random_seed)
cold_rng=np.random.default_rng(random_seed)
tf.set_random_seed(random_seed)
# if random_seed>=0:
#     np.random.seed(random_seed)
print(args)
workspace=os.path.dirname(args.output_path)
results_path=workspace+"/result.jjson"
if os.path.exists(results_path):
  print("results already here")
  sys.exit()
click_model_name = args.click_model
epochs= args.epochs
cutoff = args.cutoff
n_updates = args.n_updates
n_iteration=args.n_iteration
session_aware=args.session_aware
query_least_size=max(args.query_least_size,cutoff)
intervene_strategy=args.intervene_strategy
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
q_aver_length=dataset.get_query_aver_length(data)
q_batch_size=batch_size//q_aver_length
max_ranking_size = np.min((cutoff, data.max_query_size()))

click_model = clk.get_click_model(click_model_name)

alpha, beta = click_model(np.arange(max_ranking_size))

data.train.filtered_query_sizes(query_least_size)
data.validation.filtered_query_sizes(query_least_size)
data.test.filtered_query_sizes(query_least_size)
max_label=np.max(data.train.label_vector)
true_train_doc_weights = data.train.label_vector/max_label
true_vali_doc_weights = data.validation.label_vector/max_label
true_test_doc_weights = data.test.label_vector/max_label

train_clicks = np.zeros(data.train.num_docs())
train_exp_alpha = np.zeros(data.train.num_docs())
train_exp_beta = np.zeros(data.train.num_docs())

vali_clicks = np.zeros(data.validation.num_docs())
vali_exp_alpha = np.zeros(data.validation.num_docs())
vali_exp_beta = np.zeros(data.validation.num_docs())

test_clicks = np.zeros(data.test.num_docs())
test_exp_alpha = np.zeros(data.test.num_docs())
test_exp_beta = np.zeros(data.test.num_docs())

train_q_weights=np.ones(data.train.num_docs())*ips_initialize
vali_q_weights=np.ones(data.validation.num_docs())*ips_initialize
test_q_weights=np.ones(data.test.num_docs())*ips_initialize

train_clicks = np.zeros(data.train.num_docs())
vali_clicks = np.zeros(data.validation.num_docs())
test_clicks = np.zeros(data.test.num_docs())

train_ctr = np.zeros(data.train.num_docs())
vali_ctr = np.zeros(data.validation.num_docs())
test_ctr = np.zeros(data.test.num_docs())

if add_behavior:
    if add_ips:
        data.train.feature_matrix=np.concatenate([data.train.feature_matrix,train_q_weights[:,None]],axis=1)
        data.validation.feature_matrix=np.concatenate([data.validation.feature_matrix,vali_q_weights[:,None]],axis=1)
        data.test.feature_matrix=np.concatenate([data.test.feature_matrix,test_q_weights[:,None]],axis=1) 
    elif add_ctr:
        data.train.feature_matrix=np.concatenate([data.train.feature_matrix,train_ctr[:,None]],axis=1)
        data.validation.feature_matrix=np.concatenate([data.validation.feature_matrix,vali_ctr[:,None]],axis=1)
        data.test.feature_matrix=np.concatenate([data.test.feature_matrix,test_ctr[:,None]],axis=1)
    else:
        data.train.feature_matrix=np.concatenate([data.train.feature_matrix,train_clicks[:,None]],axis=1)
        data.validation.feature_matrix=np.concatenate([data.validation.feature_matrix,vali_clicks[:,None]],axis=1)
        data.test.feature_matrix=np.concatenate([data.test.feature_matrix,test_clicks[:,None]],axis=1)
test_feature_orig=np.copy(data.test.feature_matrix)

model_params = {'hidden units': [32, 32],}
model = nn.init_model_dropout(model_params)
logging_model = nn.init_model_dropout(model_params)

model.build(input_shape=data.train.feature_matrix.shape)
logging_model.build(input_shape=data.train.feature_matrix.shape)
if args.pretrained_model:
    model.load_weights(args.pretrained_model)
    logging_model.load_weights(args.pretrained_model)
else:
    logging_model.set_weights(model.get_weights())
if args.supervised_model:
  logging.info("supervised_model evaluation")
  supervised_model=nn.init_model_dropout(model_params)
  supervised_model.build(input_shape=data.train.feature_matrix.shape)
  supervised_model.load_weights(args.supervised_model)
  supervised_model_score=nn.get_score_matrix_by_dropout_batch_query(data.test,supervised_model,1,\
                                                              use_GPU=args.use_GPU,training=False,batch_size=q_batch_size)[:,0]
    # train_policy_scores_dropout=nn.get_score_matrix_by_dropout_batch_query(data.train.feature_matrix,model_dropout,n_dropout,training=True,batch_size=q_batch_size)
  supervised_model_metrics = evl.test_ndcg(
                                supervised_model_score,
                                data.test,
                                true_test_doc_weights,
                                cutoff
                              )
  print("supervised_model_metrics",supervised_model_metrics)
  logging.info("supervised_model finished")
get_model_params={"model_params":model_params,
                "input_shape":data.train.feature_matrix.shape,
                "weight":model.get_weights(),
                "dropout_fixed":True}

model_dropout=nn.get_dropout_model(**get_model_params)
init_weights = model.get_weights()
# init_opt_weights = optimizer.get_weights()

n_sampled = 0



n_eval = args.n_eval
(eval_points,
 update_incl_points) = misc.efficient_spacing(n_eval, n_updates, 2, np.log10(n_iteration),linspace_sampling)
update_points = update_incl_points
# if eval_points.shape[0] > update_incl_points.shape[0]:
#   iter_points = eval_points
# else:
iter_points = eval_points
output=defaultdict(list)
logging_metrics=defaultdict(list)
training_metrics=[]
results = []
setting_output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'click model': click_model_name,
  'initial model': 'random initialization',
  'run name': 'time-aware estimator',
  'number of updates': n_updates,
  'number of evaluation points': n_eval,
  'update iterations': [int(x) for x in update_points],
  'evaluation iterations': [int(x) for x in eval_points],
  'model hyperparameters': model_params,
}

# if args.pretrained_model:
setting_output['initial model'] = args.pretrained_model
initial_model_score=nn.get_score_matrix_by_dropout_batch_query(data.test,logging_model,1,use_GPU=args.use_GPU,\
                                                               training=False,batch_size=q_batch_size)[:,0]

initial_model_metrics = evl.test_ndcg(
                                initial_model_score,
                                data.test,
                                true_test_doc_weights,
                                cutoff
                              )
output["initial_test_ndcg"]=[initial_model_metrics]*len(update_points)
print("initial logging model ndcg",initial_model_metrics)
with open(args.output_path, 'w') as f:
  json.dump(setting_output, f)
# output={
#   'iterations': [],
#   'logging_ndcg': [],
#   "test_ndcg":[],
#   "vali_ndcg":[],
#   "overall_logging_ndcg":[],
#   "shown_result_mean": [],
#   "shown_result_std": [],
#   "unshown_result_mean": [],
#   "unshown_result_std": [],
#   "dropout_correlation":[],
#   "mean_std_corr":[],
#   "label_std_corr":[],
#   "label_mean_corr":[]
# }



# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# logging_ndcg_list=[]

# queue_len=int(iter_points[-1]-iter_points[-2])
# train_queue = deque([], maxlen = queue_len)
# vali_queue = deque([], maxlen = queue_len)


train_list = np.array([])
vali_list = np.array([])
test_list = np.array([])
train_set = np.array([])
vali_set = np.array([])
test_set = np.array([])

# vali_policy_scores = logging_model(data.validation.feature_matrix)[:, 0].numpy()
# train_policy_scores = logging_model(data.train.feature_matrix)[:, 0].numpy()

train_policy_scores =nn.get_score_matrix_by_dropout_batch_query(data.train,logging_model,1,use_GPU=args.use_GPU,training=False,batch_size=q_batch_size)[:,0]
vali_policy_scores = nn.get_score_matrix_by_dropout_batch_query(data.validation,logging_model,1,use_GPU=args.use_GPU,training=False,batch_size=q_batch_size)[:,0]
test_policy_scores = nn.get_score_matrix_by_dropout_batch_query(data.test,logging_model,1,use_GPU=args.use_GPU,training=False,batch_size=q_batch_size)[:,0]
print(train_policy_scores.shape)
# train_policy_scores[:] = logging_model(data.train.feature_matrix)[:, 0].numpy()
# vali_policy_scores[:] = logging_model(data.validation.feature_matrix)[:, 0].numpy()

train_n_doc=data.train.num_docs()
train_n_query=data.train.num_queries()

vali_n_doc=data.validation.num_docs()
vali_n_query=data.validation.num_queries()

test_n_doc=data.test.num_docs()
test_n_query=data.test.num_queries()

input_feature=data.test.feature_matrix.shape
mean_std_param={"model":model,"model_params":model_params,"input_feature":input_feature,
        "data_split":data.test,"n_dropout":n_dropout,"q_batch_size":q_batch_size}

initial_doc_freq_value=1
train_doc_freq=np.ones(train_n_doc)*initial_doc_freq_value
train_query_freq=np.ones(train_n_query)

vali_doc_freq=np.ones(vali_n_doc)*initial_doc_freq_value
vali_query_freq=np.ones(vali_n_query)

test_doc_freq=np.ones(test_n_doc)*initial_doc_freq_value
test_query_freq=np.ones(test_n_query)

train_doc_query_freq=np.zeros(train_n_doc)
vali_doc_query_freq=np.zeros(vali_n_doc)
test_doc_query_freq=np.zeros(test_n_doc)
# train_q_weights_cur=np.ones(train_n_doc)
# vali_q_weights_cur=np.ones(vali_n_doc)
# test_q_weights_cur=np.ones(test_n_doc)

train_ranking_param={"policy_scores":train_policy_scores,
                  "intervene_strategy":intervene_strategy,
                  "select_fn":data.train.query_values_from_vector,
                  "doc_freq":train_doc_freq,
                  "query_freq":train_query_freq,
                  "ucb_param_c":ucb_param_c,
                  "random_param_c":random_param_c,
                  "risk_averse_param":risk_averse_param,
                  "risk_averse_param_v1":risk_averse_param_v1,
                  "tradeoff_param":tradeoff_param,
                  "cold_start":cold_start,
                  "shown_prob":shown_prob,
                    "ips_score":train_q_weights,
                    "doc_query_freq":train_doc_query_freq}
vali_ranking_param={"policy_scores":vali_policy_scores,
                   "intervene_strategy":intervene_strategy,
                   "select_fn":data.validation.query_values_from_vector,
                   "doc_freq":vali_doc_freq,
                  "query_freq":vali_query_freq,
                  "tradeoff_param":tradeoff_param,
                  "cold_start":cold_start,
                  "shown_prob":shown_prob,
                    "ips_score":vali_q_weights,
                    "doc_query_freq":vali_doc_query_freq}
test_ranking_param={"policy_scores":test_policy_scores,
                  "intervene_strategy":intervene_strategy,
                  "select_fn":data.test.query_values_from_vector,
                  "doc_freq":test_doc_freq,
                  "query_freq":test_query_freq,
                  "tradeoff_param":tradeoff_param,
                  "cold_start":cold_start,
                  "shown_prob":shown_prob,
                    "ips_score":test_q_weights,
                    "doc_query_freq":test_doc_query_freq}
if cold_start:
  train_ranking_param["mask"]=dataset.get_mask(data.train,cold_rng)
  train_ranking_param["cold_rng"]=cold_rng
  vali_ranking_param["mask"]=dataset.get_mask(data.validation,cold_rng)
  vali_ranking_param["cold_rng"]=cold_rng
  test_ranking_param["mask"]=dataset.get_mask(data.test,cold_rng)
  test_ranking_param["cold_rng"]=cold_rng
else:
  train_ranking_param["mask"]=None
  vali_ranking_param["mask"]=None 
  test_ranking_param["mask"]=None 
# logging.info("dropout scores")
if intervene_strategy in ["dropout","dropout_shuffle","portfolio","portfolio_v1","ucb_std","std_proposional","portfolio_proposional"]:
  logging.info("dropout scores")
# train_policy_scores_dropout=nn.get_score_matrix_by_dropout(data.train.feature_matrix,model_dropout,n_dropout,use_GPU=args.use_GPU,training=True)
# vali_policy_scores_dropout=nn.get_score_matrix_by_dropout(data.validation.feature_matrix,model_dropout,n_dropout,use_GPU=args.use_GPU,training=True)
  train_policy_scores_dropout=nn.get_score_matrix_by_dropout_batch_query(data.train,model_dropout\
                                                                         ,n_dropout,training=True,batch_size=q_batch_size)
  vali_policy_scores_dropout=nn.get_score_matrix_by_dropout_batch_query(data.validation,model_dropout,\
                                                                        n_dropout,training=True,batch_size=q_batch_size)

  test_policy_scores_dropout=nn.get_score_matrix_by_dropout_batch_query(data.test,model_dropout,\
                                                                      n_dropout,training=True,batch_size=q_batch_size)                    # scores=nn.get_score_matrix_by_dropout_batch_query(data_split,model_dropout,n_samples,batch_size=batch_size)
  train_ranking_param["policy_scores_dropout"]=train_policy_scores_dropout
  vali_ranking_param["policy_scores_dropout"]=vali_policy_scores_dropout
  test_ranking_param["policy_scores_dropout"]=test_policy_scores_dropout
  logging.info("dropout scores finished")
if single_training:
  iter_points=[n_iteration]
train_qids_set=[]
vali_qids_set=[]
test_qids_set=[]
model_path=workspace+"/"+str(0)+"_steps_model.h5"
model.save(model_path)

# if 
ranking_pdgd=[]
clicks_pdgd=[]
scores_pdgd=[]
qids_pdgd=[]

shown_test_id=[]
last_test_shown_id=[]
for n_queries_sampled in progressbar(iter_points):

  memory=process.memory_info().rss//2**20
#   time.sleep(1)
  logging.info(str(memory)+"before click simulation")  # in bytes
    
  logging.info("begin click simulation")
  
  (cur_train_rankings,cur_train_clicks,\
  cur_train_qids,cur_train_scores,cur_vali_rankings,\
  cur_vali_clicks,cur_vali_qids,\
   cur_test_rankings,\
  cur_test_clicks,cur_test_qids,result_logging)=clkgn.simulate_on_dataset_intervene_with_test(
                                                data.train,
                                                data.validation,
                                                data.test,
                                                n_queries_sampled - n_sampled,
                                                true_train_doc_weights,
                                                true_vali_doc_weights,
                                                true_test_doc_weights,
                                                alpha,
                                                beta,
                                              train_ranking_param=train_ranking_param,
                                              vali_ranking_param=vali_ranking_param,
                                              test_ranking_param=test_ranking_param,
                                              intervene_strategy=intervene_strategy,query_rng=query_rng)


  memory=process.memory_info().rss//2**20
#   time.sleep(1)
  logging.info(str(memory)+"finished click simulation")  # in bytes
  logging.info("finished click simulation")
  if cold_start:
    train_mask=train_ranking_param["mask"]
    vali_mask=vali_ranking_param["mask"]
    test_mask=test_ranking_param["mask"]
    num_existing_item=np.sum(train_mask==1)+np.sum(vali_mask==1)+np.sum(test_mask==1)
    output["num_existing_item"].append(int(num_existing_item))
    output["num_existing_train_item"].append(int(np.sum(train_mask==1)))
    output["num_existing_vali_item"].append(int(np.sum(vali_mask==1)))
    output["num_existing_test_item"].append(int(np.sum(test_mask==1)))
    print(num_existing_item,"num_existing_item")
    print(train_ranking_param["mask"][:20])
  # logging_ndcg={}
  # logging.info("begin training ndcg")
  # cur_logging_ndcg=evl.logs_ndcg(
  #           cur_train_qids,
  #           cur_train_rankings,
  #           true_train_doc_weights,
  #           data.train)
  # cur_vali_logging_ndcg=evl.logs_ndcg(
  #         cur_vali_qids,
  #         cur_vali_rankings,
  #         true_vali_doc_weights,
  #         data.validation)
  for metric_key in result_logging.keys():
    cur_logging_metrics=np.concatenate(result_logging[metric_key])
    # print(cur_logging_ndcg.shape,"ndcg shape")
    logging_metrics[metric_key].append(cur_logging_metrics)
    cur_logging_metrics_mean=np.mean(cur_logging_metrics)
    output[metric_key].append(cur_logging_metrics_mean)


  output["iterations"].append(int(n_queries_sampled))
  alpha_clip = min(10./np.sqrt(n_queries_sampled), 1.)
  beta_clip = 0.
    
  np.add.at(train_clicks,cur_train_rankings,cur_train_clicks)
  np.add.at(train_exp_alpha,cur_train_rankings,alpha)
  np.add.at(train_exp_beta,cur_train_rankings,beta)
  np.add.at(train_doc_freq,cur_train_rankings,1)  
  np.add.at(train_query_freq,cur_train_qids,1)  
    
  np.add.at(vali_clicks,cur_vali_rankings,cur_vali_clicks)
  np.add.at(vali_exp_alpha,cur_vali_rankings,alpha[None,:])
  np.add.at(vali_exp_beta,cur_vali_rankings,beta[None,:])
  np.add.at(vali_doc_freq,cur_vali_rankings,1)  
  np.add.at(vali_query_freq,cur_vali_qids,1) 


  np.add.at(test_clicks,cur_test_rankings,cur_test_clicks)
  np.add.at(test_exp_alpha,cur_test_rankings,alpha[None,:])
  np.add.at(test_exp_beta,cur_test_rankings,beta[None,:])
  np.add.at(test_doc_freq,cur_test_rankings,1)  
  np.add.at(test_query_freq,cur_test_qids,1) 
  
  if pdgd_update:
      ranking_pdgd.append(cur_train_rankings)
      clicks_pdgd.append(cur_train_clicks)
      scores_pdgd+=cur_train_scores
      qids_pdgd+=cur_train_qids

  memory=process.memory_info().rss//2**20
  time.sleep(1)
  logging.info(str(memory)+"after update statistics")  # in bytes

  cur_train_rankings=None
  cur_vali_rankings=None
  cur_test_rankings=None
  cur_train_clicks=None
  cur_vali_clicks=None
  cur_test_clicks=None
  memory=process.memory_info().rss//2**20
  time.sleep(1)
  logging.info(str(memory)+"memory, after free memroy")  # in bytes    
    
  if session_aware:
    # for cur_train_ranking in cur_train_rankings:
    #   train_queue.append(cur_train_ranking)
    # for cur_vali_ranking in cur_vali_rankings:
    #   vali_queue.append(cur_vali_ranking)
    # print(len(train_queue),"train_queue size",cur_train_rankings.shape)
#     cur_train_rankings_squeezed=cur_train_rankings.reshape((-1))
#     temp=np.concatenate([train_list,cur_train_rankings_squeezed])
#     train_list=temp.astype(np.int)
#     shown_train_set=np.unique(temp).astype(np.int)
    

#     n_docs_train=data.train.num_docs()
#     docids=np.arange(n_docs_train)
#     mask = np.ones(n_docs_train, dtype=bool)
#     mask[shown_train_set]=False
#     unshown_train_set=docids[mask].astype(np.int)
#     print(shown_train_set.shape,unshown_train_set.shape,\
#           "train_set.shape,unshown_train_set.shape,")
#     cur_vali_rankings_squeezed=cur_vali_rankings.reshape((-1))
#     temp=np.concatenate([vali_list,cur_vali_rankings_squeezed]) 
#     vali_list=temp.astype(np.int)
   
#     cur_test_rankings_squeezed=cur_test_rankings.reshape((-1))
#     temp=np.concatenate([test_list,cur_test_rankings_squeezed]) 
#     test_list=temp.astype(np.int)
#     queue_test_ids_uniq_other=np.unique(np.array(test_list).reshape((-1)))
    logging.info("finding the shown doc id")
    queue_train_ids_uniq=np.where(train_doc_freq>initial_doc_freq_value)[0]
    queue_vali_ids_uniq=np.where(vali_doc_freq>initial_doc_freq_value)[0]
    queue_test_ids_uniq=np.where(test_doc_freq>initial_doc_freq_value)[0]

#     assert np.all(queue_test_ids_uniq_other==queue_test_ids_uniq),"different uniq"
    output["num_shown_train_ids_uniq"].append(queue_train_ids_uniq.shape[0])
    output["num_shown_vali_ids_uniq"].append(queue_vali_ids_uniq.shape[0])
    output["num_shown_test_ids_uniq"].append(queue_test_ids_uniq.shape[0])
#     queue_train_ids=np.array(train_list).reshape((-1))
#     queue_vali_ids=np.array(vali_list).reshape((-1))
    queue_train_ids=queue_train_ids_uniq
    queue_vali_ids=queue_vali_ids_uniq
    
    ranked_num_docis=queue_train_ids_uniq.shape[0]+queue_vali_ids_uniq.shape[0]+queue_test_ids_uniq.shape[0]
    logging.info("finish the shown doc id")
    print("using session aware")
  else:
    cur_train_rankings_squeezed=cur_train_rankings.reshape((-1))
    temp=np.concatenate([train_list,cur_train_rankings_squeezed])
    train_list=temp.astype(np.int)
    shown_train_set=np.unique(temp).astype(np.int)
    

    n_docs_train=data.train.num_docs()
    docids=np.arange(n_docs_train)
    mask = np.ones(n_docs_train, dtype=bool)
    mask[shown_train_set]=False
    unshown_train_set=docids[mask].astype(np.int)
    # n_train_docs = data.train.num_docs()
    # n_vali_docs = data.validation.num_docs() 
    temp=np.concatenate([train_qids_set,cur_train_qids])
    train_qids_set=np.unique(temp).astype(np.int)  
    temp=np.concatenate([vali_qids_set,cur_vali_qids])
    vali_qids_set=np.unique(temp).astype(np.int)  
    print(len(train_qids_set),len(vali_qids_set))
    if  cold_start:
        train_mask=train_ranking_param["mask"]
        vali_mask=vali_ranking_param["mask"]
        queue_train_ids=dataset.get_docids_from_qids(train_qids_set,data.train,train_mask)
        queue_vali_ids=dataset.get_docids_from_qids(vali_qids_set,data.validation,vali_mask)   
    else:
        queue_train_ids=dataset.get_docids_from_qids(train_qids_set,data.train)
        queue_vali_ids=dataset.get_docids_from_qids(vali_qids_set,data.validation)  
    ranked_num_docis=queue_train_ids.shape[0]+queue_vali_ids.shape[0]
    print(queue_train_ids.shape,queue_vali_ids.shape)
    print("using session oblivious")
  #release memory


  train_q_weights_cur=(train_clicks-train_exp_beta)/(train_exp_alpha+1e-5)
  vali_q_weights_cur=(vali_clicks-vali_exp_beta)/(vali_exp_alpha+1e-5)
  test_q_weights_cur=(test_clicks-test_exp_beta)/(test_exp_alpha+1e-5)
  train_q_weights[queue_train_ids_uniq]=train_q_weights_cur[queue_train_ids_uniq]
  vali_q_weights[queue_vali_ids_uniq]=vali_q_weights_cur[queue_vali_ids_uniq]
  test_q_weights[queue_test_ids_uniq]=test_q_weights_cur[queue_test_ids_uniq]
  
    
  train_ctr = train_clicks/ train_doc_freq
  vali_ctr = vali_clicks/ vali_doc_freq
  test_ctr = test_clicks/ test_doc_freq  

  count=str(gc.get_count())
#   gc.collect()
  memory=process.memory_info().rss//2**20
#   time.sleep(1)
  logging.info(count+str(memory)+"memory, bafter ips update")  # in bytes


    
  if add_behavior:
    if add_ips:
        print("add_ips")
        data.train.feature_matrix[:,-1]=train_q_weights
        data.validation.feature_matrix[:,-1]=vali_q_weights     
        data.test.feature_matrix[:,-1]=test_q_weights    
    elif add_ctr:
        print("add ctr")
        data.train.feature_matrix[:,-1]=train_ctr
        data.validation.feature_matrix[:,-1]=vali_ctr
        data.test.feature_matrix[:,-1]=test_ctr
    else:
        print("add clicks")
        data.train.feature_matrix[:,-1]=train_clicks
        data.validation.feature_matrix[:,-1]=vali_clicks
        data.test.feature_matrix[:,-1]=test_clicks


  memory=process.memory_info().rss//2**20
  time.sleep(1)
  logging.info(str(memory)+"memory, after feature update")  # in bytes    

      
  # start_mean,start_std=nn.mean_std_dropout(**mean_std_param)
  # shown_start_mean=start_mean[shown_train_set].mean()
  # shown_start_std=start_std[shown_train_set].mean()
  # unshown_start_mean=start_mean[unshown_train_set].mean()
  # unshown_start_std=start_std[unshown_train_set].mean()
  # output["shown_start_mean"].append(shown_start_mean)
  # output["shown_start_std"].append(shown_start_std)
  # output["unshown_start_mean"].append(unshown_start_mean)
  # output["unshown_start_std"].append(unshown_start_std)


  # model.set_weights(init_weights)
  
  if single_training:
    verbose=2
  else:
    verbose=0
  if n_queries_sampled in update_points:
    cur_test_shown_id=np.unique(np.where(test_mask==1)[0])
    new_income_docid=np.setdiff1d(cur_test_shown_id, last_test_shown_id, assume_unique=True)
    last_test_shown_id=cur_test_shown_id
    shown_test_id.append(new_income_docid)
  else:
    shown_test_id.append(None)
    
    
  if "ips" not in intervene_strategy and n_queries_sampled in update_points and pdgd_update==False:

      logging.info("data loader")
      training_batch_size=return_batchid(len(queue_vali_ids),int(np.log2(batch_size)+1))
      training_generator=dataset.DataGenerator(
                        queue_train_ids,
                        data.train.feature_matrix,
                        train_q_weights,
                        batch_size=training_batch_size)
      validation_generator=dataset.DataGenerator(
                        queue_vali_ids,
                        data.validation.feature_matrix,
                        vali_q_weights,
                        batch_size=training_batch_size)
      if optimizer=="Adagrad":
        optimizer =tf.keras.optimizers.Adagrad(learning_rate=0.01)
      elif optimizer=="sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
      elif optimizer=="Adam":
        optimizer = tf.keras.optimizers.Adam()

    
      model.compile(optimizer=optimizer,
                    loss="mse", # Call the loss function with the selected layer
                    metrics=["mse"])
      callback=tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_squared_error",
        # monitor="val_loss_sigmoid",
        min_delta=0,
        patience=early_stop_patience,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )
      last_weight=model.get_weights()
      memory=process.memory_info().rss//2**20
    #   time.sleep(1)
      logging.info(str(memory)+"memory, after generator")  # in bytes
      logging.info("begin l2 fit")
#       enq = OrderedEnqueuer(training_generator, use_multiprocessing=True)
#       enq.start(workers=10, max_queue_size=20)
#       training_generator = enq.get()
#       enq = OrderedEnqueuer(validation_generator, use_multiprocessing=True)
#       enq.start(workers=10, max_queue_size=20)
#       validation_generator = enq.get()
        
      history=model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,callbacks=[callback,MyCustomCallback()],\
                  max_queue_size=1,verbose=verbose,workers=0,use_multiprocessing=False

    )
      print(gc.collect(),"gc.collect()")  
      cur_weight=model.get_weights()
      model=nn.select_best(model,
                      data.validation.feature_matrix[queue_vali_ids],
                      vali_q_weights[queue_vali_ids],
                      cur_weight,
                      last_weight,
                      use_GPU=args.use_GPU,batch_size=q_batch_size)
      print(history.history['val_mean_squared_error'])
      del training_generator,validation_generator,history
      print(gc.collect(),"gc.collect()") 
  memory=process.memory_info().rss//2**20
#   time.sleep(1)
  logging.info(str(memory)+"memory, after training")  # in bytes
    
  if n_queries_sampled in update_points   and pdgd_update:
    
      
      
      logging.info("begin PDGD fit")
#       print(len(qids_pdgd),qids_pdgd[:3],"len(qids_pdgd),qids_pdgd[:3]")
#       print(len(scores_pdgd))
      
      ranking_pdgd=np.concatenate(ranking_pdgd)
      clicks_pdgd=np.concatenate(clicks_pdgd)
#       print(clicks_pdgd.shape,clicks_pdgd[:3],"clicks_pdgd.shape,clicks_pdgd[:3]")
      ranking_pdgd=data.train.global2local(qids_pdgd,ranking_pdgd)
      optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
      iteration_zip= zip(qids_pdgd, ranking_pdgd,clicks_pdgd,scores_pdgd)
      for qid,ranking,clicks,scores in iteration_zip:
#           print(qid,ranking,clicks,scores,"qid,ranking,clicks,scores")
          PDGD.update_cold(model,
                optimizer,
                qid,
                data.train,
                ranking,
                clicks,
                scores)
      ranking_pdgd=[]
      clicks_pdgd=[]
      scores_pdgd=[]
      qids_pdgd=[]
      iteration_zip=[]
    
  logging.info("begin test ndcg")

#   learning_model_dropout_scores=nn.scores_dropout_direct_model(**mean_std_param)
#   learning_mean=learning_model_dropout_scores.mean(axis=1)
#   learning_std = learning_model_dropout_scores.std(axis=1)
# #   shown_start_mean=learning_mean[shown_train_set].mean()
# #   shown_start_std=learning_std[shown_train_set].mean()
# #   unshown_start_mean=learning_mean[unshown_train_set].mean()
# #   unshown_start_std=learning_std[unshown_train_set].mean()
#   output["learning_mean"].append(learning_mean.mean())
#   output["learning_std"].append(learning_std.mean())
#   dropout_correlation=nn.get_corre_stat(data.test,learning_model_dropout_scores)[1]
#   output["dropout_correlation"].append(dropout_correlation)
#   corr=evl.correlation(data.test,learning_model_dropout_scores)
#   output["mean_std_corr"].append(corr["mean_std_corr"])
#   output["label_std_corr"].append(corr["label_std_corr"])
#   output["label_mean_corr"].append(corr["label_mean_corr"])



  memory=process.memory_info().rss//2**20
#   time.sleep(1)
  logging.info(str(memory)+"memory, before evaluation")  # in bytes
  if "ips" in intervene_strategy:
        test_scores=np.zeros_like(test_q_weights)
        test_scores[:]=test_q_weights
  elif "merge" in intervene_strategy:
        test_scores_ips=test_q_weights
        model_test_scores=nn.get_score_matrix_by_dropout_batch_query(data.test,model,1,\
                                                             use_GPU=args.use_GPU,\
                                                            training=False,\
                                                             batch_size=q_batch_size)[:,0]
        test_scores=np.zeros_like(model_test_scores)
        test_scores[:]=model_test_scores
        test_scores[queue_test_ids_uniq]=test_scores_ips[queue_test_ids_uniq]
        
  else:
        test_scores=nn.get_score_matrix_by_dropout_batch_query(data.test,model,1,\
                                                             use_GPU=args.use_GPU,\
                                                            training=False,\
                                                             batch_size=q_batch_size)[:,0]
    
  if  cold_start:
       test_mask=test_ranking_param["mask"]
       test_scores_masked=dataset.mask_items(test_mask,test_scores,flag=1)
       test_q_weights_masked=dataset.mask_items(test_mask,test_q_weights,flag=1)
       test_q_weights_masked[test_mask==0]=-np.inf
       ips_test_dcg= evl.test_dcg(
                                  test_q_weights_masked,
                                  data.test,
                                  true_test_doc_weights,
                                  cutoff
                                )
       output["ips_test_dcg_cold"].append(ips_test_dcg)
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

  model_test_scores_ignore_behav=nn.get_score_matrix_by_dropout_batch_query(data.test,model,1,\
                                                             use_GPU=args.use_GPU,\
                                                            training=False,\
                                                             batch_size=q_batch_size,feature=test_feature_orig)[:,0] 
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
                              train_ranking_param["mask"]
                            )
  output["train_unfairness"].append(train_unfairness)
  output["shown_test_item_ratio"].append(np.sum(test_mask==1)/test_mask.shape[0])
#   cur_vali_dcg= evl.test_dcg(
#                                   vali_scores,
#                                   data.validation,
#                                   true_vali_doc_weights,
#                                   cutoff
#                                 )
#   output["vali_dcg"].append(cur_vali_dcg)
#   misc.test_ips(data.validation,score=vali_scores)

  memory=process.memory_info().rss//2**20
#   time.sleep(1)
  logging.info(str(memory)+"memory,after evaluation")  # in bytes
    
  logging.info("finish test ndcg")
  if "ips" not in intervene_strategy:
    logging_model.set_weights(model.get_weights())
    model_dropout.set_weights(model.get_weights())
    train_ranking_param["policy_scores"] =nn.get_score_matrix_by_dropout_batch_query(data.train,logging_model,1,use_GPU=args.use_GPU,training=False,batch_size=q_batch_size)[:,0]
    vali_ranking_param["policy_scores"] = nn.get_score_matrix_by_dropout_batch_query(data.validation,logging_model,1,use_GPU=args.use_GPU,training=False,batch_size=q_batch_size)[:,0]
    test_ranking_param["policy_scores"] = nn.get_score_matrix_by_dropout_batch_query(data.test,logging_model,1,use_GPU=args.use_GPU,training=False,batch_size=q_batch_size)[:,0]
    if intervene_strategy in ["dropout","dropout_shuffle","portfolio","portfolio_v1",\
                              "ucb_std","std_proposional","portfolio_proposional"]:
      train_policy_scores_dropout[:]=nn.get_score_matrix_by_dropout_batch_query(data.train,model_dropout,\
                                                                             n_dropout,use_GPU=args.use_GPU,training=True,batch_size=q_batch_size)
      vali_policy_scores_dropout[:]=nn.get_score_matrix_by_dropout_batch_query(data.validation,model_dropout,n_dropout,\
                                                                            use_GPU=args.use_GPU,training=True,batch_size=q_batch_size)
      test_policy_scores_dropout[:]=nn.get_score_matrix_by_dropout_batch_query(data.test,model_dropout,n_dropout,\
                                                                      use_GPU=args.use_GPU,training=True,batch_size=q_batch_size)
        
      train_ranking_param["policy_scores_dropout"]=train_policy_scores_dropout
      vali_ranking_param["policy_scores_dropout"]=vali_policy_scores_dropout
      test_ranking_param["policy_scores_dropout"]=test_policy_scores_dropout
  if n_queries_sampled in update_points:
    print('No. query %09d, NDCG %0.5f,   UPDATE' % (
        n_queries_sampled, cur_test_ndcg), flush=True)
  else:
    print('No. query %09d, NDCG %0.5f,  ' % (
        n_queries_sampled, cur_test_ndcg), flush=True)
  model_path=workspace+"/"+str(n_queries_sampled)+"_steps_model.h5"
  model.save(model_path)
  n_sampled = n_queries_sampled
np.save(workspace+"/logging_metrics.npy", logging_metrics)
logging.info("processing cumulative metrics")
for metrics_key in logging_metrics.keys():
  print(np.concatenate(logging_metrics[metrics_key]).shape,"logging_metrics[metrics_key]")
  step_metrics=np.concatenate(logging_metrics[metrics_key])
  iterations=[len(i) for i in logging_metrics[metrics_key]]
  iterations=np.cumsum(iterations)-1
  output["discounted_sum_"+metrics_key]=evl.dicounted_metrics(step_metrics)[iterations].tolist()
  output["overall_"+metrics_key]=[np.mean(step_metrics)]*len(output["iterations"])
output["initial_test_ndcg"]=[initial_model_metrics]*len(output["iterations"])
if args.supervised_model:
  output["supervised_test_ndcg"]=[supervised_model_metrics]*len(output["iterations"])
print(output)
least_labels=list(range(1,max_label+1))

test_q_ctr=test_clicks/(test_doc_query_freq+1e-5)
for least_label in least_labels:
    result_time_stamp_clicks=evl.time_stamp_clicks(test_clicks,\
                                                   data.test.label_vector,\
                                                   shown_test_id,select_label=least_label)
    output["result_time_stamp_clicks_least_label"+str(least_label)]=result_time_stamp_clicks
    result_time_stamp_ctr=evl.time_stamp_clicks(test_q_ctr,\
                                                   data.test.label_vector,\
                                                   shown_test_id,select_label=least_label)
    output["result_time_stamp_ctr_least_label"+str(least_label)]=result_time_stamp_ctr    
    
results_path=workspace+"/result.jjson"
print('Writing results to %s' % results_path)
with open(results_path, 'w') as f:
  json.dump(output, f)
model_path=workspace+"/final_model.h5"
model.save(model_path)
