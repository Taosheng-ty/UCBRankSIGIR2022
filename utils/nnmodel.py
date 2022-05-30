# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import tensorflow as tf
from progressbar import progressbar
def init_model(model_params):
  layers = [tf.keras.layers.Dense(x, activation='sigmoid', dtype=tf.float64)
            for x in model_params['hidden units']]
  layers.append(tf.keras.layers.Dense(1, activation=None, dtype=tf.float64))
  nn_model = tf.keras.Sequential(layers)
  return nn_model
def init_model_dropout(model_params):
  layers = [tf.keras.layers.Dense(x, activation='sigmoid', dtype=tf.float64)
            for x in model_params['hidden units']]
  layers.append(tf.keras.layers.Dropout(0.5))
  layers.append(tf.keras.layers.Dense(1, activation=None, dtype=tf.float64))
  nn_model = tf.keras.Sequential(layers)
  return nn_model
def init_model_dropout_fixed(model_params):
  layers = [tf.keras.layers.Dense(x, activation='sigmoid', dtype=tf.float64)
            for x in model_params['hidden units']]
  layers.append(dropout_fixed(0.5))
  layers.append(tf.keras.layers.Dense(1, activation=None, dtype=tf.float64))
  nn_model = tf.keras.Sequential(layers)
  return nn_model

def get_score_matrix_by_dropout(feat,model, n_samples,use_GPU=False,training=True,batch_size=1024):
  score_mat=[]
#   batch_size=2048
#   feat=data_split.feature_matrix
  length=feat.shape[0]
#   for i in progressbar(range(n_samples)):
  for i in range(n_samples): 
    n_batch=np.ceil(length/batch_size).astype(int)
    # if use_GPU:
    scores_dropout=[]
    # for batch_id in progressbar(range(n_batch)):
    for batch_id in range(n_batch):
        cur,next=batch_id*batch_size,(batch_id+1)*batch_size
        scores_dropout.append(model(feat[cur:next],training=training))
    # print(len(scores_dropout),"in nn")
    scores_dropout=tf.concat(scores_dropout,axis=0)[:, 0].numpy()
        # print(scores_dropout.shape,"in nn")
    # else:
    #     scores_dropout = model(feat,training=training)[:, 0].numpy()
    score_mat.append(scores_dropout)
  score_mat=np.stack(score_mat)
  dropout_scores=score_mat.T
  
  return dropout_scores

# def get_score_matrix_by_dropout(q_feat,model,n_samples=100):
#     score_mat=[]
#     for i in progressbar(range(n_samples)):
#         # tf.random.set_random_seed(i)
#         scores_dropout = model(q_feat,training=True)[:, 0].numpy()
#         score_mat.append(scores_dropout)
#     score_mat=np.stack(score_mat)
#     score_mat=score_mat.T
#     return score_mat

def get_score_matrix_by_dropout_batch_query(data_split,model, n_samples,use_GPU=False,training=True,batch_size=3,feature=None):
  score_mat=[]
  num_batch=data_split.num_queries()//batch_size+1
  doclist_ranges=data_split.doclist_ranges
  if feature is not None:
        feat=feature
  else:
      feat=data_split.feature_matrix
  length=feat.shape[0]
#   for i in progressbar(range(n_samples)):
  for i in range(n_samples): 
    
    scores_dropout=[]
    # for batch_id in progressbar(range(n_batch)):
    num_queries=data_split.num_queries()+1
    for i in range(num_batch):
        if i*batch_size+batch_size<num_queries:
            cur,next=doclist_ranges[i*batch_size],doclist_ranges[i*batch_size+batch_size]
        else:
            cur,next=doclist_ranges[i*batch_size],doclist_ranges[-1]
        scores_dropout.append(model(feat[cur:next],training=training))
    # print(len(scores_dropout),"in nn")
    scores_dropout=tf.concat(scores_dropout,axis=0)[:, 0].numpy()
    # print(scores_dropout.shape,"in nn")
    score_mat.append(scores_dropout)
  score_mat=np.stack(score_mat)
  dropout_scores=score_mat.T
  
  return dropout_scores
def get_dropout_model(model_params,input_shape,weight_path=None,dropout_fixed=False,weight=None):
    if dropout_fixed:
      model_dropout=init_model_dropout_fixed(model_params)
    else:
      model_dropout=init_model_dropout(model_params)
    model_dropout.build(input_shape=input_shape)
    if weight_path  is not None:
        model_dropout.load_weights(weight_path)
    if weight is not None:
        model_dropout.set_weights(weight)
    return model_dropout
def sub_dropout_fixed_model(model,model_params,input_shape):
    model_dropout=init_model_dropout_fixed(model_params)
    model_dropout.build(input_shape=input_shape)
    model_dropout.set_weights(model.get_weights())
    return model_dropout
def mean_std_dropout(model,model_params,input_feature,data_split,n_dropout,q_batch_size):
    model_dropout_fix=sub_dropout_fixed_model(model,model_params,input_feature)
    train_policy_scores_dropout=get_score_matrix_by_dropout_batch_query(data_split,model_dropout_fix,n_dropout,training=True,batch_size=q_batch_size)
    mean=train_policy_scores_dropout.mean(axis=1)
    std=train_policy_scores_dropout.std(axis=1)
    return mean,std
def scores_dropout_direct_model(model,model_params,input_feature,data_split,n_dropout,q_batch_size):
    model_dropout_fix=sub_dropout_fixed_model(model,model_params,input_feature)
    train_policy_scores_dropout=get_score_matrix_by_dropout_batch_query(data_split,model_dropout_fix,n_dropout,training=True,batch_size=q_batch_size)
    return train_policy_scores_dropout
class dropout_fixed(tf.keras.layers.Layer):
    def __init__(self,dropout_rate=0.5):
        super(dropout_fixed, self).__init__()
        self.dropout_rate=dropout_rate
    def call(self, inputs,seed=None):
        # tf.random.set_random_seed(seed)
        dropout_rate=self.dropout_rate
        dropout_shape=inputs.shape[-1]
        mask=tf.keras.backend.random_binomial(shape=(dropout_shape,),p=dropout_rate,seed=seed)/dropout_rate
        mask =  tf.cast(mask, dtype= tf.float64)
        output=inputs*mask
        return output
def get_corre_stat(data_split,score):
    qids=data_split.num_queries()
    stat=[]
    # for qid in progressbar(range(qids)):
    for qid in range(qids):
        score_qid=data_split.query_values_from_vector(qid,score)
        if score_qid.shape[0]<=1:
            continue
        corref_qid_cov=np.corrcoef(score_qid)
        # np.fill_diagonal(corref_qid_cov, 0)
        abs_mean=np.mean(np.abs(corref_qid_cov))
        real_mean=np.mean(corref_qid_cov)
        stat.append([abs_mean,real_mean])
    stat=np.array(stat)
    return np.mean(stat,axis=0)



def cal_marginal_value(score_qid_mean,
                        score_qid_cov,
                        selected_item_id,
                        select_cur_id,
                        risk_preference_param,
                        w_k,
                        **param):
    current_rank=len(selected_item_id)
    First=score_qid_mean[select_cur_id]
    Second=-risk_preference_param*w_k[current_rank]*score_qid_cov[select_cur_id,select_cur_id]
    # print(select_cur_id)

    Third=-2*risk_preference_param*np.sum(score_qid_cov[select_cur_id,selected_item_id]*w_k[:current_rank])
    marginal_value_cur=First+Second+Third
    return marginal_value_cur


def select_next_marginal_one_slow(score_qid_mean,
                        score_qid_cov,
                        selected_item_id,
                        risk_preference_param,
                        w_k,
                        **param):
    n_doc=score_qid_mean.shape[0]
    value_cache_dict={}
    for select_cur_id in range(n_doc):
        # print(select_cur_id)
        if select_cur_id in selected_item_id:
            continue
        # print(select_cur_id)
        value=cal_marginal_value(score_qid_mean,
                        score_qid_cov,
                        selected_item_id,
                        select_cur_id,
                        risk_preference_param,
                        w_k,
                        )
        # print(value)
        value_cache_dict[select_cur_id]=value
    # print(value_cache_dict)
    selected_id=max(value_cache_dict, key = value_cache_dict.get) 
    return selected_id

def select_next_marginal_one(score_qid_mean,
                        score_qid_cov,
                        selected_item_id,
                        risk_preference_param,
                        w_k,
                        **param):
    # print(selected_item_id)
    n_doc=score_qid_mean.shape[0]
    current_rank=len(selected_item_id)
    value_cache_dict={}
    First=score_qid_mean
    Second=-risk_preference_param*w_k[current_rank]*score_qid_cov.diagonal()
    Third=-2*risk_preference_param*np.sum(score_qid_cov[:,selected_item_id]*w_k[None,:current_rank],axis=1)
    
    marginal_value=First+Second+Third
    marginal_value[selected_item_id]=-np.inf
    # print(marginal_value)
    selected_id=np.argmax(marginal_value)
    # print("fasht")
    return selected_id


def get_rank_profolio(qid,
                      score,
                      data_split,
                      cutoff=5,
                      risk_preference_param=0):
    # w_k=1/np.log2(np.arange(cutoff,0,-1)+2)
    w_k=1/np.log2(np.arange(cutoff)+2)
    score_qid=data_split.query_values_from_vector(qid,score)
    score_qid_cov=np.cov(score_qid)
    if len(score_qid_cov.shape)!=2:
        # print(score_qid_cov)
        score_qid_cov=np.array([[score_qid_cov]])
    score_qid_mean=np.mean(score_qid,axis=1)
    cutoff=min(cutoff,score_qid_mean.shape[0])
    selected_item_id=[]
    len_doc=score_qid.shape[0]
    marginal_value=[]
    param={"score_qid_mean":score_qid_mean,
            "score_qid_cov":score_qid_cov,
            "selected_item_id":selected_item_id,
            "risk_preference_param":risk_preference_param,
            "w_k":w_k
        }
    for i in range(cutoff):
        # print(i)
        cur_id=select_next_marginal_one(**param)
        selected_item_id.append(cur_id)
    return selected_item_id    

def get_rank_profolio_multiple(q_score,
                      cutoff=5,
                      risk_preference_param=0):
    # w_k=1/np.log2(np.arange(cutoff,0,-1)+2)
    w_k=1/np.log2(np.arange(cutoff)+2)
    q_score_cov=np.cov(q_score)
    if len(q_score_cov.shape)!=2:
        # print(score_qid_cov)
        q_score_cov=np.array([[q_score_cov]])
    q_score_mean=np.mean(q_score,axis=1)
    cutoff=min(cutoff,q_score_mean.shape[0])
    selected_item_id=np.zeros((risk_preference_param.shape[0],0)).astype(np.int)
    len_doc=q_score.shape[0]
    marginal_value=[]
    param={"q_score_mean":q_score_mean,
            "q_score_cov":q_score_cov,
            "selected_item_id":selected_item_id,
            "risk_preference_param":risk_preference_param,
            "w_k":w_k
        }
    for i in range(cutoff):
        # print(i)
        cur_id=select_next_marginal_one_multiple(**param)
        param["selected_item_id"]=np.append(param["selected_item_id"],cur_id[:,None],axis=-1)
    if len(param["selected_item_id"].shape)==1:
        param["selected_item_id"]=param["selected_item_id"][None,:]
    # print(param["selected_item_id"].shape)
    return [param["selected_item_id"]] 

def select_next_marginal_one_multiple(q_score_mean,
                        q_score_cov,
                        selected_item_id,#[n_doc_selcted,n_sample]
                        risk_preference_param,
                        w_k,
                        **param):
    # print(selected_item_id)
    # assert len(selected_item_id.shape)==2
    n_sample=len(risk_preference_param)
    n_doc=q_score_mean.shape[0]
    current_rank=selected_item_id.shape[1]
    First=q_score_mean[None,:]    #[n_doc]
    
    Second=-risk_preference_param[:,None]*w_k[current_rank]*q_score_cov.diagonal()[None,:] #[n_doc]
    
    ind=np.tile(np.arange(n_doc),(current_rank,1)).T  #[n_doc,n_doc_selcted]
    # selected_item_id=np.array(selected_item_id).T  #[n_sample,n_doc_selcted]
    # print(score_qid_cov.shape,ind.shape,selected_item_id.shape)
    # if len(selected_item_id.shape)==1:
    #     selected_item_id=selected_item_id[None,:]
    # print(score_qid_cov.shape,ind.shape,selected_item_id.shape)
    score_qid_cov_select=q_score_cov[ind[None,:,:],selected_item_id[:,None,:]] ##[n_sample,n_doc,n_doc_selcted]
    Third=-2*risk_preference_param[:,None]*np.sum(score_qid_cov_select*w_k[None,None,:current_rank],axis=-1)##[n_sample,n_doc]
    # print(First.shape,Second.shape,Third.shape)
    marginal_value=First+Second+Third##[n_sample,n_doc]
    # print(marginal_value.shape)
    ind=np.tile(np.arange(n_sample),(current_rank,1)).T  #[n_sample,n_doc_selcted]
    marginal_value[ind,selected_item_id]=-np.inf
    # print(marginal_value[1],selected_item_id[1])
    selected_id=np.argmax(marginal_value,axis=-1)
    return selected_id
def select_best(model,feature,label,cur_weight,last_weight,use_GPU=None,batch_size=1024):
    model.set_weights(last_weight)
    # print(model(feature))
    # feature=data_split.feature_matrix
    last_score=get_score_matrix_by_dropout(feature, model,1,use_GPU=use_GPU,training=False,batch_size=batch_size)[:,0]
    last_val=np.mean(np.square(last_score-label))
    model.set_weights(cur_weight)
    cur_score=get_score_matrix_by_dropout(feature, model,1,use_GPU=use_GPU,training=False,batch_size=batch_size)[:,0]
    cur_val=np.mean(np.square(cur_score-label))
    print(last_val,cur_val,"last_val,cur_val")
    if last_val<cur_val:
        model.set_weights(last_weight)
        print("select last model")
    else:
        model.set_weights(cur_weight)
        print("select cur model")
    return model