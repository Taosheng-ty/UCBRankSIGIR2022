# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl
import utils.nnmodel as nn
from sklearn.metrics import ndcg_score
import glob
import matplotlib.pyplot as plt
from progressbar import progressbar
from scipy.stats import pearsonr
from scipy.stats import spearmanr
def test_ndcg(scores,data_split,true_doc_weights,cutoff=5,model=None):
    feature_matrix=data_split.feature_matrix
    # scores=model(feature_matrix)[:,0].numpy()
    filtered_queries=data_split.get_filtered_queries()
    ndcg_list=[]
    for qid in filtered_queries:
        cur_score=data_split.query_values_from_vector(qid,scores)
        ranked_list=np.argsort(-cur_score)[:cutoff]
        ranked_list=ranked_list[None,:]
        cur_weight=data_split.query_values_from_vector(qid,true_doc_weights)
        # # print(cur_score.shape,cur_weight.shape)
        # if cur_score.shape[1]==1:   # will raise error for ndcg_score in sklearn.metrics.
        #     continue       
        # cur_ndcg=ndcg_score(cur_weight,cur_score,k=cutoff)
        cur_ndcg=pl.NDCG_based_on_samples(ranked_list,cur_weight)[0]
        ndcg_list.append(cur_ndcg)
    return np.mean(ndcg_list)
def test_dcg(scores,data_split,true_doc_weights,cutoff=5,model=None):
    feature_matrix=data_split.feature_matrix
    # scores=model(feature_matrix)[:,0].numpy()
    filtered_queries=data_split.get_filtered_queries()
    dcg_list=[]
    for qid in filtered_queries:
        cur_score=data_split.query_values_from_vector(qid,scores)
        ranked_list=np.argsort(-cur_score)[:cutoff]
        ranked_list=ranked_list[None,:]
        cur_weight=data_split.query_values_from_vector(qid,true_doc_weights)
        # # print(cur_score.shape,cur_weight.shape)
        # if cur_score.shape[1]==1:   # will raise error for ndcg_score in sklearn.metrics.
        #     continue       
        # cur_ndcg=ndcg_score(cur_weight,cur_score,k=cutoff)
        cur_dcg=pl.DCG(ranked_list,cur_weight)[0]
        dcg_list.append(cur_dcg)
    return np.mean(dcg_list)


def L2(exposure,rel):
    exposure=exposure/exposure.sum()
    rel=rel/rel.sum()
    return np.sum(np.abs(exposure-rel))/2

def test_unfairness(exposure,data_split,true_doc_weights,mask):
#     feature_matrix=data_split.feature_matrix
    # scores=model(feature_matrix)[:,0].numpy()
    filtered_queries=data_split.get_filtered_queries()
    unfairness_list=[]
    for qid in filtered_queries:
        q_mask=data_split.query_values_from_vector(qid,mask)
        q_shown_id=q_mask==1
        q_exposure=data_split.query_values_from_vector(qid,exposure)[q_shown_id]
        q_rel=data_split.query_values_from_vector(qid,true_doc_weights)[q_shown_id]
        if q_exposure.sum()<=0 or q_rel.sum()<=0:
            continue
        unfairness_list.append(L2(q_exposure,q_rel))
    unfairness_list=np.array(unfairness_list)
    return np.mean(unfairness_list)


def evaluate_policy(model,
                    data_split,
                    doc_weights,
                    click_alpha,
                    click_beta,
                    n_samples = 100,intervene_strategy="gumbel"):
  cutoff = click_alpha.shape[0]
  dcg_weights = 1./np.log2(np.arange(cutoff)+2.)
  stacked_alphas = np.stack([click_alpha,
                             click_alpha+click_beta,
                             dcg_weights,
                            ], axis=-1)
  stacked_betas = np.stack([click_beta,
                            np.zeros_like(click_beta),
                            np.zeros_like(click_beta),
                          ], axis=-1)

  norm_factors = max_score_per_query(data_split,
                                     doc_weights,
                                     stacked_alphas,
                                     stacked_betas)

  # repeat alphas to calculate both normalized and unnormalized
  stacked_alphas = stacked_alphas[:,[0,0,1,1,2,2]]
  stacked_betas = stacked_betas[:,[0,0,1,1,2,2]]
  norm_factors = np.stack([
                           norm_factors[:, 0],
                           np.ones_like(norm_factors[:, 0]),
                           norm_factors[:, 1],
                           np.ones_like(norm_factors[:, 0]),
                           norm_factors[:, 2],
                           np.ones_like(norm_factors[:, 0]),
                          ], axis=-1)
  if intervene_strategy=="dropout":
    policy_scores=nn.get_dropout_scores(model, data_split.feature_matrix,n_samples)
  else:
    policy_scores = model(data_split.feature_matrix)[:,0].numpy()

  metrics =  pl.datasplit_metrics(
                  data_split,
                  policy_scores,
                  stacked_alphas,
                  stacked_betas,
                  doc_weights,
                  n_samples=n_samples,
                  query_norm_factors=norm_factors,
                  intervene_strategy=intervene_strategy
                )
  result = {
      'NCTR':  metrics[0],
      'CTR':   metrics[1],
      'NRCTR': metrics[2],
      'RCTR':  metrics[3],
      'NDCG':  metrics[4],
      'DCG':   metrics[5],
    }
  for k, v in result.items():
    result[k] = float(v)
  return result

def max_score_per_query(data_split, weight_per_doc,
                        weight_per_rank, addition_per_rank):
  cutoff = weight_per_rank.shape[0]
  n_queries = data_split.num_queries()
  results = np.zeros((n_queries, weight_per_rank.shape[1]),)

  sort_i = np.argsort(-weight_per_rank, axis=0)
  sorted_weights = weight_per_rank[
          sort_i,
          np.arange(weight_per_rank.shape[1])[None, :],
        ]
  sorted_additions = addition_per_rank[
          sort_i,
          np.arange(weight_per_rank.shape[1])[None, :],
        ]

  for qid in range(n_queries):
    q_doc_weights = data_split.query_values_from_vector(qid, weight_per_doc)
    best_ranking = np.argsort(-q_doc_weights)[:cutoff]
    results[qid] = pl.metrics_based_on_samples(best_ranking[None, :],
                                                sorted_weights,
                                                sorted_additions,
                                                q_doc_weights[:, None])
  return results

def ndcg(true_relevance,ranking ,cutoff=5):
    # print(true_relevance.shape)
    assert len(true_relevance.shape)==1,print("only support one dimension")
    # print(ranking)
    qsuedo_score=np.zeros_like(true_relevance)
    ranking_len=ranking.shape[0]
    qsuedo_score[ranking]=np.arange(ranking_len,0,-1)
    # if true_relevance.sum()==0:
    #     cur_ndcg=0
    # else:
    cur_ndcg=ndcg_score(true_relevance[None,:],qsuedo_score[None,:],k=cutoff)
    return cur_ndcg
def logs_ndcg(cur_qids,cur_rankings,true_doc_weights,data_split):
    ndcg_list=[]
    for num, qid in enumerate(cur_qids):
        base,_=data_split.query_range(qid)
        query_local_ranking_id=cur_rankings[num]-base
        query_q_weight=data_split.query_values_from_vector(qid,true_doc_weights)
        # print(query_q_weight.shape)
        cur_ndcg=ndcg(query_q_weight,query_local_ranking_id)
        ndcg_list.append(cur_ndcg)

    return ndcg_list

import logging
def configure_logging(logging):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
def extract_tradeoff_res(cur_res_split,res_names=[],step=0):
    tradeoff_params=cur_res_split.keys()
    tradeoff_nums=[float(i.split("_")[1]) for i in tradeoff_params]
    tradeoff_nums=np.array(tradeoff_nums)
    argsort=np.argsort(tradeoff_nums)
    tradeoff_nums=tradeoff_nums[argsort]
    result=[tradeoff_nums]
    for res_name in res_names:
        res_cur_name=[cur_res_split[tradeoff_param][res_name][step].tolist() for tradeoff_param in tradeoff_params ]
        res_cur_name=np.array(res_cur_name)[argsort]
        result.append(res_cur_name)
    return result
def get_coefficient(model_dir_list,data_split,n_samples=10,batch_size=64):
    model_params={}
    model_params['hidden units']=[32,32]
    get_model_params={"model_params":model_params,
                    "input_shape":data_split.feature_matrix.shape,
                    "weight_path":None,
                    "dropout_fixed":True}

    cor_nogpus=[]
    mean_std_corrs=[]
    label_std_corrs=[]
    label_mean_corrs=[]
    results={
      "coefficient":cor_nogpus,
      "mean_std_corr":mean_std_corrs,
      "label_std_corr":label_std_corrs,
      "label_mean_corr":label_mean_corrs
    }
    for model in progressbar(model_dir_list):
        get_model_params["weight_path"]=model
        model_dropout=nn.get_dropout_model(**get_model_params)
        feat=data_split.feature_matrix
        scores_nogpu=nn.get_score_matrix_by_dropout_batch_query(data_split,model_dropout,n_samples,batch_size=batch_size)
        cor_nogpu=nn.get_corre_stat(data_split,scores_nogpu)[1]
        
        corr=correlation(data_split,scores_nogpu)
        # mean_std_corr,label_std_corr,label_mean_corr
        mean_std_corrs.append(corr["mean_std_corr"])
        label_std_corrs.append(corr["label_std_corr"])
        label_mean_corrs.append(corr["label_mean_corr"])
        cor_nogpus.append(cor_nogpu)
    return results
def plot_coefficient(model_dir_list,data_split,steps,n_samples=10,batch_size=64,save_path=None):
    # models=glob.glob(base+"*.h5")[:len(steps)]
    # print(models)
    model_params={}
    model_params['hidden units']=[32,32]
    get_model_params={"model_params":model_params,
                    "input_shape":data_split.feature_matrix.shape,
                    "weight_path":None,
                    "dropout_fixed":True}

    cor_nogpus=[]
    mean_std_corrs=[]
    label_std_corrs=[]
    label_mean_corrs=[]
    results={
      "coefficient":cor_nogpus,
      "mean_std_corr":mean_std_corrs,
      "label_std_corr":label_std_corrs,
      "label_mean_corr":label_mean_corrs
    }
    for model in model_dir_list:
        get_model_params["weight_path"]=model
        model_dropout=nn.get_dropout_model(**get_model_params)
        feat=data_split.feature_matrix
        scores_nogpu=nn.get_score_matrix_by_dropout_batch_query(data_split,model_dropout,n_samples,batch_size=batch_size)
        cor_nogpu=nn.get_corre_stat(data_split,scores_nogpu)[1]
        
        corr=correlation(data_split,scores_nogpu)
        # mean_std_corr,label_std_corr,label_mean_corr
        mean_std_corrs.append(corr["mean_std_corr"])
        label_std_corrs.append(corr["label_std_corr"])
        label_mean_corrs.append(corr["label_mean_corr"])
        cor_nogpus.append(cor_nogpu)
    print(cor_nogpus,mean_std_corrs,label_std_corrs,label_mean_corrs)
    for key in results.keys():
      result_cur=results[key]
      plt.plot(steps,result_cur)
      plt.xscale("log")
      save_title=save_path.split("/")[-1]
      # save_title=save_title.split(".")[0]
      plt.title(save_title+key)
      if save_path:
        plt.savefig(save_path+key+".pdf")
      plt.close()

def correlation(data, scores,correlation_fn=spearmanr):
    corr={"mean_std_corr":[],"label_std_corr":[],"label_mean_corr":[]}
    # scores=nn.get_score_matrix_by_dropout_batch_query(data,model,n_samples=n_sample)
    mean,std=scores.mean(axis=1), scores.std(axis=1)
    for qid in range(data.num_queries()):
        q_doc_weight= data.query_labels(qid)  
        n_doc=  q_doc_weight.shape[0]    
        if n_doc<3 or np.all(q_doc_weight == q_doc_weight[0]):
            # print(qid)
            continue
        mean_cur=data.query_values_from_vector(qid,mean)
        std_cur=data.query_values_from_vector(qid,std)
        corr["mean_std_corr"].append(correlation_fn(mean_cur,std_cur)[0])
        label_std_corr=correlation_fn(q_doc_weight,std_cur)[0]
        # print(qid,label_std_corr,q_doc_weight.shape,q_doc_weight.sum())
        
        corr["label_std_corr"].append(label_std_corr)
        corr["label_mean_corr"].append(correlation_fn(q_doc_weight,mean_cur)[0])
    for i in corr.keys():
        corr[i]=np.mean(corr[i])
    # print(corr)
    return corr

def get_best_tradeoff_param(cur_res_split,vali_name="logging_ndcg",step=-1):
    tradeoff_params=list(cur_res_split.keys())
    tradeoff_nums=[]
    tradeoff_params_valid=[]
    for i in tradeoff_params:
      if len(i.split("_"))>1:
        tradeoff_nums.append(float(i.split("_")[1]))
        tradeoff_params_valid.append(i)
    tradeoff_params=tradeoff_params_valid
    tradeoff_nums=np.array(tradeoff_nums)
    print(tradeoff_params,step,cur_res_split.keys(),"tradeoff_params,cur_res_split.keys().index,step")
    if not tradeoff_params:
        return None
    if not hasattr(cur_res_split[tradeoff_params[0]], 'index'):
        return None
    steps=cur_res_split[tradeoff_params[0]].index[step]
    performance=[]
    for tradeoff_param in tradeoff_params:
        print(tradeoff_param,vali_name,steps,"tradeoff_param,vali_name,steps")
        performance.append(cur_res_split[tradeoff_param][vali_name][steps].mean())
    performance=np.array(performance)
#     print(performance)
    argmax=np.argmax(performance)
    max_param=tradeoff_params[argmax]
    print(tradeoff_nums[argmax],"from", tradeoff_nums)
    return max_param
def dicounted_metrics(metrics,gamma=0.995):
#     Convert the metrics in a discounted sum verion. 
#     input, metrics: shape=(n), 
#     gamma: discounted coeficient,
#     output, discounted_metrics: shape=(n)
    m=metrics.shape[0]
    results=np.zeros(m)
    previous_sum=0
    for i in range(m):
        previous_sum=previous_sum*gamma+metrics[i]
        results[i]=previous_sum
    return results
def dicounted_metrics_lists(metrics,gamma=0.995):
#     Convert the metrics in a discounted sum verion. 
#     input, metrics: shape=(n), 
#     gamma: discounted coeficient,
#     output, discounted_metrics: shape=(n)
    m=metrics.shape[0]
    results=np.zeros(m)
    previous_sum=0
    for i in range(m):
        previous_sum=previous_sum*gamma+metrics[i]
        results[i]=previous_sum
    return results


def time_stamp_clicks(clicks,label,shown_id_lists,select_label):
    result=[]
    for shown_id_list in shown_id_lists:
        if  shown_id_list is None:
            result.append(None)
            continue
        selected_id=shown_id_list[label[shown_id_list]>=select_label]
        clicks_cur=clicks[selected_id]
        result.append(clicks_cur.mean())
    return result