# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.ranking as rnk
import time
import utils.dataset as dataset
import utils.nnmodel as nn
def sample_rankings(log_scores, n_samples, cutoff=None, prob_per_rank=False):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings = np.empty((n_samples, ranking_len), dtype=np.int32)
  inv_rankings = np.empty((n_samples, n_docs), dtype=np.int32)
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)

  if cutoff:
    inv_rankings[:] = ranking_len

  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    cumprobs = np.cumsum(probs, axis=1)
    random_values = np.random.uniform(size=n_samples)
    greater_equal_mask = np.greater_equal(random_values[:,None], cumprobs)
    sampled_ind = np.sum(greater_equal_mask, axis=1)

    rankings[:, i] = sampled_ind
    inv_rankings[ind, sampled_ind] = i
    rankings_prob[:, i] = probs[ind, sampled_ind]
    log_scores[ind, sampled_ind] = np.NINF

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix
  else:
    return rankings, inv_rankings, rankings_prob
def gumbel_sample_rankings(log_scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
  gumbel_scores = -(log_scores[None,:]+gumbel_samples)

  rankings, inv_rankings = rnk.multiple_cutoff_rankings(
                                gumbel_scores,
                                ranking_len,
                                invert=inverted)
#   print(log_scores,gumbel_scores,rankings,"log_scores,gumbel_scores,rankings")
  if not doc_prob:
    return rankings, inv_rankings, None, None

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)
  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    rankings_prob[:, i] = probs[ind, rankings[:, i]]
    log_scores[ind, rankings[:, i]] = np.NINF

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix
  else:
    return rankings, inv_rankings, rankings_prob, None

def random_rankings(log_scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False):
  n_docs = log_scores.shape[0]

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  random_rnk_scores = np.random.uniform(0,1,(n_samples, n_docs))
  rnk_scores = random_rnk_scores

  rankings, inv_rankings = rnk.multiple_cutoff_rankings(
                                rnk_scores,
                                ranking_len,
                                invert=inverted)

  return rankings, inv_rankings, None, None

def pl_rankings(scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False):
  n_docs = scores.shape[0]

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  assert len(scores.shape)==1,"pl ranking scores should be one dimensional"
  scores_tiled = np.exp(np.tile(scores[None,:], (n_samples, 1)))
  scores_tiled=scores_tiled/scores_tiled.sum(axis=1,keepdims=True)
  rankings=np.zeros((n_samples,0))
  ind_=np.arange(0,n_samples)
  for i in range(ranking_len):
      # ind=np.stack(rankings,(i,1))
      # scores_tiled[ind,rankings]=0
      scores_cum=np.cumsum(scores_tiled,axis=1)
      judge=np.random.uniform(0,1,(n_samples,1))
      # print(scores_tiled)
      selected_id=np.sum(scores_cum<judge,axis=1).astype(np.int)
      scores_tiled[ind_,selected_id]=0
      scores_tiled=scores_tiled/scores_tiled.sum(axis=1,keepdims=True)
      # print(selected_id)
      rankings=np.append(rankings,selected_id[:,None],axis=1)

  # for ranking in rankings:
  #     assert np.unique(ranking).shape[0]==5, print("not unique in ranked list")
      
  rankings=rankings.astype(np.int)
  return rankings, None, None, None

def arg_topk(x,K):
    ind=np.argpartition(-x,K)[:K]
    ind=ind[np.argsort(-x[ind])]
    return ind

def ucb_rankings(scores, n_samples, cutoff=None, 
                          doc_freq=None, query_freq=None,param_c=None):
  n_docs = scores.shape[0]

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  assert len(scores.shape)==1,"ucb ranking scores should be one dimensional"

  rankings=[]
  for i_sample in range(n_samples):
      ranking_scores=scores+param_c*np.sqrt(np.log(query_freq)/doc_freq)
      selected_id=arg_topk(ranking_scores,ranking_len)
      # print(selected_id)
#       doc_freq[selected_id]+=1
      rankings.append(selected_id)
#       query_freq[:]+=1
  rankings=np.array(rankings)      
  rankings=rankings.astype(np.int)
  return rankings, None, None, None

def ucb_std_rankings(scores, n_samples, cutoff=None, 
                          doc_freq=None, query_freq=None,param_c=None):
  n_docs = scores.shape[0]
  scores_mean=scores.mean(axis=1)
  scores_std=scores.std(axis=1)
  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  assert len(scores.shape)==2,"ucb_std ranking scores should be one dimensional"

  rankings=[]
  for i_sample in range(n_samples):
      ranking_scores=scores_mean+param_c*np.sqrt(np.log(query_freq)/doc_freq)*scores_std
      selected_id=arg_topk(ranking_scores,ranking_len)
      # print(selected_id)
      doc_freq[selected_id]+=1
      rankings.append(selected_id)
      query_freq[:]+=1
  rankings=np.array(rankings)      
  rankings=rankings.astype(np.int)
  return rankings, None, None, None

def std_proposional_rankings(scores, n_samples, cutoff=None, 
                          param_c=None):
  n_docs = scores.shape[0]
  scores_mean=scores.mean(axis=1)
  scores_std=scores.std(axis=1)
  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  assert len(scores.shape)==2,"ucb_std ranking scores should be one dimensional"

  rankings=[]
  for i_sample in range(n_samples):
      ranking_scores=scores_mean+param_c*scores_std
      selected_id=arg_topk(ranking_scores,ranking_len)
      # print(selected_id)
#       doc_freq[selected_id]+=1
      rankings.append(selected_id)
#       query_freq[:]+=1
  rankings=np.array(rankings)      
  rankings=rankings.astype(np.int)
  return rankings, None, None, None

def dropout_rankings(scores, n_samples, cutoff=None, 
                          doc_freq=None, query_freq=None,param_c=None):
  n_docs = scores.shape[0]

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  assert len(scores.shape)==2,"pl ranking scores should be one dimensional"
  n_dropout=scores.shape[1]
  selected_dropout=np.random.choice(n_dropout,n_samples)
  rankings=[]
  for i_dropout in selected_dropout:
      ranking_scores=scores[:,i_dropout]
      selected_id=arg_topk(ranking_scores,ranking_len)
      rankings.append(selected_id)
  rankings=np.array(rankings)      
  rankings=rankings.astype(np.int)
  return rankings, None, None, None


def direct_rankings(log_scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False):
  n_docs = log_scores.shape[0]

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs
  log_scores=np.tile(log_scores,(n_samples,1))
  rnk_scores = -(log_scores)

  rankings, inv_rankings = rnk.multiple_cutoff_rankings(
                                rnk_scores,
                                ranking_len,
                                invert=inverted)

  return rankings, inv_rankings, None, None
def rankings_n_samples(log_scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False):
  n_docs = log_scores.shape[0]
  assert len(log_scores.shape)== 2,\
    "scores should be dim=2, [n_doc,n_sample], current dime"+str(log_scores.shape)
  log_scores=log_scores.T
  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)
  zeros = np.zeros(shape=(n_samples, n_docs))
  rnk_scores = -(log_scores+zeros)

  rankings, inv_rankings = rnk.multiple_cutoff_rankings(
                                rnk_scores,
                                ranking_len,
                                invert=inverted)

  if not doc_prob:
    return rankings, inv_rankings, None, None

def random_tradeoff_sample_rankings(log_scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False,random_param_c=None):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  random_samples = np.random.uniform(0,1,size=(n_samples, n_docs))*random_param_c
  random_scores = -(log_scores[None,:]+random_samples)
#   print(np.sum(random_scores==np.inf),random_scores.shape,"np.sum(random_scores==np.inf)")
#   print(random_scores,"random_scores")
  rankings, inv_rankings = rnk.multiple_cutoff_rankings(
                                random_scores,
                                ranking_len,
                                invert=inverted)

  if not doc_prob:
    return rankings, inv_rankings, None, None

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)
  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    rankings_prob[:, i] = probs[ind, rankings[:, i]]
    log_scores[ind, rankings[:, i]] = np.NINF

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix
  else:
    return rankings, inv_rankings, rankings_prob, None

# def gumbel_sample_rankings_intervene(q_feature,model, n_samples, cutoff=None, 
#                            inverted=False, doc_prob=False,
#                            prob_per_rank=False,additional_param={}):
#  scores=model(q_feature)[:, 0].numpy()
#  return gumbel_sample_rankings(scores,n_samples,cutoff)


def metrics_based_on_samples(sampled_rankings,
                             weight_per_rank,
                             addition_per_rank,
                             q_doc_weights,):
  cutoff = sampled_rankings.shape[1]
  return np.sum(np.mean(
              q_doc_weights[sampled_rankings]*weight_per_rank[None, :cutoff],
            axis=0) + addition_per_rank[:cutoff], axis=0)
def metrics_based_on_rankings(sampled_rankings,
                              q_doc_weights,
                              click_alpha=np.ones(5),
                              click_beta=np.zeros(5)):
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
    stacked_alphas = stacked_alphas[:,[0,0,1,1,2,2]]
    stacked_betas = stacked_betas[:,[0,0,1,1,2,2]]
    results_qid = metrics_based_on_samples(sampled_rankings,
                                              stacked_alphas,
                                              stacked_betas,
                                              q_doc_weights[:,None])
    ideal_ranking=np.argsort(-q_doc_weights)[:cutoff][None,:]
    ideal_factors_qid=metrics_based_on_samples(ideal_ranking,
                                              stacked_alphas,
                                              stacked_betas,
                                              q_doc_weights[:, None])
    ideal_factors_qid[1::2]=1
    if q_doc_weights.sum()<=0:
      metrics=np.zeros_like(ideal_factors_qid)
    else:
      metrics=results_qid/ideal_factors_qid
    result = {
      'NCTR':  metrics[0],
      'CTR':   metrics[1],
      'NRCTR': metrics[2],
      'RCTR':  metrics[3],
      'NDCG':  metrics[4],
      'DCG':   metrics[5],
    }
    return result
def get_ranking(policy_scores,
                      n_samples,
                      cutoff,
                      intervene_strategy):
      # print(intervene_strategy)
      if "direct" == intervene_strategy:
        sampled_rankings=direct_rankings(policy_scores,
                                                n_samples,
                                                cutoff=cutoff)
      elif "dropout" == intervene_strategy:
        sampled_rankings=direct_rankings(policy_scores,
                                                n_samples,
                                                cutoff=cutoff)                                              
      elif "rankings_n_samples" == intervene_strategy:
        sampled_rankings=rankings_n_samples(policy_scores,
                                                n_samples,
                                                cutoff=cutoff)
      elif "gumbel" == intervene_strategy:
        sampled_rankings = gumbel_sample_rankings(policy_scores,
                                                n_samples,
                                                cutoff=cutoff)
      elif "random" == intervene_strategy:
        sampled_rankings = random_rankings(policy_scores,
                                                n_samples,
                                                cutoff=cutoff)                                             
      else:
        raise NotImplementedError("intervene strategy not found.")    
      return sampled_rankings

def DCG(sampled_rankings,
        q_doc_weights):
  cutoff = sampled_rankings.shape[1]
  dcg_weights = 1./np.log2(np.arange(cutoff)+2.)
  return np.sum(
              q_doc_weights[sampled_rankings]*dcg_weights[None, :cutoff],
            axis=1)
def cDCG(clicks):
  cutoff = clicks.shape[1]
  dcg_weights = 1./np.log2(np.arange(cutoff)+2.)
  return np.sum(
              clicks*dcg_weights[None, :cutoff],
            axis=1)
def crazyshuffle(arr):
    x, y = arr.shape
    rows = np.indices((x,y))[0]
    # print(rows)
    cols = [np.random.permutation(y) for _ in range(x)]
    return arr[rows, cols]
def NDCG_based_on_samples(sampled_rankings,
                             q_doc_weights):
    n_samples=sampled_rankings.shape[0]
    if q_doc_weights.sum()<=0:
      return np.zeros(n_samples)
    cutoff = sampled_rankings.shape[1]
    ideal_ranking=np.argsort(-q_doc_weights)[:cutoff][None,:]
    dcg=DCG(sampled_rankings,
        q_doc_weights)
    idcg=DCG(ideal_ranking,
        q_doc_weights)
    return dcg/idcg
def DCG_based_on_samples(sampled_rankings,
                             q_doc_weights):
    n_samples=sampled_rankings.shape[0]
    if q_doc_weights.sum()<=0:
      return np.zeros(n_samples)
    cutoff = sampled_rankings.shape[1]
    # ideal_ranking=np.argsort(-q_doc_weights)[:cutoff][None,:]
    dcg=DCG(sampled_rankings,
        q_doc_weights)
    return dcg


def get_ranking_intervene(**ranking_param):
    n_samples=ranking_param["n_samples"]
    select_fn=ranking_param["select_fn"]
    doc_query_freq=ranking_param["doc_query_freq"]
    qid=ranking_param["qid"]
#     print("qid_",qid)
    intervene_strategy=ranking_param["intervene_strategy"]
    if intervene_strategy in ["ips_random_tradeoff","ips_ucb"]:
        policy_scores=ranking_param["ips_score"]
        policy_scores_query=select_fn(qid,policy_scores)
    elif intervene_strategy in ["merge_random_tradeoff","merge_ucb"]:
        q_scores_ips=select_fn(qid,ranking_param["ips_score"])
        q_scores_model=select_fn(qid,ranking_param["policy_scores"])
        q_doc_freq=select_fn(qid,ranking_param["doc_freq"])
        policy_scores_query=np.zeros_like(q_scores_ips)
        policy_scores_query[:]=q_scores_ips
        no_shown_id=q_doc_freq<=1
        
        policy_scores_query[no_shown_id]=q_scores_model[no_shown_id]
    else:
        policy_scores=ranking_param["policy_scores"]
        policy_scores_query=select_fn(qid,policy_scores)
    if ranking_param.get("cold_start"):
      ranking_param["n_samples"]=1
      mask=ranking_param["mask"]
      cold_rng=ranking_param["cold_rng"]
      mask_query=select_fn(qid,mask)
      shown_prob=ranking_param["shown_prob"]
      q_doc_query_freq=select_fn(qid,doc_query_freq)
      
      flag=1
      if intervene_strategy=="poor":
        flag=-1
      rankings=[]
      scores=[]
      if "policy_scores_dropout" in ranking_param.keys():
        policy_scores_dropout=ranking_param["policy_scores_dropout"]
        q_policy_scores_dropout=select_fn(qid,policy_scores_dropout)
      # print(unshown,mask_query)
      for sample in range(n_samples):
        dataset.update_mask(mask_query,shown_prob,cold_rng)
#         print(np.sum(mask_query==0),np.sum(mask_query),"np.sum(mask_query==0),np.sum(mask_query)")
#         print(mask_query[:10],policy_scores_query[:10])
        policy_scores_masked=np.zeros_like(policy_scores_query)
        policy_scores_masked[:]=policy_scores_query
        policy_scores_masked[mask_query==0]=-np.inf*flag
        q_doc_query_freq[mask_query==1]+=1
#         print(mask_query[:10],policy_scores_query[:10],policy_scores_masked[:10])
        ranking_param["q_scores"]=policy_scores_masked
        if "policy_scores_dropout" in ranking_param.keys():
          q_policy_scores_dropout_masked=np.zeros_like(q_policy_scores_dropout)
          q_policy_scores_dropout_masked[:]=q_policy_scores_dropout
          q_policy_scores_dropout_masked[mask_query==0]=-np.inf*flag
          ranking_param["q_policy_scores_dropout"]=q_policy_scores_dropout_masked
        ranking=get_ranking_intervene_oneshot(**ranking_param)
#         print(ranking,policy_scores_masked,np.where(policy_scores_masked!=np.NINF))
#         print(ranking[0][0])
#         print(policy_scores_masked[ranking[0][0]],"policy_scores_masked[ranking[0][0]]")
#         print(np.isinf(policy_scores_masked[ranking[0][0]]))
#         print(np.any(np.isinf(policy_scores_masked[ranking[0][0]])))
        assert np.all(~np.isinf(policy_scores_masked[ranking[0][0]])),"something wrong with the mask"
        # print(ranking,"ranking.shape,")
        rankings.append(ranking[0][0])
        scores.append(policy_scores_masked)
      # print(len(rankings),"rankings.shape")
      rankings=np.array(rankings).astype(np.int)
#       if n_samples==1:
#         rankings=rankings[None,:]
      # print(unshown,"after",mask_query)
      return [rankings,scores]
#     else:
#       ranking_param["q_scores"]=policy_scores_query
#       if "policy_scores_dropout" in ranking_param.keys():
#         policy_scores_dropout=ranking_param["policy_scores_dropout"]
#         ranking_param["q_policy_scores_dropout"]=select_fn(qid,policy_scores_dropout)
#         rankings=get_ranking_intervene_oneshot(**ranking_param)
#       return [rankings]


# def get_ranking_intervene_cold_start(**ranking_param):
#       # print(intervene_strategy)
#       intervene_strategy=ranking_param["intervene_strategy"]
#       select_fn=ranking_param["select_fn"]
#       qid=ranking_param["qid"]
#       n_samples=ranking_param["n_samples"]
#       cutoff=ranking_param["cutoff"]
#       if "direct" == intervene_strategy:
#         policy_scores=ranking_param["policy_scores"]
#         q_scores=select_fn(qid,policy_scores)
#         sampled_rankings=direct_rankings(q_scores,
#                                                 n_samples,
#                                                 cutoff=cutoff)
#       elif "poor" == intervene_strategy:
#         policy_scores=ranking_param["policy_scores"]
#         q_scores=select_fn(qid,policy_scores)
#         sampled_rankings=direct_rankings(-q_scores,
#                                                 n_samples,
#                                                 cutoff=cutoff)
#       # elif "dropout" == intervene_strategy:
#       #   sampled_rankings=direct_rankings(q_feature,model,
#       #                                           n_samples,
#       #                                           cutoff=cutoff)                                              

#       elif "random_tradeoff" == intervene_strategy:
#         random_param_c=ranking_param["random_param_c"]
#         policy_scores=ranking_param["policy_scores"]
#         q_scores=select_fn(qid,policy_scores)
#         sampled_rankings = random_tradeoff_sample_rankings(q_scores,
#                                                 n_samples,
#                                                 cutoff=cutoff,random_param_c=random_param_c)
#       elif "ucb" == intervene_strategy:
#         policy_scores=ranking_param["policy_scores"]
#         q_scores=select_fn(qid,policy_scores)
#         doc_freq=select_fn(qid,ranking_param["doc_freq"])
#         query_freq=ranking_param["query_freq"][qid:qid+1]
#         ucb_param_c=ranking_param["ucb_param_c"]
#         sampled_rankings = ucb_rankings(q_scores,
#                                         n_samples,
#                                         cutoff=cutoff, 
#                                         doc_freq=doc_freq, 
#                                         query_freq=query_freq,
#                                         param_c=ucb_param_c)
#       elif "plackettluce" == intervene_strategy:
#         policy_scores=ranking_param["policy_scores"]
#         q_scores=select_fn(qid,policy_scores)
#         sampled_rankings = pl_rankings(q_scores,
#                                                 n_samples,
#                                                 cutoff=cutoff)
#       elif "dropout" == intervene_strategy:
#         policy_scores_dropout=ranking_param["policy_scores_dropout"]
#         q_scores=select_fn(qid,policy_scores_dropout)
#         sampled_rankings = dropout_rankings(q_scores,
#                                                 n_samples,
#                                                 cutoff=cutoff)
#       elif "dropout_shuffle" == intervene_strategy:
#         policy_scores_dropout=ranking_param["policy_scores_dropout"]
#         q_scores=select_fn(qid,policy_scores_dropout)
#         q_scores=crazyshuffle(q_scores)
#         sampled_rankings = dropout_rankings(q_scores,
#                                                 n_samples,
#                                                 cutoff=cutoff)
#       elif "portfolio" == intervene_strategy:
#         policy_scores_dropout=ranking_param["policy_scores_dropout"]
#         risk_averse_param=ranking_param["risk_averse_param"]
#         risk_averse_param=np.random.uniform(-risk_averse_param,risk_averse_param,size=(n_samples))
#         q_scores=select_fn(qid,policy_scores_dropout)
#         sampled_rankings = nn.get_rank_profolio_multiple(q_scores,
#                                             cutoff=cutoff,
#                                             risk_preference_param=risk_averse_param)
#       elif "portfolio_v1" == intervene_strategy:
#         policy_scores_dropout=ranking_param["policy_scores_dropout"]
#         risk_averse_param=ranking_param["risk_averse_param_v1"]
#         risk_averse_param=np.random.uniform(risk_averse_param,10,size=(n_samples))
#         q_scores=select_fn(qid,policy_scores_dropout)
#         sampled_rankings = nn.get_rank_profolio_multiple(q_scores,
#                                             cutoff=cutoff,
#                                             risk_preference_param=risk_averse_param)
                                          
#       elif "ucb_std" == intervene_strategy:
#         doc_freq=select_fn(qid,ranking_param["doc_freq"])
#         query_freq=ranking_param["query_freq"][qid:qid+1]
#         tradeoff_param=ranking_param["tradeoff_param"]
#         policy_scores_dropout=ranking_param["policy_scores_dropout"]
#         q_scores=select_fn(qid,policy_scores_dropout)
#         sampled_rankings = ucb_std_rankings(q_scores,
#                                         n_samples,
#                                         cutoff=cutoff, 
#                                         doc_freq=doc_freq, 
#                                         query_freq=query_freq,
#                                         param_c=tradeoff_param)
                                    
#         # print(sampled_rankings.shape,"ranking shape")
#       elif "random" == intervene_strategy:
#         policy_scores=ranking_param["policy_scores"]
#         q_scores=select_fn(qid,policy_scores)
#         sampled_rankings = random_rankings(q_scores,
#                                                 n_samples,
#                                                 cutoff=cutoff)                                          
#       else:
#         raise NotImplementedError("intervene strategy not found.")    
#       return sampled_rankings



def get_ranking_intervene_oneshot(**ranking_param):
      # print(intervene_strategy)
      intervene_strategy=ranking_param["intervene_strategy"]
      select_fn=ranking_param["select_fn"]
      qid=ranking_param["qid"]
      n_samples=ranking_param["n_samples"]
      cutoff=ranking_param["cutoff"]
      if "direct" == intervene_strategy:
        q_scores=ranking_param["q_scores"]
        sampled_rankings=direct_rankings(q_scores,
                                                n_samples,
                                                cutoff=cutoff)
      elif "poor" == intervene_strategy:
        q_scores=ranking_param["q_scores"]
        sampled_rankings=direct_rankings(-q_scores,
                                                n_samples,
                                                cutoff=cutoff)
      # elif "dropout" == intervene_strategy:
      #   sampled_rankings=direct_rankings(q_feature,model,
      #                                           n_samples,
      #                                           cutoff=cutoff)                                              

      elif intervene_strategy in ["random_tradeoff","ips_random_tradeoff","merge_random_tradeoff"] :
        random_param_c=ranking_param["tradeoff_param"]
        q_scores=ranking_param["q_scores"]
        sampled_rankings = random_tradeoff_sample_rankings(q_scores,
                                                n_samples,
                                                cutoff=cutoff,random_param_c=random_param_c)
      elif intervene_strategy in ["ucb","ips_ucb","merge_ucb"]:
        # policy_scores=ranking_param["policy_scores"]
        # q_scores=select_fn(qid,policy_scores)
        q_scores=ranking_param["q_scores"]
        doc_freq=select_fn(qid,ranking_param["doc_freq"])
        query_freq=ranking_param["query_freq"][qid:qid+1]
        ucb_param_c=ranking_param["tradeoff_param"]
        sampled_rankings = ucb_rankings(q_scores,
                                        n_samples,
                                        cutoff=cutoff, 
                                        doc_freq=doc_freq, 
                                        query_freq=query_freq,
                                        param_c=ucb_param_c)
      elif "plackettluce" == intervene_strategy:
        q_scores=ranking_param["q_scores"]
        sampled_rankings = pl_rankings(q_scores,
                                                n_samples,
                                                cutoff=cutoff)
      elif "dropout" == intervene_strategy:
        q_scores=ranking_param["q_policy_scores_dropout"]
        # q_scores=select_fn(qid,policy_scores_dropout)
        sampled_rankings = dropout_rankings(q_scores,
                                                n_samples,
                                                cutoff=cutoff)
      elif "dropout_shuffle" == intervene_strategy:
        q_scores=ranking_param["q_policy_scores_dropout"]
        q_scores=crazyshuffle(q_scores)
        sampled_rankings = dropout_rankings(q_scores,
                                                n_samples,
                                                cutoff=cutoff)
      elif "portfolio" == intervene_strategy:
        q_scores=ranking_param["q_policy_scores_dropout"]
        risk_averse_param=ranking_param["tradeoff_param"]
        risk_averse_param=np.random.uniform(-risk_averse_param,risk_averse_param,size=(n_samples))
        sampled_rankings = nn.get_rank_profolio_multiple(q_scores,
                                            cutoff=cutoff,
                                            risk_preference_param=risk_averse_param)
      elif "portfolio_proposional" == intervene_strategy:
        q_scores=ranking_param["q_policy_scores_dropout"]
        risk_averse_param=-ranking_param["tradeoff_param"]
        risk_averse_param=np.array([risk_averse_param]*n_samples)
#         risk_averse_param=np.random.uniform(-risk_averse_param,risk_averse_param,size=(n_samples))
        sampled_rankings = nn.get_rank_profolio_multiple(q_scores,
                                            cutoff=cutoff,
                                            risk_preference_param=risk_averse_param)
      elif "portfolio_v1" == intervene_strategy:
        q_scores=ranking_param["q_policy_scores_dropout"]
        risk_averse_param=ranking_param["tradeoff_param"]
        risk_averse_param=np.random.uniform(risk_averse_param,10,size=(n_samples))
        # q_scores=select_fn(qid,policy_scores_dropout)
        sampled_rankings = nn.get_rank_profolio_multiple(q_scores,
                                            cutoff=cutoff,
                                            risk_preference_param=risk_averse_param)
                                          
      elif "ucb_std" == intervene_strategy:
        doc_freq=select_fn(qid,ranking_param["doc_freq"])
        query_freq=ranking_param["query_freq"][qid:qid+1]
        tradeoff_param=ranking_param["tradeoff_param"]
        q_scores=ranking_param["q_policy_scores_dropout"]
        sampled_rankings = ucb_std_rankings(q_scores,
                                        n_samples,
                                        cutoff=cutoff, 
                                        doc_freq=doc_freq, 
                                        query_freq=query_freq,
                                        param_c=tradeoff_param)
      elif "std_proposional" == intervene_strategy:
        doc_freq=select_fn(qid,ranking_param["doc_freq"])
        query_freq=ranking_param["query_freq"][qid:qid+1]
        tradeoff_param=ranking_param["tradeoff_param"]
        q_scores=ranking_param["q_policy_scores_dropout"]
        sampled_rankings = std_proposional_rankings(q_scores,
                                        n_samples,
                                        cutoff=cutoff, 
                                        param_c=tradeoff_param)
                                    
        # print(sampled_rankings.shape,"ranking shape")
      elif "random" == intervene_strategy:
        q_scores=ranking_param["q_scores"]
#         q_scores=select_fn(qid,policy_scores)
        sampled_rankings = random_rankings(q_scores,
                                                n_samples,
                                                cutoff=cutoff)
      elif intervene_strategy in ["gumbel","PDGD"]:
        q_scores=ranking_param["q_scores"]
#         q_scores=select_fn(qid,policy_scores)
        sampled_rankings = gumbel_sample_rankings(q_scores,
                                                n_samples,
                                                cutoff=cutoff)    
      else:
        raise NotImplementedError("intervene strategy not found.")    
      return sampled_rankings

def datasplit_metrics(data_split,
                      policy_scores,
                      weight_per_rank,
                      addition_per_rank,
                      weight_per_doc,
                      query_norm_factors=None,
                      n_samples=1000,
                      intervene_strategy="gumbel"):
  cutoff = weight_per_rank.shape[0]
  n_queries = data_split.num_queries()
  results = np.zeros((n_queries, weight_per_rank.shape[1]),)
  for qid in range(n_queries):
    q_doc_weights = data_split.query_values_from_vector(qid, weight_per_doc)
    if not np.all(np.equal(q_doc_weights, 0.)):
      q_policy_scores = data_split.query_values_from_vector(qid, policy_scores)

      sampled_rankings = get_ranking(q_policy_scores,
                                      n_samples,
                                      cutoff=cutoff,\
                                      intervene_strategy=intervene_strategy)[0]
      results[qid] = metrics_based_on_samples(sampled_rankings,
                                              weight_per_rank,
                                              addition_per_rank,
                                              q_doc_weights[:, None])
  if query_norm_factors is not None:
    results /= query_norm_factors

  return np.mean(results, axis=0)


def gradient_based_on_samples(sampled_rankings,
                              weight_per_rank,
                              log_scores):
  
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  cutoff = sampled_rankings.shape[1]

  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)
  result = np.zeros((n_docs, n_docs))
  log_scores = np.tile(log_scores[None,:], (n_samples, 1))

  cumulative_grad = np.zeros((n_samples, n_docs))
  cur_grad = np.zeros((n_docs, n_docs))

  for i in range(cutoff):
    cur_grad[:] = 0.
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    denom = np.log(np.sum(np.exp(log_scores), axis=1))
    cur_doc_prob = np.exp(log_scores[:,:] - denom[:, None])

    cur_grad[doc_ind, doc_ind] += np.mean(cur_doc_prob, axis=0)
    cur_grad -= np.mean(cur_doc_prob[:, :, None]*cur_doc_prob[:, None, :], axis=0)
    if i > 0:
      cur_grad += np.mean(cur_doc_prob[:, :, None]*cumulative_grad[:, None, :], axis=0)

    result += weight_per_rank[i]*cur_grad

    if i < n_docs - 1:
      cumulative_grad[sample_ind, sampled_rankings[:, i]] += 1
      cumulative_grad -= cur_doc_prob

      log_scores[sample_ind, sampled_rankings[:, i]] = np.NINF

  return result

def fast_gradient_based_on_samples(sampled_rankings,
                              weight_per_rank,
                              weight_per_doc,
                              log_scores,
                              cutoff=None):
  
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  ranking_len = min(n_docs, cutoff)

  rank_ind = np.arange(ranking_len)
  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)

  log_scores = np.tile(log_scores[None, None,:], (n_samples, ranking_len, 1))

  inf_mask = np.zeros((n_samples, ranking_len, n_docs))
  inf_mask[sample_ind[:, None], rank_ind[None, 1:], sampled_rankings[:, :-1]] = np.NINF

  log_scores += np.cumsum(inf_mask, axis=1)
  log_scores += 18 - np.amax(log_scores, axis=2)[:, :, None]
  denom = np.log(np.sum(np.exp(log_scores), axis=2))
  doc_prob_per_sample = np.exp(log_scores[:,:,:] - denom[:, :, None]) ## placket-luce score

  # # delete very large matrices
  del inf_mask
  del log_scores
  del denom

  doc_grad_per_rank = np.zeros((n_samples, ranking_len, n_docs))
  doc_grad_per_rank[sample_ind[:,None],
                    rank_ind[None,:],
                    sampled_rankings] += 1
  doc_grad_per_rank[sample_ind[:,None],
                    rank_ind[None,:], :] -= doc_prob_per_sample

  cum_grad = np.cumsum(doc_grad_per_rank, axis=1)
  cum_grad *= weight_per_rank[None, :ranking_len, None]
  cum_grad *= weight_per_doc[sampled_rankings][:, :, None]
  
  return np.sum(np.mean(cum_grad, axis=0), axis=0)

def slow_gradient_based_on_samples(sampled_rankings,
                              weight_per_rank,
                              log_scores,
                              cutoff=None):
  
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  ranking_len = min(n_docs, cutoff)

  rank_ind = np.arange(ranking_len)
  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)

  log_scores = np.tile(log_scores[None, None,:], (n_samples, ranking_len, 1))

  inf_mask = np.zeros((n_samples, ranking_len, n_docs))
  inf_mask[sample_ind[:, None], rank_ind[None, 1:], sampled_rankings[:, :-1]] = np.NINF

  log_scores += np.cumsum(inf_mask, axis=1)
  log_scores += 18 - np.amax(log_scores, axis=2)[:, :, None]
  denom = np.log(np.sum(np.exp(log_scores), axis=2))
  doc_prob_per_sample = np.exp(log_scores[:,:,:] - denom[:, :, None])

  # # delete very large matrices
  del inf_mask
  del log_scores
  del denom

  result = np.zeros((n_docs, n_docs))

  cur_doc_grad = np.zeros((ranking_len, n_docs, n_docs))
  cur_doc_grad[rank_ind[:, None],
               doc_ind[None, :],
               doc_ind[None, :]] = np.mean(doc_prob_per_sample, axis=0)
  cur_doc_grad -= np.mean(doc_prob_per_sample[:, :, :, None]
                          *doc_prob_per_sample[:, :, None, :],
                          axis=0)
  cur_doc_grad *= weight_per_rank[:ranking_len, None, None]

  result += np.sum(cur_doc_grad, axis=0)

  del cur_doc_grad

  cumulative_grad = np.zeros((n_samples, ranking_len-1, n_docs))
  cumulative_grad[sample_ind[:, None], rank_ind[None, :-1],
                  sampled_rankings[:, :-1]] += 1
  cumulative_grad -= doc_prob_per_sample[:, :-1, :]
  cumulative_grad = np.cumsum(cumulative_grad, axis=1)
  cumulative_grad *= weight_per_rank[None, 1:ranking_len, None]

  per_doc_cum_grad = np.mean(cumulative_grad[:,:,None,:]
                             *doc_prob_per_sample[:,:-1,:,None],
                             axis=0)
  result += np.sum(per_doc_cum_grad, axis=0)

  return result

def hybrid_gradient_based_on_samples(
                              sampled_rankings,
                              weight_per_rank,
                              weight_per_doc,
                              log_scores,
                              cutoff=None):
  
  n_docs = log_scores.shape[0]
  n_samples = sampled_rankings.shape[0]
  ranking_len = min(n_docs, cutoff)

  rank_ind = np.arange(ranking_len)
  doc_ind = np.arange(n_docs)
  sample_ind = np.arange(n_samples)

  log_scores = np.tile(log_scores[None, None,:], (n_samples, ranking_len, 1))

  inf_mask = np.zeros((n_samples, ranking_len, n_docs))
  inf_mask[sample_ind[:, None], rank_ind[None, 1:], sampled_rankings[:, :-1]] = np.NINF

  log_scores += np.cumsum(inf_mask, axis=1)
  log_scores += 18 - np.amax(log_scores, axis=2)[:, :, None]
  denom = np.log(np.sum(np.exp(log_scores), axis=2))
  doc_prob_per_sample = np.exp(log_scores[:,:-1,:] - denom[:, :-1, None])

  # delete very large matrices
  del inf_mask
  del denom

  final_scores = log_scores[:, -1, :]
  final_denom = np.log(np.sum(np.exp(log_scores[:, -1, :]), axis=1))
  final_prob = np.exp(final_scores - final_denom[:, None])

  # delete very large matrices
  del log_scores
  del final_scores
  del final_denom

  doc_grad_per_rank = np.zeros((n_samples, ranking_len-1, n_docs), dtype=np.float64)
  doc_grad_per_rank[sample_ind[:,None],
                    rank_ind[None,:-1],
                    sampled_rankings[:,:-1]] += 1
  doc_grad_per_rank[sample_ind[:,None],
                    rank_ind[None,:-1], :] -= doc_prob_per_sample

  cum_grad = np.cumsum(doc_grad_per_rank, axis=1)
  weighted_cum_grad = cum_grad*weight_per_rank[None, :ranking_len-1, None]
  weighted_cum_grad *= weight_per_doc[sampled_rankings[:,:-1]][:, :, None]
  
  sample_based_grad = np.sum(np.mean(weighted_cum_grad, axis=0), axis=0)

  rel_mask = np.not_equal(weight_per_doc, 0)
  rel_ind = doc_ind[rel_mask]
  n_rel = rel_ind.shape[0]

  final_grad = np.zeros((n_samples, n_rel, n_docs), dtype=np.float64)
  final_grad -= final_prob[:,None,:]
  final_grad[sample_ind[:,None],
             doc_ind[None,:n_rel],
             rel_ind[None,:]] += 1.
  final_grad += cum_grad[:, -1, None, :]
  final_grad *= final_prob[:, rel_ind, None]
  final_grad = np.mean(final_grad, axis=0)
  final_grad *= weight_per_doc[rel_ind, None]

  final_grad = np.sum(final_grad, axis=0)
  final_grad *= weight_per_rank[ranking_len-1] 

  return sample_based_grad+final_grad
