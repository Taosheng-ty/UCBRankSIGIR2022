# Copyright (C) H.R. Oosterhuis 2020.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl
import utils.dataset as dataset
from collections import defaultdict
from progressbar import progressbar
import gc
# def simulate_on_dataset(data_train,
#                         data_validation,
#                         n_samples,
#                         train_doc_weights,
#                         validation_doc_weights,
#                         alpha,
#                         beta,
#                         model=None,
#                         train_policy_scores=None,
#                         vali_policy_scores=None,
#                         return_display=False,
#                         store_per_rank=False,
#                         intervene_strategy="gumbel",
#                         training_metrics=[]
#                         ):
#   n_train_queries = data_train.num_queries()
#   n_vali_queries = data_validation.num_queries()

#   train_ratio = n_train_queries/float(n_train_queries + n_vali_queries)

#   sampled_queries = np.random.choice(2, size=n_samples,
#                                      p=[train_ratio, 1.-train_ratio])
#   samples_per_split = np.zeros(2, dtype=np.int64)
#   np.add.at(samples_per_split, sampled_queries, 1)

#   (train_clicks,
#    train_displays,
#    train_samples_per_query) = simulate_queries(
#                      data_train,
#                      samples_per_split[0],
#                      train_doc_weights,
#                      alpha,
#                      beta,
#                      model=model,
#                      all_policy_scores=train_policy_scores,
#                      return_display=return_display,
#                      store_per_rank=store_per_rank,
#                      intervene_strategy=intervene_strategy,
#                      training_metrics=training_metrics)

#   (validation_clicks,
#    validation_displays,
#    validation_samples_per_query) = simulate_queries(
#                      data_validation,
#                      samples_per_split[0],
#                      validation_doc_weights,
#                      alpha,
#                      beta,
#                      model=model,
#                      all_policy_scores=vali_policy_scores,
#                      return_display=return_display,
#                      store_per_rank=store_per_rank,
#                      intervene_strategy=intervene_strategy)

#   return (train_clicks, train_displays,
#           train_samples_per_query,
#           validation_clicks, validation_displays,
#           validation_samples_per_query)


# def simulate_queries(data_split,
#                      n_samples,
#                      doc_weights,
#                      alpha,
#                      beta,
#                      model=None,
#                      all_policy_scores=None,
#                      return_display=False,
#                      store_per_rank=False,
#                      intervene_strategy="gumbel",
#                      training_metrics=[]):
  
#   n_queries = data_split.num_queries()
#   n_docs = data_split.num_docs()
#   sampled_queries = np.random.choice(n_queries, size=n_samples)

#   samples_per_query = np.zeros(n_queries, dtype=np.int32)
#   np.add.at(samples_per_query, sampled_queries, 1)

#   if all_policy_scores is None and n_samples > n_queries*0.8:
#     all_policy_scores = model(data_split.feature_matrix)[:, 0].numpy()

#   if store_per_rank:
#     cutoff = alpha.shape[0]
#     all_clicks = np.zeros((n_docs, cutoff), dtype=np.int64)
#     all_displays = np.zeros((n_docs, cutoff), dtype=np.int64)
#   else:
#     all_clicks = np.zeros(n_docs, dtype=np.int64)
#     all_displays = np.zeros(n_docs, dtype=np.int64)
#   for qid in progressbar(np.arange(n_queries)[np.greater(samples_per_query, 0)]):
#     q_clicks = data_split.query_values_from_vector(
#                                   qid, all_clicks)
#     q_displays = data_split.query_values_from_vector(
#                                   qid, all_displays)
#     (new_clicks,
#      new_displays) = single_query_generation(
#                           qid,
#                           data_split,
#                           samples_per_query[qid],
#                           doc_weights,
#                           alpha,
#                           beta,
#                           model=model,
#                           all_policy_scores=all_policy_scores,
#                           return_display=return_display,
#                           store_per_rank=store_per_rank,
#                           intervene_strategy=intervene_strategy,
#                           training_metrics=training_metrics)

#     q_clicks += new_clicks
#     if store_per_rank:
#       q_displays += new_displays
#     else:
#       q_displays += samples_per_query[qid]

#   return all_clicks, all_displays, samples_per_query

# def single_query_generation(
#                     qid,
#                     data_split,
#                     n_samples,
#                     doc_weights,
#                     alpha,
#                     beta,
#                     model=None,
#                     all_policy_scores=None,
#                     return_display=False,
#                     store_per_rank=False,
#                     intervene_strategy="gumbel",
#                     training_metrics=[]):
#   assert model is not None or policy_scores is not None

#   n_docs = data_split.query_size(qid)
#   cutoff = min(alpha.shape[0], n_docs)

#   if all_policy_scores is None:
#     q_feat = data_split.query_feat(qid)
#     policy_scores = model(q_feat)[:,0].numpy()
#   else:
#     policy_scores = data_split.query_values_from_vector(
#                                   qid, all_policy_scores)

#   rankings = pl.get_ranking(
#                       policy_scores,
#                       n_samples,
#                       cutoff,
#                       intervene_strategy=intervene_strategy)[0]
  
#   q_doc_weights = data_split.query_values_from_vector(
#                                   qid, doc_weights)
#   # for ranking in rankings:
#   #   metrics=pl.metrics_based_on_rankings(ranking[None,:],q_doc_weights,alpha,beta)
#   #   training_metrics.append(metrics)
#   clicks = generate_clicks(
#                           rankings,
#                           q_doc_weights,
#                           alpha, beta)

#   if store_per_rank:
#     store_cutoff = alpha.shape[0]
#     clicks_per_doc = np.zeros((n_docs, store_cutoff),
#                               dtype=np.int32)
#     ind_tile = np.tile(np.arange(cutoff)[None,:], (n_samples, 1))
#     np.add.at(clicks_per_doc, (rankings[clicks],
#                                ind_tile[clicks]), 1)
#   else:
#     clicks_per_doc = np.zeros(n_docs, dtype=np.int32)
#     np.add.at(clicks_per_doc, rankings[clicks], 1)

#   if return_display:
#     if store_per_rank:
#       displays_per_doc = np.zeros((n_docs, store_cutoff),
#                                   dtype=np.int32)
#       np.add.at(displays_per_doc, (rankings, ind_tile), 1)
#     else:
#       displays_per_doc = np.zeros(n_docs, dtype=np.int32)
#       np.add.at(displays_per_doc, rankings, 1)
#   else:
#     displays_per_doc = None

#   return clicks_per_doc, displays_per_doc




# @lru_cache(maxsize=None)
def simulate_on_dataset_intervene_with_test(data_train,
                        data_validation,
                        data_test,
                        n_samples,
                        train_doc_weights,
                        validation_doc_weights,
                        test_doc_weights,
                        alpha,
                        beta,
                        # model=None,
                        train_ranking_param=None,
                        vali_ranking_param=None,
                        test_ranking_param=None,
                        return_display=False,
                        store_per_rank=False,
                        intervene_strategy="gumbel",
                        query_rng=np.random.default_rng(0)
                        ):
  n_train_queries =len(data_train.get_filtered_queries())
  n_vali_queries = len(data_validation.get_filtered_queries())
  n_test_queries = len(data_test.get_filtered_queries())
  total_n_query=n_train_queries+n_vali_queries+n_test_queries
  query_ratio = np.array([n_train_queries,n_vali_queries,n_test_queries])/total_n_query

  sampled_queries = query_rng.choice(3, size=n_samples,
                                     p=query_ratio)
  samples_per_split = np.zeros(3, dtype=np.int64)
  np.add.at(samples_per_split, sampled_queries, 1)
  result_logging = defaultdict(list)
  # result_logging["ndcg"]=[]
  # result_logging["dcg"]=[]
  (train_clicks,
   train_rankings,
   train_qids,train_scores) = simulate_queries_intervene(
                     data_train,
                     samples_per_split[0],
                     train_doc_weights,
                     alpha,
                     beta,
                     ranking_param=train_ranking_param,
                     return_display=return_display,
                     store_per_rank=store_per_rank,
                     intervene_strategy=intervene_strategy,
                     result_logging=result_logging,
                     query_rng=query_rng,
                      metrics_prefix="train_")
  (validation_clicks,
   validation_rankings,
   validation_qids,_) = simulate_queries_intervene(
                     data_validation,
                     samples_per_split[1],
                     validation_doc_weights,
                     alpha,
                     beta,
                     ranking_param=vali_ranking_param,
                     return_display=return_display,
                     store_per_rank=store_per_rank,
                     intervene_strategy=intervene_strategy,
                     result_logging=result_logging,
                      query_rng=query_rng,
                      metrics_prefix="vali_"
                     )
  (test_clicks,
   test_rankings,
   test_qids,_) = simulate_queries_intervene(
                     data_test,
                     samples_per_split[2],
                     test_doc_weights,
                     alpha,
                     beta,
                     ranking_param=test_ranking_param,
                     return_display=return_display,
                     store_per_rank=store_per_rank,
                     intervene_strategy=intervene_strategy,
                     result_logging=result_logging,
                      query_rng=query_rng,
                      metrics_prefix="test_"
                     )
  return (train_clicks, train_rankings,train_qids, train_scores,
          validation_clicks,validation_rankings,validation_qids,
          test_clicks, test_rankings,test_qids,result_logging)






def simulate_on_dataset_intervene(data_train,
                        data_validation,
                        n_samples,
                        train_doc_weights,
                        validation_doc_weights,
                        alpha,
                        beta,
                        # model=None,
                        train_ranking_param=None,
                        vali_ranking_param=None,
                        return_display=False,
                        store_per_rank=False,
                        intervene_strategy="gumbel",
                        query_rng=np.random.default_rng(0)
                        ):
  n_train_queries =len(data_train.get_filtered_queries())
  n_vali_queries = len(data_validation.get_filtered_queries())

  train_ratio = n_train_queries/float(n_train_queries + n_vali_queries)

  sampled_queries = query_rng.choice(2, size=n_samples,
                                     p=[train_ratio, 1.-train_ratio])
  samples_per_split = np.zeros(2, dtype=np.int64)
  np.add.at(samples_per_split, sampled_queries, 1)
  result_logging = defaultdict(list)
  # result_logging["ndcg"]=[]
  # result_logging["dcg"]=[]
  (train_clicks,
   train_rankings,
   train_qids) = simulate_queries_intervene(
                     data_train,
                     samples_per_split[0],
                     train_doc_weights,
                     alpha,
                     beta,
                    #  model=model,
                     ranking_param=train_ranking_param,
                     return_display=return_display,
                     store_per_rank=store_per_rank,
                     intervene_strategy=intervene_strategy,
                     result_logging=result_logging,
                     query_rng=query_rng)
#   print(train_qids[:10])
  (validation_clicks,
   validation_rankings,
   validation_qids) = simulate_queries_intervene(
                     data_validation,
                     samples_per_split[1],
                     validation_doc_weights,
                     alpha,
                     beta,
                    #  model=model,
                     ranking_param=vali_ranking_param,
                     return_display=return_display,
                     store_per_rank=store_per_rank,
                     intervene_strategy=intervene_strategy,
                     result_logging=result_logging,
                      query_rng=query_rng
                     )

  return (train_clicks, train_rankings,
          train_qids,
          validation_clicks, validation_rankings,
          validation_qids,result_logging)


def simulate_queries_intervene(data_split,
                     n_samples,
                     doc_weights,
                     alpha,
                     beta,
                    #  model=None,
                     ranking_param=None,
                     return_display=False,
                     store_per_rank=False,
                     intervene_strategy="gumbel",
                     result_logging={},
                     query_rng=np.random.default_rng(0),metrics_prefix=""):
  
  n_queries = data_split.num_queries()
  n_docs = data_split.num_docs()

  
  # cutoff = min(alpha.shape[0], n_docs)
  cutoff = alpha.shape[0]
  filtered_queries=data_split.get_filtered_queries()
  # n_queries=selected_query
  sampled_queries = query_rng.choice(filtered_queries, size=n_samples)

  samples_per_query = np.zeros(n_queries, dtype=np.int32)
  np.add.at(samples_per_query, sampled_queries, 1)

  # if all_policy_scores is None and n_samples > n_queries*0.8:
  #   all_policy_scores = model(data_split.feature_matrix)[:, 0].numpy()

  all_clicks = np.zeros((n_samples, cutoff), dtype=np.int64)
  # all_alpha = np.zeros((n_docs, cutoff), dtype=np.float32)
  # all_beta = np.zeros((n_docs, cutoff), dtype=np.float32)
  all_rankings=np.zeros((n_samples, cutoff), dtype=np.int64)
  all_scores=[None]*n_samples
  qid_list=[]
  cur_sample=0
  # for qid in progressbar(np.arange(n_queries)[np.greater(samples_per_query, 0)]):
  for qid in np.arange(n_queries)[np.greater(samples_per_query, 0)]:
    # print(qid)
    # q_clicks = data_split.query_values_from_vector(
    #                               qid, all_clicks)
    # q_displays = data_split.query_values_from_vector(
    #                               qid, all_displays)

    (new_clicks,new_ranking,new_scores) = single_query_generation_intervene(
                          qid,
                          data_split,
                          samples_per_query[qid],
                          doc_weights,
                          alpha,
                          beta,
                          # model=model,
                          ranking_param=ranking_param,
                          return_display=return_display,
                          store_per_rank=store_per_rank,
                          intervene_strategy=intervene_strategy)
    q_doc_weights = data_split.query_values_from_vector(
                                  qid, doc_weights)    
    ndcg_list=pl.NDCG_based_on_samples(new_ranking,q_doc_weights)
    dcg_list=pl.DCG_based_on_samples(new_ranking,q_doc_weights)
    cdcg_list=pl.cDCG(new_clicks)

    result_logging[metrics_prefix+"ndcg"].append(ndcg_list)
    result_logging[metrics_prefix+"dcg"].append(dcg_list)
    result_logging[metrics_prefix+"cdcg"].append(cdcg_list)
    base=data_split.query_range(qid)[0]
    new_ranking=new_ranking+base
    all_rankings[cur_sample:samples_per_query[qid]+cur_sample]=new_ranking
    all_clicks[cur_sample:samples_per_query[qid]+cur_sample]=new_clicks
    all_scores[cur_sample:samples_per_query[qid]+cur_sample]=new_scores
    qid_list+=[qid]*samples_per_query[qid]
    cur_sample=cur_sample+samples_per_query[qid]
#     qid_list[cur_sample:samples_per_query[qid]+cur_sample]=[qid]*samples_per_query[qid]
    
    
  return all_rankings,all_clicks,qid_list,all_scores
def single_query_generation_intervene(
                    qid,
                    data_split,
                    n_samples,
                    doc_weights,
                    alpha,
                    beta,
                    # model=None,
                    ranking_param=None,
                    return_display=False,
                    store_per_rank=False,
                    intervene_strategy="gumbel"):
  # assert model is not None or policy_scores is not None

  n_docs = data_split.query_size(qid)
  cutoff = min(alpha.shape[0], n_docs)
  ranking_param=ranking_param.copy()
  ranking_param["qid"]=qid
  ranking_param["n_samples"]=n_samples
  ranking_param["cutoff"]=cutoff


  rankings,scores = pl.get_ranking_intervene(
                      **ranking_param)

#   print(rankings.shape,"rankings.shape",len(scores))
  q_doc_weights = data_split.query_values_from_vector(
                                  qid, doc_weights)
  # for ranking in rankings:
  #   metrics=pl.metrics_based_on_rankings(ranking[None,:],q_doc_weights,alpha,beta)
  #   training_metrics.append(metrics)
  clicks = generate_clicks(
                          rankings,
                          q_doc_weights,
                          alpha, beta)
#   print(clicks.shape,"clicks.shape")
  # if store_per_rank:
  #   store_cutoff = alpha.shape[0]
  #   clicks_per_doc = np.zeros((n_docs, store_cutoff),
  #                             dtype=np.int32)
  #   ind_tile = np.tile(np.arange(cutoff)[None,:], (n_samples, 1))
  #   np.add.at(clicks_per_doc, (rankings[clicks],
  #                              ind_tile[clicks]), 1)
  # else:
  #   clicks_per_doc = np.zeros(n_docs, dtype=np.int32)
  #   np.add.at(clicks_per_doc, rankings[clicks], 1)

  # if return_display:
  #   if store_per_rank:
  #     displays_per_doc = np.zeros((n_docs, store_cutoff),
  #                                 dtype=np.int32)
  #     np.add.at(displays_per_doc, (rankings, ind_tile), 1)
  #   else:
  #     displays_per_doc = np.zeros(n_docs, dtype=np.int32)
  #     np.add.at(displays_per_doc, rankings, 1)
  # else:
  #   displays_per_doc = None

  # return clicks_per_doc, displays_per_doc
  return clicks,rankings,scores


def single_ranking_generation_cold(
                     qid,
                     data_split,
                     doc_weights,
                     alpha,
                     beta,
                     model=None,
                     all_policy_scores=None,
                     return_scores=False,intervene_strategy="Gumbel",\
    result_logging={},mask=None,mask_rng=None,prob=1.0,prefix=""):
    assert model is not None or all_policy_scores is not None

    n_docs = data_split.query_size(qid)
    cutoff = min(alpha.shape[0], n_docs)

    if all_policy_scores is None:
        q_feat = data_split.query_feat(qid)
        policy_scores = model(q_feat)[:,0].numpy()
    else:
        policy_scores = data_split.query_values_from_vector(
                                      qid, all_policy_scores)
    if intervene_strategy=="dropout":

        ranking_scores=model(q_feat,training=True)[:,0].numpy()
    else:
        ranking_scores=policy_scores
    dataset.update_mask(mask,prob=prob,cold_rng=mask_rng)
    policy_scores=dataset.mask_items(mask,policy_scores,flag=1)
#     print(mask.shape,np.sum(mask==0),np.sum(policy_scores==-np.inf),\
#           "np.sum(mask==1),np.sum(policy_scores==-np.inf)")
    rankings = pl.gumbel_sample_rankings(
                        policy_scores,
                        1,
                        cutoff)[0]

#     rankings = pl.get_ranking(
#                       ranking_scores,
#                       1,
#                       cutoff,
#                       intervene_strategy=intervene_strategy)[0]
    q_doc_weights = data_split.query_values_from_vector(
                                  qid, doc_weights)
    clicks = generate_clicks(
                          rankings,
                          q_doc_weights,
                          alpha, beta)
    ndcg_list=pl.NDCG_based_on_samples(rankings,q_doc_weights)
    dcg_list=pl.DCG_based_on_samples(rankings,q_doc_weights)
    cdcg_list=pl.cDCG(clicks)

    result_logging[prefix+"ndcg"].append(ndcg_list)
    result_logging[prefix+"dcg"].append(dcg_list)
    result_logging[prefix+"cdcg"].append(cdcg_list)
    if return_scores:
        return rankings[0,:], clicks[0,:], policy_scores
    else:
        return rankings[0,:], clicks[0,:]

def single_ranking_generation(
                     qid,
                     data_split,
                     doc_weights,
                     alpha,
                     beta,
                     model=None,
                     all_policy_scores=None,
                     return_scores=False,intervene_strategy="Gumbel",result_logging={},):
    assert model is not None or all_policy_scores is not None

    n_docs = data_split.query_size(qid)
    cutoff = min(alpha.shape[0], n_docs)

    if all_policy_scores is None:
        q_feat = data_split.query_feat(qid)
        policy_scores = model(q_feat)[:,0].numpy()
    else:
        policy_scores = data_split.query_values_from_vector(
                                      qid, all_policy_scores)
    if intervene_strategy=="dropout":

        ranking_scores=model(q_feat,training=True)[:,0].numpy()
    else:
        ranking_scores=policy_scores

    rankings = pl.gumbel_sample_rankings(
                        policy_scores,
                        1,
                        cutoff)[0]

#     rankings = pl.get_ranking(
#                       ranking_scores,
#                       1,
#                       cutoff,
#                       intervene_strategy=intervene_strategy)[0]
    q_doc_weights = data_split.query_values_from_vector(
                                  qid, doc_weights)
    clicks = generate_clicks(
                          rankings,
                          q_doc_weights,
                          alpha, beta)
    ndcg_list=pl.NDCG_based_on_samples(rankings,q_doc_weights)
    dcg_list=pl.DCG_based_on_samples(rankings,q_doc_weights)
    cdcg_list=pl.cDCG(clicks)

    result_logging["ndcg"].append(ndcg_list)
    result_logging["dcg"].append(dcg_list)
    result_logging["cdcg"].append(cdcg_list)
    if return_scores:
        return rankings[0,:], clicks[0,:], policy_scores
    else:
        return rankings[0,:], clicks[0,:]

def generate_clicks(sampled_rankings,
                    doc_weights,
                    alpha, beta):
  # print(sampled_rankings.shape,"ranking shape")
  cutoff = min(sampled_rankings.shape[1], alpha.shape[0])
  ranked_weights = doc_weights[sampled_rankings]
  click_prob = ranked_weights*alpha[None, :cutoff] + beta[None, :cutoff]

  noise = np.random.uniform(size=click_prob.shape)
  return noise < click_prob

def multiplt_ranking_generation(data_split,
                     n_samples,
                     true_doc_weights,
                     alpha,
                     beta,
                     model=None,
                     intervene_strategy="gumbel"):
    n_queries = data_split.num_queries()
    n_docs = data_split.num_docs()
    sampled_queries = np.random.choice(n_queries, size=n_samples)
    cutoff = alpha.shape[0]
    all_clicks = np.zeros((n_samples, cutoff), dtype=np.int64)
    # all_alpha = np.zeros((n_docs, cutoff), dtype=np.float32)
    # all_beta = np.zeros((n_docs, cutoff), dtype=np.float32)
    all_rankings=np.zeros((n_samples, cutoff), dtype=np.int64)
    qid_list=[]
    for cur_sample in progressbar(range(n_samples)):
        query_length=0
        while query_length<cutoff:
            qid = np.random.choice(n_queries, size=1)[0]
            query_length= data_split.query_size(qid)  ## for simpliciy we only select query length greater than cutoff.
        
        new_ranking,new_clicks=single_ranking_generation(
                            qid,
                            data_split,
                            true_doc_weights,
                            alpha,
                            beta,
                            model=model,                                                             
                            intervene_strategy=intervene_strategy)
        qid_list.append(qid)
        base=data_split.query_range(qid)[0]
        new_ranking=new_ranking+base
        all_rankings[cur_sample]=new_ranking
        all_clicks[cur_sample]+=new_clicks
        # all_alpha[new_ranking]+=alpha
        # all_beta[new_ranking]+=beta
    return all_rankings,all_clicks,qid_list

def simulate_on_dataset_sigmoid(data_train,
                        data_validation,
                        n_samples,
                        true_train_doc_weights,
                        true_validation_doc_weights,
                        alpha,
                        beta,
                        model=None,
                        intervene_strategy="gumbel"
                        ):
  n_train_queries = data_train.num_queries()
  n_vali_queries = data_validation.num_queries()
  train_ratio = n_train_queries/float(n_train_queries + n_vali_queries)

  sampled_queries = np.random.choice(2, size=n_samples,
                                     p=[train_ratio, 1.-train_ratio])
  samples_per_split = np.zeros(2, dtype=np.int64)
  np.add.at(samples_per_split, sampled_queries, 1)
  (train_rankings,train_clicks,train_qids) = multiplt_ranking_generation(
                     data_train,
                     samples_per_split[0],
                     true_train_doc_weights,
                     alpha,
                     beta,
                     model=model,
                     intervene_strategy=intervene_strategy,
                     )
  
  (vali_rankings,vali_clicks,vali_qids) = multiplt_ranking_generation(
                     data_validation,
                     samples_per_split[1],
                     true_validation_doc_weights,
                     alpha,
                     beta,
                     model=model,
                     intervene_strategy=intervene_strategy)
  return (train_rankings,train_clicks,train_qids,
        vali_rankings,vali_clicks,vali_qids)