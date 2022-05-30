import sys
sys.path.append("../")
import utils.dataset as dataset
import utils.result_organization as rog
import utils.evaluation as evl
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import pandas as pd
from utils.evaluation import *
from utils.BEL import results_org

def tradeoff_plot(result:dict,metrics_name,step,ax=None,xlim=None,savepath=None):
    if ax:
        plot=ax
    else:
        plot=plt

    for key, value in result.items():
        result_list=evl.extract_tradeoff_res(value,[metrics_name],step)
        param,result_metric=result_list[0],result_list[1]
        if len(result_metric)>1:
#                 print(result_metric[i])
            mean=np.mean(result_metric,axis=1)
            std=np.std(result_metric,axis=1)
            ind=np.arange(mean.shape[0])
            if xlim:
                ind=param<=xlim
#                     print(ind)
            plot.errorbar(param[ind],mean[ind],yerr=std[ind],label=key)
        else:
            plot.plot(param,result_metric,label=key)

if __name__ == '__main__':
#     path="addbehaviour_cold_29oct_fair/show_prob5"
    path="/raid/datasets/shared/tao_tem"
    result,result_mean=results_org.get_result_df(path,groupby="iterations")
    vali_name="discounted_sum_vali_cdcg"
    step=99  
    data_rename={
                 "MSLR-WEB30k_beh_rm":"MSLR-WEB30k",\
                 "MSLR-WEB10k_beh_rm":"MSLR-WEB10k",\
                "NP2003":"NP2003",\
                "NP2004":"NP2004",\
                "MQ2007":"MQ2007",\
                "MQ2008":"MQ2008"}
    path_dir=Path("local_output/figure1")
    path_dir.mkdir(parents=True, exist_ok=True)
    for datasets,data_name_cur in data_rename.items():
        update="n_updates_20"
        # result,result_mean=get_result_df(path,groupby="iterations")
        result_validated={}
        # result_validated["ips_random"]=result['no_behav'][datasets]["ips_random_tradeoff"][update]
        # result_validated["ips_ucbrandom"]=result['no_behav'][datasets]["ips_ucb"][update]
        # result_validated["merge_random"]=result['no_behav'][datasets]["merge_random_tradeoff"][update]
        # result_validated["merge_ucb"]=result['no_behav'][datasets]["merge_ucb"][update]
        # result_validated["w_click_random"]=result['add_click'][datasets]["random_tradeoff"][update]
        result_validated["Feature-IE(UCB)"]=result['no_behav'][datasets]["merge_ucb"][update]
        result_validated["Feature-IE(EpsilonRank)"]=result['no_behav'][datasets]["merge_random_tradeoff"][update]
      
        result_validated["Feature-w-ctr(EpsilonRank)"]=result['add_ctr'][datasets]["random_tradeoff"][update]
        # result_validated["w_click_std"]=result['add_click'][datasets]["std_proposional"][update]

        # result_validated["w_ips_std"]=result['add_ips'][datasets]["std_proposional"][update]
        result_validated["Feature-w-$\Delta$(EpsilonRank)"]=result['add_ips'][datasets]["random_tradeoff"][update]

        result_validated["Feature-w/o-behav.(EpsilonRank)"]=result['no_behav'][datasets]["random_tradeoff"][update]
        # result_validated["Feature-w/o-behav._std"]=result['no_behav'][datasets]["std_proposional"][update]
        #

        # tradeoff_plot(result_validated,["vali_ndcg"],xlim=100,step=21,savepath="local_output/")
        fig, axs = plt.subplots(2,figsize=(5,4), sharex=True)
        tradeoff_plot(result_validated,"test_cold_ndcg_masked",ax=axs[0],xlim=100,step=99)

        # ax=axs[0]
        # axins = ax.inset_axes([0.5, 0.5, 0.18, 0.18])
        # tradeoff_plot(result_validated,["test_cold_ndcg_masked"],ax=axins,xlim=100,step=19)
        # # sub region of the original image
        # x1, x2, y1, y2 = 8, 50, 0.475, 0.485
        # axins.set_xlim(x1, x2)
        # axins.set_ylim(y1, y2)
        # axins.set_xticklabels('')
        # axins.set_yticklabels('')

        # ax.indicate_inset_zoom(axins, edgecolor="black")

        tradeoff_plot(result_validated,"test_warm_ndcg_masked",ax=axs[1],xlim=100,step=99)
        fig.tight_layout(h_pad=0)
        
        axs[0].set_xscale("log")
        axs[0].set_ylabel("NDCG@5-Cold")
        axs[1].set_xlabel("tradeoff parameter")
        axs[1].legend(bbox_to_anchor=(1.1, 1.05))
        axs[1].set_xscale("log")
        axs[1].set_ylabel("NDCG@5-Warm")
        plt.savefig("local_output/figure1/"+data_name_cur+"performance_along_tradeoff_param.pdf", dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close()