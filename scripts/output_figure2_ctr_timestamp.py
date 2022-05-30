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

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

if __name__ == '__main__':
#     path="addbehaviour_cold_29oct_fair/show_prob5"
    path="/raid/datasets/shared/tao_tem"
    result,result_mean=results_org.get_result_df(path,groupby="iterations")
    vali_name="discounted_sum_vali_cdcg"
    step=99  
    data_rename={
                 "MSLR-WEB30k_beh_rm":"MSLR-WEB30k",\
                 "MSLR-WEB10k_beh_rm":"MSLR-WEB10k",\
#                 "NP2003":"NP2003",\
#                 "NP2004":"NP2004",\
                "MQ2007":"MQ2007",\
                "MQ2008":"MQ2008"}
    data_maximum_label={
                 "MSLR-WEB30k_beh_rm":"4",\
                 "MSLR-WEB10k_beh_rm":"4",\
#                 "NP2003":"NP2003",\
#                 "NP2004":"NP2004",\
                "MQ2007":"2",\
                "MQ2008":"2"}    
    path_dir=Path("local_output/figure2_ctr_timestamp")
    path_dir.mkdir(parents=True, exist_ok=True)
    for datasets,data_name_cur in data_rename.items():
        update="n_updates_20"
        # result,result_mean=get_result_df(path,groupby="iterations")
        result_reorg={
                    "UCBRank+Feature-IE":result['no_behav'][datasets]["merge_ucb"][update],
                    "EpsilonRank+Feature-IE":result['no_behav'][datasets]["merge_random_tradeoff"][update],

        }
        vali_name="discounted_sum_vali_cdcg"
        step=99
        result_validated=rog.get_validation(result_reorg,vali_name,step)
        result_validated["Top-k+Feature-IE"]=result['no_behav'][datasets]["merge_random_tradeoff"][update]["tradeoff_0.0"]
        result_validated["Random-k+Feature-IE"]=result['no_behav'][datasets]["merge_random_tradeoff"][update]["tradeoff_100.0"]
        graph_param={}
        graph_param["title"]="Items' ctr vs. entering time"
        plots_y_partition="result_time_stamp_ctr_least_label"+data_maximum_label[datasets]
        results_org.plot_metrics(result_validated,plots_y_partition=plots_y_partition,\
                     graph_param=graph_param,smoooth_fn=smooth)
        plt.ylabel("ctr")
        plt.xlabel("timestamp of entering")
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.locator_params(axis="x", nbins=5)
#         fig.tight_layout(h_pad=0)
        
        plt.savefig("local_output/figure2_ctr_timestamp/"+data_name_cur+"ctr_timestamp.pdf", dpi=600, bbox_inches = 'tight', pad_inches = 0.05)
        plt.close()