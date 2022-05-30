import sys
sys.path.append("../")
import utils.dataset as dataset
import utils.result_organization as rog
import matplotlib.pyplot as plt 
import numpy as np
from pathlib import Path
import pandas as pd
from utils.evaluation import *
from utils.BEL.results_org import *
if __name__ == '__main__':
#     path="addbehaviour_cold_29oct_fair/show_prob5"
    path="/raid/datasets/shared/tao_tem"
    paths=[path]
    paths_data_dict={path:["MQ2007","MQ2008","NP2003","NP2004","MSLR-WEB10k_beh_rm","MSLR-WEB30k_beh_rm"]}
    paths_data_result={}
    for path in paths_data_dict:
        result_all,result_mean=get_result_df(path,groupby="iterations")
        paths_data_result[path]=result_all

    vali_name="discounted_sum_vali_cdcg"
    step=99
    metric_dict={"cumulative-ndcg":"discounted_sum_test_ndcg","ndcg-warm":"test_warm_ndcg_masked","ndcg-cold":"test_cold_ndcg_masked"}
    for name, metric_name in metric_dict.items():
        path_dir=Path("local_output/"+name)
        path_dir.mkdir(parents=True, exist_ok=True)
        update="n_updates_20"
        setting_dict={"paths_data_result":paths_data_result,
                      "paths_data_dict":paths_data_dict,
                     "vali_name":vali_name,\
                     "step":step,\
                     "metric_name":metric_name,\
                     "update":update}
        result=rog.return_table(**setting_dict)
        r,rstd=rog.to_latex(result)
        columnsTitles = ['MQ2007', 'MQ2008','MSLR-WEB10k',"MSLR-WEB30k","NP2003","NP2004",]
        # columnsTitles = ['MQ2007', 'MQ2008']
        r = r.reindex(columns=columnsTitles)
        rstd = rstd.reindex(columns=columnsTitles)
        output_path=os.path.join("local_output",name,"mean.csv")
        rstd.to_csv(output_path)
        output_path=os.path.join("local_output",name,"mean_std.csv")
        rstd.to_csv(output_path)
        freq=rog.to_freq(result)
        freq = freq.reindex(columns=columnsTitles)
        output_path=os.path.join("local_output",name,"num_trials.csv")
        freq.to_csv(output_path)
        detailed=rog.to_round(result)
        output_path=os.path.join("local_output",name,"detailed_results.csv")
        detailed.to_csv(output_path)