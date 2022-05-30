import glob
import os
import pandas as pd
import json
import matplotlib.pyplot as plt 
from progressbar import progressbar
import numpy as np
from pathlib import Path
import pickle
def read_json(file_path):
    with open(file_path,"r") as f:
        result=json.load(f)
    return result
def find_max_len(result):
    length=1
    for i in result.keys():
        length=max(length,len(result[i]))
    return length
def append_single(result,max_len):
    for i in result.keys():
        if len(result[i])<max_len:
            result[i]=result[i]+[result[i][-1]]*(max_len-len(result[i]))
def write_back(result,file_path):
    with open(file_path,"w") as f:
        json.dump(result,f)
def get_path_sets(root_path,same_length=False):
    paths=glob.glob(root_path+'/**/*.jjson', recursive=True)
    # print(paths)
    path_sets=set()
    for path in paths:
        pp=Path(paths[0])
        # path_root=str(pp.parent.parent)
        path_root=os.path.join(*path.split("/")[:-2])
#         print(path_root,path_root,"path_root")
        if same_length:
            result_cur=read_json(path)
            max_len=find_max_len(result_cur)
            append_single(result_cur,max_len)
            write_back(result_cur,path)
        # print(result_cur)
        # print(path_root)
        path_sets.add(path_root)
    return path_sets
def merge_single_experiment_results(root_path):
    paths=glob.glob(root_path+'/**/*.jjson', recursive=True)
    # print(paths,root_path)
    df_result_cur=pd.DataFrame()
    for path in paths:
#         print(path)
        with open(path, "r") as read_file:
        #     print("Converting JSON encoded data into Python dictionary")
            developer = json.load(read_file)
            pd_frame=pd.DataFrame(developer).fillna(np.nan)    
        df_result_cur=df_result_cur.append(pd_frame)
#     print(df_result_cur)
    return df_result_cur

def merge_multiple_experiment_results(root_path):
    path_sets=get_path_sets(root_path)
    for path_set in progressbar(path_sets):
        df_result_cur=merge_single_experiment_results(path_set)
        df_result_cur.to_csv(path_set+"/result.ccsv")

class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
def get_node(root_path,path):
    # print(root_path,path,"root_path,path")
    all_node=path.split("/")
    start_folder=root_path.split("/")[-1]
    # print(root_path.split("/"),all_node)
    ind=all_node.index(start_folder)
    start_node=all_node[ind+1:]
    # print(start_node)
    return start_node
def set_node_val(node_list,multi_level_dict,val):
    for i in node_list[:-1]:
        multi_level_dict=multi_level_dict[i]
    last_node=node_list[-1]
    multi_level_dict[last_node]=val
def get_result_df(root_path,same_length=False,groupby="iterations",filter_list=[],only_mean=False,rerun=False):
    results_path=os.path.join(root_path,"results.pickle")
    results_path_exist=glob.glob(results_path)
#     print(results_path)
    if results_path_exist and not rerun:
        results_path=results_path_exist[0]
        print("found saved results")
        with open(results_path, 'rb') as handle:
            result,result_mean = pickle.load(handle)
    else:
        path_sets=get_path_sets(root_path,same_length)
        # print(path_sets)
        result=AutoVivification()
        result_mean=AutoVivification()
        for path_set in path_sets:
            node=get_node(root_path,path_set)
            result_cur=merge_single_experiment_results(path_set)
            set_node_val(node,result,result_cur)
            result_merged_cur=result_cur.groupby(groupby).mean().reset_index()
            set_node_val(node,result_mean,result_merged_cur)
        with open(results_path, 'wb') as handle:
            pickle.dump([result,result_mean], handle, protocol=pickle.HIGHEST_PROTOCOL)
    if not only_mean:
        return result,result_mean
    else:
        return result

import itertools
def splitSerToArr(ser):
    return [ser.index, ser.as_matrix()]


def plot_metrics(name_results_pair:dict,plots_y_partition:str="metrics_NDCG",errbar=True,
plots_x_partition:str="iterations",groupby="iterations",ax=None,graph_param=None,smoooth_fn=None)->None:
    
    '''    
        name_results_pair:{method_name:result_dataframe}
        plots_partition: key name in each result_dataframe which need to be plotted
    '''
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_list = prop_cycle.by_key()['color']
    colors=itertools.cycle(colors_list)
    marker = itertools.cycle((',', '+', '.', 'o', '*')) 
    if ax:
        plot=ax
    else:
        plot=plt
    for algo_name in name_results_pair:
            algo_result=name_results_pair[algo_name]
#             print(type(algo_result),"*"*100)
            mean_orig=algo_result.groupby(groupby).mean().reset_index()
           
            std=algo_result.groupby(groupby).std().reset_index()
            mean = mean_orig[mean_orig[plots_y_partition].notna()]
            if smoooth_fn is not None:
                mean[plots_y_partition]=smoooth_fn(mean[plots_y_partition])
                errbar=False
            
            std = std[mean_orig[plots_y_partition].notna()]
            if plots_x_partition not in mean.keys() or plots_y_partition not in mean.keys() :
                continue
#             assert plots_y_partition in algo_result, algo_name+" doesn't contain the partition "+plots_y_partition
            if not errbar:
                plot.plot(mean[plots_x_partition],mean[plots_y_partition], marker = next(marker),color=next(colors), label=algo_name)
            else:
                plot.errorbar(mean[plots_x_partition],mean[plots_y_partition], yerr=std[plots_y_partition], marker = next(marker),color=next(colors), label=algo_name)
    if ax is None:
        gca=plot.gca()
        gca.set(**graph_param)
        plot.legend()