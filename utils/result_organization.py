import sys
from utils.BEL.results_org import *
import numpy as np   
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
    steps=cur_res_split[tradeoff_params[0]].index[step]
    performance=[cur_res_split[tradeoff_param][vali_name][steps].mean() for tradeoff_param in tradeoff_params]
    performance=np.array(performance)
#     print(performance)
    argmax=np.argmax(performance)
    max_param=tradeoff_params[argmax]
    print(tradeoff_nums[argmax],"from", tradeoff_nums)
    return max_param
def get_validation(result:dict,vali_name,step):
    result_vali={}
    for key, value in result.items():
        tradeoff_param=get_best_tradeoff_param(value,vali_name,step)
        if tradeoff_param:
            result_vali[key]=value[tradeoff_param]
        else:
            result_vali[key]=None
    return result_vali
def get_performance(vali_name,metric_name,step,result,feature_settings,feature_settings_rename,datasets,data_rename,update):
    result_final=[]
    for feature_setting in feature_settings:
        
        for dataset in  datasets:
            print(result.keys(),"keys")
            if feature_setting not in result or dataset not in result[feature_setting] or "random_tradeoff" not in result[feature_setting][dataset]:
                continue
            print(feature_setting,dataset,"feature_setting,dataset,")
            result_validated={}
            result_validated[feature_settings_rename[feature_setting]+"Top-k"]= result[feature_setting][dataset]["random_tradeoff"][update]["tradeoff_0.0"]
            result_validated[feature_settings_rename[feature_setting]+"Random-k"]= result[feature_setting][dataset]["random_tradeoff"][update]["tradeoff_100.0"]

            result_cur={
#                 feature_settings_rename[feature_setting]+"_ucb":result[feature_setting][dataset]["std_proposional"][update],\
                       feature_settings_rename[feature_setting]+"_EpsilonRank":result[feature_setting][dataset]["random_tradeoff"][update],\
                       }
            if "std_proposional" in result[feature_setting][dataset]:
                result_cur[feature_settings_rename[feature_setting]+"_ucb"]=result[feature_setting][dataset]["std_proposional"][update]
            print(dataset,feature_setting)
            result_validated.update(get_validation(result_cur,vali_name,step))
#             print(result_validated,"fdas")
#             print("begin pdgd")
            if "PDGD" in result[feature_setting][dataset]:
                result_validated[feature_settings_rename[feature_setting]+"pdgd"]= result[feature_setting][dataset]["PDGD"][update]["tradeoff_1.0"]
            result_validated=extract_step_metric(result_validated,metric_name,step,data_rename[dataset])
            print(result_validated,"result_validated")
            result_final=result_final+result_validated
    #         print(result_validated)
    for dataset in  datasets:
        print(dataset,"merge_ucb")
        result_cur={"Feature-IE_ucb":result["no_behav"][dataset]["merge_ucb"][update],\
                    "Feature-IE_EpsilonRank":result["no_behav"][dataset]["merge_random_tradeoff"][update],\
                           }
        result_validated=get_validation(result_cur,vali_name,step)
        result_validated["Feature-IE_Top-k"]= result["no_behav"][dataset]["merge_random_tradeoff"][update]["tradeoff_0.0"]
        result_validated["Feature-IE_Random-k"]= result["no_behav"][dataset]["merge_random_tradeoff"][update]["tradeoff_1000.0"]
        result_validated=extract_step_metric(result_validated,metric_name,step,data_rename[dataset])
        result_final=result_final+result_validated
#     for dataset in  datasets:
#         print(dataset,"ips_ucb")
#         result_cur={"ips_ucb":result["no_behav"][dataset]["ips_ucb"][update],\
#                     "ips_random_tradeoff":result["no_behav"][dataset]["ips_random_tradeoff"][update],
#                            }
#         result_validated=get_validation(result_cur,vali_name,step)
#         result_validated["ips_Top-k"]= result["no_behav"][dataset]["ips_random_tradeoff"][update]["tradeoff_0.0"]
#         result_validated["ips_Random-k"]= result["no_behav"][dataset]["ips_random_tradeoff"][update]["tradeoff_100.0"]
#         result_validated=extract_step_metric(result_validated,metric_name,step,data_rename[dataset])
#         result_final=result_final+result_validated
         
        
    result_dfram=pd.DataFrame(result_final, columns=["method","datasets","metrics"])
    print(result_dfram)
    result_dfram=result_dfram.pivot(index='method', columns='datasets', values='metrics')
#     print(result_dfram)
    return result_dfram
def latex_two_f(x, y):                      # this is a demo function that takes in two ints and 
    if x>1:
        x="{:#.4G}".format(x)
    else:
        x="{:.3f}".format(x)
    if y>1:
        y="{:#.4G}".format(y)
    else:
        y="{:.3f}".format(y)
    return "&"+str(x) + "\$_{("+str(y)+")}\$"        # concatenate them as str
vec_latex_two_f = np.vectorize(latex_two_f) 
def latex_single_f(x):                      # this is a demo function that takes in two ints and 
    if x>1:
        x="{:#.3G}".format(x)
    else:
        x="{:.3f}".format(x)
    return "&"+str(x)        # concatenate them as str
vec_latex_single_f = np.vectorize(latex_single_f) 
def to_latex(result_dataframe):
    std=result_dataframe.applymap(func=np.std)
    mean=result_dataframe.applymap(func=np.mean)
    result=pd.DataFrame(vec_latex_single_f(mean),index=mean.index,columns=mean.columns)
    result_std=pd.DataFrame(vec_latex_two_f(mean, std),index=mean.index,columns=mean.columns) 
    return result,result_std
def get_freq_singe(x):
    if not isinstance(x, list):
        if pd.isnull(x):
            return 0
        else:
            return 1
    else:
        return len(x)
def get_round_value(x):
    if not isinstance(x, list):
        return 1
    else:
        return [round(i, 2) for i in x]
def to_round(result_dataframe):
    frequency=result_dataframe.applymap(func=get_round_value)
    return frequency
def to_freq(result_dataframe):
    frequency=result_dataframe.applymap(func=get_freq_singe)
    return frequency
def extract_step_metric(result:dict,metric_name,step,data_name):
    result_return=[]
    for key, value in result.items():
        try:
            result_return.append([key,data_name,value[metric_name][step].tolist()])
        except ValueError: 
            print(value[metric_name][step])
            print(dir(value[metric_name][step]))
#         if "tolist" in dir(value[metric_name][step]):
#             result_return.append([key,data_name,value[metric_name][step].tolist()])
#         else:
#             result_return.append([key,data_name,[]])
    return result_return      
def return_table(paths_data_result,paths_data_dict,vali_name,step,metric_name,update):
    result_table=None
    for path in paths_data_dict:
        datasets=paths_data_dict[path]
        result=paths_data_result[path]
        feature_settings=["no_behav",'add_click',"add_ips",'add_ctr']
        feature_settings=["no_behav","add_ips",'add_ctr']
        feature_settings_rename={"no_behav":"Feature-w/o-behav.",'add_ips':"Feature-w-$\Delta$",'add_click':"w-click",'add_ctr':"Feature-w-ctr"}
        data_rename={"MSLR-WEB30k_beh_rm1%":"MSLR-toy",\
                     "MSLR-WEB30k_beh_rm":"MSLR-WEB30k",\
                     "MSLR-WEB10k_beh_rm":"MSLR-WEB10k",\
                    "istella-s":"istella-s",\
                    "Webscope_C14_Set1":"Yahoo!",
                    "NP2003":"NP2003",\
                    "NP2004":"NP2004",\
                    "MQ2007":"MQ2007",\
                    "MQ2008":"MQ2008"}
        setting_dict={"feature_settings":feature_settings,\
                      "feature_settings_rename":feature_settings_rename,\
                     "datasets":datasets,\
                     "data_rename":data_rename,\
                     "vali_name":vali_name,\
                     "step":step,\
                     "result":result,\
                     "metric_name":metric_name,\
                     "update":update}
        result_cur=get_performance(**setting_dict)
#         print(type(result_cur),"type")
        if result_table is not None:
            result_table=result_table.join(result_cur)
        else:
            result_table=result_cur
    return result_table