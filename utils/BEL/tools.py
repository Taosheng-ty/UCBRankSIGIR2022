import os
import json
import logging
def configure_logging(logging):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
def iterate_settings(ind, all_settings,cur_setting={},path="."):
    setting_name_list=list(all_settings.keys())
    if ind==len(setting_name_list):
        cur_setting["log_dir"]=path
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+"/setting.json", 'w+') as f:
            json.dump(cur_setting, f)            
        return
    cur_setting=dict(cur_setting)
    cur_setting_name=setting_name_list[ind]
    for i in all_settings[cur_setting_name]:
        cur_path=path+"/"+cur_setting_name+"_"+str(i)
        cur_setting[cur_setting_name]=i
        iterate_settings(ind+1, all_settings,cur_setting,cur_path)