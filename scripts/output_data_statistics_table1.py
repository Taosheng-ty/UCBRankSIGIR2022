import sys
from pathlib import Path
import os
sys.path.append("../")
import utils.dataset as dataset
if __name__ == '__main__':
    data_name_list=["NP2003","NP2004","MQ2007","MQ2008","MSLR-WEB10k_beh_rm","MSLR-WEB30k_beh_rm"]
    stas_pd=dataset.get_mutiple_data_statics(data_name_list)
    path_dir=Path("local_output/datasets_statistics_table1")
    path_dir.mkdir(parents=True, exist_ok=True)
    output_path=os.path.join("local_output/datasets_statistics_table1","Datasets-statistics.csv")
    stas_pd.to_csv(output_path)
    print(stas_pd)