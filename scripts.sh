

##counterfactual no intervention
python3 interventionaware_run.py local_output/interventionaware.txt --n_updates 0 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1 
##counterfactual 100 intervention
python3 interventionaware_run.py local_output/interventionaware_100interventions.txt --n_updates 100 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1

##counterfactual 1 intervention
python3 interventionaware_run.py local_output/interventionaware_1interventions.txt --n_updates 1 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1
##counterfactual 10 intervention
python3 interventionaware_run.py local_output/interventionaware_10interventions.txt --n_updates 10 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1


##counterfactual no intervention with direct intervene strategy
python3 interventionaware_run.py local_output/interventionaware_direct.txt --n_updates 0 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1 --intervene_strategy direct
##counterfactual 1 intervention with direct intervene strategy
python3 interventionaware_run.py local_output/interventionaware_1interventions_direct.txt --n_updates 1 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1 --intervene_strategy direct
##counterfactual 10 intervention with direct intervene strategy
python3 interventionaware_run.py local_output/interventionaware_10interventions_direct.txt --n_updates 10 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1 --intervene_strategy direct


##counterfactual no intervention with random intervene strategy
python3 interventionaware_run.py local_output/interventionaware_random.txt --n_updates 0 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1 --intervene_strategy random
##counterfactual 1 intervention with random intervene strategy
python3 interventionaware_run.py local_output/interventionaware_1interventions_random.txt --n_updates 1 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1 --intervene_strategy random
##counterfactual 10 intervention with random intervene strategy
python3 interventionaware_run.py local_output/interventionaware_10interventions_random.txt --n_updates 10 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1 --intervene_strategy random



##PDGD online
python3 PDGD_run.py local_output/PDGD_online.txt  --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1
## supervised_run
python3 supervised_run.py local_output/Supervised.txt   --dataset Webscope_C14_Set1
## interventionoblivious_run
python3 --output_path interventionoblivious_run.py local_output/interventionoblivious.txt  --pretrained_model local_output/pretrained_model.h5  --dataset Webscope_C14_Set1 --n_updates 100


## supervised_run withdropout model
python3 supervised_run.py local_output/Supervised_dropout.txt   --dataset Webscope_C14_Set1 --dropout



##toy setup

nohup python3 interventionaware_run.py local_output/interventionaware_gumbel_toy.txt --n_updates 0 --pretrained_model local_output/pretrained_model.h5 --dataset Webscope_C14_Set1  --Maximum_iteration 10000 --n_eval 10 &




##batch scripts 
slurm_python --CODE_PATH=. --Cmd_file=interventionaware_run.py --JSON_PATH=/home/ec2-user/documents/uncertainty/2021wsdm-unifying-LTR/local_output/toy_settings --json2args --plain_script --jobs_limit=30


slurm_python --CODE_PATH=. --Cmd_file=interventionaware_run_sigmoid_loss.py  --JSON_PATH=/home/ec2-user/documents/uncertainty/2021wsdm-unifying-LTR/local_output/settings_sigmoid_1000000/loss_session_obliv/cutoff_5/  --json2args --plain_script --jobs_limit=12

##kill gpu jobs
nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 kill -9
nvidia-smi -q -d PIDS | grep -P "Process ID +: [0-9]+" | grep -Po "[0-9]+" | xargs kill -9

kill -9 $(ps aux | grep -e myProcessName | awk '{ print $2 }') 


slurm_python --CODE_PATH=. --Cmd_file=supervised_run_point.py  --JSON_PATH=/home/ec2-user/documents/uncertainty/2021wsdm-unifying-LTR/prerained_fully  --json2args --plain_script --jobs_limit=30 --python_ver=tf15


##get correlation
slurm_python --CODE_PATH=. --Cmd_file=get_correlation.py  --JSON_PATH=/home/ec2-user/documents/uncertainty/2021wsdm-unifying-LTR/pretrained/correlation/  --json2args --plain_script   --python_ver=tf15  --secs_each_sub=1  --memory_usage=50


## run point supervised
slurm_python --CODE_PATH=. --Cmd_file=supervised_run_point.py  --JSON_PATH=pretrained/MSLR-WEB30k_beh_rm/  --json2args --plain_script --jobs_limit=30 --secs_each_sub=1 --python_ver=ultra_p36