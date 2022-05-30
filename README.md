# Uncertainty-aware Online  Learning to Rank.
This repository contains the code used for the experiments borrow a lot code from https://github.com/HarrieO/2021wsdm-unifying-LTR


License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Usage
-------

This code makes use of [Python 3](https://www.python.org/) and the [numpy](https://numpy.org/) package, make sure they are installed.

Below is the link you can dowlownd the preprocessed datasets.
https://drive.google.com/file/d/17NDVk354G2Zv9_e2_63id_Ng0T-V9E6u/view?usp=sharing
In the link, there are four datasets. MQ2008 and MQ2007 originally don't contain any behaviour features. MSLR-WEB30K and MSLR-WEB10K originally contain behaviour features i.e., feature
134-136).  Here we remove the those features and rename the datasets as MSLR-WEB30K_rm_clicks and MSLR-WEB10K_rm_clicks.

A file is required that explains the location and details of the LTR datasets available on the system. You can copy the file:
```
cp example_datasets_info.txt local_dataset_info.txt
```
Open this copy and edit the paths to the folders where the train/test/vali files are placed (where you put the downloaded datasets).

Here are some command-line examples that illustrate how the results in the paper can be replicated.
First create a folder to store the resulting models:
```
mkdir local_output
```


For the uncertainty-aware code, it is mainly within interventionaware_run_sigmoid_loss.py, For instance, the following command will use ucb_std internvention strategy:
```
python  interventionaware_run_sigmoid_loss.py   --n_eval=10 --n_iteration=10000 --cutoff=5 --epochs=100 --session_aware=True --query_least_size=5 --optimizer=Adam --use_GPU=True --early_stop_patience=5 --batch_size=65536 --tradeoff_param=0.0  --intervene_strategy=ucb_std --n_updates=10 --dataset=MQ2008 --output_path=local_output/ucb_std_output.txt
```

Reproduce
-------
After we finish the experiments, 
you can run the following to produce figure 1
```
python scripts/output_figure1.py
```
run the following to produce table 1
```
python scripts/output_data_statistics_table1.py
```
run the following to produce table 2 and 3
```
python scripts/output_table2-3.py
```