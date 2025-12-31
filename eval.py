from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'clam_mh', 'ABMIL', 'ABMIL_CLAM', 'TransMIL', 'TransMIL_CLAM'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'HPV', 'HPV2', 
            'HPV512', 'HPV_cesc', 'HPV_cesc_sur', 'HPV_cesc_lusc', 'HPV_cesc_lusc_alone', 'HPV_cesc_subtypes', 'CPTAC_HNSCC_CESC', 'CPTAC_HNSCC'],)
parser.add_argument('--drop_out', type=float, default=0.25, help='dropout')
parser.add_argument('--embed_dim', type=int, default=1024)
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])
    
elif args.task == 'HPV':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/HNSCC.csv',
                            data_dir= os.path.join(args.data_root_dir, 'HPV_UNI2_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'HPV-':0, 'HPV+':1},
                            patient_strat= False,
                            label_col = 'hpv_status',
                            ignore=[])
    
elif args.task == 'HPV2':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/HNSCC.csv',
                            data_dir= os.path.join(args.data_root_dir, 'HPV_UNI2_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Deceased':0, 'Survived':1},
                            patient_strat= False,
                            label_col = 'survival',
                            ignore=[])
    
elif args.task == 'HPV512':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/HNSCC.csv',
                            data_dir= os.path.join(args.data_root_dir, 'HPV512'),
                            shuffle = False, 
                            # seed = args.seed, 
                            print_info = True,
                            label_dict = {'HPV-':0, 'HPV+':1},
                            patient_strat= False,
                            label_col = 'hpv_status',
                            ignore=[])

# elif args.task == 'HPV_cesc':
#     args.n_classes=2
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CESC2.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'HPV_CESC'),
#                             shuffle = False,  
#                             print_info = True,
#                             label_dict = {'HPV-':0, 'HPV+':1},
#                             patient_strat= False,
#                             label_col = 'hpv_status',
#                             ignore=[])

elif args.task == 'HPV_cesc':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CESC_high_conf_HPVs.csv',
                            data_dir= os.path.join(args.data_root_dir, 'HPV_CESC'),
                            shuffle = False,  
                            print_info = True,
                            label_dict = {'HPV-':0, 'HPV+':1},
                            patient_strat= False,
                            label_col = 'hpv_status',
                            ignore=[])
    
elif args.task == 'HPV_cesc_sur':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CESC2.csv',
                            data_dir= os.path.join(args.data_root_dir, 'HPV_CESC'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'Deceased':0, 'Survived':1},
                            patient_strat= False,
                            label_col = 'survival',
                            ignore=[])
    
elif args.task == 'HPV_cesc_lusc':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CESC_LUSC_combined.csv',
                            data_dir= os.path.join(args.data_root_dir, 'HPV_CESC_LUSC'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'HPV-':0, 'HPV+':1},
                            patient_strat= False,
                            label_col = 'hpv_status',
                            ignore=[])
    
elif args.task == 'HPV_cesc_lusc_alone':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CESC_LUSC_combined.csv',
                            data_dir= os.path.join(args.data_root_dir, 'HPV_CESC_LUSC'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'HPV-':0, 'HPV+':1},
                            patient_strat= False,
                            label_col = 'hpv_status',
                            ignore=[])

elif args.task == 'HPV_cesc_subtypes':
    args.n_classes=4
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CESC_with_subtypes_binned.csv',
                            data_dir= os.path.join(args.data_root_dir, 'HPV_CESC'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'HPV16':0, 'HPV18_45':1, 'OtherHR':2, 'Negative':3},
                            patient_strat= False,
                            label_col = 'hpv_subtype_bin',
                            ignore=[])
    
elif args.task == 'CPTAC_HNSCC_CESC':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CPTAC_HNSCC_CESC.csv',
                            data_dir= os.path.join(args.data_root_dir, 'CPTAC_HNSCC_CESC'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'HPV-':0, 'HPV+':1},
                            patient_strat= False,
                            label_col = 'hpv_status',
                            ignore=[])

elif args.task == 'CPTAC_HNSCC':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/CPTAC_HNSCC_filtered.csv',
                            data_dir= os.path.join(args.data_root_dir, 'CPTAC_HNSCC'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'HPV-':0, 'HPV+':1},
                            patient_strat= False,
                            label_col = 'hpv_status',
                            ignore=[])
# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_auc = []
    all_acc = []
    all_patient_results = []
    all_sensitivity = []
    all_specificity = []
    all_precision = []
    all_f1 = []
    for ckpt_idx in range(len(ckpt_paths)):
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        model, patient_results, test_error, auc, df, extra  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        
        all_auc.append(auc)
        all_acc.append(1-test_error)
        all_patient_results.append(patient_results)
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
        all_sensitivity.append(extra['sensitivity'])
        all_specificity.append(extra['specificity'])
        all_precision.append(extra['precision'])
        all_f1.append(extra['f1'])
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc, 
                            'sensitivity': all_sensitivity, 'specificity': all_specificity,
                            'precision': all_precision, 'f1': all_f1,
                            })
    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))
    # print(all_patient_results)
