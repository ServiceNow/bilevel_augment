import torch
import torchvision
import tqdm
import pandas as pd
import pprint
import itertools
import os
import random
import pylab as plt
import exp_configs
import time
import numpy as np

import argparse
import torch.nn as nn
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj

from src.models import get_model
from src.datasets import get_dataset, get_train_val_dataloader

import pprint

from src.utils import get_slope

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainval(exp_dict, savedir_base, datadir_base, reset=False, num_workers=0, pin_memory=False, ngpu=1, cuda_deterministic=False):
    # bookkeeping
    # ==================

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)

    # create folder and save the experiment dictionary
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    if DEVICE.type == "cuda":
        if cuda_deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True

    # Dataset
    # ==================
    trainset = get_dataset(exp_dict['dataset'], 'train',
                           exp_dict=exp_dict, datadir_base=datadir_base,
                           n_samples=exp_dict['dataset_size']['train'],
                           transform_lvl=exp_dict['dataset']['transform_lvl'],
                           colorjitter=exp_dict['dataset'].get('colorjitter')
                           )

    valset = get_dataset(exp_dict['dataset'], 'validation',
                         exp_dict=exp_dict, datadir_base=datadir_base,
                         n_samples=exp_dict['dataset_size']['train'],
                         transform_lvl=0,
                         val_transform=exp_dict['dataset']['val_transform'])

    testset = get_dataset(exp_dict['dataset'], 'test',
                          exp_dict=exp_dict, datadir_base=datadir_base,
                          n_samples=exp_dict['dataset_size']['test'],
                          transform_lvl=0,
                          val_transform=exp_dict['dataset']['val_transform'])
    print("Dataset defined.")

    # define dataloaders
    if exp_dict['dataset']['name'] == 'bach':
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory)
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=exp_dict['batch']['size'],
                                                 shuffle=False,
                                                 num_workers=num_workers,
                                                 pin_memory=pin_memory)

    print("Testloader  defined.")

    # Model
    # ==================
    model = get_model(exp_dict, trainset, device=DEVICE)

    print("Model loaded")

    model_path = os.path.join(savedir, 'model.pth')
    model_best_path = os.path.join(savedir, 'model_best.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')

    # checkpoint management
    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = len(score_list)
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # define and log random seed for reproducibility
    assert('fixedSeed' in exp_dict)
    seed = exp_dict['fixedSeed']

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Seed defined.")

    # Train & Val
    # ==================
    print("Starting experiment at epoch %d/%d" % (s_epoch, exp_dict['niter']))

    for epoch in range(s_epoch, exp_dict['niter']):
        s_time = time.time()
        # Sample new train val
        trainloader, valloader = get_train_val_dataloader(exp_dict,
                                                          trainset, valset,
                                                          mixtrainval=exp_dict['mixTrainVal'],
                                                          num_workers=num_workers,
                                                          pin_memory=pin_memory)
        # Train & validate
        train_dict = model.train_on_loader(trainloader, valloader, epoch=epoch,
                                           exp_dict=exp_dict)

        # Test phase
        train_dict_2 = model.test_on_loader(trainloader)
        val_dict = model.test_on_loader(valloader)
        test_dict = model.test_on_loader(testloader)

        # Vis phase
        # model.vis_on_loader(epoch, visloader, savedir_images=os.path.join(savedir, 'images'))
        model.vis_on_loader('train', trainset, savedir_images=os.path.join(
            savedir, 'images'), epoch=epoch)
        # model.vis_on_loader('validation', valset, savedir_images=os.path.join(savedir, 'images'), epoch=epoch)
        # model.vis_on_loader('test', testset, savedir_images=os.path.join(savedir, 'images'), epoch=epoch)

        score_dict = {}
        score_dict["epoch"] = epoch
        score_dict["test_acc"] = test_dict['acc']
        score_dict["test_acc5"] = test_dict['acc5']
        # score_dict["test_acc"] = val_dict['acc']
        score_dict["val_acc"] = val_dict['acc']
        score_dict["train_acc"] = train_dict_2['acc']
        score_dict["train_loss"] = train_dict['loss']
        score_dict["time_taken"] = time.time() - s_time
        score_dict["netC_lr"] = train_dict['netC_lr']

        if exp_dict['model']['netA'] is not None:
            if 'transformations_mean' in train_dict:
                for i in range(len(train_dict['transformations_mean'])):
                    score_dict[str(
                        i) + "_mean"] = train_dict['transformations_mean'][i].item()
            if 'transformations_std' in train_dict:
                for i in range(len(train_dict['transformations_std'])):
                    score_dict[str(
                        i) + "_std"] = train_dict['transformations_std'][i].item()

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail(), "\n")
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

        # Update best score
        if epoch == 0 or (score_dict["test_acc"] >= score_df["test_acc"][:-1].max()):
            hu.save_pkl(os.path.join(
                savedir, "score_list_best.pkl"), score_list)
            hu.torch_save(os.path.join(savedir, "model_best.pth"),
                          model.get_state_dict())

            print("Saved Best: %s" % savedir)

    print('experiment completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir_base', default=None)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)
    parser.add_argument('-v', '--view_results', default=None)
    parser.add_argument('-j', '--run_jobs', default=None)

    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--gpumem', type=int, default=12,
                        help='amount of GPU RAM to use')
    parser.add_argument('--ncpu', type=int, default=10,
                        help='number of CPUs to use')
    parser.add_argument('--mem', type=int, default=16,
                        help='amount of RAM to use')
    parser.add_argument('--num_workers', type=int,
                        help='number of data loading workers', default=0)
    parser.add_argument('--cuda_deterministic', action='store_true',
                        help='activate cuda deterministic mode')
    parser.add_argument('--pin_memory', action='store_true',
                        help='use GPU pinned memory')

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))

        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # Run experiments
    # ----------------------------
    if args.run_jobs:
        # launch jobs
        from haven import haven_jobs as hjb
        
        jm = hjb.JobManager(exp_list=exp_list, 
                    savedir_base=args.savedir_base, 
                    
                    account_id='75ce4cee-6829-4274-80e1-77e89559ddfb',
                    workdir=os.path.dirname(os.path.realpath(__file__)),

                    job_config={
                                'image': 'registry.console.elementai.com/eai.issam/ssh',
                                'data': ['c76999a2-05e7-4dcb-aef3-5da30b6c502c:/mnt/home',
                                         '20552761-b5f3-4027-9811-d0f2f50a3e60:/mnt/results',
                                         '9b4589c8-1b4d-4761-835b-474469b77153:/mnt/datasets'],
                                'preemptable':True,
                                'resources': {
                                    'cpu': 4,
                                    'mem': 8,
                                    'gpu': 1
                                },
                                'interactive': False,
                                },
                    )

        command = ("python trainval.py -ei <exp_id> -sb {savedir_base} -d {datadir_base} "
                       "--ngpu {ngpu} "
                       "{cuda_deterministic} "
                       "{pin_memory} "
                       "--num_workers {num_workers}".format(savedir_base=args.savedir_base,
                                                            ngpu=args.ngpu,
                                                            cuda_deterministic="--cuda_deterministic " if args.cuda_deterministic else "",
                                                            pin_memory="--pin_memory" if args.pin_memory else "",
                                                            num_workers=args.num_workers,
                                                            datadir_base=args.datadir_base))
        print(command)
        jm.launch_menu(command=command)
        

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                     savedir_base=args.savedir_base,
                     datadir_base=args.datadir_base,
                     reset=args.reset,
                     num_workers=args.num_workers,
                     pin_memory=args.pin_memory,
                     ngpu=args.ngpu,
                     cuda_deterministic=args.cuda_deterministic,
                     )
