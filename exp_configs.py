
from haven import haven_utils as hu
from itertools import product

    
EXP_GROUPS = {}

# This EXP Groups 94.5% acc on cifar10
EXP_GROUPS['cifar'] = hu.cartesian_exp_group({
        "dataset": [{'name': 'cifar10', 'transform_lvl':1.5, 'colorjitter': False, 'val_transform':'identity'}],
        "dataset_size": [ 
                        {'train':None, 'test':None}
        ],
        "valratio": [0.2],
        'model':
        [{'name':'blvl', 
                'netC':{"name": "resnet18_meta_2", 
                        "opt":{'name':'sgd', 'momentum':0.9,
                                'sched':True, 
                                'lr':0.1,
                                 "weight_decay": 5e-4}},

                'netA':netA
                } for netA in [{"name": 'small_affine', 
                                    "opt":{'name':'sgd', 
                                           'lr':0.2,
                                           'sched':False,
                                           'momentum':0.9,
                                           "weight_decay": 0.01}, 
                                    "transform" : "affine", 
                                    "factor": 1}, 
                                   {"name": 'affine_color', 
                                    "opt":{'name':'sgd', 
                                           'lr':0.2,
                                           'sched':False,
                                           'momentum':0.9,
                                           "weight_decay": 0.01}, 
                                    "transform" : "affine", 
                                    "factor": 1}, 
                                    None]
         ],
        "n_inner_iter": [1],
        "batch": {"size": 128, "factor": 1},
        "niter": [201],
        "fixedSeed": [6442],
        "predParams": [None],
        "mixTrainVal": [True],
        "testTimeDA": [0],
        }) 


EXP_GROUPS['bach'] = hu.cartesian_exp_group({
        "dataset": 
                {'name': 'bach',
                'transform_lvl': 0,
                'colorjitter': False,
                'val_transform':'identity',
                'fold': 4,
                'patch_size':'512'
                },
        "dataset_size": [ 
                        {'train': None, 'test': None}],
        "valratio": [0.2],
        'model': [{'name':'blvl', 
                'netC':{"name": "resnet18_meta", 
                        "opt":{'name':'sgd', 'momentum':0.9,
                                'sched':True, 
                                'lr':0.1,
                                 "weight_decay": 5e-4}},

                'netA':netA
                } for netA in [None, 
                                {"name": 'small_affine', 
                                "opt":{'name':'sgd', 
                                        'lr':0.2,
                                        'sched':False,
                                        'momentum':0.9,
                                        "weight_decay": 0.01}, 
                                "transform" : "affine", 
                                "factor": 1},
                                {"name": 'affine_color', 
                                "opt":{'name':'sgd', 
                                        'lr':0.2,
                                        'sched':False,
                                        'momentum':0.9,
                                        "weight_decay": 0.01}, 
                                "transform" : "affine", 
                                "factor": 1},
                                ]
                ],
        "n_inner_iter": [1],
        "batch": {"size": 16, "factor": 1},
        "niter": [40],
        "fixedSeed": [6442],
        "predParams": [None],
        "mixTrainVal": [True],
        "testTimeDA": [0],
        }) 

EXP_GROUPS['imagenet'] =  hu.cartesian_exp_group({
        "dataset": [
        {'name': 'imagenet', 'transform_lvl':2, 'colorjitter': False, 'val_transform':'identity'},        
        ],        
        "dataset_size": [{'train':None, 'test':None}],
        "valratio": [0.2],
        'model': [
        {'name':'blvl', 
                'netC':{"name": "resnet50_meta",
                        "pretrained": False,
                        "RNDepth": 28,
                        "RNWidth": 10, "RNDO": 0.3,
                        "opt":{'name':'sgd', 'momentum':0.9,
                                'sched':True, 
                                'lr':0.1,
                                 "weight_decay": 1e-4}},

                'netA':{"name": 'small_affine', 
                                    "opt":{'name':'sgd', 
                                           'lr':0.1,
                                           'sched':False,
                                           'momentum':0.9,
                                           "weight_decay": 0.1}, 
                                    "transform" : "affine", 
                                    "factor": 1}},
       {'name':'blvl', 
                'netC':{"name": "resnet50_meta",
                        "pretrained": False,
                        "RNDepth": 28,
                        "RNWidth": 10, "RNDO": 0.3,
                        "opt":{'name':'sgd', 'momentum':0.9,
                                'sched':True, 
                                'lr':0.1,
                                 "weight_decay": 5e-4}},

                'netA':{"name": 'affine_color', 
                                    "opt":{'name':'sgd', 
                                           'lr':0.1,
                                           'sched':False,
                                           'momentum':0.9,
                                           "weight_decay": 0.1}, 
                                    "transform" : "affine", 
                                    "factor": 1}},
                                   
        ],
        "n_inner_iter": [1],
        "batch": {"size": 800, "factor": 1},
        "niter": [90],
        "fixedSeed": [6442],
        "mixTrainVal": [True],
        "testTimeDA": [0]
        }) 