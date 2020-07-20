ACCOUNT_ID = '75ce4cee-6829-4274-80e1-77e89559ddfb'

JOB_CONFIG = {
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
                                }