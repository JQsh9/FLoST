from simulation_functions import *
from simulation_class import *
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    message="Casting complex values to real discards the imaginary part"
)
import pandas as pd

simulation_dict={'K':[], 'T':[], 'p':[],
                 'test_rmse':[], 'train_rmse':[], 'time':[],
                 'method':[], 'seed':[]}

def write_simulation_dict(simulation_dict, T,K,p,seed,
                          method,test_error, train_error, time):
    simulation_dict['K'].append(K)
    simulation_dict['T'].append(T)
    simulation_dict['p'].append(p)
    simulation_dict['method'].append(method)
    simulation_dict['seed'].append(seed)
    simulation_dict['test_rmse'].append(test_error)
    simulation_dict['train_rmse'].append(train_error)
    simulation_dict['time'].append(time)
    return simulation_dict

'''Change the parameters here'''
jobname = 'B2'
T=500
K=int(T/20)
missing=0.5


d1, d2 = 100,100
r=5
T0 = generate_tensor_simulated(d1, d2, T, k = 10, s=0.2,r=r)
for my_seed in tqdm(range(0,100)):
    A_T100 = simulate_data(T0, my_seed, d1=d1, d2=d2, d3=T, missing_rate=missing, k=K, r=r)
    A_T100.RCGD()
    simulation_dict = write_simulation_dict(simulation_dict, T, K, missing, my_seed,'RCGD',
                                            A_T100.te_rmse_rcgd, A_T100.tr_rmse_rcgd, A_T100.time_rcgd,
                                            )
    A_T100.FLoST()
    simulation_dict = write_simulation_dict(simulation_dict, T, K, missing, my_seed,'FLoST',
                                            A_T100.te_rmse_flost, A_T100.tr_rmse_flost, A_T100.time_flost,
                                            )
    A_T100.FLT()
    simulation_dict = write_simulation_dict(simulation_dict, T, K, missing, my_seed,'FLT',
                                            A_T100.te_rmse_flt, A_T100.tr_rmse_flt, A_T100.time_flt,
                                            )
import json
with open(jobname+'.json', "w") as fp:
    json.dump(simulation_dict, fp) 
