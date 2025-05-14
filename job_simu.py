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

T=500
K=100
p=0.5
d1, d2 = 100,100
T0 = generate_tensor_simulated(d1, d2, T, k = 10, s=0.2,r=5)
for my_seed in tqdm(range(0,10)):
    simu_class = simulate_data(T0, my_seed, d1=d1, d2=d2, d3=T, missing_rate=p, k=K, s=0.1, r=10)
    simu_class.RCGD()
    simulation_dict = write_simulation_dict(simulation_dict, T, K, p, my_seed,'RCGD',
                                            simu_class.te_rmse_rcgd, simu_class.tr_rmse_rcgd, simu_class.time_rcgd,
                                            )
    simu_class.FLoST()
    simulation_dict = write_simulation_dict(simulation_dict, T, K, p, my_seed,'FLoST',
                                            simu_class.te_rmse_flost, simu_class.tr_rmse_flost, simu_class.time_flost,
                                            )
    simu_class.FLT()
    simulation_dict = write_simulation_dict(simulation_dict, T, K, p, my_seed,'FLT',
                                            simu_class.te_rmse_flost, simu_class.tr_rmse_flost, simu_class.time_flost,
                                            )
jobname = 'A_T500_K100_p5'
import json
with open(jobname+'.json', "w") as fp:
    json.dump(simulation_dict, fp) 
