import matplotlib.pyplot as plt
import numpy as np
import math

def error_plot(tensor, tensor_hat, title, points=9, metric='RMSE'):
    def get_error(tensor, tensor_hat, mask):
        if metric == 'RMSE':
            return np.linalg.norm(tensor_hat[mask] - tensor[mask]) / math.sqrt(np.sum(mask))
        elif metric == 'RSE':
            return np.linalg.norm(tensor_hat[mask] - tensor[mask]) / np.linalg.norm(tensor[mask])
    error_99= []
    error_95= []
    error_75= []
    error_0= []
    thr_99 = np.percentile(tensor, 99)
    thr_95 = np.percentile(tensor, 95)
    thr_75 = np.percentile(tensor, 75)
    len = tensor.shape[2]//points
    for t in range(0, tensor.shape[2], len):
        mask= tensor[:,:,t:t+len] > thr_99
        error_99.append(get_error(tensor[:,:,t:t+len], tensor_hat[:,:,t:t+len], mask))
        
        mask= tensor[:,:,t:t+len] > thr_95
        error_95.append(get_error(tensor[:,:,t:t+len], tensor_hat[:,:,t:t+len], mask))

        mask= tensor[:,:,t:t+len] > thr_75
        error_75.append(get_error(tensor[:,:,t:t+len], tensor_hat[:,:,t:t+len], mask))

        mask= tensor[:,:,t:t+len] > np.min(tensor)
        error_0.append(get_error(tensor[:,:,t:t+len], tensor_hat[:,:,t:t+len], mask))
    slice = [x for x in range(len,  tensor.shape[2]+1, len)]
    plt.figure()
    plt.plot(slice, error_0,  label='Q 0%',  marker='x', color='red')
    plt.plot(slice, error_75, label='Q 75%', marker='x', color='blue')
    plt.plot(slice, error_95, label='Q 95%', marker='x', color='green')
    plt.plot(slice, error_99, label='Q 99%', marker='x', color='black')
    plt.xlabel('slices')
    plt.ylabel( metric+ ' (TECU)')
    plt.title(title)
    plt.legend(frameon=True, loc='center left', bbox_to_anchor=(1, 0.5))  # put legend outside
    plt.savefig(title + '.png', bbox_inches='tight')

def compare_maps(tensor0, tensor1, tensor2, tensor3,
                t , title, n_rows =4, n_cols=4,vmin=0, vmax =  55,
                row_tags = ['VISTA Map', 'Imputed', 'Normalized' ,'Mean-Filtered'],  
                col_titles = ['2017/09/2 00:00 UT', '2017/09/2 21:30 UT','2017/09/3 15:30 UT', '2017/09/4 11:00 UT'],  
                ):
    fig, axes = plt.subplots(n_rows, n_cols,figsize=(10,5),sharex=True, sharey=True,constrained_layout=True)
    for i in range(n_rows):
        for j in range(n_cols):
            if i == 0:
                im = axes[i, j].imshow(tensor0[:,:,t[j]],
                               origin='lower', aspect='auto',vmin=vmin, vmax=vmax, cmap='jet')
                axes[i, j].set_title(col_titles[j], fontsize=9, pad=4)
            elif i == 1:
                im = axes[i, j].imshow(tensor1[:,:,t[j]],
                               origin='lower', aspect='auto',vmin=vmin, vmax=vmax, cmap='jet')
            elif i == 2:
                im = axes[i, j].imshow(tensor2[:,:,t[j]],
                               origin='lower', aspect='auto',vmin=vmin, vmax=vmax, cmap='jet')        
            elif i == 3:
                im = axes[i, j].imshow(tensor3[:,:,t[j]],
                               origin='lower', aspect='auto',vmin=vmin, vmax=vmax, cmap='jet')
                axes[i, j].set_xticks(np.arange(0, 240+1, 60))
                axes[i, j].set_xticklabels(np.arange(6, 22+1, 4), fontsize=8)
                axes[i, j].set_xlabel('LT', fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(f'{row_tags[i]}\nLatitude',fontsize=9, labelpad=4)
                axes[i, j].set_yticks(np.arange(0, 90+1, 30))
                axes[i, j].set_yticklabels(np.arange(-30, 60+1, 30), fontsize=8)
            axes[i, j].text(0.02, 0.92, f'({chr(97 + i*n_cols + j)})',
                        transform=axes[i, j].transAxes,
                        color='white', fontsize=8, fontweight='bold')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                    shrink=0.8, label='TEC (TECU)')
    plt.savefig('compare_maps_'+title+'.png', bbox_inches='tight')
        
