import argparse, sys, os, glob, time, json
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import matplotlib.pyplot as plt
from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_readers import Poly5Reader

def read_poly(fname):
    if fname[-6:] != '.poly5':
        data = Poly5Reader(fname+'.poly5')
    else:
        data = Poly5Reader(fname)
    samples = data.samples
    return samples

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/ nyq, highcut/ nyq], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def filt_GRID_rect(data, lowcut=20, highcut=500, fs=2000, order=2, notch_fs = 50, notch_q = 30,low_rect= 0.1, high_rect = 5):
    filt_out = np.zeros_like(data)
    for i in range(data.shape[0]):
        filt_out[i,:] = notch_filter(butter_bandpass_filter(data[i,:], lowcut, highcut, fs, order=3), notch_fs, fs, notch_q)
    emg_rectified = abs(filt_out) - np.min(abs(filt_out),axis =0).reshape(1,-1)
    low_pass = high_rect/(fs/2)
    b2, a2 = butter(4, low_pass, btype='low')
    emg_envelope = np.zeros_like(data)
    for i in range(data.shape[0]):
        emg_envelope[i,:] = filtfilt(b2,a2,emg_rectified[i,:])
    return emg_envelope

def filt_GRID(data, lowcut=20, highcut=500, fs=2000, order=3, notch_fs = 50, notch_q = 30):
    filt_out = np.zeros_like(data)
    for i in range(data.shape[0]):
        filt_out[i,:] = notch_filter(butter_bandpass_filter(data[i,:], lowcut, highcut, fs, order=order), notch_fs, fs, notch_q)
    return filt_out

def notch(notch_freq, samp_freq, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, samp_freq)
    return b, a

def notch_filter(data, notch_fs, fs, q=30):
    b, a = notch(notch_fs, fs, q)
    y = filtfilt(b, a, data)
    return y

def segment_trigs(trigs):
    stim_idx = {}
    trigs = np.diff(trigs)
    events = np.where(trigs<0)[0]
    stim_idx["start"] = events[0]
    stim_idx['stims'] = events[1:]
    return stim_idx

def plot_grid(args, f_grid, event_dict, out_path, flag):
    f_grid = filt_GRID(f_grid, fs = args.fs).T
    s_ms_factor = args.fs/1000
    plot_data_dict = np.empty((len(event_dict['stims']), int((args.vis_win_L+args.vis_win_U)*s_ms_factor),f_grid.shape[1]))
    trial_SD = np.empty((len(event_dict['stims']), f_grid.shape[1]))
    for i, event_idx in enumerate(event_dict['stims']):
        event_data = f_grid[int(event_idx-args.vis_win_L*s_ms_factor):int(event_idx+args.vis_win_U*s_ms_factor),:]
        event_data[int((args.vis_win_L-args.blank_win_L)*s_ms_factor):int((args.vis_win_L+args.blank_win_U)*s_ms_factor),:] = np.zeros((int((args.blank_win_L+args.blank_win_U)*s_ms_factor),event_data.shape[1]))
        plot_data_dict[i,:,:] = event_data
        trial_SD[i,:] = np.std(f_grid[int(event_idx-1050):int(event_idx-50),:],axis = 0)
    
    rows = 8; cols =8
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(25, 25))
    fig.suptitle(flag)
    max_val = np.max(np.abs(plot_data_dict))
    x_axis = np.linspace(-args.vis_win_L,args.vis_win_U,plot_data_dict.shape[1])
    ctr = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(plot_data_dict.shape[0]):
                axes[i][j].plot(x_axis, plot_data_dict[k,:,ctr], alpha = 0.5, linewidth = 0.5)
            mean_sig = np.mean(plot_data_dict[:,:,ctr],axis = 0)
            std_sig = np.std(plot_data_dict[:,:,ctr],axis = 0)
            std_base = np.mean(trial_SD[:,ctr],axis = 0)
            yerr = mean_sig + std_sig
            axes[i][j].axvline(x=0, ymin=0.0, ymax=1.0, color='k')
            axes[i][j].axvline(x=20, ymin=0.0, ymax=1.0, color='c',alpha = 0.25)

            axes[i][j].axhline(y=-std_base*5.5, xmin=0.0, xmax=1.0, color='k',alpha = 0.25)
            axes[i][j].axhline(y=std_base*5.5, xmin=0.0, xmax=1.0, color='k',alpha = 0.25)

            ctr+=1
            axes[i][j].plot(x_axis, mean_sig,c='k',alpha = 0.75)
            axes[i][j].fill_between(x_axis, -yerr,yerr,color='k',alpha = 0.2)
            # axes[i][j].fill_between(x_axis, mean_sig, yerr,color='k',alpha = 0.05)
            axes[i][j].set_xlim([-args.vis_win_L,args.vis_win_U])
            # axes[i][j].get_xaxis().set_visible(False) # Hide tick marks and spines
            axes[i][j].spines["right"].set_visible(False)
            axes[i][j].spines["top"].set_visible(False)
            max_val =  max(std_base*6, max_val)
            axes[i][j].set_ylim([-max_val,max_val])
            axes[i][j].set_xticks(np.linspace(-args.vis_win_L,args.vis_win_U,7,dtype = int))
            axes[i][j].set_xticklabels(axes[i][j].get_xticks(), rotation = 45)
    # axes[-1][0].get_xaxis().set_visible(True) 
    plt.savefig(os.path.join(out_path,flag)+'.png', bbox_inches="tight",dpi = 300)
    # plt.show()
    return

def plot_aux(args, f_grid, event_dict, out_path, incl_chan, dev_map):
    f_grid = filt_GRID(f_grid, fs = args.fs).T
    s_ms_factor = args.fs/1000
    plot_data_dict = np.empty((len(event_dict['stims']), int((args.vis_win_L+args.vis_win_U)*s_ms_factor),f_grid.shape[1]))
    trial_SD = np.empty((len(event_dict['stims']), f_grid.shape[1]))
    for i, event_idx in enumerate(event_dict['stims']):
        event_data = f_grid[int(event_idx-args.vis_win_L*s_ms_factor):int(event_idx+args.vis_win_U*s_ms_factor),:]
        event_data[int((args.vis_win_L-args.blank_win_L)*s_ms_factor):int((args.vis_win_L+args.blank_win_U)*s_ms_factor),:] = np.zeros((int((args.blank_win_L+args.blank_win_U)*s_ms_factor),event_data.shape[1]))
        plot_data_dict[i,:,:] = event_data
        trial_SD[i,:] = np.std(f_grid[int(event_idx-1050):int(event_idx-50),:],axis = 0)

    if len(incl_chan)%2:
        rows = int(np.sqrt(len(incl_chan)))+1; cols = int(np.sqrt(len(incl_chan)))
    else:
        rows = int(np.sqrt(len(incl_chan))); cols = int(np.sqrt(len(incl_chan)))
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))
    fig.suptitle(dev_map["GRID"])
    x_axis = np.linspace(-args.vis_win_L,args.vis_win_U,plot_data_dict.shape[1])
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i <rows+cols:
            for k in range(plot_data_dict.shape[0]):
                ax.plot(x_axis, plot_data_dict[k,:,i], alpha = 0.5, linewidth = 0.5)
            mean_sig = np.mean(plot_data_dict[:,:,i],axis = 0)
            std_sig = np.std(plot_data_dict[:,:,i],axis = 0)
            std_base = np.mean(trial_SD[:,i],axis = 0)
            yerr = mean_sig + std_sig
            ax.set_title(dev_map[incl_chan[i]])
            ax.axvline(x=0, ymin=0.0, ymax=1.0, color='k')
            ax.axvline(x=20, ymin=0.0, ymax=1.0, color='c',alpha = 0.25)

            ax.axhline(y=-std_base*5.5, xmin=0.0, xmax=1.0, color='k',alpha = 0.25)
            ax.axhline(y=std_base*5.5, xmin=0.0, xmax=1.0, color='k',alpha = 0.25)

            ax.plot(x_axis, mean_sig,c='k',alpha = 0.75)
            ax.fill_between(x_axis, -yerr,yerr,color='k',alpha = 0.2)
            ax.set_xlim([-args.vis_win_L,args.vis_win_U])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            max_val =  max(std_base*6, np.max(plot_data_dict[:,:,i]))
            ax.set_ylim([-max_val,max_val])
            ax.set_xticks(np.linspace(-args.vis_win_L,args.vis_win_U,7,dtype = int))
            ax.set_xticklabels(ax.get_xticks(), rotation = 45)
    plt.savefig(os.path.join(out_path,dev_map["GRID"])+'_AUX.png', bbox_inches="tight",dpi = 300)

def gen_MEP_vis(args):

    with open(os.path.join(args.data_dir,args.particiapnt_ID, args.exp_date,'musclemap.json'), 'r') as j:
        muscle_map = json.loads(j.read())

    for key in muscle_map.keys():
        if args.MEP:
            file_ID, time_ID = args.fname.split('-')
            file_path = os.path.join(args.data_dir,args.particiapnt_ID, args.exp_date, 'MEPs', key+'-'+time_ID)
            out_path = os.path.join(args.data_dir,args.particiapnt_ID, args.exp_date, 'MEPs','plots',time_ID)
        else:
            file_ID, time_ID = args.fname.split('-')
            file_path = os.path.join(args.data_dir,args.particiapnt_ID, args.exp_date, file_ID[:-3]+key+'-'+time_ID)
            out_path = os.path.join(args.data_dir,args.particiapnt_ID, args.exp_date,'plots',time_ID)
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        incl_chan = []
        for chan in muscle_map[key].keys():
            if muscle_map[key][chan] != 'N/A':
                incl_chan.append(chan)

        data = read_poly(file_path)
        f_trigs = data[-3,:].T
        event_dict = segment_trigs(f_trigs)

        if len(data)>67 and args.AUX:
            f_aux = data[64:-3,:]
            plot_aux(args, f_aux, event_dict, out_path, incl_chan[1:],muscle_map[key])

        if args.GRID:
            f_grid = data[:64,:]
            plot_grid(args, f_grid, event_dict, out_path, muscle_map[key]["GRID"])
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description='Poly5_2_pkl')

    parser.add_argument('--data_dir',default='data', type=str,
                        help= "Data directory")

    parser.add_argument('--fs',default=2000, type=int,
                    help= "sampling freq of data")
    
    parser.add_argument('--vis_win_L',default=50, type=int,
                    help= "time in ms before stim")
    
    parser.add_argument('--vis_win_U',default=100, type=int,
                        help= "time in ms after stim")
    
    parser.add_argument('--blank_win_L',default=2, type=int,
                        help= "time in ms before stim to blank")
    
    parser.add_argument('--blank_win_U',default=10, type=int,
                        help= "time in ms after stim to blank")
    
    parser.add_argument('--particiapnt_ID',default='PX', type=str,
                        help= "Data directory")
    
    parser.add_argument('--GRID',default=True, type=bool,
                        help= "Plot grid")
    
    parser.add_argument('--AUX',default=True, type=bool,
                        help= "Plot other chans")

    today = time.strftime("%Y%m%d")
    parser.add_argument('--exp_date',default=today, type=str,
                        help= "Data directory")
    
    parser.add_argument('--MEP',default=True, type=bool,
                        help= "Is the file an MEP scan")
    
    parser.add_argument('--fname',default="EXT-20240229_125918", type=str,
                        help= "File name of the trial")
    

    args = parser.parse_args(sys.argv[1:])
    gen_MEP_vis(args)

    