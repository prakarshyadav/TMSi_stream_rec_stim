import os
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt_zi,sosfilt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
from scipy.io import savemat
from tmsi_dual_interface.tmsi_libraries.TMSiFileFormats.file_writer import FileWriter, FileFormat
from tmsi_dual_interface.tmsi_libraries.TMSiSDK.device import ChannelType
import math
import pylsl
import nidaqmx
import nidaqmx.system
from nidaqmx.constants import LineGrouping

"""
The code seems to be unstable if alternating between trail and rec mode
Also sometimes stream does not close properly (issue with sample_data_server.py) which causes code to crash after a while
"""

plot_duration = 15  # how many seconds of data to show
update_interval = 30  # ms between screen updates
pull_interval = 100  # ms between each pull operation

class Inlet:
    def __init__(self, info: pylsl.StreamInfo):
        self.inlet = pylsl.StreamInlet(info, max_buflen=plot_duration,
                                       processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
        self.name = info.name()
        self.channel_count = info.channel_count()

    def pull_and_plot(self,):
        pass

class DataInlet(Inlet):
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):#, plt: pg.PlotItem):
        super().__init__(info)
        bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
        self.buffer = np.empty(bufsize, dtype=self.dtypes[info.channel_format()])

    def pull_and_plot(self,):
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        return self.buffer
class DataInlet_reset(Inlet):
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo):#, plt: pg.PlotItem):
        super().__init__(info)
        self.bufsize = (2 * math.ceil(info.nominal_srate() * plot_duration), info.channel_count())
        self.info = info.channel_format()
        self.buffer = np.empty(self.bufsize, dtype=self.dtypes[self.info])

    def pull_and_plot(self,):
        _, ts = self.inlet.pull_chunk(timeout=0.0,
                                      max_samples=self.buffer.shape[0],
                                      dest_obj=self.buffer)
        out = self.buffer
        self.buffer = np.empty(self.bufsize, dtype=self.dtypes[self.info])
        return out
    
class check_MEPs_win(tk.Toplevel):
    def __init__(self, parent, task_trial, task_stim, target_profile_x,target_profile_y,stim_profile_x,stim_profile_y, trial_params,dev_select='FLX', vis_chan_mode='avg', vis_chan = 10,vis_chan_mode_check='single', vis_chan_check = 35,record = False,):
        super().__init__(parent)

        self.vis_buffer_len = 10
        self.vis_xlim_pad = 3
        self.EMG_avg_win = 100 #in samples
        self.vis_chan_mode  = vis_chan_mode
        self.vis_chan = int(vis_chan)
        self.vis_chan_mode_check  = vis_chan_mode_check
        self.vis_chan_check = int(vis_chan_check)
        self.task_trial = task_trial
        self.task_stim = task_stim
        self.force_holder = deque(list(np.empty(self.vis_buffer_len)))
        self.trig_holder = deque(list(np.empty(self.vis_buffer_len,dtype= bool)))
        self.stim_profile_x = stim_profile_x
        self.x_axis = np.linspace(0,1,self.vis_buffer_len)
        self.kill = False

        self.attributes('-fullscreen', True)
        self.title('Force Visualization')
        self.trial_params = trial_params
        self.rec_flag = record
        self.parent = parent
        if self.rec_flag:
            self.parent.dump_trig = []
            self.parent.dump_force = []
            self.parent.dump_time = []

        if self.vis_chan_mode == 'single':
            self.vis_chan_slice = np.array([int(self.vis_chan)])
        elif self.vis_chan_mode == 'aux':
            self.vis_chan_slice = np.array([int(self.vis_chan) + self.parent.UNI_count-1])
        else:
            self.vis_chan_slice = np.arange(int(self.vis_chan))

        if self.vis_chan_mode_check == 'single':
            self.vis_chan_slice_check = np.array([int(self.vis_chan_check)])
        elif self.vis_chan_mode_check == 'aux':
            self.vis_chan_slice_check = np.array([int(self.vis_chan_check) + self.parent.UNI_count-1])
        else:
            self.vis_chan_slice_check = np.arange(int(self.vis_chan_check))

        fig_data = Figure()
        self.disp_target = fig_data.add_subplot(111)
        
        fig_MEP = Figure()
        self.check_MEP_fig = fig_MEP.add_subplot(111)
        
        
        self.main_frame = tk.Frame(self, borderwidth=2, relief= 'solid')
        self.main_frame.pack(side="bottom", expand=True, fill="both")
        self.main_frame.grid_columnconfigure(0, weight=1,uniform=1)
        self.main_frame.grid_rowconfigure(0, weight=1,uniform=1)
        self.main_frame.grid_rowconfigure(1, weight=1,uniform=1)

        self.frame1=tk.Frame(self.main_frame,bg="red")
        self.frame1.grid(row=0, column=0, sticky='nsew')
        self.frame2=tk.Frame(self.main_frame,bg="black")
        self.frame2.grid(column=0,row=1,sticky='nsew')

        self.canvas_disp_target = FigureCanvasTkAgg(fig_data, master=self.frame1,)  
        self.canvas_disp_target.draw()
        self.canvas_disp_target.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.canvas_check_MEP_fig = FigureCanvasTkAgg(fig_MEP, master=self.frame2,)  
        self.canvas_check_MEP_fig.draw()
        self.canvas_check_MEP_fig.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("EMG (mV)/Torque (Nm)", fontsize=14)
        self.check_MEP_fig.set_xlabel("Time (ms)", fontsize=14)
        self.check_MEP_fig.set_ylabel("EMG (mV)/AUX (V)", fontsize=14)

        self.l_target = self.disp_target.plot(target_profile_x, target_profile_y, linewidth = 50, color = 'r')
        self.l_current = self.disp_target.plot(self.x_axis, self.force_holder, linewidth = 13, color = 'b',)
        
        self.x_axis_MEP = np.linspace(self.trial_params['MEP_winL'],self.trial_params['MEP_winU'],(self.trial_params['MEP_winU'] - self.trial_params['MEP_winL'])*2)#
        self.stim_line_y0 = self.check_MEP_fig.vlines(0,-1000,1000, linewidth = 3, color = 'k')
        self.stim_line_20 = self.check_MEP_fig.vlines(20,-1000,1000, linewidth = 1, color = 'c')
        self.stim_line_x0 = self.check_MEP_fig.hlines(0,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 0.5, color = 'c')
        self.stim_line_sd_U = self.check_MEP_fig.hlines(0.1,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 1, color = 'k', alpha =0.5)
        self.stim_line_sd_L = self.check_MEP_fig.hlines(-0.1,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 1, color = 'k',alpha = 0.5)
        self.MEP_amp = self.check_MEP_fig.text(0,0,'MEP: '+str(0))
        self.vis_MEP = self.check_MEP_fig.plot(self.x_axis_MEP, np.zeros_like(self.x_axis_MEP), linewidth = 2, color = 'r',)
        
        self.disp_target.set_xlim([0,self.trial_params['duration']])
        self.disp_target.set_ylim([0,self.trial_params['MVF']*0.5])

        self.check_MEP_fig.set_xlim([self.trial_params['MEP_winL'],self.trial_params['MEP_winU']])
        self.check_MEP_fig.set_ylim([-self.trial_params['MVF']*0.5,self.trial_params['MVF']*0.5])

        self.canvas_disp_target.draw()
        self.canvas_check_MEP_fig.draw()

        self.stream_vis_button = tk.Button(self, text='START TRIAL', bg ='yellow')
        self.stream_vis_button['command'] = self.start_vis
        self.stream_vis_button.pack()
        self.stream_vis_button.place(x=100, y=100)

        self.stream_vis_button = tk.Button(self, text='STOP TRIAL', bg ='red')
        self.stream_vis_button['command'] = self.stop_vis
        self.stream_vis_button.pack()
        self.stream_vis_button.place(x=100, y=150)


        print("finding stream")
        stream = pylsl.resolve_stream('name', dev_select)
        for info in stream:
            print('name: ', info.name())
            print('channel count:', info.channel_count())
            print('sampling rate:', info.nominal_srate())
            print('type: ', info.type())
        self.inlet = DataInlet(stream[0])    
        self.inlet_STA = DataInlet_reset(stream[0])    
    def stop_vis(self):
        self.kill = True
        self.inlet.inlet.close_stream()
        self.inlet_STA.inlet.close_stream()
        self.destroy()


    def start_vis(self):
        self.inlet.inlet.open_stream()
        self.inlet_STA.inlet.open_stream()
        if self.rec_flag:
            self.task_stim.write(False)
        self.task_trial.write(True)
        data_STA = self.inlet_STA.pull_and_plot()#
        array_data = self.inlet.pull_and_plot()#
        if self.vis_chan_mode == 'aux':
            sos_raw = butter(3, [0.2, 20], 'bandpass', fs=2000, output='sos')
            sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
            z_sos0 = sosfilt_zi(sos_raw)
            z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
            z_sos0 = sosfilt_zi(sos_env)
            z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
        else:
            sos_raw = butter(3, [20, 500], 'bandpass', fs=2000, output='sos')
            sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
            z_sos0 = sosfilt_zi(sos_raw)
            z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
            z_sos0 = sosfilt_zi(sos_env)
            z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
        
        STA_raw = sosfilt(sos_raw,data_STA[:,self.vis_chan_slice_check].T)
        
        samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
        samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
        _, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)

        # if self.vis_chan_mode == 'aux':
        #     array_data = self.inlet.pull_and_plot()
        #     array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])
        #     array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
        #     baseline =  abs(np.mean(array_data_scaled))


        t0 = time.time()
        stim_ctr = 0
        curr_pulse_time = 1e16
        MEP_update = False
        baseline_list = []
        data_STA = self.inlet_STA.pull_and_plot()#
        if stim_ctr<len(self.stim_profile_x)-1:
            curr_pulse_time = self.stim_profile_x[stim_ctr]
        while time.time()-t0 < self.trial_params['duration'] and not self.kill:
            time.sleep(0.0001)
            self.trig_holder.popleft()
            
            stim = False
            if time.time()-t0 > curr_pulse_time and stim_ctr<len(self.stim_profile_x):
                stim = True
                MEP_update = False
                self.task_stim.write(True)
                stim_ctr+=1
                if stim_ctr<len(self.stim_profile_x):
                    curr_pulse_time = self.stim_profile_x[stim_ctr]
                else:
                    curr_pulse_time += int(self.parent.stim_rate.get())
                self.trig_holder.append(1)
            self.trig_holder.append(0)
            if time.time()-t0 > (curr_pulse_time-3.5) and not MEP_update and stim_ctr > 0:
                MEP_update = True
                data_STA = self.inlet_STA.pull_and_plot()#
                trigs = data_STA[:,-3]
                plot_event_idx = np.where(np.abs(np.diff(trigs))>0)[0]#[-1]
                print("updating MEPs", plot_event_idx)
                if len(plot_event_idx)>0:
                    if self.vis_chan_mode_check == 'aux':
                        data_STA_filt = np.abs(data_STA[:,self.vis_chan_slice_check])
                    else:
                        data_STA_filt = sosfilt(sos_raw, data_STA[:,self.vis_chan_slice_check].T)
                    data_STA_scaled = np.nan_to_num(data_STA_filt,nan=0,posinf=0,neginf=0).reshape(-1)
                    plot_data = data_STA_scaled[plot_event_idx[-2]+self.trial_params['MEP_winL']*2:plot_event_idx[-2]+self.trial_params['MEP_winU']*2]
                    SD_bound = np.std(data_STA_scaled[plot_event_idx[-2]-1050:plot_event_idx[-2]-50])
                    l_cut = 4; u_cut = 10
                    plot_data[abs(self.trial_params['MEP_winL']*2)-l_cut:abs(self.trial_params['MEP_winL']*2)+u_cut] = np.zeros(l_cut+u_cut)
                    y_MEP = max(0.05,np.max(np.abs(plot_data)))
                    self.check_MEP_fig.set_ylim([-y_MEP*1.1,y_MEP*1.1])
                    self.vis_MEP[0].set_data(self.x_axis_MEP,plot_data)
                    self.stim_line_sd_U.remove()
                    self.stim_line_sd_U = self.check_MEP_fig.hlines(SD_bound*5.5,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 1, color = 'k', alpha =0.5)
                    #set_segments([np.array([[SD_bound, self.trial_params['MEP_winL']], [SD_bound,self.trial_params['MEP_winU']]])])
                    self.stim_line_sd_L.remove()
                    self.stim_line_sd_L = self.check_MEP_fig.hlines(-SD_bound*5.5,self.trial_params['MEP_winL'],self.trial_params['MEP_winU'], linewidth = 1, color = 'k', alpha =0.5)

                    #set_segments([np.array([[-SD_bound, self.trial_params['MEP_winL']], [-SD_bound,self.trial_params['MEP_winU']]])])

                    p2p_amp = np.max(plot_data) - np.min(plot_data)
                    self.MEP_amp.set_text("P2P: "+str(abs(np.float16(p2p_amp))))
                    # plot_data = np.abs(np.diff(trigs))
                    # self.check_MEP_fig.set_xlim([0,1])
                    max_val =  max(SD_bound*6, np.max(np.abs(plot_data)*1.1))
                    self.check_MEP_fig.set_ylim([-max_val,max_val])
                    # self.vis_MEP[0].set_data(np.linspace(0,1,plot_data.shape[0]),plot_data)
                    self.canvas_check_MEP_fig.draw()
                else:
                    print("Warning Trigs not detected")

                
            self.force_holder.popleft()
            array_data = self.inlet.pull_and_plot()
            if self.vis_chan_mode == 'aux':
                array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])
                # samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                # samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                # array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
            else:
                samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
            array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
            
            if self.vis_chan_mode == 'aux' and time.time()-t0 < 3:
                force = abs(np.mean(array_data_scaled)) 
                baseline_list.append(force)
                baseline = np.mean(baseline_list)
                force = force*float(self.parent.conv_factor.get())
                print("setting", baseline)
            elif self.vis_chan_mode == 'aux' and time.time()-t0 > 3:
                force = abs(np.mean(array_data_scaled)) - baseline
                print("using", baseline)
                force = force*float(self.parent.conv_factor.get())
            else:
                force = abs(np.mean(array_data_scaled))
            # force = np.median(array_data_scaled)
            # print(force)
            self.force_holder.append(force)
            t_prev = time.time()-t0
            if stim==True:
                print(time.time()-t0,curr_pulse_time,stim,force)
            if self.rec_flag:
                self.task_stim.write(False)
                self.parent.dump_time.append(t_prev)
                self.parent.dump_trig.append(self.trig_holder[-1])
                self.parent.dump_force.append(force)
            disp_force = sorted(self.force_holder)
            self.l_current[0].set_data(self.x_axis*(time.time()-t0-t_prev-0.1)+t_prev,np.mean(disp_force)*np.ones(self.vis_buffer_len))
            self.disp_target.set_xlim([time.time()-t0-self.vis_xlim_pad,time.time()-t0+self.vis_xlim_pad])
            self.canvas_disp_target.draw()
            self.update()

        self.inlet_STA.inlet.close_stream()
        self.inlet.inlet.close_stream()
        self.destroy()

class display_force_data(tk.Toplevel):
    def __init__(self, parent, task_trial, task_stim, target_profile_x,target_profile_y,stim_profile_x,stim_profile_y, trial_params,dev_select='FLX', vis_chan_mode='avg', vis_chan = 10,record = False):
        super().__init__(parent)

        self.vis_buffer_len = 10
        self.vis_xlim_pad = 3
        self.EMG_avg_win = 100 #in samples
        self.vis_chan_mode  = vis_chan_mode
        self.vis_chan = int(vis_chan)
        self.task_trial = task_trial
        self.task_stim = task_stim
        self.force_holder = deque(list(np.empty(self.vis_buffer_len)))
        self.trig_holder = deque(list(np.empty(self.vis_buffer_len,dtype= bool)))
        self.stim_profile_x = stim_profile_x
        self.x_axis = np.linspace(0,1,self.vis_buffer_len)
        self.kill = False

        self.attributes('-fullscreen', True)
        self.title('Force Visualization')
        self.trial_params = trial_params
        self.rec_flag = record
        self.parent = parent
        if self.rec_flag:
            self.parent.dump_trig = []
            self.parent.dump_force = []
            self.parent.dump_time = []

        if self.vis_chan_mode == 'single':
            self.vis_chan_slice = np.array([int(self.vis_chan)])
        elif self.vis_chan_mode == 'aux':
            self.vis_chan_slice = np.array([int(self.vis_chan) + self.parent.UNI_count-1])
        else:
            self.vis_chan_slice = np.arange(int(self.vis_chan))

        fig = Figure()
        self.disp_target = fig.add_subplot(111)
        
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        
        self.main_frame = tk.Frame(self, borderwidth=2, relief= 'solid')
        self.main_frame.pack(side="bottom", expand=True, fill="both")


        self.canvas_disp_target = FigureCanvasTkAgg(fig, master=self.main_frame,)  
        self.canvas_disp_target.draw()
        self.canvas_disp_target.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("EMG (mV)/Torque (Nm)", fontsize=14)
        self.l_target = self.disp_target.plot(target_profile_x, target_profile_y, linewidth = 50, color = 'r')
        self.l_current = self.disp_target.plot(self.x_axis, self.force_holder, linewidth = 13, color = 'b',)
        self.disp_target.set_xlim([0,self.trial_params['duration']])
        self.disp_target.set_ylim([0,self.trial_params['MVF']*0.5])

        self.canvas_disp_target.draw()

        self.stream_vis_button = tk.Button(self, text='START TRIAL', bg ='yellow')
        self.stream_vis_button['command'] = self.start_vis
        self.stream_vis_button.pack()
        self.stream_vis_button.place(x=100, y=100)
        self.stream_vis_button = tk.Button(self, text='STOP TRIAL', bg ='red')
        self.stream_vis_button['command'] = self.stop_vis
        self.stream_vis_button.pack()
        self.stream_vis_button.place(x=100, y=150)

        print("finding stream")
        stream = pylsl.resolve_stream('name', dev_select)
        for info in stream:
            print('name: ', info.name())
            print('channel count:', info.channel_count())
            print('sampling rate:', info.nominal_srate())
            print('type: ', info.type())
        self.inlet = DataInlet(stream[0])    
        
    def stop_vis(self):
        self.kill = True
        self.inlet.inlet.close_stream()
        self.destroy()

    def start_vis(self):
        self.inlet.inlet.open_stream()

        if self.rec_flag:
            self.task_stim.write(False)
        self.task_trial.write(True)

        array_data = self.inlet.pull_and_plot()#
        if self.vis_chan_mode == 'aux':
            sos_raw = butter(3, [0.2, 20], 'bandpass', fs=2000, output='sos')
            sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
            z_sos0 = sosfilt_zi(sos_raw)
            z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
            z_sos0 = sosfilt_zi(sos_env)
            z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
        else:
            sos_raw = butter(3, [20, 500], 'bandpass', fs=2000, output='sos')
            sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
            z_sos0 = sosfilt_zi(sos_raw)
            z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
            z_sos0 = sosfilt_zi(sos_env)
            z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
        
        
        samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
        samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
        _, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
        # time.sleep(0.1)
        # if self.vis_chan_mode == 'aux':
        #     array_data = self.inlet.pull_and_plot()
        #     array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])
        #     array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
        #     baseline =  abs(np.mean(array_data_scaled))
        baseline_list = []
        t0 = time.time()
        stim_ctr = 0
        curr_pulse_time = 1e16
        if stim_ctr<len(self.stim_profile_x)-1:
            curr_pulse_time = self.stim_profile_x[stim_ctr]
        while time.time()-t0 < self.trial_params['duration'] and not self.kill:
            time.sleep(0.0001)
            self.trig_holder.popleft()
            
            stim = False
            if time.time()-t0 > curr_pulse_time and stim_ctr<len(self.stim_profile_x):
                stim = True
                self.task_stim.write(True)
                stim_ctr+=1
                if stim_ctr<len(self.stim_profile_x):
                    curr_pulse_time = self.stim_profile_x[stim_ctr]
                else:
                    curr_pulse_time += int(self.parent.stim_rate.get())
                self.trig_holder.append(1)
            self.trig_holder.append(0)
            
            self.force_holder.popleft()
            array_data = self.inlet.pull_and_plot()
            if self.vis_chan_mode == 'aux':
                array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])
            else:
                samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
            array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
            
            if self.vis_chan_mode == 'aux' and time.time()-t0 < 3:
                force = abs(np.mean(array_data_scaled)) 
                baseline_list.append(force)
                baseline = np.mean(baseline_list)
                force = force*float(self.parent.conv_factor.get())
                print("setting", baseline)
            elif self.vis_chan_mode == 'aux' and time.time()-t0 > 3:
                force = abs(np.mean(array_data_scaled)) - baseline
                print("using", baseline)
                force = force*float(self.parent.conv_factor.get())
            else:
                force = abs(np.mean(array_data_scaled))
            # force = np.median(array_data_scaled)
            self.force_holder.append(force)
            t_prev = time.time()-t0
            if stim==True:
                print(time.time()-t0,curr_pulse_time,stim,force)
            if self.rec_flag:
                self.task_stim.write(False)
                self.parent.dump_time.append(t_prev)
                self.parent.dump_trig.append(self.trig_holder[-1])
                self.parent.dump_force.append(force)
            disp_force = sorted(self.force_holder)
            self.l_current[0].set_data(self.x_axis*(time.time()-t0-t_prev-0.1)+t_prev,np.mean(disp_force)*np.ones(self.vis_buffer_len))
            self.disp_target.set_xlim([time.time()-t0-self.vis_xlim_pad,time.time()-t0+self.vis_xlim_pad])
            self.canvas_disp_target.draw()
            self.update()
        self.inlet.inlet.close_stream()
        self.destroy()

class APP(tk.Toplevel):
    def __init__(self,parent,tmsi,dump_path):
        super().__init__(parent)
        self.title('Force Ramp Interface')
        self.geometry('1000x1000')
        """
        Buttons
        """
        self.tmsi_dev = tmsi
        self.dump_path = dump_path
        self.start_rec_button = tk.Button(self, text='START', bg ='green')
        self.start_rec_button['command'] = self.start_rec
        self.start_rec_button.pack()
        self.start_rec_button.place(x=10, y=10)

        self.stop_rec_button = tk.Button(self, text='STOP', bg ='red')
        self.stop_rec_button['command'] = self.stop_rec
        self.stop_rec_button.pack()
        self.stop_rec_button.place(x=60, y=10)

        options = list(self.tmsi_dev.keys())
        self.vis_TMSi = tk.StringVar() 
        self.vis_TMSi.set(options[0])
        self.lbl_vis = ttk.Label(self, text='Select TMSi to visualize:')
        self.lbl_vis.pack(fill='x', expand=True)
        self.lbl_vis.place(x=200, y=10)
        self.vis_tmsi_drop = tk.OptionMenu( self , self.vis_TMSi , *options) #tk.Button(self, text='START', bg ='green')
        self.vis_tmsi_drop.pack()
        self.vis_tmsi_drop.place(x=350, y=10)
        
        self.lbl_vis_mode = ttk.Label(self, text='Select vis mode:')
        self.lbl_vis_mode.pack(fill='x', expand=True)
        self.lbl_vis_mode.place(x=430, y=10)
        self.vis_chan_mode = tk.StringVar() 
        self.vis_mode_option1 = tk.Radiobutton(self, text="Single Chan", variable=self.vis_chan_mode, value="single", command=self.set_vis_mode)
        self.vis_mode_option1.pack(fill='x', expand=True)
        self.vis_mode_option1.place(x=530, y=10)
        self.vis_mode_option2 = tk.Radiobutton(self, text="Average", variable=self.vis_chan_mode, value="avg", command=self.set_vis_mode)
        self.vis_mode_option2.pack(fill='x', expand=True)
        self.vis_mode_option2.place(x=530, y=30)
        self.vis_mode_option3 = tk.Radiobutton(self, text="Aux", variable=self.vis_chan_mode, value="aux", command=self.set_vis_mode)
        self.vis_mode_option3.pack(fill='x', expand=True)
        self.vis_mode_option3.place(x=530, y=50)

        options = [0,36]
        self.vis_chan = tk.StringVar() 
        self.vis_chan.set(options[1])
        self.vis_chan_drop = tk.OptionMenu( self , self.vis_chan , *options) #tk.Button(self, text='START', bg ='green')
        self.vis_chan_drop.pack()
        self.vis_chan_drop.place(x=650, y=10)
        
        self.trial_ID = tk.StringVar()
        self.lbl_trial_ID = ttk.Label(self, text='Trial Num:')
        self.lbl_trial_ID.pack(fill='x', expand=True)
        self.lbl_trial_ID.place(x=10, y=40)
        self.t_trial_ID = tk.Entry(self, textvariable=self.trial_ID)
        self.t_trial_ID.insert(0, "1")
        self.t_trial_ID.pack(fill='x', expand=True)
        self.t_trial_ID.focus()
        self.t_trial_ID.place(x=150, y=40, width = 50)

        # self.check_dir_button = tk.Button(self, text='CHECK DIR', bg ='yellow')
        # self.check_dir_button['command'] = self.check_dir
        # self.check_dir_button.pack()
        # self.check_dir_button.place(x=250, y=40)

        # today = time.strftime("%Y%m%d")

        # self.dump_path = tk.StringVar()
        # self.lbl_dump_path = ttk.Label(self, text='Dump Path:')
        # self.lbl_dump_path.pack(fill='x', expand=True)
        # self.lbl_dump_path.place(x=10, y=70)
        # self.t_dump_path = tk.Entry(self, textvariable=self.dump_path)
        # self.t_dump_path.insert(0, "data/PX/"+today)
        # self.t_dump_path.pack(fill='x', expand=True)
        # self.t_dump_path.focus()
        # self.t_dump_path.place(x=150, y=70, width = 500)

        self.daq_name = tk.StringVar()
        self.lbl_daq_name = ttk.Label(self, text='DAQ ID:')
        self.lbl_daq_name.pack(fill='x', expand=True)
        self.lbl_daq_name.place(x=10, y=100)
        self.t_daq_name = tk.Entry(self, textvariable=self.daq_name)
        self.t_daq_name.insert(0, "Dev3")
        self.t_daq_name.pack(fill='x', expand=True)
        self.t_daq_name.focus()
        self.t_daq_name.place(x=150, y=100, width = 100)

        self.stim_chan = tk.StringVar()
        self.lbl_Ach_name = ttk.Label(self, text='Stim init Chans:')
        self.lbl_Ach_name.pack(fill='x', expand=True)
        self.lbl_Ach_name.place(x=10, y=130)
        self.t_Ach_name = tk.Entry(self, textvariable=self.stim_chan)
        self.t_Ach_name.insert(0, "port0/line0")
        self.t_Ach_name.pack(fill='x', expand=True)
        self.t_Ach_name.focus()
        self.t_Ach_name.place(x=150, y=130, width = 100)

        self.trial_chan = tk.StringVar()
        self.lbl_Dch_name = ttk.Label(self, text='Trial init Chans:')
        self.lbl_Dch_name.pack(fill='x', expand=True)
        self.lbl_Dch_name.place(x=10, y=160)
        self.t_Dch_name = tk.Entry(self, textvariable=self.trial_chan)
        self.t_Dch_name.insert(0, "port1/line0:1")
        self.t_Dch_name.pack(fill='x', expand=True)
        self.t_Dch_name.focus()
        self.t_Dch_name.place(x=150, y=160, width = 100)

        self.start_daq_button = tk.Button(self, text='START DAQ', bg ='yellow')
        self.start_daq_button['command'] = self.start_DAQ
        self.start_daq_button.pack()
        self.start_daq_button.place(x=10, y=190)

        # self.stream_daq_button = tk.Button(self, text='STREAM DAQ', bg ='yellow')
        # self.stream_daq_button['command'] = self.stream_DAQ
        # self.stream_daq_button.pack()
        # self.stream_daq_button.place(x=200, y=190)

        self.test_force_read_button = tk.Button(self, text='TRAINING', bg ='yellow')
        self.test_force_read_button['command'] = self.test_force_read
        self.test_force_read_button.pack()
        self.test_force_read_button.place(x=110, y=10)

        self.conv_factor = tk.StringVar()
        self.lbl_conv_factor = ttk.Label(self, text='Torque Const.:')
        self.lbl_conv_factor.pack(fill='x', expand=True)
        self.lbl_conv_factor.place(x=10, y=220)
        self.t_conv_factor = tk.Entry(self, textvariable=self.conv_factor)
        self.t_conv_factor.insert(0, "0.26959694")
        self.t_conv_factor.pack(fill='x', expand=True)
        self.t_conv_factor.focus()
        self.t_conv_factor.place(x=150, y=220, width = 100)

        self.MVC_duration = tk.StringVar()
        self.lbl_MVC_len = ttk.Label(self, text='Duration of MVC (s):')
        self.lbl_MVC_len.pack(fill='x', expand=True)
        self.lbl_MVC_len.place(x=10, y=250)
        self.t_MVC_len = tk.Entry(self, textvariable=self.MVC_duration)
        self.t_MVC_len.insert(0, "5")
        self.t_MVC_len.pack(fill='x', expand=True)
        self.t_MVC_len.focus()
        self.t_MVC_len.place(x=150, y=250, width = 100)

        self.start_MVC_button = tk.Button(self, text='START MVC', bg ='yellow')
        self.start_MVC_button['command'] = self.get_MVC
        self.start_MVC_button.pack()
        self.start_MVC_button.place(x=10, y=280)

        self.trl_duration = tk.StringVar()
        self.lbl_trl_duration = ttk.Label(self, text='Trial Duration (s):')
        self.lbl_trl_duration.pack(fill='x', expand=True)
        self.lbl_trl_duration.place(x=10, y=330)
        self.t_trl_duration = tk.Entry(self, textvariable=self.trl_duration)
        self.t_trl_duration.insert(0, "40")
        self.t_trl_duration.pack(fill='x', expand=True)
        self.t_trl_duration.focus()
        self.t_trl_duration.place(x=150, y=330, width = 100)

        self.init_wait = tk.StringVar()
        self.lbl_init_wait = ttk.Label(self, text='Ramp Delay (s):')
        self.lbl_init_wait.pack(fill='x', expand=True)
        self.lbl_init_wait.place(x=10, y=360)
        self.t_init_wait = tk.Entry(self, textvariable=self.init_wait)
        self.t_init_wait.insert(0, "5")
        self.t_init_wait.pack(fill='x', expand=True)
        self.t_init_wait.focus()
        self.t_init_wait.place(x=150, y=360, width = 100)

        self.peak_ramp_force = tk.StringVar()
        self.lbl_peak_ramp_force = ttk.Label(self, text='Max Ramp Force (x MVC):')
        self.lbl_peak_ramp_force.pack(fill='x', expand=True)
        self.lbl_peak_ramp_force.place(x=310, y=360)
        self.t_peak_ramp_force = tk.Entry(self, textvariable=self.peak_ramp_force)
        self.t_peak_ramp_force.insert(0, "0.3")
        self.t_peak_ramp_force.pack(fill='x', expand=True)
        self.t_peak_ramp_force.focus()
        self.t_peak_ramp_force.place(x=450, y=360, width = 100)

        self.lbl_max_force = ttk.Label(self, text="Max Force",font=('Helvetica 16 bold'))
        self.lbl_max_force.pack(fill='x', expand=True)
        self.lbl_max_force.place(x=400, y=150)
        self.max_force = tk.StringVar()
        self.max_force.set('10')
        
        self.lbl_max_force_num = ttk.Label(self, textvariable=self.max_force,font=('Helvetica 30 bold'))
        self.lbl_max_force_num.pack(fill='x', expand=True)
        self.lbl_max_force_num.place(x=400, y=200)

        self.t_max_force_num = tk.Entry(self, textvariable=self.max_force)
        self.t_max_force_num.pack(fill='x', expand=True)
        self.t_max_force_num.focus()
        self.t_max_force_num.place(x=400, y=250, width = 200)

        self.manualMVC_button = tk.Button(self, text='PUSH MVC', bg ='yellow')
        self.manualMVC_button['command'] = self.manualMVC
        self.manualMVC_button.pack()
        self.manualMVC_button.place(x=400, y=280)
        
        self.start_vanilla_button = tk.Button(self, text='PUSH VANILLA', bg ='yellow')
        self.start_vanilla_button['command'] = self.do_vanilla
        self.start_vanilla_button.pack()
        self.start_vanilla_button.place(x=10, y=400)
        
        self.start_sombrero_button = tk.Button(self, text='PUSH SOMBRERO', bg ='yellow')
        self.start_sombrero_button['command'] = self.do_sombrero
        self.start_sombrero_button.pack()
        self.start_sombrero_button.place(x=310, y=400)

        self.sombrero_width = tk.StringVar()
        self.lbl_sombrero_width = ttk.Label(self, text='Sombrero hold (s):')
        self.lbl_sombrero_width.pack(fill='x', expand=True)
        self.lbl_sombrero_width.place(x=310, y=430)
        self.t_sombrero_width = tk.Entry(self, textvariable=self.sombrero_width)
        self.t_sombrero_width.insert(0, "10")
        self.t_sombrero_width.pack(fill='x', expand=True)
        self.t_sombrero_width.focus()
        self.t_sombrero_width.place(x=500, y=430, width = 100)

        self.sombrero_ramp = tk.StringVar()
        self.lbl_sombrero_ramp = ttk.Label(self, text='Sombrero ramp (s):')
        self.lbl_sombrero_ramp.pack(fill='x', expand=True)
        self.lbl_sombrero_ramp.place(x=310, y=460)
        self.t_sombrero_ramp = tk.Entry(self, textvariable=self.sombrero_ramp)
        self.t_sombrero_ramp.insert(0, "5")
        self.t_sombrero_ramp.pack(fill='x', expand=True)
        self.t_sombrero_ramp.focus()
        self.t_sombrero_ramp.place(x=500, y=460, width = 100)

        self.sombrero_force = tk.StringVar()
        self.lbl_sombrero_force = ttk.Label(self, text='Sombrero Ramp Force (x MVC):')
        self.lbl_sombrero_force.pack(fill='x', expand=True)
        self.lbl_sombrero_force.place(x=310, y=490)
        self.t_sombrero_force = tk.Entry(self, textvariable=self.sombrero_force)
        self.t_sombrero_force.insert(0, "0.15")
        self.t_sombrero_force.pack(fill='x', expand=True)
        self.t_sombrero_force.focus()
        self.t_sombrero_force.place(x=500, y=490, width = 100)

        self.target_profile_x = [0]
        self.target_profile_y = [0]
        self.stim_profile_x = np.empty(0)
        self.stim_profile_y = np.empty(0)

        self.stim_rate = tk.StringVar()
        self.lbl_stim_rate = ttk.Label(self, text='Interval b/w stim (s):')
        self.lbl_stim_rate.pack(fill='x', expand=True)
        self.lbl_stim_rate.place(x=710, y=20)
        self.t_stim_rate = tk.Entry(self, textvariable=self.stim_rate)
        self.t_stim_rate.insert(0, "5")
        self.t_stim_rate.pack(fill='x', expand=True)
        self.t_stim_rate.focus()
        self.t_stim_rate.place(x=850, y=20, width = 100)

        self.stim_start = tk.StringVar()
        self.lbl_stim_start = ttk.Label(self, text='Start time for stim (s):')
        self.lbl_stim_start.pack(fill='x', expand=True)
        self.lbl_stim_start.place(x=710, y=50)
        self.t_stim_start = tk.Entry(self, textvariable=self.stim_start)
        self.t_stim_start.insert(0, "5")
        self.t_stim_start.pack(fill='x', expand=True)
        self.t_stim_start.focus()
        self.t_stim_start.place(x=850, y=50, width = 100)

        self.stim_stop = tk.StringVar()
        self.lbl_stim_stop = ttk.Label(self, text='Stop time for stim (s):')
        self.lbl_stim_stop.pack(fill='x', expand=True)
        self.lbl_stim_stop.place(x=710, y=80)
        self.t_stim_stop = tk.Entry(self, textvariable=self.stim_stop)
        self.t_stim_stop.insert(0, "30")
        self.t_stim_stop.pack(fill='x', expand=True)
        self.t_stim_stop.focus()
        self.t_stim_stop.place(x=850, y=80, width = 100)

        self.pushstim_button = tk.Button(self, text='PUSH STIM', bg ='yellow')
        self.pushstim_button['command'] = self.stim_push
        self.pushstim_button.pack()
        self.pushstim_button.place(x=800, y=140)
        
        self.clearstim_button = tk.Button(self, text='Clear STIM', bg ='yellow')
        self.clearstim_button['command'] = self.stim_clear
        self.clearstim_button.pack()
        self.clearstim_button.place(x=900, y=140)

        self.check_MEPs_button = tk.Button(self, text='Check MEPs', bg ='yellow')
        self.check_MEPs_button['command'] = self.check_MEPs
        self.check_MEPs_button.pack()
        self.check_MEPs_button.place(x=900, y=190)


        fig = Figure(figsize=(7, 4), dpi=100)
        self.disp_target = fig.add_subplot(111)
        
        self.disp_target.set_title("Ramp profile", fontsize=14)
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        
        self.canvas_disp_target = FigureCanvasTkAgg(fig, master=self)  
        self.canvas_disp_target.draw()
        self.canvas_disp_target.get_tk_widget().pack(side=tk.BOTTOM, fill='x', expand=True)
        self.canvas_disp_target.get_tk_widget().place(y=550,)


        self.lbl_vis_mode_check = ttk.Label(self, text='Select feedback mode (Check):')
        self.lbl_vis_mode_check.pack(fill='x', expand=True)
        self.lbl_vis_mode_check.place(x=650, y=250)
        self.vis_chan_mode_check = tk.StringVar() 
        self.vis_mode_option1_check = tk.Radiobutton(self, text="Single Chan", variable=self.vis_chan_mode_check, value="single", command=self.set_vis_mode_check)
        self.vis_mode_option1_check.pack(fill='x', expand=True)
        self.vis_mode_option1_check.place(x=850, y=250)
        self.vis_mode_option3_check = tk.Radiobutton(self, text="Aux", variable=self.vis_chan_mode_check, value="aux", command=self.set_vis_mode_check)
        self.vis_mode_option3_check.pack(fill='x', expand=True)
        self.vis_mode_option3_check.place(x=850, y=280)

        options = [x for x in range(1,65)]
        self.vis_chan_check = tk.StringVar() 
        self.vis_chan_check.set(options[35])
        self.vis_chan_drop_check = tk.OptionMenu( self , self.vis_chan_check , *options) #tk.Button(self, text='START', bg ='green')
        self.vis_chan_drop_check.pack()
        self.vis_chan_drop_check.place(x=850, y=310)
        
        self.do_vanilla()

    def start_rec(self,):
        self.task_trial.write(False)
        print('starting')
        self.start_tmsi()
        start_time = time.time()

        trial_params = {
            "duration": float(self.trl_duration.get()),
            "MVF": float(self.max_force.get()),
            }
        
        # ### TODO SET DAQ Trigs for trail start
        # self.task_stim= [] 
        # self.task_trial = []
        window = display_force_data(self, self.task_trial, 
                                    self.task_stim, 
                                    self.target_profile_x,
                                    self.target_profile_y,
                                    self.stim_profile_x,
                                    self.stim_profile_y,
                                    trial_params,
                                    dev_select=self.vis_TMSi.get(),
                                    vis_chan_mode = self.vis_chan_mode.get(),
                                    vis_chan = self.vis_chan.get(),
                                    record=True
                                    )
        window.grab_set()
        self.wait_window(window)

        out_mat = {
            "time": np.array(self.dump_time),
            "force": np.array(self.dump_force),
            "trigs": np.array(self.dump_trig),
            "target_profile": np.array((self.target_profile_x,self.target_profile_y)).T,
            "MVC": float(self.max_force.get())
                   }
        savemat(os.path.join(self.dump_path,'trial_'+ self.trial_ID.get()+'_'+str(start_time)+'_profiles'+".mat"), out_mat)
        self.task_trial.write(False)

        self.stop_tmsi()
        self.trial_ID.set(str(int(self.trial_ID.get())+1))
        current_trial = int(self.trial_ID.get())
        self.t_trial_ID.delete(0, 'end')
        self.t_trial_ID.insert(0, str(current_trial))
        self.update()

    def set_vis_mode(self):
        self.vis_chan_drop['menu'].delete(0, 'end')
        
        ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
        self.UNI_count = 0
        self.AUX_count = 0
        self.BIP_count = 0
        self.DUD_count = 0
        for idx, ch in enumerate(ch_list):
            if (ch.type.value == ChannelType.UNI.value):
                if ch.enabled == True:
                    self.UNI_count+=1
            elif (ch.type.value == ChannelType.AUX.value):
                if ch.enabled == True:
                    self.AUX_count += 1
            elif (ch.type.value == ChannelType.BIP.value):
                if ch.enabled == True:
                    self.BIP_count += 1
            else :
                self.DUD_count += 1

        if self.vis_chan_mode.get() == 'single':
            options = [x for x in range(1,65)]
            self.vis_chan.set(options[1])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
        elif self.vis_chan_mode.get() == 'aux':
            options = [x for x in range(1,self.AUX_count+self.BIP_count+1)]
            self.vis_chan.set(options[0])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
        else:
            ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
            options = [x for x in reversed(range(1,self.UNI_count))]
            self.vis_chan.set(options[1])
            for choice in options:
                self.vis_chan_drop['menu'].add_command(label=choice,command=tk._setit(self.vis_chan, choice))
 
    def set_vis_mode_check(self):
        self.vis_chan_drop_check['menu'].delete(0, 'end')
        
        ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
        self.UNI_count = 0
        self.AUX_count = 0
        self.BIP_count = 0
        self.DUD_count = 0
        for idx, ch in enumerate(ch_list):
            if (ch.type.value == ChannelType.UNI.value):
                if ch.enabled == True:
                    self.UNI_count+=1
            elif (ch.type.value == ChannelType.AUX.value):
                if ch.enabled == True:
                    self.AUX_count += 1
            elif (ch.type.value == ChannelType.BIP.value):
                if ch.enabled == True:
                    self.BIP_count += 1
            else :
                self.DUD_count += 1

        if self.vis_chan_mode_check.get() == 'single':
            options = [x for x in range(1,65)]
            self.vis_chan_check.set(options[1])
            for choice in options:
                self.vis_chan_drop_check['menu'].add_command(label=choice,command=tk._setit(self.vis_chan_check, choice))
        elif self.vis_chan_mode_check.get() == 'aux':
            options = [x for x in range(1,self.AUX_count+self.BIP_count+1)]
            self.vis_chan_check.set(options[0])
            for choice in options:
                self.vis_chan_drop_check['menu'].add_command(label=choice,command=tk._setit(self.vis_chan_check, choice))
        else:
            ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
            options = [x for x in reversed(range(1,self.UNI_count))]
            self.vis_chan_check.set(options[1])
            for choice in options:
                self.vis_chan_drop_check['menu'].add_command(label=choice,command=tk._setit(self.vis_chan_check, choice))
 
    def manualMVC(self):
        self.manualMVC_button.config(bg = 'green')
        self.max_force.set(self.max_force.get())
        self.update()

    def stop_rec(self,):
        print('stopping')

        self.trial_ID.set(str(int(self.trial_ID.get())+1))
        current_trial = int(self.trial_ID.get())
        self.t_trial_ID.delete(0, 'end')
        self.t_trial_ID.insert(0, str(current_trial))
        self.update()

    def test_force_read(self):
        print('starting')
        self.test_force_read_button.config(bg = 'green')
        self.start_tmsi(flag = "no_rec")

        trial_params = {
            "duration": float(self.trl_duration.get()),
            "MVF": float(self.max_force.get()),
            }
        # self.task_stim= [] 
        # self.task_trial = []
        window = display_force_data(self, self.task_trial, 
                                    self.task_stim, 
                                    self.target_profile_x,
                                    self.target_profile_y,
                                    stim_profile_x =  np.empty(0),
                                    stim_profile_y =  np.empty(0),
                                    trial_params=trial_params,
                                    dev_select=self.vis_TMSi.get(),
                                    vis_chan_mode = self.vis_chan_mode.get(),
                                    vis_chan = self.vis_chan.get(),
                                    record=False
                                    )
        window.grab_set()
        self.wait_window(window)
        self.stop_tmsi(flag = "no_rec")
        self.test_force_read_button.config(bg = 'yellow')

    def check_MEPs(self):
        self.task_trial.write(False)
        print('starting')
        self.start_tmsi(flag='check')
        start_time = time.time()
        trial_params = {
            "duration": float(self.trl_duration.get()),
            "MVF": float(self.max_force.get()),
            "MEP_winU": 100,
            "MEP_winL": -50,
            }
        window = check_MEPs_win(self, self.task_trial, 
                                    self.task_stim, 
                                    self.target_profile_x,
                                    self.target_profile_y,
                                    self.stim_profile_x,
                                    self.stim_profile_y,
                                    trial_params,
                                    dev_select=self.vis_TMSi.get(),
                                    vis_chan_mode = self.vis_chan_mode.get(),
                                    vis_chan = self.vis_chan.get(),
                                    vis_chan_mode_check=self.vis_chan_mode_check.get(), 
                                    vis_chan_check = self.vis_chan_check.get(),
                                    record=True,
                                    )
        self.task_trial.write(False)
        window.grab_set()
        self.wait_window(window)
        self.stop_tmsi(flag='rec')
        self.update()

    # def check_dir(self):
    #     dump_name = self.dump_path.get()
    #     if not os.path.isdir(dump_name):
    #         print("Dir not found, making it")
    #         # os.makedirs(dump_name)
    #         os.makedirs(dump_name+'/MEPs')
    #         os.makedirs(dump_name+'/MVC')
    #     self.check_dir_button.config(bg = 'green')

    def start_DAQ(self):
        daq_name = self.daq_name.get()
        self.task_trial = nidaqmx.Task("trial_trig")
        self.task_trial.do_channels.add_do_chan( daq_name+"/" + self.trial_chan.get(),line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task_trial.start()
        self.task_trial.write(False)
        self.task_stim = nidaqmx.Task("stim_trig")
        self.task_stim.do_channels.add_do_chan( daq_name+"/" + self.stim_chan.get(),line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        self.task_stim.start()
        self.task_stim.write(False)
        self.start_daq_button.config(bg = 'green')

    def stream_DAQ(self):
        self.stream_daq_button.config(bg = 'red')
        t0 = time.time()
        while time.time()-t0 < 5:
            print("trigs", self.task_trig.read(number_of_samples_per_channel=10))
            print("force", abs(np.mean(self.in_stream_force.read(number_of_samples_per_channel=10)))*float(self.conv_factor.get()))
        self.stream_daq_button.config(bg = 'yellow')

    def start_tmsi(self,flag = "rec"):
        start_time = time.time()
        trial_num = self.trial_ID.get()
        dump_path = self.dump_path
        self.streams = {}
        self.file_writers = {}
        for key in self.tmsi_dev.keys():
            self.streams[key] = FileWriter(FileFormat.lsl, self.tmsi_dev[key].dev_name)
            self.streams[key].open(self.tmsi_dev[key].dev)
        
        # self.stream_2 = FileWriter(FileFormat.lsl, self.tmsi_dev[keysList[1]].dev_name)
        # self.stream_2.open(self.tmsi_dev[keysList[1]].dev)
            if flag != "no_rec" and flag == "MVC":
                save_path = os.path.join(dump_path,'MVC','MVC_'+key+'.poly5')
                # save_path2 = os.path.join(dump_path,'trial_MVC_'+str(start_time)+'_'+keysList[1]+'.poly5')
                self.file_writers[key] = FileWriter(FileFormat.poly5, save_path)
                self.file_writers[key].open(self.tmsi_dev[key].dev)
                # self.file_writer2 = FileWriter(FileFormat.poly5, save_path2)
                # self.file_writer2.open(self.tmsi_dev[keysList[1]].dev)
            
            elif flag != "no_rec" and flag == "check":
                save_path = os.path.join(dump_path,'MEPs',key+'.poly5')
                # save_path2 = os.path.join(dump_path,'MEPs',str(start_time)+'_'+keysList[1]+'.poly5')
                self.file_writers[key] = FileWriter(FileFormat.poly5, save_path)
                self.file_writers[key].open(self.tmsi_dev[key].dev)
                # self.file_writer2 = FileWriter(FileFormat.poly5, save_path2)
                # self.file_writer2.open(self.tmsi_dev[keysList[1]].dev)
            
                
            elif flag != "no_rec" and flag == "rec":
                save_path = os.path.join(dump_path,'trial_'+str(trial_num)+'_'+key+'.poly5')
                # save_path2 = os.path.join(dump_path,'trial_'+str(trial_num)+'_'+str(start_time)+'_'+keysList[1]+'.poly5')
                self.file_writers[key] = FileWriter(FileFormat.poly5, save_path)
                self.file_writers[key].open(self.tmsi_dev[key].dev)
                # self.file_writer2 = FileWriter(FileFormat.poly5, save_path2)
                # self.file_writer2.open(self.tmsi_dev[keysList[1]].dev)
                
            self.tmsi_dev[key].dev.start_measurement()
            # self.tmsi_dev[keysList[1]].dev.start_measurement()
        time.sleep(0.5)

    def stop_tmsi(self,flag='rec'):
        keysList = list(self.tmsi_dev.keys())
        time.sleep(0.2)
        for key in self.tmsi_dev.keys():
            if flag == "rec":
                self.file_writers[key].close()
                # self.file_writer2.close()
            self.streams[key].close()
            # self.stream_2.close()
            time.sleep(0.5)
            self.tmsi_dev[key].dev.stop_measurement()
            # self.tmsi_dev[keysList[1]].dev.stop_measurement()

    def get_MVC(self):
        self.task_trial.write(False)
        self.EMG_avg_win = 100
        trial_len = int(self.MVC_duration.get())
        max_force = 0
        self.start_MVC_button.config(bg = 'red')

        ch_list = self.tmsi_dev[self.vis_TMSi.get()].dev.config.channels
        self.UNI_count = 0
        self.AUX_count = 0
        self.BIP_count = 0
        self.DUD_count = 0
        for idx, ch in enumerate(ch_list):
            if (ch.type.value == ChannelType.UNI.value):
                if ch.enabled == True:
                    self.UNI_count+=1
            elif (ch.type.value == ChannelType.AUX.value):
                if ch.enabled == True:
                    self.AUX_count += 1
            elif (ch.type.value == ChannelType.BIP.value):
                if ch.enabled == True:
                    self.BIP_count += 1
            else :
                self.DUD_count += 1
        if self.vis_chan_mode.get() == 'single':
            self.vis_chan_slice = np.array([int(self.vis_chan.get())])
        elif self.vis_chan_mode.get() == 'aux':
            self.vis_chan_slice = np.array([int(self.vis_chan.get()) + self.UNI_count-1])
        else:
            self.vis_chan_slice = np.arange(int(self.vis_chan.get()))

        if self.vis_chan_mode.get() == 'aux':
            sos_raw = butter(3, [0.2, 20], 'bandpass', fs=2000, output='sos')
            sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
            z_sos0 = sosfilt_zi(sos_raw)
            z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
            z_sos0 = sosfilt_zi(sos_env)
            z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
        else:
            sos_raw = butter(3, [20, 500], 'bandpass', fs=2000, output='sos')
            sos_env= butter(3, 5, 'lowpass', fs=2000, output='sos')
            z_sos0 = sosfilt_zi(sos_raw)
            z_sos_raw=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
            z_sos0 = sosfilt_zi(sos_env)
            z_sos_env=np.repeat(z_sos0[:, np.newaxis, :], len(self.vis_chan_slice), axis=1)
        
        self.start_tmsi(flag = "MVC")
        print("finding stream")
        stream = pylsl.resolve_stream('name', self.vis_TMSi.get())
        for info in stream:
            print('name: ', info.name())
            print('channel count:', info.channel_count())
            print('sampling rate:', info.nominal_srate())
            print('type: ', info.type())
        self.inlet = DataInlet(stream[0])    
        

        self.inlet.inlet.open_stream()
        showinfo(title='START MVC', message="START MVC")
        self.task_trial.write(True)
        array_data = self.inlet.pull_and_plot()#

        samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
        samples = np.abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
        _, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
        t0 = time.time()
        ctr = 0 # this counter prevents initial values (with filter artifact) from being saved into MVC
        while time.time()-t0 < trial_len:
            if ctr>3:
                time.sleep(0.1)
                array_data = self.inlet.pull_and_plot()
                if self.vis_chan_mode.get() == 'aux':
                    array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])#sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                    # samples = abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                    # array_data_filt, z_sos_env= sosfilt(samples_raw.T, samples, zi=z_sos_env)
                else:
                    samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                    samples = np.abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                    array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
                array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
                curr_force = np.median(array_data_scaled)
                
                if self.vis_chan_mode.get() == 'aux':
                    curr_force = curr_force*float(self.conv_factor.get())
                print(curr_force)
                if curr_force > max_force:
                    max_force = curr_force
                    self.max_force.set(str(max_force))
                    self.update()
            else:
                ctr+=1
                time.sleep(0.1)
                array_data = self.inlet.pull_and_plot()
                if self.vis_chan_mode.get() == 'aux':
                    array_data_filt = np.abs(array_data[:self.EMG_avg_win,self.vis_chan_slice])
                else:
                    samples_raw, z_sos_raw= sosfilt(sos_raw, array_data[:self.EMG_avg_win,self.vis_chan_slice].T, zi=z_sos_raw)
                    samples = np.abs(samples_raw) - np.min(abs(samples_raw),axis =0).reshape(1,-1)
                    array_data_filt, z_sos_env= sosfilt(sos_env, samples, zi=z_sos_env)
                array_data_scaled = np.abs(np.nan_to_num(array_data_filt,nan=0,posinf=0,neginf=0)).T
                curr_force = np.median(array_data_scaled)
                if self.vis_chan_mode.get() == 'aux':
                    curr_force = curr_force*float(self.conv_factor.get())
                print("not saved",curr_force)
        
        self.task_trial.write(False)
        self.inlet.inlet.close_stream()

        self.stop_tmsi()
        # showinfo(title='STOP MVC', message="STOP MVC")
        self.start_MVC_button.config(bg = 'green')

    def do_sombrero(self):
        max_force = float(self.max_force.get())
        peak_ramp_force = float(self.peak_ramp_force.get())
        trl_duration = float(self.trl_duration.get())
        init_wait = float(self.init_wait.get())
        sombrero_width = float(self.sombrero_width.get())
        sombrero_ramp = float(self.sombrero_ramp.get())
        sombrero_force = float(self.sombrero_force.get())

        self.target_profile_x = [0, init_wait, init_wait+sombrero_ramp, init_wait+sombrero_ramp+sombrero_width, trl_duration//2, 
                                 trl_duration-init_wait-sombrero_ramp-sombrero_width, trl_duration-init_wait-sombrero_ramp, trl_duration-init_wait, trl_duration]
        self.target_profile_y = [0, 0, max_force*sombrero_force, max_force*sombrero_force, max_force*peak_ramp_force, max_force*sombrero_force, max_force*sombrero_force, 0, 0]
        assert len(self.target_profile_x) == len(self.target_profile_y)

        self.stim_profile_x = np.empty(0)
        self.stim_profile_y = np.empty(0)
        self.disp_target.clear()
        
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        self.disp_target.plot(self.target_profile_x, self.target_profile_y, linewidth = 5, color = 'r')
        self.canvas_disp_target.draw()
        self.start_vanilla_button.config(bg = 'yellow')
        self.start_sombrero_button.config(bg = 'green')

    def do_vanilla(self):
        max_force = float(self.max_force.get())
        peak_ramp_force = float(self.peak_ramp_force.get())
        trl_duration = float(self.trl_duration.get())
        init_wait = float(self.init_wait.get())

        self.target_profile_x = [0, init_wait, trl_duration//2, trl_duration-init_wait, trl_duration]
        self.target_profile_y = [0, 0, peak_ramp_force*max_force, 0, 0]
        assert len(self.target_profile_x) == len(self.target_profile_y)
        self.stim_profile_x = np.empty(0)
        self.stim_profile_y = np.empty(0)

        self.disp_target.clear()
        self.disp_target.set_xlabel("Time (s)", fontsize=14)
        self.disp_target.set_ylabel("Torque (Nm)", fontsize=14)
        self.disp_target.plot(self.target_profile_x, self.target_profile_y, linewidth = 5, color = 'r')
        self.canvas_disp_target.draw()

        self.start_sombrero_button.config(bg = 'yellow')
        self.start_vanilla_button.config(bg = 'green')

    def stim_push(self):
        
        stim_rate = float(self.stim_rate.get())
        stim_stop = float(self.stim_stop.get())
        stim_start = float(self.stim_start.get())
        trial_stim_profile_x = np.arange(stim_start,stim_stop,stim_rate)
        trial_stim_profile_y = np.zeros_like(self.stim_profile_x)
        
        self.stim_profile_x = np.unique(np.concatenate((self.stim_profile_x,trial_stim_profile_x)))
        self.stim_profile_y = np.concatenate((self.stim_profile_y,trial_stim_profile_y))
        self.disp_target.vlines(self.stim_profile_x,0,np.max(self.target_profile_y), linewidth = 3, color = 'k')
        self.canvas_disp_target.draw()

        self.pushstim_button.config(bg = 'green')

    def stim_clear(self):
        self.stim_profile_x = np.empty(0)
        self.stim_profile_y = np.empty(0)
        assert len(self.target_profile_x) == len(self.target_profile_y)
        self.disp_target.clear()
        self.canvas_disp_target.draw()


def main():
    tk_trial = APP([],{"FLX":[],"EXT":[]})
    tk_trial.mainloop()
    return None

if __name__ == "__main__":
    main()