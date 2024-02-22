# import os
import tkinter as tk
from tkinter import ttk

from vis_feedback import render_emg
from tmsi_dual_interface import TMSi_gui
# import PySide2

# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class APP_main(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Main interface for gesture decoding')
        self.geometry('500x200')
        
        self.tmsi_init_button = tk.Button(self, text='Init TMSi', bg ='yellow')
        self.tmsi_init_button['command'] = self.open_TMSi_GUI
        self.tmsi_init_button.pack()
        self.tmsi_init_button.place(x=300, y=10)

        self.lbltmsi = ttk.Label(self, text='1. Click to initialize TMSi as "tmsi_dev"') 
        self.lbltmsi.pack(fill='x', expand=True)
        self.lbltmsi.place(x=10, y=15)

        self.prompt_init_button = tk.Button(self, text='Init Rec', bg ='yellow')
        self.prompt_init_button['command'] = self.open_disp
        self.prompt_init_button.pack()
        self.prompt_init_button.place(x=300, y=60)
        
        self.lblprompt = ttk.Label(self, text='2. Click to initialize recording')
        self.lblprompt.pack(fill='x', expand=True)
        self.lblprompt.place(x=10, y=65)
        # self.open_disp()
    def open_TMSi_GUI(self):
        window = TMSi_gui.TMSi_GUI(self)
        window.grab_set()
        self.wait_window(window)
        self.tmsi_dev = window.device_dict
        self.tmsi_init_button.config(bg = 'green')

    def open_disp(self):
        # self.tmsi_dev ={"FLX":[],"EXT":[]}
        window = render_emg.APP(self, self.tmsi_dev)
        window.grab_set()
        self.wait_window(window)
        self.prompt_init_button.config(bg = 'green')



def main():
    tk_trial = APP_main()
    tk_trial.mainloop()
    return None

if __name__ == "__main__":
    main()