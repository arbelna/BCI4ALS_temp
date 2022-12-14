import os
import random
import sys
from datetime import datetime
from tkinter import messagebox
from tkinter.filedialog import askdirectory
from tkinter import Tk, Entry, Label, Button, ttk
from random import randrange
from psychopy import visual, core

import brainflow
import numpy as np


class Experiment:
    def __init__(self, eeg):
        self.ask_num_blocks()
        self.ask_num_trials()
        self.eeg = eeg

        # if self.eeg.board_id == brainflow.BoardIds.SYNTHETIC_BOARD:
        #     messagebox.showwarning(title="bci4als WARNING", message="You are running a synthetic board!")
        #     self.debug = True
        # else:
        #     self.debug = False

        self.cue_length = None
        self.trial_length = None
        self.session_directory = None
        self.enum_image = {0: 'yes', 1: 'no', 2: 'none'}
        self.experiment_type = None
        self.skip_after = None

        #     labels
        self.labels = []
        self._init_labels()

    def _init_labels(self, keys=(0, 1, 2)):
        """
        This method creates dict containing a stimulus vector
        :return: the stimulus in each trial (list)
        """
        for i in range(self.num_blocks):
            self.labels.append([])
            for j in range(self.num_trials):
                temp = randrange(7)
                if temp == 0:
                    self.labels[i].append(0)
                elif temp == 1:
                    self.labels[i].append(1)
                else:
                    self.labels[i].append(2)

    # def _ask_subject_directory(self):
    #     """
    #     init the current subject directory
    #     :return: the subject directory
    #     """
    #
    #     # get the CurrentStudy recording directory
    #     if not messagebox.askokcancel(title='bci4als',
    #                                   message="Welcome to the motor imagery EEG recorder."
    #                                           "\n\nNumber of trials: {}\n\n"
    #                                           "Please select the subject directory:".format(self.num_trials)):
    #         sys.exit(-1)
    #
    #     # show an "Open" dialog box and return the path to the selected file
    #     init_dir = os.path.join(os.path.split(os.path.abspath(''))[0], 'recordings')
    #     subject_folder = askdirectory(initialdir=init_dir)
    #     if not subject_folder:
    #         sys.exit(-1)
    #     return subject_folder

    def ask_num_trials(self):
        # Define a function to return the Input data
        def get_num_trials():
            input = entry.get()
            try:
                self.num_trials = int(input)
            except:
                self.num_trials = None
            win.destroy()

        def error(message):
            err.geometry("400x300")
            Label(err, text=message, font=('Helvetica 14 bold')).pack(pady=20)
            # Create a button in the main Window to open the popup
            Button(err, text="Ok", command=cont).pack()
            err.mainloop()

        def cont():
            err.destroy()
            pass

        self.num_trials = None
        while True:

            win = Tk()
            win.geometry('400x300')
            entry = Entry(win, width=42)
            entry.place(relx=.5, rely=.2, anchor='center')
            label = Label(win, text="Enter the number of trials you want.", font=('Helvetica 13'))
            label.pack()
            Button(win, text="submit", command=get_num_trials).place(relx=.5, rely=.3)
            win.mainloop()

            if self.num_trials is not None:
                break
            err = Tk()
            error("You should enter a number!")

    def ask_num_blocks(self):
        # Define a function to return the Input data
        def get_num_block():
            input = entry.get()
            try:
                self.num_blocks = int(input)
            except:
                self.num_blocks = None
            win.destroy()

        def error(message):
            err.geometry("400x300")
            Label(err, text=message, font=('Helvetica 14 bold')).pack(pady=20)
            # Create a button in the main Window to open the popup
            Button(err, text="Ok", command=cont).pack()
            err.mainloop()

        def cont():
            err.destroy()
            pass

        self.num_blocks = None
        while True:

            win = Tk()
            win.geometry('400x300')
            entry = Entry(win, width=42)
            entry.place(relx=.5, rely=.2, anchor='center')
            label = Label(win, text="Enter the number of blocks you want.", font=('Helvetica 13'))
            label.pack()
            Button(win, text="submit", command=get_num_block).place(relx=.5, rely=.3)
            win.mainloop()

            if self.num_blocks is not None:
                break
            err = Tk()
            error("You should enter a number!")



    def run_experiment(self):
        for i in range(self.num_blocks):
            core.wait(1.0)
            mywin = visual.Window([800, 600], monitor="testMonitor", units="deg")
            yes = visual.TextStim(mywin, f'Block number {i+1}', color=(1, 1, 1), colorSpace='rgb')
            yes.draw()
            mywin.update()
            core.wait(1.0)
            mywin.close()
            mywin = visual.Window([800, 600], monitor="testMonitor", units="deg")
            for j in range(self.num_trials):
                core.wait(3.0)
                yes = visual.ImageStim(win=mywin, image=f'Pictures/{self.enum_image[self.labels[i][j]]}.jpg')
                yes.draw()
                mywin.update()
                core.wait(3.0)
            mywin.close()

