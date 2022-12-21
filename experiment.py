import os

import sys
from datetime import datetime
from tkinter import messagebox
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import Tk, Entry, Label, Button
from random import randrange
from psychopy import visual, core, logging
import random
import pandas as pd
from eeg import Eeg
import time

import brainflow
import numpy as np

import eeg


class Experiment:
    def __init__(self, eeg):

        self.num_blocks = None
        self.num_trials = None
        self.ask_num_blocks()
        self.ask_num_trials()
        self.eeg = eeg
        self.results = []

        # if self.eeg.board_id == brainflow.BoardIds.SYNTHETIC_BOARD:
        #     messagebox.showwarning(title="bci4als WARNING", message="You are running a synthetic board!")
        #     self.debug = True
        # else:
        #     self.debug = False

        self.cue_length = None
        self.trial_length = None
        self.session_directory = None
        self.enum_image = {0: 'furious', 1: 'sad', 2: 'happy'}
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
                if temp == 2:
                    self.labels[i].append(2)
                elif temp == 1:
                    self.labels[i].append(1)
                else:
                    self.labels[i].append(0)

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

        def get_num_trials_ent(event):
            return get_num_trials(entry.get())

        def get_num_trials(input=None):
            if input is None:
                input1 = entry.get()
            else:
                input1 = input
            try:
                self.num_trials = int(input1)
            except:
                self.num_trials = None
            win.destroy()

        def error(message):
            err.geometry("400x300")
            Label(err, text=message, font=('Helvetica 14 bold')).pack(pady=20)
            # Create a button in the main Window to open the popup
            Button(err, text="Ok", command=cont).pack()
            err.bind("<Return>", cont)
            err.after(1, lambda: err.focus_force())
            err.mainloop()

        def cont(input):
            err.destroy()
            pass

        self.num_trials = None
        while True:

            win = Tk()
            win.geometry('400x300')
            entry = Entry(win, width=42)
            entry.place(relx=.5, rely=.2, anchor='center')
            entry.after(1, lambda: entry.focus_force())
            label = Label(win, text="Enter the number of trials you want.", font=('Helvetica 13'))
            label.pack()
            Button(win, text="submit", command=get_num_trials).place(relx=.5, rely=.3)
            win.bind("<Return>", get_num_trials_ent)
            win.mainloop()

            if self.num_trials is not None:
                break
            err = Tk()
            error("You should enter a number!")

    def ask_num_blocks(self):
        # Define a function to return the Input data
        def get_num_block_ent(event):
            return get_num_block(entry.get())

        def get_num_block(input=None):
            if input is None:
                input1 = entry.get()
            else:
                input1 = input
            try:
                self.num_blocks = int(input1)
            except:
                self.num_blocks = None
            win.destroy()

        def error(message):
            err.geometry("400x300")
            Label(err, text=message, font=('Helvetica 14 bold')).pack(pady=20)
            # Create a button in the main Window to open the popup
            Button(err, text="Ok", command=cont).pack()
            err.bind("<Return>", cont)
            err.after(1, lambda: err.focus_force())
            err.mainloop()

        def cont(input):
            err.destroy()
            pass

        self.num_blocks = None
        while True:

            win = Tk()
            win.geometry('400x300')
            entry = Entry(win, width=42)
            entry.place(relx=.5, rely=.2, anchor='center')
            entry.after(1, lambda: entry.focus_force())
            label = Label(win, text="Enter the number of blocks you want.", font=('Helvetica 13'))
            label.pack()
            Button(win, text="submit", command=get_num_block).place(relx=.5, rely=.3)
            win.bind("<Return>", get_num_block_ent)
            win.mainloop()

            if self.num_blocks is not None:
                break
            err = Tk()
            error("You should enter a number!")

    def run_experiment(self, eeg):

        # overwrite (filemode='w') a detailed log of the last run in this dir
        lastLog = logging.LogFile("lastRun.log", level=logging.CRITICAL, filemode='w')

        for i in range(self.num_blocks):
            mywin = visual.Window([800, 800], monitor="testMonitor", units="deg")
            look = random.randint(1, 2)
            yes = visual.TextStim(mywin, f'Block number {i + 1} \n\n\n {self.enum_image[look]}', color=(1, 1, 1),
                                  colorSpace='rgb')
            yes.draw()
            mywin.logOnFlip(level=logging.CRITICAL, msg=f'+{i + 1}')
            mywin.flip(clearBuffer=True)
            core.wait(3.0)
            mywin.close()
            mywin = visual.Window([800, 800], monitor="testMonitor", units="deg")
            for j in range(self.num_trials):
                wait = random.uniform(0.3, 0.6)
                core.wait(wait)
                yes = visual.ImageStim(win=mywin, image=f'Pictures/{self.enum_image[self.labels[i][j]]}.png')
                yes.draw()
                # status: str, label: int, index: int
                mywin.logOnFlip(level=logging.CRITICAL, msg=f'{self.labels[i][j]} {time.time()}')
                mywin.flip(clearBuffer=True)
                # eeg.insert_marker(status='start', label=self.labels[i][j], index=j)
                core.wait(0.2)
                yes = visual.ImageStim(win=mywin)
                yes.draw()
                mywin.flip()
                wait = 1 - 0.2 - wait
                core.wait(wait)
            mywin.close()

        with open('lastRun.log') as file:
            file = [line.rstrip('\n').split('\t') for line in file]
        pre_dataframe = []
        for i in range(self.num_blocks * self.num_trials + self.num_blocks):
            if i % (self.num_blocks + 1) == 0:
                curr_block = int(i / self.num_blocks) + 1
                continue
            temp = file[i][2].split(" ")
            pre_dataframe.append([curr_block, i % (self.num_blocks + 1), temp[0], file[i][0], temp[1]])
        self.results = np.array([np.array(x) for x in pre_dataframe])
        self.results = pd.DataFrame(self.results)
        self.results = self.results.set_axis(['Block', 'Trial', 'Label', 'Time', 'Unix time'], axis=1)
