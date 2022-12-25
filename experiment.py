# importing relevant libraries
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
        """
        This is the constructor method that is called when an object of this class is created.
        It initializes several instance variables
        """
        self.num_blocks = None
        self.num_trials = None
        self.ask_num_blocks()
        self.ask_num_trials()
        self.eeg = eeg
        self.results = []

        self.enum_image = {0: 'furious', 1: 'sad', 2: 'happy'}

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

    def ask_num_trials(self):
        # Define a function to return the Input data
        """
        This method prompts the user to enter the number of trials they want in their experiment.
         If the input is not a valid number, it displays an error message.
         :param: input: none
        :return:  the number of trials
        """
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
        """
        This method prompts the user to enter the number of blocks they want in their experiment.
         If the input is not a valid number, it displays an error message.
        :return: the number of desired blocks
        """
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
        """
        This method runs the experiment by displaying images to the user and collecting their responses.
         It stores the results in the results instance variable.
        :param eeg:
        :return: csv file with expermient results
        Target: 1=happy 2= sad
        image: 0= distractor/furious 1=happy 2=sad, when 1 or 2 are target or non target
        """

        # overwrite (filemode='w') a detailed log of the last run in this dir
        lastLog = logging.LogFile("lastRun.log", level=logging.CRITICAL, filemode='w')

        for i in range(self.num_blocks):
            mywin = visual.Window([800, 800], monitor="testMonitor", units="deg")
            look = (i % 2) + 1
            start_block_win = visual.TextStim(mywin, f'Block number {i + 1} \n\n\n {self.enum_image[look]}', color=(1, 1, 1),
                                  colorSpace='rgb')
            start_block_win.draw()
            mywin.logOnFlip(level=logging.CRITICAL, msg=f'+{i + 1}')
            mywin.flip(clearBuffer=True)
            core.wait(3.0)
            mywin.close()
            mywin = visual.Window([800, 800], monitor="testMonitor", units="deg")
            for j in range(self.num_trials):
                wait = random.uniform(0.3, 0.6)
                core.wait(wait)
                start_block_win = visual.ImageStim(win=mywin, image=f'Pictures/{self.enum_image[self.labels[i][j]]}.png')
                start_block_win.draw()
                # status: str, label: int, index: int
                mywin.logOnFlip(level=logging.CRITICAL, msg=f'{self.labels[i][j]} {time.time()} {look}')
                mywin.flip(clearBuffer=True)
                eeg.insert_marker(status='start', label=self.labels[i][j], index=j)
                core.wait(0.2)
                start_block_win = visual.ImageStim(win=mywin)
                start_block_win.draw()
                mywin.flip()
                wait = 1 - 0.2 - wait
                core.wait(wait)
            mywin.close()

        with open('lastRun.log') as file:
            file = [line.rstrip('\n').split('\t') for line in file]
        pre_dataframe = []
        curr_block = 0

        for line in file:
            temp = line[2].split(" ")
            if len(temp) == 1:
                curr_block += 1
                curr_trial = 0
                continue
            curr_trial += 1
            pre_dataframe.append([curr_block, curr_trial, temp[0], temp[2], line[0], temp[1]])
        self.results = np.array([np.array(x) for x in pre_dataframe])
        self.results = pd.DataFrame(self.results)
        self.results = self.results.set_axis(['Block', 'Trial', 'Label', 'Target', 'Time', 'Unix time'], axis=1)
