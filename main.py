import experiment as ex

exp = ex.Experiment('eeg')
exp.run_experiment()

df = exp.results
print(exp)
