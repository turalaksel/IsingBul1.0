#!/usr/bin/env python

from ising import *
directory = "."
ising_file       = directory+"/"+"files.dat"
constraints_file = directory+"/"+"constraints.dat"
init_file        = directory+"/"+"init_params.dat"
boot_file        = directory+"/"+"bootstrap.dat"
param_file       = directory+"/"+"saved_params.npy"
con_ANKs = IsingDen()
con_ANKs.set_temp(293.15)
con_ANKs.prepare_all(ising_file,constraints_file)
con_ANKs.check_solvability()
con_ANKs.initialize_params(init_file)
con_ANKs.print_params()
best_params = con_ANKs.fit(max_steps=10000)
con_ANKs.save_params(param_file)
con_ANKs.print_params()
con_ANKs.simulate()
con_ANKs.plot_fit()
tic = time.clock()
con_ANKs.bootstrap(param_file,boot_file,num_steps = 1000)
tac = time.clock()
print (tac-tic),'seconds'
