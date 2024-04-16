import time
import tensorflow as tf
import numpy as np
from Training2DRNN_2D_ClassicalIsingModel_Annealing_VANILLAorGRU_per import run_2DTFIM, Ising2D_energies, Ising2D_local_purities
from RNNwavefunction_VANILLAorGRU_per import RNNwavefunction, MDRNNcell, MDRNNGRUcell

"""
Code that performs the training of the RNN as per the code in
Training2DRNN_2D_ClassicalIsingModel_Annealing_VANILLAorGRU_per.py and
RNNwavefunction_VANILLAorGRU_per.py
"""

if __name__ == '__main__':
    '''
    
    '''
    from argparse import ArgumentParser

    # Let us keep track of the execution time
    start_time = time.time()
    
    parser = ArgumentParser(description='Set parameters')
    parser.add_argument('--numsteps', default=1*10**3)
    parser.add_argument('--Nx', default=3)
    parser.add_argument('--Ny', default=3)
    #parser.add_argument('--Bx', default=1)
    parser.add_argument('--num_units', default=50)
    parser.add_argument('--numsamples', default=500)
    parser.add_argument('--numsamples_end', default=10**4) # numsamples used for calculation of minimized F using the RNN trained after numsteps
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--seed', default=111)
    parser.add_argument('--Tmin', default=0.1)
    parser.add_argument('--Tmax', default=1.0)
    parser.add_argument('--Tstep', default=0.1)
    #parser.add_argument('--slurm_nr', default=0)
    parser.add_argument('--cell', default=MDRNNcell) # other option is 'MDRNNGRUcell'
    #parser.add_argument('--ix', default=1)
    #parser.add_argument('--iy', default=1)
    parser.add_argument('--FMorAFM', default='FM')
    parser.add_argument('--BC', default = 'Periodic')
    parser.add_argument('--alpha', default = 2)
    parser.add_argument('--periodicRNN', default = 'NO')
    

    
    args = parser.parse_args()
    numsteps = int(args.numsteps)
    Nx = int(args.Nx)
    Ny = int(args.Ny)
    #Bx = float(args.Bx)
    num_units = int(args.num_units)
    numsamples = int(args.numsamples)
    numsamples_end = int(args.numsamples_end)
    lr= float(args.lr)
    seed = int(args.seed)
    Tmin = float(args.Tmin)
    Tmax = float(args.Tmax)
    Tstep = float(args.Tstep)
    #slurm_nr = int(args.slurm_nr) # should be from 0-99
    cellForRun = str(args.cell)
    #ix = int(args.ix)
    #iy = int(args.iy)
    FMorAFM = str(args.FMorAFM)
    BC = str(args.BC)
    alpha = float(args.alpha)
    periodicRNN = str(args.periodicRNN)
    

    # Define constants that will be input into local energies and local purities functions
    input_dim = 2
    Jz = tf.ones((Nx,Ny)) #Ferromagnetic couplings
    temperatures = np.arange(Tmin,Tmax+0.00001,Tstep)

    savename = '_2D_ClassicalIsingModel'
    # We don't really need to save the temperatures in a .npy file, but do it anyways.
    np.save('temperatures_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units) + '_Tmin' + '{:.3f}'.format(Tmin) + '_Tmax' + '{:.3f}'.format(Tmax) + '_dT' + '{:.3f}'.format(Tstep) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy', temperatures)

    for i in range(len(temperatures)):
        # This next bit of code is all about printing the number of GPUs our simulation is using. Recall, I ask for a GPU number in my sbatch / slurm
        # files. tf.config.list_physical_devices('GPU) should be a list or tuple of the GPUs being used- the length of this list is the number of GPUs.
        physical_devices = tf.config.list_physical_devices('GPU')
        print("Num GPUs:", len(physical_devices))
        '''
        if a job gets killed halfway down the temperatures list, when we restart the simulation, this loop will restart at the first temp
        unfortunately, but the positive is that the training data for each temperature for which training has already been performed is
        saved via the checkpointing done in the run_2DTFIM function in the
        Training2DRNN_2D_ClassicalIsingModel_Annealing_VANILLAorGRU_per.py file. So the code in this file runs
        through the temperatures where the training has already been done, uses the saved final model for each of those
        temperatures to compute the minimized free energy without any training having to be re-done, and goes from there.
        It's not too big of an extra time cost, I think, and then we reach the first temperature in this loop for which
        no training has as of yet been done, and training proceeds normally from there.
        '''
        
        T = temperatures[len(temperatures)-1-i]
        print()
        print("T =",T) # Keep track of loop execution
        print()
        
        
        trainedRNN_freeEnergy = []
        trainedRNN_Energy = []
        trainedRNN_Purity = []
        trainedRNN_varEnergy = []
        trainedRNN_varPurity = []
        
        # DEFINE THE INITIAL WAVEFUNCTION.
        if i == 0:
            units=[num_units]
            if cellForRun == 'MDRNNcell':
                wf_T = RNNwavefunction(Nx,Ny,cell=MDRNNcell,units=units,seed = seed)
            elif cellForRun == 'MDRNNGRUcell':
                wf_T = RNNwavefunction(Nx,Ny,cell=MDRNNGRUcell,units=units,seed = seed)
        
        '''
        NOTE: when jobs are killed halfway through runs, when you then do the checkpointing, even if the optimized wf_T
        from the previous temperature is fed into the run_2DTFIM function for the next temperature, the model for this
        next temperature is restored from the checkpoint inside the run_2DTFIM function and you start the training
        from the saved model rather than the wf_T wavefunction that is input into the function. If we weren't annealing,
        we would have a line of code that re-initializes wf every time run_2DTFIM is called, but the model having been
        saved means the saved RNN parameters are restored. No need to worry.
        '''
        
        # Perform the training
        wf_T = run_2DTFIM(wf_T,numsteps,Nx,Ny,num_units,numsamples,lr,seed,T,FMorAFM,BC,alpha,periodicRNN)
        
        # Generate numsamples_end samples for final observable calculations
        if periodicRNN == "NO":
            samples = wf_T.sample(numsamples_end,input_dim)
        elif periodicRNN == "YES":
            samples = wf_T.sample_periodicRNN(numsamples_end,input_dim)
            
        Eloc = Ising2D_energies(Jz, Nx, Ny, samples, FMorAFM, BC)
        Ploc = Ising2D_local_purities(samples,wf_T,alpha,periodicRNN)

        # Energy Calculation
        meanE = np.mean(Eloc)

        # Purity Calculation
        meanP = np.mean(Ploc)

        # Variances
        # varEnergy Calculation
        varE = np.var(Eloc)

        # varPurity Calculation
        varP = np.var(Ploc)

        # Free Energy Calculation -
        meanF = meanE - T / (1-alpha) * np.log(meanP)
    
        #temperatures.append(T.numpy())
        trainedRNN_freeEnergy.append(meanF)
        
        trainedRNN_Energy.append(meanE)
        trainedRNN_Purity.append(meanP)
        trainedRNN_varEnergy.append(varE)
        trainedRNN_varPurity.append(varP)
        
        #print("temperatures =",temperatures)
        print()
        print("FYI. T =",T) # Keep track of loop execution
        print()
        print("periodicRNN =",periodicRNN)
        print("trained RNN free energy =",meanF)
        print("trained RNN energy =",meanE)
        print("trained RNN purity =",meanP)
        print("trained RNN varEnergy =",varE)
        print("trained RNN varPurity =",varP)
    
        '''
        Now calculate the magnetization per spin, for samples1, which has numsamples = numsamples_end
        '''
        trainedRNN_magPerSpinAbs = []
        trainedRNN_varmagPerSpinAbs = []
        mN_tensor = wf_T.magPerSpin_abs(samples) # tf.Tensor of shape
        mN = np.mean(mN_tensor)
        var_mN = np.var(mN_tensor)
        trainedRNN_magPerSpinAbs.append(mN)
        trainedRNN_varmagPerSpinAbs.append(var_mN)
        print("trained RNN magPerSpinAbs =",mN)
        print("trained RNN varmagPerSpinAbs =",var_mN)
        
        
        
        
        #.npy file & SAVING MODEL
        np.save('trainedRNN_freeEnergy_2DRNN_'+str(Nx)+'x'+ str(Ny) + '_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_sampEnd'+str(numsamples_end)+'_steps'+str(numsteps)+'_units'+str(num_units)+'_seed'+str(seed) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + '_T' + '{:.3f}'.format(T) + savename +'.npy', trainedRNN_freeEnergy)
        np.save('trainedRNN_Energy_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_sampEnd'+str(numsamples_end)+'_steps'+str(numsteps)+'_units'+str(num_units)+'_seed'+str(seed) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + '_T' + '{:.3f}'.format(T) + savename +'.npy', trainedRNN_Energy)
        np.save('trainedRNN_Purity_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_sampEnd'+str(numsamples_end)+'_steps'+str(numsteps)+'_units'+str(num_units)+'_seed'+str(seed) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + '_T' + '{:.3f}'.format(T) + savename +'.npy', trainedRNN_Purity)
        np.save('trainedRNN_varEnergy_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_sampEnd'+str(numsamples_end)+'_steps'+str(numsteps)+'_units'+str(num_units)+'_seed'+str(seed) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + '_T' + '{:.3f}'.format(T) + savename +'.npy', trainedRNN_varEnergy)
        np.save('trainedRNN_varPurity_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_sampEnd'+str(numsamples_end)+'_steps'+str(numsteps)+'_units'+str(num_units)+'_seed'+str(seed) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + '_T' + '{:.3f}'.format(T) + savename +'.npy', trainedRNN_varPurity)
        np.save('trainedRNN_magPerSpinAbs_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_sampEnd'+str(numsamples_end)+'_steps'+str(numsteps)+'_units'+str(num_units)+'_seed'+str(seed) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + '_T' + '{:.3f}'.format(T) + savename +'.npy', trainedRNN_magPerSpinAbs)
        np.save('trainedRNN_varmagPerSpinAbs_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_sampEnd'+str(numsamples_end)+'_steps'+str(numsteps)+'_units'+str(num_units)+'_seed'+str(seed) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + '_T' + '{:.3f}'.format(T) + savename +'.npy', trainedRNN_varmagPerSpinAbs)
        

    
    # Execution time
    print("------------")
    print("Execution time = --- %s seconds ---" % (time.time() - start_time))
    
    
    