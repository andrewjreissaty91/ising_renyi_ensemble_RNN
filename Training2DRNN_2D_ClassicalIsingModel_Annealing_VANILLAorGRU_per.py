import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import time
import random
from math import ceil

#from RNNwavefunction_VANILLAorGRU_per import RNNwavefunction

"""
This implementation is based on RNN Wave Functions code of https://github.com/mhibatallah/RNNWavefunctions
Adapted by Andrew Jreissaty for Tensorflow 2.0 and for the problem at hand (2D RNN, periodic RNN structure)
"""

def Ising2D_energies(Jz, Nx, Ny, samples, FMorAFM, BC):
    """
    To get the energies of the classical 2D TFIM (PBC) given a set of set of samples in parallel!

    Returns: The energies that correspond to each sample/configuration in "samples"
    
    Inputs:
    - samples: (numsamples, Nx,Ny)
    - Jz: (Nx,Ny) np array 
    - FMorAFM: if == 'FM', then we are working with the FM TFIM. If == 'AFM', then we are working with the AFM TFIM. This will
      inform the prefactor below
    - BC: if == 'Open', then open boundary conditions. if == 'Periodic', then periodic boundary conditions
    - Nx and Ny: Ny = number of rows, Nx = number of columns.

    """

    numsamples = samples.shape[0]

    #N = Nx*Ny #Total number of spins
    
    if FMorAFM == 'FM':
        prefactor = -1
    elif FMorAFM == 'AFM':
        prefactor = 1
    

    energies = tf.zeros(numsamples,dtype=tf.float32)
    
    # We will need the below tf.Tensor to change the values of 0,1,2 and to +1,-1 and +1 after using one-hot encoding
    # below.
    diagonalEnergyTensor = tf.constant([+1.0,-1.0,+1.0]) # I think floats are needed for whatever reason
    for i in range(Nx-1): #diagonal elements (right neighbours)
        values = samples[:,i]+samples[:,i+1] # the ith plane + the (i+1)th plane (plane being in the x-z plane)
                                                           # This gives us a tf.tensor of shape (numsamples,Ny)
        
        # We will actually use tf.one_hot here. valuesT is a tf.tensor of shape (numsamples,Ny). Its values are 0,1,2.
        # We want the values of 0 and 2 to be changed to +1, the value of 1 to be changed to -1. To do this, we use
        # one_hot encoding to represent 0 as (1,0,0), 1 as (0,1,0) and 2 as (0,0,1).
        valuesT = tf.one_hot(values,depth=3,dtype=tf.float32) # now valuesT is of shape numsamples,Ny,3
        valuesT = tf.reduce_sum(tf.multiply(valuesT,diagonalEnergyTensor),axis=2)
        
        # Assume Jz is a tf.Tensor of shape(Nx,Ny), filled with ones (not an np array)
        valuesT = valuesT * Jz[i,:] # I tested this in testFile.py
        
        energies_i = tf.reduce_sum(valuesT,axis=1)
        energies = energies + energies_i
        
        
    for i in range(Ny-1): #diagonal elements (downward neighbours)
        values = samples[:,:,i]+samples[:,:,i+1] # summing xy-planes of spins. Values is of dimension numsamples,Nx
        
        # We will actually use tf.one_hot here. valuesT is a tf.tensor of shape (numsamples,Nx). Its values are 0,1,2.
        # We want the values of 0 and 2 to be changed to +1, the value of 1 to be changed to -1. To do this, we use
        # one_hot encoding to represent 0 as (1,0,0), 1 as (0,1,0) and 2 as (0,0,1).
        valuesT = tf.one_hot(values,depth=3,dtype=tf.float32) # now valuesT is of shape numsamples,Nx,3
        valuesT = tf.reduce_sum(tf.multiply(valuesT,diagonalEnergyTensor),axis=2)

        # Assume Jz is a tf.Tensor of shape(Nx,Ny), filled with ones (not an np array)
        valuesT = valuesT * Jz[:,i] # I tested this in testFile.py
        
        energies_i = tf.reduce_sum(valuesT,axis=1)
        energies = energies + energies_i
    
    '''
    Account for periodic boundary conditions
    '''
    if BC == 'Periodic':
        '''
        Horizontal
        '''
        # samples[:,i] is equivalent to samples[:,i,:]
        values_horiz_periodic = samples[:,0]+samples[:,Nx-1] # the 0th plane + the (Nx-1)th plane (plane being in the x-z plane)
                                                             # This gives us a tf.tensor of shape (numsamples,Ny)

        valuesT = tf.one_hot(values_horiz_periodic,depth=3,dtype=tf.float32) # now valuesT is of shape numsamples,Ny,3
        valuesT = tf.reduce_sum(tf.multiply(valuesT,diagonalEnergyTensor),axis=2)
        
        valuesT = valuesT * Jz[Nx-1,:] # I tested this in testFile.py
        
        energies_i = tf.reduce_sum(valuesT,axis=1)
        energies = energies + energies_i
        '''
        Vertical
        '''
        values_vert_periodic = samples[:,:,0]+samples[:,:,Ny-1] # summing xy-planes of spins. Values is of dimension numsamples,Nx

        valuesT = tf.one_hot(values_vert_periodic,depth=3,dtype=tf.float32) # now valuesT is of shape numsamples,Nx,3
        valuesT = tf.reduce_sum(tf.multiply(valuesT,diagonalEnergyTensor),axis=2)

        
        valuesT = valuesT * Jz[:,Ny-1] # I tested this in testFile.py
        
        energies_i = tf.reduce_sum(valuesT,axis=1)
        energies = energies + energies_i
    
    '''
    Account for choice between FM and AFM.
    '''
    energies = prefactor * energies
    
    return energies



'''
"Local Purities" is the quantity that is being averaged to estimate the generalized alpha-purity
(generalized purity = sum over sigma of P(sigma)^alpha).
'''
def Ising2D_local_purities(samples,wf,alpha,periodicRNN):
    """
    To get the local purities of 2D TFIM given a set of samples in parallel!
    The quantity that is being averaged is what we are calling "local purities" for now.
    
    Returns: The local purities that correspond to samples
    
    Inputs:
    - samples: (numsamples, Nx,Ny)
    - wf: an instance of the RNNwavefunction class that will be created in the training function.
    - alpha: Renyi index
    - periodicRNN: "YES" or "NO"
    
    """
    numsamples = samples.shape[0]
    #Nx = samples.shape[1]
    #Ny = samples.shape[2]
    local_purities = tf.zeros(numsamples,dtype=tf.float32)
    
    

    steps = ceil(numsamples/25000) #Get a maximum of 25000 configurations in batch size to not allocate too much memory
    
    for i in range(steps):
        
        if i < steps-1:
            cut = slice((i*numsamples)//steps,((i+1)*numsamples)//steps)
        else:
            cut = slice((i*numsamples)//steps,numsamples)
        


        if periodicRNN == "NO":
            log_probs_i = wf.log_probability(samples[cut],inputdim=2)
        elif periodicRNN == "YES":
            log_probs_i = wf.log_probability_periodicRNN(samples[cut],inputdim=2)
            
        if i == 0:
            log_probs = log_probs_i
        else:
            log_probs = tf.concat([log_probs,log_probs_i],0)
        
    
    '''
    log_probs should be a tf.Tensor of shape numsamples.
    '''
    local_purities = tf.math.exp((alpha-1) * log_probs)
    return local_purities # shape numsamples


# ---------------- Running VMC with 2DRNNs -------------------------------------
# Input values are just default values
def run_2DTFIM(wf_T,numsteps = 2*10**4, systemsize_x = 5, systemsize_y = 5, num_units = 50, numsamples = 500, learningrate = 5e-3, seed = 111, temp = 1, FMorAFM = 'FM',BC = 'Periodic',alpha=2, periodicRNN = "NO"):
    '''
    wf_T: an instance of the RNNwavefunction class, trained at the previous temperature.
    systemsize_x = Nx (number of cols), systemsize_y = Ny (number of rows)
    num_units = number of memory units (size of hidden state)
    numsamples = number of samples used for training
    learningrate = self-explanatory
    seed = seed used for random number generators
    temp = temperature
    MorAFM: if == 'FM', then we are working with the FM TFIM. If == 'AFM', then we are working with the AFM TFIM.
    BC = boundary conditions of the Ising model. "Periodic" or "Open"
    alpha = Renyi index
    periodicRNN = "NO" (not using the periodic RNN structure) or "YES" (indeed using the periodic RNN structure)
    '''
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.random.set_seed(seed)  # tensorflow pseudo-random generator

    # Intitializing the RNN-----------
    units=[num_units] #list containing the number of hidden units for each layer of the networks (We only support one layer for the moment)

    Nx=systemsize_x #x dim
    Ny=systemsize_y #y dim
    #N = Nx*Ny
    
    Jz = tf.ones((Nx,Ny)) #Ferromagnetic couplings
    lr=np.float64(learningrate)

    input_dim=2 #Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples_=20 #number of samples only for initialization
    if periodicRNN == "NO":
        wf_T.sample(numsamples_,input_dim) # I think this is equivalent to sampling=wf.sample(numsamples_,input_dim)
    elif periodicRNN == "YES":
        wf_T.sample_periodicRNN(numsamples_,input_dim) 
    
    
    trainable_variables = []
    ''' Tensorflow 1 code:
        
    for cell in wf.rnn:
        trainable_variables.extend(cell.trainable_variables)
    for node in wf.dense:
        trainable_variables.extend(node.trainable_variables)
    '''
    trainable_variables.extend(wf_T.rnn.trainable_variables) # trainable variables always have to be tf.Variables, I think.
    trainable_variables.extend(wf_T.dense.trainable_variables) # we didn't explicitly define the parameters of the dense layer.
    
    # Creating training variables. Initialize global_step to 0
    global_step = tf.Variable(0, name="global_step", trainable=False) # I think this just tracks the # of training steps...
    # tf.Variables automatically assumed to be trainable.
    
    
    # Creating the optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr) # Remember Adam optimizer is a replacement for SGD optimizers


    #path=os.getcwd() # AJ Comment: Python method getcwd() returns current working directory of a process.

    print('Training with numsamples = ', numsamples)
    print('\n')
    
    ending='_units'
    for u in units:
        ending+='_{0}'.format(u) # e.g. ending turns out to be ending_10_10 if there are 2 layers with 10 units each
                                 # However, there is only one layer in this implementation...
    
    # Checkpointing Essentials
    
    # First we need the temperature filename
    filename = './Check_Points/2DClassicalIsingModel/RNNwavefunction_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_temp'+'{:.2f}'.format(temp)+ending
    savename = '_2D_ClassicalIsingModel'
    
    if not os.path.exists(filename): # if the folders/directories don't already exist, create them
        os.makedirs(filename)

    # Create a TF function for the training step.
    @tf.function()
    def train_step():
        print("TRAIN_STEP")
        with tf.GradientTape() as tape:
            # Start by generating samples
            if periodicRNN == "NO":
                samples = wf_T.sample(numsamples,input_dim)
                log_probs_samples = wf_T.log_probability(samples,input_dim)
            elif periodicRNN == "YES":
                samples = wf_T.sample_periodicRNN(numsamples,input_dim)
                log_probs_samples = wf_T.log_probability_periodicRNN(samples,input_dim)
            
            # "local energies". In the classical Ising model, these are just the energies of the configurations.
            Eloc = Ising2D_energies(Jz, Nx, Ny, samples, FMorAFM, BC)
            
            # local purities
            Ploc = Ising2D_local_purities(samples,wf_T,alpha,periodicRNN)
            
            
            
            # cost function - i.e. the expression to be differentiated. tf.stop_gradient means don't differentiate the argument
            # Including variance reduction terms
            cost = tf.reduce_mean(
                tf.multiply(log_probs_samples,tf.stop_gradient(Eloc))) - tf.reduce_mean(
                tf.stop_gradient(Eloc)) * tf.reduce_mean(log_probs_samples) - (alpha*temp/(1-alpha)) * (1 / (tf.reduce_mean(
                tf.stop_gradient(Ploc))) * tf.reduce_mean(tf.multiply(log_probs_samples,tf.stop_gradient(Ploc))) - tf.reduce_mean(
                log_probs_samples))
                    
            #cost = tf.reduce_mean(
            #    tf.multiply(log_probs_samples,tf.stop_gradient(Eloc))) - tf.reduce_mean(
            #    tf.stop_gradient(Eloc))*tf.reduce_mean(log_probs_samples1) + (2*temp/tf.reduce_mean(
            #    tf.stop_gradient(Ploc))) * tf.reduce_mean(tf.multiply(log_probs_samples1,tf.stop_gradient(Ploc)))

        gradients = tape.gradient(cost, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables), global_step=global_step) # this is the step (updates the parameters)
        return cost, Eloc, Ploc # Eloc are just the local energies
    #cost, Eloc, Ploc = train_step()

    # Checkpointing
    '''
    The whole point here is to store the model and not have to restart the optimization in the event of the job getting killed, as sometimes
    happens on a given cluster
    '''
    # all tf.Variables are "kwags" (keyword arguments) of tf.train.Checkpoint
    # ckpt clearly stores the network weights/parameters, i.e. the trainable_variables
    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, variables=trainable_variables)
    manager = tf.train.CheckpointManager(ckpt, filename, max_to_keep=1) # max_to_keep = number of checkpoints to keep.

    
    # Load old model / initialize lists to store data if there is no old model to load
    # .restore is what restores the trainable_variables to the values they were at when the job was killed...
    #if checkpoint:
    ckpt.restore(manager.latest_checkpoint) 
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        print(f"Continuing at step {ckpt.step.numpy()}")

        # To store data
        meanEnergy=list(np.load('meanEnergy_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy'))
        meanPurity=list(np.load('meanPurity_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy'))
        meanFreeEnergy=list(np.load('meanFreeEnergy_2DRNN_'+str(Nx)+'x'+ str(Ny)+'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy'))
        varEnergy=list(np.load('varEnergy_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy'))
        varPurity=list(np.load('varPurity_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy'))
        #varFreeEnergy = []
    else: # no checkpoint so initialize lists
        # To store data
        meanEnergy=[]
        meanPurity=[]
        meanFreeEnergy=[]
        varEnergy=[]
        varPurity=[]
        #varFreeEnergy = []
        print("Initializing from scratch.")

    
    # Training Loop
    while True:
        start_time_trainingStep = time.time() # to measure time it takes to compute the gradient, update the parameters, etc.
        
        it = global_step.numpy() # where exactly is global_step incremeneted? Why does this happen automatically?        
        #global step gets updated every iteration of this loop
        
        if it == numsteps: 
            break
        
        cost, Eloc, Ploc = train_step() # the parameters are also updated in the train_step() function (see apply_gradients)
        
        total_time_trainingStep = time.time() - start_time_trainingStep
        
        if it%2000==0:
            print()
            print("Training Step Time =",total_time_trainingStep) # every  500 training steps, pring the time it took to compute the gradients
                                                                  # and update the parameters
            print()
        
        meanE = np.mean(Eloc)
        meanP = np.mean(Ploc)
        meanF = meanE - temp / (1-alpha) * np.log(meanP)
        varE = np.var(Eloc)
        varP = np.var(Ploc)
        

        #adding elements to be saved
        meanEnergy.append(meanE)
        meanPurity.append(meanP)
        meanFreeEnergy.append(meanF)
        varEnergy.append(varE)
        varPurity.append(varP)

        # Every 2000 hundred steps, print the energy, purity and free energy, to see where we are at in the optimization process.
        if it%2000==0:
            print()
            print("T =", temp)
            print()
            print("it =",it)
            print()
            print("<H> =",meanE)
            print("Purity =",meanP)
            print()
            print('mean(F): {0}, var(E): {1}, var(P): {2}, #samples {3}, #Step {4} \n\n'.format(meanF,varE,varP,numsamples,it))
            #print()

        
        # Save the performance, every 100 steps.
        if int(ckpt.step) % 100 == 0: #ckpt.step is the global_step which updates after every iteration
            # Saving the per
            np.save('meanEnergy_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy', meanEnergy)
            np.save('meanPurity_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy', meanPurity)
            np.save('meanFreeEnergy_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy', meanFreeEnergy)
            np.save('varEnergy_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy', varEnergy)
            np.save('varPurity_2DRNN_'+str(Nx)+'x'+ str(Ny) +'_lr'+'{:.4f}'.format(lr)+'_samp'+str(numsamples)+'_units'+str(num_units)+'_seed'+str(seed)+'_temp'+'{:.3f}'.format(temp) + BC + FMorAFM + '_alpha'+'{:.2f}'.format(alpha) + savename +'.npy', varPurity)
            filename = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), filename))
            
            
            
        
    return wf_T

