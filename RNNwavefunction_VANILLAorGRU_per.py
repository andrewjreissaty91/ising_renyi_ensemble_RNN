import tensorflow as tf

"""
This implementation is based on RNN Wave Functions code of https://github.com/mhibatallah/RNNWavefunctions
Adapted by Andrew Jreissaty for Tensorflow 2.0 and for the problem at hand (2D RNN, periodic RNN structure)
"""

'''
Vanilla Cell
'''
class MDRNNcell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """An implementation of the most basic Vanilla RNN cell. 2DRNN for bulk sites, 4DRNN for edge sites.
        Args:
            num_units (int): The number of units in the RNN cell, hidden layer size.
            num_in: Input vector size, input layer size.
    """
    def __init__(self, num_units = None, num_in = None, name=None, dtype = None, reuse=None):
        super(MDRNNcell, self).__init__(_reuse=reuse, name=name)

        self._num_in = num_in # dimension of input vector
        self._num_units = num_units # number of hidden units
        self._state_size = num_units # dimensions of hidden state = num_units
        self._output_size = num_units # dimension of output of RNN cell is also num_units, clearly (output = hidden state)
        
        self.Wh = tf.Variable(name="Wh_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units, num_units]),
                             dtype = dtype,trainable=True)
        
        self.Uh = tf.Variable(name="Uh_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_in, num_units]),
                             dtype = dtype,trainable=True)
        
        self.Wv = tf.Variable(name="Wv_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units, num_units]),
                             dtype = dtype,trainable=True)
        
        self.Uv = tf.Variable(name="Uv_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_in, num_units]),
                             dtype = dtype,trainable=True)
        
        # Bias vector
        self.b = tf.Variable(name="b_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units]),
                             dtype = dtype,trainable=True)
        
        '''
        Additional weights needed for boundary sites, which receive four inputs in a periodic RNN structure
        '''
        
        self.Wx = tf.Variable(name="Wx_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units, num_units]),
                             dtype = dtype,trainable=True)
        
        self.Ux = tf.Variable(name="Ux_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_in, num_units]),
                             dtype = dtype,trainable=True)

        self.Wy = tf.Variable(name="Wy_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units, num_units]),
                             dtype = dtype,trainable=True)
        
        self.Uy = tf.Variable(name="Uy_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_in, num_units]),
                             dtype = dtype,trainable=True)
    
    @property # this property here is essential for variable training
    def trainable_variables(self):
        return [self.Wh, self.Uh, self.Wv, self.Uv, self.Wx, self.Ux, self.Wy, self.Uy, self.b]
    
    # needed properties
    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real
    
    # Call the RNN cell

    def call(self, inputs, states):
        
        input_mul_horizontal_1 = tf.matmul(inputs[0], self.Uh) # [batch_sz, num_units] #Horizontal
        input_mul_vertical_1 = tf.matmul(inputs[1], self.Uv) # [batch_sz, num_units] #Vertical

        state_mul_horizontal_1 = tf.matmul(states[0], self.Wh)  # [batch_sz, num_units] #Horizontal
        state_mul_vertical_1 = tf.matmul(states[1], self.Wv) # [batch_sz, num_units] #Vectical
        
        if len(inputs) == 2: # Bulk spin
        
            preact = input_mul_horizontal_1 + state_mul_horizontal_1 + input_mul_vertical_1 + state_mul_vertical_1 + self.b #Calculating the preactivation
            output = tf.nn.elu(preact) # [batch_sz, num_units] # output is of shape (numsamples,num_units)

            new_state = output
            
        elif len(inputs) == 4: # Boundary spin treated periodically
            '''
            = 1 means treat the boundary spins periodically in terms of the structure of the RNN. The implications is that inputs now
            has 4 elements and so does states.
            '''
            input_mul_horizontal_2 = tf.matmul(inputs[2], self.Ux) # [batch_sz, num_units] #Horizontal (other side)
            input_mul_vertical_2 = tf.matmul(inputs[3], self.Uy) # [batch_sz, num_units] #Vertical (other side)

            state_mul_horizontal_2 = tf.matmul(states[2], self.Wx)  # [batch_sz, num_units] #Horizontal (other side)
            state_mul_vertical_2 = tf.matmul(states[3], self.Wy) # [batch_sz, num_units] #Vectical (other side)

            preact = input_mul_horizontal_1 + state_mul_horizontal_1 + input_mul_vertical_1 + state_mul_vertical_1 + \
                     input_mul_horizontal_2 + state_mul_horizontal_2 + input_mul_vertical_2 + state_mul_vertical_2 + \
                     self.b
                     
            output = tf.nn.elu(preact) # [batch_sz, num_units] # output is of shape (numsamples,num_units)

            new_state = output 

        return output, new_state # 2 copies of the hidden state of shape numsamples, num_units

'''
RNN with GRU
'''
class MDRNNGRUcell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """An implementation of a GRU cell, based on https://arxiv.org/pdf/2207.14314.pdf (see page 6)
        Args:
            num_units (int): The number of units in the RNN cell, hidden layer size.
            num_in: Input vector size, input layer size.
    """
    def __init__(self, num_units = None, num_in = None, name=None, dtype = None, reuse=None):
        super(MDRNNGRUcell, self).__init__(_reuse=reuse, name=name)

        self._num_in = num_in # dimension of input vector
        self._num_units = num_units # number of hidden units
        self._state_size = num_units # dimensions of hidden state = num_units, clearly
        self._output_size = num_units # dimension of output of RNN cell is also num_units, clearly (output = hidden state)
        
        '''
        # TENSORS AND MATRICES TO DEFINE ARE THE SAME AS IN THIS PAPER:
            
        https://arxiv.org/pdf/2207.14314.pdf (see page 6)
        '''
        self.T = tf.Variable(name="T_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units, 2*num_in, 2*num_units]),
                             dtype = dtype,trainable=True)
        
        self.Tg = tf.Variable(name="Tg_" + name,
                              initial_value=tf.keras.initializers.GlorotNormal()([num_units, 2*num_in, 2*num_units]),
                              dtype = dtype,trainable=True)
        
        self.W = tf.Variable(name="W_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units, 2*num_units]),
                             dtype = dtype,trainable=True)
        
        '''
        Need new tensors to account for boundary spins in a periodic RNN
        '''
        self.T_prime = tf.Variable(name="T_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units, 2*num_in, 2*num_units]),
                             dtype = dtype,trainable=True)
        
        self.Tg_prime = tf.Variable(name="Tg_" + name,
                              initial_value=tf.keras.initializers.GlorotNormal()([num_units, 2*num_in, 2*num_units]),
                              dtype = dtype,trainable=True)
        
        self.W_prime = tf.Variable(name="W_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units, 2*num_units]),
                             dtype = dtype,trainable=True)

        
        # Don't forget the b and bg bias vectors
        self.b = tf.Variable(name="b_" + name,
                             initial_value=tf.keras.initializers.GlorotNormal()([num_units]),
                             dtype = dtype,trainable=True)
        
        self.bg = tf.Variable(name="bg_" + name,
                              initial_value=tf.keras.initializers.GlorotNormal()([num_units]),
                              dtype = dtype,trainable=True)
        
    
    @property
    def trainable_variables(self):
        return [self.T, self.Tg, self.W, self.T_prime, self.Tg_prime, self.W_prime, self.b, self.bg]
    
    # needed properties
    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real
    
    # Call RNN

    def call(self, inputs, states):
        # inputs[k], [=] numsamples x num_in, for k = 0,1,2
        # states [=] numsamples x num_units
        
        inputs_concat = tf.concat([inputs[0],inputs[1]],axis=1) # shape numsamples x (2*num_in)
        states_concat = tf.concat([states[0],states[1]],axis=1) # shape numsamples x (2*num_units)
        
        # Contracting a tensor of shape (num_units, 2*num_in, 2*num_units) with tensor of shape
        # (numsamples, 2*num_units) to leave us with a tensor of shape (num_units, 2*num_in, numsamples).
        T_states_concat = tf.tensordot(self.T,states_concat,axes=[[2],[1]]) # shape (num_units, 2*num_in, numsamples)
        # axes=[[2],[1]] --> meaning contract over index [2] of T and index [1] of states_concat.
        
        # inputs_concat = shape (numsamples, 2*num_in): ij
        # T_states_concat = shape (num_units, 2*num_in, numsamples): kji
        # i = numsamples, j = 2*num_in, k = num_units... we want to end up with shape ik = (numsamples, num_units)
        inputs_concat_T_states_concat = tf.einsum('ij,kji->ik',inputs_concat,T_states_concat) # shape (numsamples, num_units)
        
        # Contracting a tensor of shape (num_units, 2*num_in, 2*num_units) with tensor of shape
        # (numsamples, 2*num_units) to leave us with a tensor of shape (num_units,2*num_in, numsamples).
        Tg_states_concat = tf.tensordot(self.Tg,states_concat,axes=[[2],[1]]) # shape (num_units, 2*num_in, numsamples)
        inputs_concat_Tg_states_concat = tf.einsum('ij,kji->ik',inputs_concat,Tg_states_concat) # shape (numsamples, num_units)        
        W_states_concat = tf.tensordot(states_concat,self.W,axes=[[1],[1]]) # shape (numsamples, num_units)
        
        
        if len(inputs) == 2:
            '''
            Bulk spin
            '''
            state_tilde = tf.nn.tanh(inputs_concat_T_states_concat + self.b)  # shape (numsamples, num_units)
            u = tf.nn.sigmoid(inputs_concat_Tg_states_concat + self.bg)  # shape (numsamples, num_units)
            
            new_state = u * state_tilde + (1. - u) * W_states_concat # these are element wise operations. Result has shape
                                                                     # (numsamples, num_units)
            output = new_state
        
        elif len(inputs) == 4:
            '''
            Boundary spin with periodic BC reflected in cell structure
            '''
            inputs_concat_prime = tf.concat([inputs[2],inputs[3]],axis=1) # shape numsamples x (2*num_in)
            states_concat_prime = tf.concat([states[2],states[3]],axis=1) # shape numsamples x (2*num_units)
            
            T_states_concat_prime = tf.tensordot(self.T_prime,states_concat_prime,axes=[[2],[1]]) # shape (num_units, 2*num_in, numsamples)
            
            inputs_concat_T_states_concat_prime = tf.einsum('ij,kji->ik',inputs_concat_prime,T_states_concat_prime) # shape (numsamples, num_units)
            
            state_tilde = tf.nn.tanh(inputs_concat_T_states_concat + inputs_concat_T_states_concat_prime + self.b)  # shape (numsamples, num_units)
            
            Tg_states_concat_prime = tf.tensordot(self.Tg_prime,states_concat_prime,axes=[[2],[1]]) # shape (num_units, 2*num_in, numsamples)
            inputs_concat_Tg_states_concat_prime = tf.einsum('ij,kji->ik',inputs_concat_prime,Tg_states_concat_prime) # shape (numsamples, num_units)
            
            u = tf.nn.sigmoid(inputs_concat_Tg_states_concat + inputs_concat_Tg_states_concat_prime + self.bg)  # shape (numsamples, num_units)
            
            W_states_concat_prime = tf.tensordot(states_concat_prime,self.W_prime,axes=[[1],[1]]) # shape (numsamples, num_units)
            
            new_state = u * state_tilde + (1. - u) * (W_states_concat + W_states_concat_prime) # these are element wise operations. 
                                                                                               # Result has shape (numsamples, num_units)
            output = new_state

        return output, new_state # 2 copies of the hidden state of shape (numsamples, num_units)
            
        



# OK I don't think much needs to change here. We need to change the MDRNNcell.py code to have each RNN cell be able to accept
# three hidden states and three spin states as inputs.
class RNNwavefunction(object):
    def __init__(self,systemsize_x: int, systemsize_y:int,cell=MDRNNGRUcell,units=[10],scope='RNNwavefunction',seed = 111):
        """
            systemsize_x:  int
                         number of sites for x-axis
            systemsize_y:  int
                         number of sites for y-axis         
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            seed:        pseudo-random number generator 
        """
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #size of x direction in the 2d model (i.e. number of cols)
        self.Ny=systemsize_y
        
        self.rnn=cell(num_units = units[0], num_in = 2 ,name="rnn_"+str(0), dtype=tf.float32) # num_in = dimension of input vector
        self.dense = tf.keras.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense', dtype = tf.float32)
              
              

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int
                             samples to be produced
                             
            inputdim:        int
                             hilbert space dimension of one spin

            ------------------------------------------------------------------------
            Returns:      

            samples:         Originally, this was a tf.Tensor of shape (numsamples,systemsize_x, systemsize_y)
                             the samples in integer encoding.
                             ...
                             NOW, we need a 4th dimension because each sample is comprised of one 2D lattice stacked on top of the other.
                             So I think samples is a tf.Tensor of shape (numsamples,systemsize_x,systemsize_y,2)
        """


        self.inputdim=inputdim
        self.outputdim=self.inputdim
        self.numsamples=numsamples

        samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
        rnn_states = {} # initializing a dictionary
        inputs = {} # initializing a dictionary

        # Zero states / inputs
        for ny in range(self.Ny): #Loop over the boundary
            if ny%2==0:
                nx = -1 # i.e. the zero input vector / hidden state being fed into the cell will come from the left, at a position 0-1=-1
                rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) # Roeland uses float32

            if ny%2==1:
                nx = self.Nx
                # print(nx,ny)
                rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)

        for nx in range(self.Nx): #Loop over the boundary
            ny = -1
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)

                
        
        '''
        Now, we have defined all the zero hidden states and spin states that are fed into the 2D layer/lattice along
        all boundaries (x-,y-boundaries)
        '''
    

        # OK ready for sampling.
        for ny in range(self.Ny):
            
            if ny%2 == 0:

                for nx in range(self.Nx): #left to right
                    
                    rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx-1)+str(ny)],
                                                                        inputs[str(nx)+str(ny-1)]),
                                                                       (rnn_states[str(nx-1)+str(ny)],
                                                                        rnn_states[str(nx)+str(ny-1)]))

                    output=self.dense(rnn_output) # output has shape (numsamples, 2), each row is a pair of probabilities
                    sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1, ])
                    samples[ny][nx] = sample_temp # samples was a list of lists... I get it. sample_temp is a tf.tensor of shape numsamples
                    inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float32)


            if ny%2 == 1:

                for nx in range(self.Nx-1,-1,-1): #right to left # AJ Comment: makes sense
                    # Basically the same idea as above
                    rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx+1)+str(ny)],
                                                                        inputs[str(nx)+str(ny-1)]),
                                                                       (rnn_states[str(nx+1)+str(ny)],
                                                                        rnn_states[str(nx)+str(ny-1)]))
                    
                    
                    output=self.dense(rnn_output)
                    
                    sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1, ])
                    samples[ny][nx] = sample_temp
                    inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float32)

        self.samples=tf.transpose(tf.stack(values=samples,axis=0), perm = [2,1,0])
        return self.samples
    
    
    
    def sample_periodicRNN(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int                             
                             samples to be produced
                             
            inputdim:        int
                             hilbert space dimension of one spin

            ------------------------------------------------------------------------
            Returns:      

            samples:         Originally, this was a tf.Tensor of shape (numsamples,systemsize_x, systemsize_y)
                             the samples in integer encoding.
                             ...
                             NOW, we need a 4th dimension because each sample is comprised of one 2D lattice stacked on top of the other.
                             So I think samples is a tf.Tensor of shape (numsamples,systemsize_x,systemsize_y,2)
        """

        #Initial input to feed to the 2drnn

        self.inputdim=inputdim
        self.outputdim=self.inputdim
        self.numsamples=numsamples

        '''
        Removing the extra dimension for the 2DRNN
        '''
        #samples=[[[[] for nz in range(2)] for nx in range(self.Nx)] for ny in range(self.Ny)]
        samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
        rnn_states = {} # initializing a dictionary
        inputs = {} # initializing a dictionary
        
        '''
        Initialization of zero states. It's more complicated for the "periodicRNN". The assumption is:
        Sampling starts from the top left spin and moves from left to right in the first row, right to left
        in the second row, left to right again in the third row, etc.
        '''
        # For the periodic RNN as per Mohamed's strategy in the topological order paper, there are many spins that must
        # be initialized to zero.
        # TOP BOUNDARY, except first spin that we sample (top left)
        ny = 0
        for nx in range(1,self.Nx):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        # BOTTOM BOUNDARY
        ny = self.Ny - 1
        for nx in range(self.Nx):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        #LEFT BOUNDARY
        nx = 0
        for ny in range(self.Ny):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        #RIGHT BOUNDARY
        nx = self.Nx - 1
        for ny in range(self.Ny):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        # Next, we need the full second row initialized to 0 (because it feeds the top boundary). The edge spins
        # have already been initialized to 0, hence range(1,self.Nx-1).
        ny = 1
        for nx in range(1,self.Nx-1):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
            
        # Finally, for row 2, 4, 6, etc. (where the top row is row 0) the second spin must be initialized to zero.
        # For row 3, 5, 7. etc. the penultimate spin (counting from the left) must be initialized to zero.
        nx = 1
        for ny in range(2,self.Ny-1,2):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units                                                                              
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)

        nx = self.Nx - 2
        for ny in range(3,self.Ny-1,2):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units                                                                              
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)

        
        
        '''
        Sampling
        '''

        for ny in range(self.Ny):
            
            if ny%2 == 0:

                for nx in range(self.Nx): #left to right
                    if (ny == 0) or (ny == self.Ny - 1) or (nx == 0) or (nx == self.Nx - 1):
                        # Then we have a boundary spin, so 4 inputs/states into the RNN
                        #self.Periodic_Boundary_Spin_RNN = 1
                        
                        # rnn_output, rnn_states[...], these are tf.tensors. rnn_state[] is a tf.tensor of shape (numsamples,num_units)
                        # rnn_output is another copy of rnn_state[]. In this case, self.rnn takes two tuples as arguments, one tuple
                        # containing 3 spin inputs and another containing 3 hidden state inputs
                        ix_left = (nx-1)%self.Nx
                        iy_left = ny
                        ix_top = nx
                        iy_top = (ny-1)%self.Ny
                        ix_right = (nx+1)%self.Nx
                        iy_right = ny
                        ix_bottom = nx
                        iy_bottom = (ny+1)%self.Ny
                        rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(ix_left)+str(iy_left)],
                                                                            inputs[str(ix_top)+str(iy_top)],
                                                                            inputs[str(ix_right)+str(iy_right)],
                                                                            inputs[str(ix_bottom)+str(iy_bottom)]),
                                                                           (rnn_states[str(ix_left)+str(iy_left)],
                                                                            rnn_states[str(ix_top)+str(iy_top)],
                                                                            rnn_states[str(ix_right)+str(iy_right)],
                                                                            rnn_states[str(ix_bottom)+str(iy_bottom)]))
                    else: #Not a Boundary spin- instead, a bulk spin that takes inputs from the left and top spins only
                        #self.Periodic_Boundary_Spin_RNN = 0
                        rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx-1)+str(ny)],
                                                                            inputs[str(nx)+str(ny-1)]),
                                                                           (rnn_states[str(nx-1)+str(ny)],
                                                                            rnn_states[str(nx)+str(ny-1)]))
          
                    output=self.dense(rnn_output) # output has shape (numsamples, 2), each row is a pair of probabilities
                    sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1, ])
                    samples[ny][nx] = sample_temp
                    inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float32)
                    
            
            if ny%2 == 1:

                for nx in range(self.Nx-1,-1,-1): #right to left # AJ Comment: makes sense
                
                    if (ny == self.Ny - 1) or (nx == 0) or (nx == self.Nx - 1):
                        # Then we have a boundary spin
                        #self.Periodic_Boundary_Spin_RNN = 1
                    
                        # rnn_output, rnn_states[...], these are tf.tensors. rnn_state[] is a tf.tensor of shape (numsamples,num_units)
                        # rnn_output is another copy of rnn_state[]. In this case, self.rnn takes two tuples as arguments, one tuple
                        # containing 3 spin inputs and another containing 3 hidden state inputs
                        ix_left = (nx-1)%self.Nx
                        iy_left = ny
                        ix_top = nx
                        iy_top = (ny-1)%self.Ny
                        ix_right = (nx+1)%self.Nx
                        iy_right = ny
                        ix_bottom = nx
                        iy_bottom = (ny+1)%self.Ny
                        rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(ix_left)+str(iy_left)],
                                                                            inputs[str(ix_top)+str(iy_top)],
                                                                            inputs[str(ix_right)+str(iy_right)],
                                                                            inputs[str(ix_bottom)+str(iy_bottom)]),
                                                                           (rnn_states[str(ix_left)+str(iy_left)],
                                                                            rnn_states[str(ix_top)+str(iy_top)],
                                                                            rnn_states[str(ix_right)+str(iy_right)],
                                                                            rnn_states[str(ix_bottom)+str(iy_bottom)]))
                    else: #Not a Boundary spin- instead, a bulk spin that takes inputs from the right and top spins only
                        #self.Periodic_Boundary_Spin_RNN = 0
                        rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx+1)+str(ny)],
                                                                            inputs[str(nx)+str(ny-1)]),
                                                                           (rnn_states[str(nx+1)+str(ny)],
                                                                            rnn_states[str(nx)+str(ny-1)]))
                    
                    output=self.dense(rnn_output)
                    
                    sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1, ])
                    samples[ny][nx] = sample_temp
                    inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float32)


        self.samples=tf.transpose(tf.stack(values=samples,axis=0), perm = [2,1,0])
        return self.samples
    
    
    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of each sample in ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize_x,system_size_y)
                             containing the input samples in integer encoding
                             
            inputdim:        int
                             dimension of the input space

            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        self.inputdim=inputdim
        self.outputdim=self.inputdim
        self.numsamples=tf.shape(samples)[0]
        
        
        samples_=tf.transpose(samples, perm = [1,2,0]) # now samples_ is of shape (Nx,Ny,numsamples)
        rnn_states = {}
        inputs = {}

        # Zero states / input
        for ny in range(self.Ny): #Loop over the boundary
            if ny%2==0:
                nx = -1
                rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32) # Roeland uses float32
                
            if ny%2==1:
                nx = self.Nx
                rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)

        for nx in range(self.Nx): #Loop over the boundary
            ny = -1
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32)
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        
        
        '''
        Now we have created our zero hidden states and spin state inputs that will be fed into RNN cells along the boundary
        '''
        probs=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)] # shape (Ny,Nx,2)
        
        #Begin estimation of log probs
        for ny in range(self.Ny):

            if ny%2 == 0:

                for nx in range(self.Nx): #left to right
                    rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx-1)+str(ny)],
                                                                        inputs[str(nx)+str(ny-1)]),
                                                                       (rnn_states[str(nx-1)+str(ny)],
                                                                        rnn_states[str(nx)+str(ny-1)]))

                    output=self.dense(rnn_output) # output has shape (numsamples, 2)
                    probs[ny][nx] = output
                    inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim, dtype = tf.float32)
                    

            if ny%2 == 1:

                for nx in range(self.Nx-1,-1,-1): #right to left # AJ Comment: makes sense
                    # Basically the same idea as above
                    rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx+1)+str(ny)],
                                                                        inputs[str(nx)+str(ny-1)]),
                                                                       (rnn_states[str(nx+1)+str(ny)],
                                                                        rnn_states[str(nx)+str(ny-1)]))
                    
                    
                    output=self.dense(rnn_output)
                    probs[ny][nx] = output
                    inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim, dtype = tf.float32)
                
        
        probs=tf.transpose(tf.stack(values=probs,axis=0),perm=[2,1,0,3]) 
        one_hot_samples = tf.one_hot(samples,depth=self.inputdim, dtype = tf.float32)
        
        #N = self.Nx * self.Ny
        probs_tf = tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=3) # this is an array of shape numsamples,Nx,Ny
        self.log_probs = tf.reduce_sum(tf.reduce_sum(tf.math.log(probs_tf),axis=2),axis=1)
     
        return self.log_probs
          
    
    def log_probability_periodicRNN(self,samples,inputdim):
        """
            calculate the log-probabilities of each sample in ```samples``, according to the periodic RNN structure
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,systemsize_x,system_size_y,2) (2 = top and bottom lattices)
                             containing the input samples in integer encoding
                             
            inputdim:        int
                             dimension of the input space

            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        self.inputdim=inputdim
        self.outputdim=self.inputdim
        self.numsamples=tf.shape(samples)[0] # This line of code can be changed to samples.shape[0] I think... but maybe not necessary
        

        samples_=tf.transpose(samples, perm = [1,2,0]) # now samples_ is of shape (Nx,Ny,numsamples)
        rnn_states = {}
        inputs = {}

        # Zero states / inputs for periodic RNN
        
        # TOP BOUNDARY, except first spin that we sample (top left)
        ny = 0
        for nx in range(1,self.Nx):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        # BOTTOM BOUNDARY
        ny = self.Ny - 1
        for nx in range(self.Nx):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        #LEFT BOUNDARY
        nx = 0
        for ny in range(self.Ny):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        #RIGHT BOUNDARY
        nx = self.Nx - 1
        for ny in range(self.Ny):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        # Next, we need the full second row initialized to 0 (because it feeds the top boundary). The edge spins
        # have already been initialized to 0, hence range(1,self.Nx-1).
        ny = 1
        for nx in range(1,self.Nx-1):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
            
        # Finally, for row 2, 4, 6, etc. (where the top row is row 0) the second spin must be initialized to zero.
        # For row 3, 5, 7. etc. the penultimate spin (counting from the left) must be initialized to zero.
        nx = 1
        for ny in range(2,self.Ny-1,2):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        nx = self.Nx - 2
        for ny in range(3,self.Ny-1,2):
            rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float32) # shape numsamples x num_units
            inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float32)
        
        
        '''
        Now we have created our zero hidden states and spin state inputs that will be fed into RNN cells along the boundary
        '''
        probs=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)] # shape (Ny,Nx,2)

        
        #Begin estimation of log probs
        for ny in range(self.Ny):

            if ny%2 == 0:

                for nx in range(self.Nx): #left to right
                    if (ny == 0) or (ny == self.Ny - 1) or (nx == 0) or (nx == self.Nx - 1):
                        # Then we have a boundary spin
                        ix_left = (nx-1)%self.Nx
                        iy_left = ny
                        ix_top = nx
                        iy_top = (ny-1)%self.Ny
                        ix_right = (nx+1)%self.Nx
                        iy_right = ny
                        ix_bottom = nx
                        iy_bottom = (ny+1)%self.Ny
                        rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(ix_left)+str(iy_left)],
                                                                            inputs[str(ix_top)+str(iy_top)],
                                                                            inputs[str(ix_right)+str(iy_right)],
                                                                            inputs[str(ix_bottom)+str(iy_bottom)]),
                                                                           (rnn_states[str(ix_left)+str(iy_left)],
                                                                            rnn_states[str(ix_top)+str(iy_top)],
                                                                            rnn_states[str(ix_right)+str(iy_right)],
                                                                            rnn_states[str(ix_bottom)+str(iy_bottom)]))
                    else: #Not a Boundary spin- instead, a bulk spin that takes inputs from the left and top spins only
                        rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx-1)+str(ny)],
                                                                            inputs[str(nx)+str(ny-1)]),
                                                                           (rnn_states[str(nx-1)+str(ny)],
                                                                            rnn_states[str(nx)+str(ny-1)]))
                        
                    output=self.dense(rnn_output) # output has shape (numsamples, 2), each row is a pair of probabilities
                    probs[ny][nx] = output
                    inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim, dtype = tf.float32)
  
                    

            if ny%2 == 1:

                for nx in range(self.Nx-1,-1,-1): #right to left
                    # Basically the same idea as above
                    if (ny == 0) or (ny == self.Ny - 1) or (nx == 0) or (nx == self.Nx - 1):
                        # Then we have a boundary spin
                        ix_left = (nx-1)%self.Nx
                        iy_left = ny
                        ix_top = nx
                        iy_top = (ny-1)%self.Ny
                        ix_right = (nx+1)%self.Nx
                        iy_right = ny
                        ix_bottom = nx
                        iy_bottom = (ny+1)%self.Ny
                        rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(ix_left)+str(iy_left)],
                                                                            inputs[str(ix_top)+str(iy_top)],
                                                                            inputs[str(ix_right)+str(iy_right)],
                                                                            inputs[str(ix_bottom)+str(iy_bottom)]),
                                                                           (rnn_states[str(ix_left)+str(iy_left)],
                                                                            rnn_states[str(ix_top)+str(iy_top)],
                                                                            rnn_states[str(ix_right)+str(iy_right)],
                                                                            rnn_states[str(ix_bottom)+str(iy_bottom)]))
                    else: #Not a Boundary spin- instead, a bulk spin that takes inputs from the right and top spins only
                        rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn((inputs[str(nx+1)+str(ny)],
                                                                            inputs[str(nx)+str(ny-1)]),
                                                                           (rnn_states[str(nx+1)+str(ny)],
                                                                            rnn_states[str(nx)+str(ny-1)]))
                    
                    output=self.dense(rnn_output)
                    probs[ny][nx] = output
                    inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim, dtype = tf.float32)

        probs=tf.transpose(tf.stack(values=probs,axis=0),perm=[2,1,0,3])
        one_hot_samples = tf.one_hot(samples,depth=self.inputdim, dtype = tf.float32)
        #N = self.Nx * self.Ny
        probs_tf = tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=3) # this is an array of shape numsamples,Nx,Ny
        self.log_probs = tf.reduce_sum(tf.reduce_sum(tf.math.log(probs_tf),axis=2),axis=1)
        return self.log_probs # tf.Tensor of shape numsamples.
            
    
    def magPerSpin_abs(self,samples):
        
        """
            Calculate magnetization per spin of each sample in samples.
            ------------------------------------------------------------------------
            Parameters:
            samples:        tf.Tensor
                            a tf.Tensor of shape (number of samples,systemsize_x,system_size_y)
                            containing the input samples in integer encoding


            ------------------------------------------------------------------------
            Returns:
            magPerSpin_samples             tf.Tensor of shape (numsamples)
                                                    
            """
        
        N = self.Nx * self.Ny
        
        upSpins_samples = tf.reduce_sum(tf.reduce_sum(samples,axis=2),axis=1)
        downSpins_samples = N - upSpins_samples
        
        magPerSpin_samples = (upSpins_samples - downSpins_samples) / N
        magPerSpin_abs_samples = tf.math.abs(magPerSpin_samples)
        return magPerSpin_abs_samples # tf.Tensor of shape (numsamples)
    
    