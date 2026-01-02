@keras.utils.register_keras_serializable(name="EfmLSTM")

class EfmLSTM(Layer):

    def __init__(
        self,
        units: int,
        signature_depth: int = 2,
        signature_input_size: int = 5,
        return_sequences: bool = False,
        return_state: bool = False,
        unroll_level: Union[bool,int] = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.signature_depth = signature_depth
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.signature_input_size = signature_input_size

        # Will be set in build()
        self.state_size = units
        self.signature_dim = None
        self.forget_kernel = None  # Kernel for signature-based forget gate
        self.input_kernel = None   # Standard kernels for other gates
        self.recurrent_kernel = None
        self.bias = None
        self.unroll_level = unroll_level

    def build(self, input_shape):
        batch_size, seq_len, features = input_shape

        # Calculate signature dimension
        self.signature = SigLayer(self.signature_depth, stream=True)

        self.signature.build((batch_size, seq_len, self.signature_input_size))
        self.signature_dim = int(ta.number_of_words_up_to_trunc(self.signature_depth, signature_input_size))  # number of signature elements of order <= trunc)

        # Kernel for signature-based forget gate
        self.forget_kernel = self.add_weight(
            shape=(self.signature_dim, self.units),
            initializer='glorot_uniform',
            name='forget_kernel'
        )

        # Standard LSTM kernels for input, cell, and output gates
        self.input_kernel = self.add_weight(
            shape=(features, self.units * 3),  # 3 because forget gate is handled separately
            initializer='glorot_uniform',
            name='input_kernel'
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),  # 3 because forget gate is handled separately
            initializer='glorot_uniform',
            name='recurrent_kernel'
        )

        # Bias for all gates (including forget gate)
        self.bias = self.add_weight(
            shape=(self.units * 4,),  # Still 4 for all gates
            initializer='zeros',
            name='bias'
        )
        self.linear_preprocess_inputs_for_sig = Dense(self.signature_input_size, 'linear', use_bias=False)## Permet de faire la linearisation des inputs
        self.linear_preprocess_inputs_for_sig.build(input_shape)
        super().build(input_shape)

    def get_initial_state(self, inputs): ## Inputs doit etre jnp.array : initialise les gate (hiddenet cell state)
        batch_size = ops.shape(inputs)[0]
        return [
            ops.zeros((batch_size, self.units)),  # hidden state
            ops.zeros((batch_size, self.units))   # cell state
        ]

    def _normalize_signature_by_time(self, signatures): # Signature doit etre aussi np.array : Normalise la signature
        """Normalize signature values by dividing by time index."""
        seq_length = signatures.shape[1]
        time_indices = ops.arange(1, seq_length + 1, dtype=signatures.dtype)
        time_indices = ops.reshape(time_indices, (1, -1, 1))
        # Add zero signature for first timestep
        return ops.concatenate([
            ops.zeros_like(signatures[:,:1,:]),
            signatures / time_indices
        ], axis=1)

    def call(self, inputs, initial_state=None, training=None):
        if True or keras.backend.backend() == 'tensorflow':
            return self._call_tensorflow(inputs, initial_state=initial_state, training=training)
        else:
            return self._call_generic(inputs, initial_state=initial_state, training=training)
#############################CALCUL DE SIGNATURE######################################
    ## Calculs des signature avec batch
   def sig_one_path(self,input):
        return ta.path_to_fm_signature(
        path=input,
        trunc=1,
        t_grid= jnp.linspace(0, 1,input.shape[1] ),
        lam=lam
    ) #### Calcul des signatures chemin

    def EfmLayer(self,input): # A retester
      # Prend en compte , le batch, time_step
      T= input.shape[1] ## Il prend le nombre de ligne de l'input (la base de donnée: 1 car la base est un jnp.array)
      t_grid = np.linspace(0, input.shape[0], T)         # Il va creer des temps afin de l'ajouter dans la path
      t_grid1 = t_grid[None, :, None]         # (1, 4, 1)
      t_grid1 = np.repeat(t_grid1, 2, axis=0)  # (
      path_with_time =jnp.concatenate([t_grid1, input], axis=2)
      sig_batch = jax.vmap(sig_one_path)(path_with_time) ## ca sort un tensor avec le clacul des batchs mais le 1 n'apparait pas attention.


      return sig_batch.array

""" paths = jnp.array([
    [[0.,0],[0.5,0.2],[0.8,0.4],[1.,0.5]],
    [[0.,1],[0.2,0.34],[0.8,0.9],[1,0.5]]
])"""



######################

    def _call_tensorflow(self, inputs, initial_state=None, training=None):
        """Process inputs using hybrid signature/standard LSTM computation.

        Args:
            inputs: Input tensor of shape (batch_size, time_steps, features)
            initial_state: Optional initial state tuple (h_0, c_0)

        Returns:
            If return_sequences=False: Final output (batch_size, units)
            If return_sequences=True: Full sequence (batch_size, time_steps, units)
            If return_state=True: Tuple of (output, h_n, c_n)
        """
        # Compute signatures for forget gate
        signatures = self.EfmLayer(self.linear_preprocess_inputs_for_sig(inputs)) ## A tester
        normalized_signatures = self._normalize_signature_by_time(signatures) ## a tester



        # Get sequence length and pre-compute input transformations for all timesteps
        time_steps = ops.shape(inputs)[1] ## Le nombre de ligne par batch

        # Pre-compute all input transformations at once using einsum
        all_x_transformed = ops.einsum('bti,ij->btj', inputs, self.input_kernel) ## a tester

        # Initialize states
        if initial_state is None:
            h_tm1, c_tm1 = self.get_initial_state(inputs)
        else:
            h_tm1, c_tm1 = initial_state

        # Initialize list for sequence outputs if needed
        sequence_outputs = [] if self.return_sequences else None

        # Get biases for each gate
        b_i, b_f, b_c, b_o = ops.split(self.bias, 4)

        # Process sequence
        for t in range(time_steps):
            # Get current signature and pre-computed input transform
            current_sig = normalized_signatures[:, :, t] ## Selectionne la t ieme colonne par batch car mes signature par pbatch sont en colonne
            current_x_transformed = all_x_transformed[:, t] ### Je dois revoir ca

            # Standard LSTM computations for input, cell, and output gates
            h_tm1_transformed = ops.dot(h_tm1, self.recurrent_kernel)

            # Combine transformations for standard gates
            gates_standard = ops.add(h_tm1_transformed, current_x_transformed)

            # Split standard gates
            i_t, c_t, o_t = ops.split(gates_standard, 3, axis=-1)

            # Compute forget gate using signature
            f_t = ops.dot(current_sig, self.forget_kernel) + b_f

            # Apply activations
            i_t = ops.sigmoid(i_t + b_i)      # input gate
            f_t = ops.sigmoid(f_t)            # forget gate (signature-based)
            c_t = ops.tanh(c_t + b_c)        # cell gate
            o_t = ops.sigmoid(o_t + b_o)      # output gate

            # Update cell state
            c_tm1 = f_t * c_tm1 + i_t * c_t

            # Update hidden state
            h_tm1 = o_t * ops.tanh(c_tm1)

            # Store output if returning sequences
            if self.return_sequences:
                sequence_outputs.append(h_tm1)

        if self.return_sequences:
            final_outputs = ops.stack(sequence_outputs, axis=1)
        else:
            final_outputs = h_tm1

        if self.return_state:
            return final_outputs, h_tm1, c_tm1
        return final_outputs


 def _call_generic(self, inputs, initial_state=None, training=None):
        """Process inputs using hybrid signature/standard LSTM computation with scan operations.

        Args:
            inputs: Input tensor of shape (batch_size, time_steps, features)
            initial_state: Optional initial state tuple (h_0, c_0)

        Returns:
            If return_sequences=False: Final output (batch_size, units)
            If return_sequences=True: Full sequence (batch_size, time_steps, units)
            If return_state=True: Tuple of (output, h_n, c_n)
        """
        # Compute signatures for forget gate
        signatures = self.EfmLayer(self.linear_preprocess_inputs_for_sig(inputs)) ## A tester
        normalized_signatures = self._normalize_signature_by_time(signatures) ## a tester

        # Pre-compute input transformations for all timesteps
        all_x_transformed = ops.einsum('bti,ij->btj', inputs, self.input_kernel) ### A REVOIR

        # Initialize states
        if initial_state is None:
            initial_h, initial_c = self.get_initial_state(inputs)
        else:
            initial_h, initial_c = initial_state

        # Get biases for each gate
        b_i, b_f, b_c, b_o = ops.split(self.bias, 4)

        # Define scan function for processing sequence
        def step_fn(states, step_inputs):
            h_prev, c_prev = states
            current_sig, current_x_transformed = step_inputs

            # Standard LSTM computations
            h_prev_transformed = ops.dot(h_prev, self.recurrent_kernel)
            gates_standard = ops.add(h_prev_transformed, current_x_transformed)

            # Split standard gates
            i_t, c_t, o_t = ops.split(gates_standard, 3, axis=-1)

            # Compute forget gate using signature
            f_t = ops.dot(current_sig, self.forget_kernel) + b_f

            # Apply activations
            i_t = ops.sigmoid(i_t + b_i)
            f_t = ops.sigmoid(f_t)
            c_t = ops.tanh(c_t + b_c)
            o_t = ops.sigmoid(o_t + b_o)

            # Update states
            c_new = f_t * c_prev + i_t * c_t
            h_new = o_t * ops.tanh(c_new)

            if self.return_sequences:
                return (h_new, c_new), h_new
            return (h_new, c_new), None

        # Prepare inputs for scan
        scan_inputs = (
            ops.moveaxis(normalized_signatures, 1, 0),  # (time_steps, batch, sig_dim)
            ops.moveaxis(all_x_transformed, 1, 0)      # (time_steps, batch, transformed_dim)
        )

        # Process sequence using scan
        (final_h, final_c), sequence = ops.scan(
            step_fn,
            init=(initial_h, initial_c),
            xs=scan_inputs,
            unroll=self.unroll_level,
            length=ops.shape(inputs)[1]
        )

        if self.return_sequences:
            # Restore time axis to middle position
            outputs = ops.moveaxis(sequence, 0, 1)
        else:
            outputs = final_h

        if self.return_state:
            return outputs, final_h, final_c
        return outputs

def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'signature_depth': self.signature_depth,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'unroll_level': self.unroll_level
        })
        return config

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        if input_shape is None or input_shape[0] is None:
            batch_size = None
        else:
            batch_size = input_shape[0]

        if self.return_sequences:
            output_shape = (batch_size, input_shape[1], self.units)
        else:
            output_shape = (batch_size, self.units)

        if self.return_state:
            return [output_shape, (batch_size, self.units), (batch_size, self.units)]
        return output_shape
