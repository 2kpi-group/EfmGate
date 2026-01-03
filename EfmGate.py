import keras
from keras import ops
from keras.layers import Layer, Dense
import numpy as np
from typing import Optional, Tuple, Union



@keras.utils.register_keras_serializable(name="EfmLSTM")
class EfmLSTM(Layer):

    def __init__(
        self,
        units: int,
        signature_depth: int = 2,
        signature_input_size: int = 5,
        return_sequences: bool = False,
        return_state: bool = False,
        unroll_level: Union[bool, int] = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.signature_depth = signature_depth
        self.signature_input_size = signature_input_size
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.unroll_level = unroll_level

        self.state_size = units
        self.signature_dim = None
        self.forget_kernel = None
        self.input_kernel = None
        self.recurrent_kernel = None
        self.bias = None

    def build(self, input_shape):
        batch_size, seq_len, features = input_shape

       # self.signature = SigLayer(self.signature_depth, stream=True)
       # self.signature.build((batch_size, seq_len, self.signature_input_size))

        self.signature_dim = int(
            ta.number_of_words_up_to_trunc(
                self.signature_depth,
                self.signature_input_size
            )
        )

        self.forget_kernel = self.add_weight( # self.signature_dim → taille de l’entrée.
            shape=(self.signature_dim, self.units), # self.units → nombre de neurones ou d’unités dans le layer
            initializer="glorot_uniform",
            name="forget_kernel"
        )

        self.input_kernel = self.add_weight(
            shape=(features, self.units * 3),
            initializer="glorot_uniform",
            name="input_kernel"
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            initializer="glorot_uniform",
            name="recurrent_kernel"
        )

        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer="zeros",
            name="bias"
        )

        self.linear_preprocess_inputs_for_sig = Dense(
            self.signature_input_size,
            activation="linear",
            use_bias=False
        )
        self.linear_preprocess_inputs_for_sig.build(input_shape)

        super().build(input_shape)

    def get_initial_state(self, inputs):
        inputs = jnp.asarray(inputs) # 1
        batch_size = ops.shape(inputs)[0]
        return [
            ops.zeros((batch_size, self.units)),
            ops.zeros((batch_size, self.units)),
        ]

    def _normalize_signature_by_time(self, signatures):
        seq_length = signatures.shape[1] ## Donc avant la normalisation  pas de transposer
        time_indices = ops.arange(1, seq_length + 1)
        time_indices = ops.reshape(time_indices, (1, -1, 1))

        return signatures / time_indices

    # ================= SIGNATURE COMPUTATION =================

    def sig_one_path(self, path):
        return ta.path_to_fm_signature(
            path=path,
            trunc=1,
            t_grid=jnp.linspace(0, 1, path.shape[1]),
            lam=0.2, # Input a mettre plus haut
        )

    def EfmLayer(self, inputs): ## Calcul de signature batché
        inputs = jnp.asarray(inputs)
        T = inputs.shape[1]
        t_grid = np.linspace(0, 1, T) # inputs.shape[0] a la place de 1
        t_grid = t_grid[None, :, None]
        t_grid = np.repeat(t_grid, inputs.shape[0], axis=0)

        path_with_time = jnp.concatenate([t_grid, inputs], axis=2)

        sig_batch = jax.vmap(self.sig_one_path)(path_with_time)
        return sig_batch

    # ================= MAIN CALL =================

    def call(self, inputs, initial_state=None, training=None):
        inputs = jnp.asarray(inputs) #2
        return self._call_tensorflow(
            inputs,
            initial_state=initial_state,
            training=training,
        )

    def _call_tensorflow(self, inputs, initial_state=None, training=None):
        inputs = jnp.asarray(inputs) #3
        signatures = self.EfmLayer(
            inputs #self.linear_preprocess_inputs_for_sig(inputs)
        )
        ## CA CA signatures = signatures.array
        
        signatures = signatures.array          # ✔ UNE SEULE FOIS

        normalized_signatures = self._normalize_signature_by_time(signatures)
        normalized_signatures = tf.transpose(normalized_signatures, perm=(0, 2, 1))

      #  signatures = signatures.array 
      #  normalized_signatures = self._normalize_signature_by_time(signatures)
      #  normalized_signatures=normalized_signatures.array
      # normalized_signatures = jnp.transpose(normalized_signatures, (0, 2, 1))
        
        
        time_steps = normalized_signatures.shape[-1]
        all_x_transformed = ops.einsum(
            "bti,ij->btj", inputs, self.input_kernel
        )

        if initial_state is None:
            h_tm1, c_tm1 = self.get_initial_state(inputs)
        else:
            h_tm1, c_tm1 = initial_state

        sequence_outputs = [] if self.return_sequences else None
        b_i, b_f, b_c, b_o = ops.split(self.bias, 4)

        for t in range(time_steps):
            current_sig = signatures[:, t] # normalized_signatures[:, t]
            current_x_transformed = all_x_transformed[:, t]

            h_tm1_transformed = ops.dot(h_tm1, self.recurrent_kernel)
            gates_standard = ops.add(h_tm1_transformed, current_x_transformed) 

            i_t, c_t, o_t = ops.split(gates_standard, 3, axis=-1)
            f_t = ops.dot(current_sig, self.forget_kernel) + b_f

            i_t = ops.sigmoid(i_t + b_i)
            f_t = ops.sigmoid(f_t)
            c_t = ops.tanh(c_t + b_c)
            o_t = ops.sigmoid(o_t + b_o)

            c_tm1 = f_t * c_tm1 + i_t * c_t
            h_tm1 = o_t * ops.tanh(c_tm1)

            if self.return_sequences:
                sequence_outputs.append(h_tm1)

        outputs = (
            ops.stack(sequence_outputs, axis=1)
            if self.return_sequences
            else h_tm1
        )

        if self.return_state:
            return outputs, h_tm1, c_tm1

        return outputs

    # ================= CONFIG =================

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "signature_depth": self.signature_depth,
                "signature_input_size": self.signature_input_size,
                "return_sequences": self.return_sequences,
                "return_state": self.return_state,
                "unroll_level": self.unroll_level,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        if self.return_sequences:
            output_shape = (batch_size, input_shape[1], self.units)
        else:
            output_shape = (batch_size, self.units)

        if self.return_state:
            return [
                output_shape,
                (batch_size, self.units),
                (batch_size, self.units),
            ]
        return output_shape

