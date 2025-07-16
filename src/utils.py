import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Any, Optional, Dict, List, Generator, Union, Tuple

# Use non-interactive backend for matplotlib (suitable for headless environments)
matplotlib.use('Agg') 

# Import TensorFlow
import tensorflow as tf

# ==============================
# Global Constants and Helper Functions
# ==============================

# DNA character mapping (A:0, C:1, G:2, T:3)
charmap: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
# Reverse mapping (0:A, 1:C, 2:G, 3:T)
rev_charmap: Dict[int, str] = {v: k for k, v in charmap.items()}

def check_folder(log_dir: str) -> str:
    """Ensure directory exists, create if not.
    
    Args:
        log_dir: Directory path to check/create
        
    Returns:
        str: Verified directory path
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def feed(data: np.ndarray, batch_size: int, reuse: bool = True) -> Generator[tf.Tensor, None, None]:
    """Generate batches of data indefinitely.
    
    Args:
        data: Input data array
        batch_size: Number of samples per batch
        reuse: Whether to restart from beginning after last batch
        
    Yields:
        tf.Tensor: Batch of data as TensorFlow tensor
    """
    num_batches = len(data) // batch_size
    reshaped_data = data
    while True:
        for ctr in range(num_batches):
            yield tf.convert_to_tensor(
                reshaped_data[ctr * batch_size : (ctr + 1) * batch_size],
                dtype=tf.float32
            )
        # Stop if not reusing after final batch
        if not reuse and ctr == num_batches - 1:
            yield None

def balanced_batch(train_data: np.ndarray, 
                   train_label: np.ndarray, 
                   batch_size: int, 
                   random_seed: Optional[int] = None) -> Generator[tf.Tensor, None, None]:
    """Generate balanced batches by oversampling minority classes.
    
    Args:
        train_data: Input features
        train_label: Target labels
        batch_size: Total samples per batch
        random_seed: Optional seed for reproducibility
        
    Yields:
        tf.Tensor: Balanced batch of data
    """
    # Create 16 bins based on label distribution
    bins = np.arange(
        min(train_label),
        max(train_label),
        (max(train_label) - min(train_label)) / 16
    )
    indices = np.digitize(train_label, bins)
    
    # Group samples by bin
    groupby_levels: Dict[int, List[int]] = {i+1: [] for i in range(16)}
    for i in range(len(indices)):
        groupby_levels[indices[i]].append(i)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    while True:
        balanced_copy_idx = []
        # Oversample each bin equally
        for _, gb_idx in groupby_levels.items():
            over_sample_idx = np.random.choice(
                gb_idx, 
                size=batch_size // 16, 
                replace=True
            ).tolist()
            balanced_copy_idx += over_sample_idx
        
        np.random.shuffle(balanced_copy_idx)
        yield tf.convert_to_tensor(
            [train_data[i] for i in balanced_copy_idx],
            dtype=tf.float32
        )

def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float64) -> np.ndarray:
    """One-hot encode DNA sequence.
    
    Args:
        sequence: Input DNA sequence string
        alphabet: Valid DNA characters
        neutral_alphabet: Characters to treat as neutral
        neutral_value: Value to use for neutral characters
        dtype: Output data type
        
    Returns:
        np.ndarray: One-hot encoded sequence array
    """
    def to_uint8(string: str) -> np.ndarray:
        return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
    
    # Create mapping table
    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    return hash_table[to_uint8(sequence)]

def save_samples(logdir: str, samples: np.ndarray, iteration: int):
    """Save generated DNA sequences to file.
    
    Args:
        logdir: Output directory
        samples: Generated sequences (one-hot encoded)
        iteration: Training iteration number for filename
    """
    # Convert one-hot to sequence strings
    argmax = np.argmax(samples, 2)
    samples_dir = os.path.join(logdir, "samples")
    check_folder(samples_dir)
    
    with open(os.path.join(samples_dir, f"samples_{iteration}"), "w") as f:
        for line in argmax:
            s = "".join(rev_charmap[d] for d in line) + "\n"
            f.write(s)

def plot(y: List[float], 
         x: List[float], 
         logdir: str, 
         name: str,
         xlabel: Optional[str] = None, 
         ylabel: Optional[str] = None, 
         title: Optional[str] = None):
    """Save training curve plot to file.
    
    Args:
        y: Y-axis values
        x: X-axis values
        logdir: Output directory
        name: Filename prefix
        xlabel: Optional X-axis label
        ylabel: Optional Y-axis label
        title: Optional plot title
    """
    plt.close()
    plt.plot(x, y)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.savefig(os.path.join(logdir, f"{name}.png"))
    plt.close()

# ==============================
# Lion Optimizer (TF 2.5 Implementation)
# ==============================

class Lion(tf.keras.optimizers.Optimizer):
    """Lion optimizer implementation for TensorFlow 2.x.
    
    Based on: "Symbolic Discovery of Optimization Algorithms" (https://arxiv.org/abs/2302.06675)
    """
    
    def __init__(self,
                 learning_rate: Union[float, tf.Tensor] = 0.0001,
                 beta1: Union[float, tf.Tensor] = 0.9,
                 beta2: Union[float, tf.Tensor] = 0.99,
                 weight_decay: Optional[Union[float, tf.Tensor]] = 0.0,
                 name: str = "Lion",
                 **kwargs):
        """Initialize Lion optimizer.
        
        Args:
            learning_rate: Learning rate
            beta1: First moment exponential decay rate
            beta2: Second moment exponential decay rate
            weight_decay: Optional weight decay coefficient
            name: Optimizer name
            **kwargs: Additional base class arguments
        """
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta1", beta1)
        self._set_hyper("beta2", beta2)
        
        # Handle weight decay (disable if negative)
        if weight_decay is None or (isinstance(weight_decay, float) and weight_decay < 0):
            self.weight_decay = None
        else:
            self._set_hyper("weight_decay", weight_decay)
    
    def _create_slots(self, var_list):
        # Create momentum slot for each variable
        for var in var_list:
            self.add_slot(var, "m")
    
    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        
        # Get hyperparameters in proper dtype
        local_apply_state = apply_state[(var_device, var_dtype)]
        beta1_t = tf.identity(self._get_hyper("beta1", var_dtype))
        beta2_t = tf.identity(self._get_hyper("beta2", var_dtype))
        local_apply_state.update({
            "beta1_t": beta1_t,
            "beta2_t": beta2_t,
            "one_minus_beta1_t": 1 - beta1_t,
            "one_minus_beta2_t": 1 - beta2_t,
        })
        
        # Handle weight decay if enabled
        if self.weight_decay is not None:
            wd_t = tf.identity(self._get_hyper("weight_decay", var_dtype))
            local_apply_state["weight_decay_t"] = wd_t
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        
        # Get optimizer state
        m = self.get_slot(var, "m")
        
        # Compute update direction
        update_direction = tf.sign(
            m * coefficients["beta1_t"] + 
            grad * coefficients["one_minus_beta1_t"]
        )
        
        # Apply weight decay if enabled
        if self.weight_decay is not None:
            update_direction += var * coefficients["weight_decay_t"]
        
        # Update variable
        var_update = var.assign_sub(
            coefficients["lr_t"] * update_direction,
            use_locking=self._use_locking
        )
        
        # Update momentum
        with tf.control_dependencies([var_update]):
            m_update = m.assign(
                m * coefficients["beta2_t"] + 
                grad * coefficients["one_minus_beta2_t"]
            )
        
        return tf.group(var_update, m_update)

    def get_config(self):
        """Get optimizer configuration for serialization."""
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta1": self._serialize_hyperparameter("beta1"),
            "beta2": self._serialize_hyperparameter("beta2"),
            "weight_decay": self._serialize_hyperparameter("weight_decay") if self.weight_decay is not None else None,
        })
        return config