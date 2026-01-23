import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len).reshape(seq_len, 1)
    
    i = np.arange(d_model).reshape(1, d_model)
    
    div_term = np.power(base, (2 * (i // 2)) / d_model)
    
    angles = pos / div_term
    
    PE = np.zeros((seq_len, d_model), dtype=float)
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    
    return PE

def add_positional_encoding(x, base=10000.0):
    """
    Add PE to input x of shape (B, T, d_model); return same shape.
    """
    if(x.ndim != 3):
        raise ValueError
    
    batch, seq_len, d_model = x.shape
    
    PE = positional_encoding(seq_len, d_model, base=base)
    
    x_pe = x + PE 
    
    return x_pe