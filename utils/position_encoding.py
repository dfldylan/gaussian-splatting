import torch

# Positional encoding
class Embedder:

    def __init__(self, multires, input_dims=3, include_input=True, log_sampling=True,
                 periodic_fns=[torch.sin, torch.cos]):
        self.embed_fns = []
        out_dim = 0

        # Add input itself if include_input is True
        if include_input:
            self.embed_fns.append(lambda x: x)
            out_dim += input_dims

        # Calculate frequency bands
        max_freq = multires - 1
        N_freqs = multires

        if log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        # Create embedding functions for each frequency and periodic function
        for freq in freq_bands:
            for p_fn in periodic_fns:
                self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += input_dims

        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], dim=-1)
