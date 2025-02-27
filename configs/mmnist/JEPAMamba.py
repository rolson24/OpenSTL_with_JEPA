method = 'MNIST_JEPA_mamba'
# model
in_shape = [3, 1, 32, 32]
# encoder
patch_size = 4
embed_dim = 64
depths = [2, 4, 8, 4]
num_heads = [2, 4, 8, 16]
mlp_ratio = 4
qkv_bias = True
drop_rate = 0.0
drop_path_rate = 0.2
ssd_expansion = 2
ssd_ngroups = 1
ssd_chunk_size = 256
linear_attn_duality = True
attn_types = ['mamba2', 'mamba2', 'mamba2', 'standard']
d_state = 64
# predictor
latent_tensor_mode = 2
latent_tensor_size = 20
sparsity_reg = 0.2
n_steps_inf = 10
lrt_z = 1
tolerance = 1e-6
alpha = 0.1
beta = 0.1
# decoder
train_decoder = True
lrt_decoder = 0.01
aft_seq_length = 12
pre_seq_length = 3
total_length = 15
# training
lr = 1e-4
batch_size = 16
sched = 'onecycle'
epoch = 100

