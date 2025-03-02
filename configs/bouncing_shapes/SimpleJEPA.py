method = 'simple_mnist_jepa'
# model
# in_shape = [3, 3, 32, 32]
embed_dim = 512
latent_tensor_mode = 0
latent_tensor_size = 20
sparsity_reg = 0.2
n_steps_inf = 100
lrt_z = 1
tolerance = 1e-6
alpha = 0.1
beta = 0.1
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