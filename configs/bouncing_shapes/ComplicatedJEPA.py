method = 'complicated_jepa'
# model
in_shape = [10, 1, 64, 64]
embed_dim = 128

alpha = 0.1
beta = 0.1
train_decoder = True
lrt_decoder = 0.01
aft_seq_length = 10
pre_seq_length = 10
total_length = 20
hid_S = 32
hid_T = 128
N_T = 4
N_S = 8
# training
lr = 1e-4
batch_size = 16
sched = 'onecycle'
epoch = 100