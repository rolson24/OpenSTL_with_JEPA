method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
# model_type = None  # define `model_type` in args
hid_S = 16
hid_T = 256
N_T = 4
N_S = 4
# training
lr = 1e-3
batch_size = 1
drop_path = 0
sched = 'onecycle'

# EMAHook = dict(
#     momentum=0.999,
#     priority='NORMAL',
# )
