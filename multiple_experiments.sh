#!/bin/bash
# #!/geom/bin/python

# gnn, batch_size, num_hops, lr, dim_h, dropout, wandb_disable
# accelerate launch --num_processes=4 train.py PMTGNN 64 8 '0.0003' 108 '0.4' true
accelerate launch --num_processes=4 train.py PMTGNN_VN 32 4 '0.0003' 128 '0.2' false
# accelerate launch --num_processes=4 train.py PMTGNN_GT 32 4 '0.0003' 64 '0.1'