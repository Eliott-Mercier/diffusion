#!/bin/bash
source "$(dirname "$0")/common.sh"
run name=epochs200 epochs=200 checkpoint_every=5 sample_every=25

run name=K1_resunet_lr1e-3 model=resunet lr=1e-3
run name=K2_K1+cosine model=resunet lr=1e-3 schedule=cosine
run name=K3_K2+batch64 model=resunet lr=1e-3 schedule=cosine batch_size=64
run name=K4_K3+emb256 model=resunet lr=1e-3 schedule=cosine batch_size=64 emb_dim=256
run name=K5_unet64_combo base_ch=64 lr=1e-3 schedule=cosine batch_size=64 emb_dim=256
