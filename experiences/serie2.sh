#!/bin/bash
source "$(dirname "$0")/common.sh"
run name=lr1e-4 lr=1e-4
run name=lr5e-4 lr=5e-4
run name=lr1e-3 lr=1e-3
run name=lr2e-3 lr=2e-3

run name=T100 T=100 beta_end=0.189
run name=T200 T=200 beta_end=0.0977
run name=T500 T=500 beta_end=0.0398

run name=beta_end0.01 beta_end=0.01
run name=beta_end0.05 beta_end=0.05

run name=beta_start1e-5 beta_start=1e-5
run name=beta_start1e-3 beta_start=1e-3

run name=batch64 batch_size=64
run name=batch256 batch_size=256

run name=base_ch16 base_ch=16
run name=base_ch64 base_ch=64

run name=emb32 emb_dim=32
run name=emb256 emb_dim=256

run name=resunet model=resunet
run name=unet_ch40_control base_ch=40
run name=unet_noskip model=unet_noskip
run name=unet_not model=unet_not
