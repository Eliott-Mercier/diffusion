#!/bin/bash
source "$(dirname "$0")/common.sh"
run name=seed1 seed=1
run name=seed2 seed=2

run name=cosine schedule=cosine

run name=T100_beta0.02 T=100

run name=epochs5 epochs=5
run name=epochs50 epochs=50
