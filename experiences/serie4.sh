#!/bin/bash
source "$(dirname "$0")/common.sh"
run name=baseline28 image_size=28

run name=C1_resunet_lr1e-3_28 "${C[@]}"
run name=C2_C1+cosine_28 "${C[@]}" schedule=cosine
run name=C3_C2+batch64_28 "${C[@]}" schedule=cosine batch_size=64
