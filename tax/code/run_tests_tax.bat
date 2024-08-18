@echo off
setlocal enabledelayedexpansion

set datasets=("bpic15_1_orig.csv" "bpic15_1_baseline.csv" "bpic15_1_cvT_0.95_0.95_dc2_augm.csv" "bpic15_1_cvT_0.95_0.95_dc3_augm.csv")

for %%d in %datasets% do (
    set dataset=%%~d
    set folder=!dataset:.csv=!
    python train.py --dataset ..\data\!folder!\!dataset! --train --test
)