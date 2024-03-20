#!/bin/bash

CL_BS=32
FMAPS=64
CL_LR=0.01 #### cifar 10 learning rate 
CL_EPOCHS=5
NUM_ROUNDS=100
SAMP_CLIENT=0.1
NUM_CLIENTS=100
ALPHA=0.5
DATASET=cifar10

PYTHON_CMD="python main_ray.py --num_rounds $NUM_ROUNDS --num_clients $NUM_CLIENTS  --feature_maps $FMAPS --samp_rate $SAMP_CLIENT --alpha $ALPHA --nworkers 4"

#fedlora - groupnorm - cifar10 --ablation r=16
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 1234 --lora_ablation_mode 0 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 3923 --lora_ablation_mode 0 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 9128 --lora_ablation_mode 0 

$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 1234 --lora_ablation_mode 1 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 3923 --lora_ablation_mode 1 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 9128 --lora_ablation_mode 1 

$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 1234 --lora_ablation_mode 2 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 3923 --lora_ablation_mode 2 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 9128 --lora_ablation_mode 2

$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 1234 --lora_ablation_mode 3 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 3923 --lora_ablation_mode 3 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 16 --lora_alpha 256 --seed 9128 --lora_ablation_mode 3

#fedlora - groupnorm - cifar10 --ablation r=32
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 1234 --lora_ablation_mode 0 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 3923 --lora_ablation_mode 0 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 9128 --lora_ablation_mode 0 

$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 1234 --lora_ablation_mode 1 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 3923 --lora_ablation_mode 1 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 9128 --lora_ablation_mode 1 

$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 1234 --lora_ablation_mode 3 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 3923 --lora_ablation_mode 3 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 9128 --lora_ablation_mode 3

#fedavg group-norm
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --lora_r 0 --seed 1234 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --lora_r 0 --seed 3923 
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --lora_r 0 --seed 9128

# #fedlora - r effect
declare -a R_TEST_CASES=(8 16 32 64 128)
for R in "${R_TEST_CASES[@]}"
do
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $R --lora_alpha $((2 * R)) --seed 1234 
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $R --lora_alpha $((2 * R)) --seed 3923 
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $R --lora_alpha $((2 * R)) --seed 9128 
done

# #fedlora - s effect
declare -a S_TEST_CASES=(8 16 32 64 128)
for RS in "${S_TEST_CASES[@]}"
do
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $RS --lora_alpha $((16 * $RS)) --seed 1234 
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $RS --lora_alpha $((16 * $RS)) --seed 3923 
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $RS --lora_alpha $((16 * $RS)) --seed 9128 
done

#fedavg - groupnorm - cifar100
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 9128

#fedavg - groupnorm - cinic10
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 9128

#fedlora - cifar100 - r=16
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 32 --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 32 --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 32 --wandb --wandb_prj_name eusipco_24_lora --seed 9128

$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 9128

#fedlora - cifar100 - r=32
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 64 --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 64 --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 64 --wandb --wandb_prj_name eusipco_24_lora --seed 9128

$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 9128

$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 1024 --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 1024 --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cifar100" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 1024 --wandb --wandb_prj_name eusipco_24_lora --seed 9128

#fedlora - cinic10 - r=16

$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 16 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 9128

$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 256 --wandb --wandb_prj_name eusipco_24_lora --seed 9128

$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 512 --wandb --wandb_prj_name eusipco_24_lora --seed 1234 
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 512 --wandb --wandb_prj_name eusipco_24_lora --seed 3923
$PYTHON_CMD --dataset "cinic10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --lora_r 32 --lora_alpha 512 --wandb --wandb_prj_name eusipco_24_lora --seed 9128

#fedlora - resnet18 zerofl
declare -a R_TEST_CASES=(16 32 64)
for R in "${R_TEST_CASES[@]}"
do
    $PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $R --lora_alpha $((16 * R)) --seed 1234
    $PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $R --lora_alpha $((16 * R)) --seed 3923
    $PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r $R --lora_alpha $((16 * R)) --seed 9128
done

#pruning - resnet18 zerofl
declare -a PRUNING_RATE=(0.5 0.8 0.9 0.95)
for p in "${PRUNING_RATE[@]}"
do
    $PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 1234 --prune --prate $p --prune_srv
    $PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 3923 --prune --prate $p --prune_srv
    $PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedavg --wandb --wandb_prj_name eusipco_24_lora --seed 9128 --prune --prate $p --prune_srv
done

#fedlora quant r=16
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 1234 --apply_quant --quant_bits 8
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 3923 --apply_quant --quant_bits 8
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 9128 --apply_quant --quant_bits 8

$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 1234 --apply_quant --quant_bits 4
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 3923 --apply_quant --quant_bits 4
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 9128 --apply_quant --quant_bits 4

$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 1234 --apply_quant --quant_bits 2
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 3923 --apply_quant --quant_bits 2
$PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name simple_quant --lora_r 16 --lora_alpha 256 --seed 9128 --apply_quant --quant_bits 2

#fedlora quant r=32
declare -q S_TEST_CASES=(2 4 8)
for Q in "${S_TEST_CASES[@]}"
do
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 1234 --apply_quant --quant_bits $Q
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 3923 --apply_quant --quant_bits $Q
    $PYTHON_CMD --dataset "cifar10" --cl_lr $CL_LR --cl_epochs $CL_EPOCHS --cl_bs $CL_BS --model resnet8 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --lora_r 32 --lora_alpha 512 --seed 9128 --apply_quant --quant_bits $Q
done

# fedlora + quant vs zerofl 
$PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --seed 1234 --lora_r 16 --lora_alpha 256 --apply_quant --quant_bits 8
$PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --seed 1234 --lora_r 32 --lora_alpha 512 --apply_quant --quant_bits 8

$PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --seed 3923 --lora_r 16 --lora_alpha 256 --apply_quant --quant_bits 8
$PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --seed 9128 --lora_r 16 --lora_alpha 256 --apply_quant --quant_bits 8

$PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --seed 3923 --lora_r 32 --lora_alpha 512 --apply_quant --quant_bits 8
$PYTHON_CMD --dataset "cifar10" --alpha 0.1 --cl_lr $CL_LR --cl_epochs 1 --cl_bs $CL_BS --model resnet18 --strategy fedlora --wandb --wandb_prj_name eusipco_24_lora --seed 9128 --lora_r 32 --lora_alpha 512 --apply_quant --quant_bits 8