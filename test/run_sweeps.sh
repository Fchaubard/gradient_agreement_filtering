#!/bin/bash

WANDB_API_KEY=<INSERT>

# Example run:
#WANDB_API_KEY=<INSERT WANDB_API_KEY> python ../examples/1_cifar_100_train_loop_exposed.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --learning_rate 0.01 --momentum 0.9 --nesterov True --wandb True --verbose True --num_samples_per_class_per_batch 2 --num_batches_to_force_agreement 2 --label_error_percentage 0.15 --cos_distance_thresh 0.97

# Kill existing screen sessions that start with "gaf_"
screen -ls | grep '\.gaf_' | awk '{print $1}' | xargs -I{} screen -S {} -X quit

# HPP SWEEP
num_samples_per_class_per_batch=(1 2 4)
num_batches_to_force_agreement=(2 4 10)
label_error_percentage=(0 .1 .5)
cos_distance_thresh=(.95 .97 .99 1.0 1.01 1.05 2.0)


# Non-GAF runs
for a in "${label_error_percentage[@]}"; do
    sleep 30 # this allows the GPU mem to fill up and to download any files needed
    screen_name="gaf_baseline_${a}"
    screen -dmS "$screen_name" bash -c "export WANDB_API_KEY=${WANDB_API_KEY}; \
    python ../examples/1_cifar_100_train_loop_exposed.py --GAF False \
    --optimizer \"SGD+Nesterov+val_plateau\" \
    --learning_rate 0.01 --momentum 0.9 --nesterov True \
    --wandb True --verbose True \
    --label_error_percentage ${a}" 
done

# GAF runs
for a in "${num_samples_per_class_per_batch[@]}"; do
    for b in "${num_batches_to_force_agreement[@]}"; do
        for c in "${label_error_percentage[@]}"; do
            for d in "${cos_distance_thresh[@]}"; do
                sleep 0.1 
                screen_name="gaf_${a}_${b}_${c}_${d}"
                screen -dmS "$screen_name" bash -c "export WANDB_API_KEY=${WANDB_API_KEY}; \
                python ../examples/1_cifar_100_train_loop_exposed.py \
                --GAF True --optimizer \"SGD+Nesterov+val_plateau\" \
                --learning_rate 0.01 --momentum 0.9 --nesterov True \
                --wandb True --verbose True \
                --num_samples_per_class_per_batch ${a} \
                --num_batches_to_force_agreement ${b} \
                --label_error_percentage ${c} \
                --cos_distance_thresh ${d}"\
                
            done
        done
    done
done

