ulimit -u 2048

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export PYTHONPATH=$(pwd):$PYTHONPATH

# Paper: total batch size 128 across 2 H100 -> 2 GPUs x 64 per device (effective 128).
GPUS=2
PER_DEVICE_BATCH_SIZE=64
NUM_NODES=${NUM_NODES:-1}

# Build accelerate arguments based on GPU count
if [ $GPUS -eq 1 ]; then
    echo "Single GPU mode"
    ACCELERATE_ARGS="--num_processes 1 --num_machines 1"
elif [ $GPUS -gt 1 ] && [ $NUM_NODES -eq 1 ]; then
    echo "Multi-GPU single node mode"
    ACCELERATE_ARGS="--multi_gpu --num_processes ${GPUS} --num_machines 1 --machine_rank 0"
else
    echo "Multi-GPU multi-node mode"
    ACCELERATE_ARGS="--multi_gpu --num_processes ${GPUS} --num_machines ${NUM_NODES} --machine_rank ${SLURM_NODEID}  --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"
fi

# datasets
dataset=${1:-experiments/13_robocasa365/data_configs/data-robocasa365-opendrawer-point.yaml}
dataset_name=$(basename ${dataset%.*})-rot6d-image.leftview

# hparams
lr=5e-5
mlr=5e-5
vlr=2e-5

chunk_size=16
epoch=50

model_name_or_path=
run_name=${dataset_name}_ck${chunk_size}_lr${lr}_gpu${GPUS}_bs${PER_DEVICE_BATCH_SIZE}_epoch${epoch}

output_dir=$SCRATCH/datasets/PointAct_exprs/robocasa365/pointact/VLAEncDec3DWithActionRegressionModel-utonia-${run_name}-freeze.vlm

# Weights & Biases. On Jean Zay compute nodes (no internet) this logs offline; sync from a
# login node with `wandb sync` (WANDB_MODE / WANDB_DIR are set by train.slurm).
export WANDB_ENTITY=diffusion4robots
export WANDB_PROJECT=pointact-robocasa365
export WANDB_NAME=utonia-${run_name}

# Determine TF32 support
TF32_SUPPORT="False"
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
echo "GPU Compute Capability: $COMPUTE_CAP"
# Check if Ampere or newer (compute capability >= 8.0)
if [ "$MAJOR" -ge 8 ]; then
    TF32_SUPPORT="True"
fi
echo "TF32_SUPPORT: $TF32_SUPPORT"


accelerate launch $ACCELERATE_ARGS scripts/train.py \
    --model_class VLAEncDec3DWithActionRegressionModel \
    --output_dir ${output_dir} \
    ${model_name_or_path:+--model-name-or-path $model_name_or_path} \
    --vlm-name-or-path $DSDIR/HuggingFace_Models/Qwen/Qwen2.5-VL-3B-Instruct \
    --data-path ${dataset} \
    --chunk-size ${chunk_size} \
    --dataloader-num-workers 8 \
    --freeze-vision-tower True \
    --freeze-llm True \
    --freeze-merger True \
    --bf16 True \
    --tf32 ${TF32_SUPPORT} \
    --fp16 False \
    --num-train-epochs ${epoch} \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --learning-rate ${lr} \
    --merger-lr ${mlr} \
    --vision-lr ${vlr} \
    --weight-decay 0.1 \
    --warmup-steps 0.03 \
    --lr-scheduler-type cosine \
    --gradient-checkpointing True \
    --save-strategy steps \
    --logging-steps 10 \
    --save-steps 1000 \
    --save-total-limit 10 \
    --run-name ${run_name} \
    --attn-implementation flash_attention_2 \
    --log_level info \
    --report-to wandb tensorboard \
    --color_aug True \
    --max_grad_norm 3 \
    --use_robot_state True \
    --ctx_embed_size 512 \
    --ptv3_backend utonia \
    --ptv3_patch_size 1024 \
    --ptv3_enc_mode True \
    --ptv3_enc_channels 54 108 216 432 576 \
    --ptv3_enc_depths 3 3 3 12 3 \
    --ptv3_enc_num_head 3 6 12 24 32 \
    --ptv3_input_channels 6 \
    --action_regression_loss l2 \
    --regression_head_heatmap_temp 0.1 \
    --action_head_pos_center zero \
    --ptv3_apply_point_ca False \
    --ptv3_init_ckpt_file $SCRATCH/models/Pointcept-Utonia/utonia.pth
