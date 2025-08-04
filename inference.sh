MODEL_TYPE="x052"
N_LAYER="8"
N_EMBD="512"
CTX_LEN="1024" # !!! change magic_prime if you change ctx_len !!!
PROJ_DIR="/pvc/RWKVout/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE"1024" # set output folder
M_BSZ="8" # takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
LR_INIT="1e-4"
LR_FINAL="1e-5"
RWKV_JIT_ON='1'
GRAD_CP=0 # 1 => slower, save VRAM; 0 => faster, more VRAM
EPOCH_SAVE=1
N_NODE=1 # number of nodes
GPU_PER_NODE=1 # number of GPUs per node
#
DS_BUCKET_MB=2 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)
#
python inference.py --load_model "rwkv-final.pth" --wandb "rna-rwkv" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
 --test_json "/pvc/dataset/rna_test.json"  --output_json "/home/jovyan/result/etherna100.json" --my_exit_tokens 114800000 --magic_prime 1793731 \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --my_pile_edecay 0 --data_type "numpy" --vocab_size 9 \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB --topk=1 --temperature 0.01 --gc_target 0.25 --gc_strength 50 --div_test 0