
import torch
from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO)
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import pytorch_lightning as pl
import re
import os, warnings, math, datetime, sys, time,random
import numpy as np
import json
import subprocess
from difflib import SequenceMatcher
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
rank_zero_info("########## work in progress ##########")
id_to_token = {
    0: '.', 1: '(', 2: ')', 3: 'A', 4: 'C', 5: 'G', 6: 'U', 7: '\n', 8: 'PAD'
}
vocab = {v: k for k, v in id_to_token.items()}
rna_tokens = {3, 4, 5, 6}  
def generate_rna_sequence(
    model, input_ids, rna_tokens, vocab, id_to_token,
    ctx_len=256, top_k=4, temperature=0.00001,
    gc_target=0.5, gc_strength=1.0
):
    rna_seq = ""

    for i in range(ctx_len):
        if input_ids.shape[1] > ctx_len:
            break
        
        logits = model(input_ids)[:, -1, :]  
        logits = logits / temperature
        
        if len(rna_seq) > 0:
            gc_count = rna_seq.count('G') + rna_seq.count('C')
            gc_ratio = gc_count / len(rna_seq)
            for token_id in rna_tokens:
                token = id_to_token[token_id]
                if token in ['G', 'C']:
                    if gc_ratio < gc_target:
                        logits[0, token_id] += gc_strength 
                    elif gc_ratio > gc_target:
                        logits[0, token_id] -= gc_strength  

        logits = logits / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
  
        

        topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(topk_probs, num_samples=1)
        token_id = topk_indices[0, next_token.item()].item()
        if token_id in rna_tokens:
            rna_seq += id_to_token[token_id]
        elif token_id == vocab['PAD']:
            break

        next_token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    return rna_seq
def load_model(args):
    

    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_TRAIN_TYPE"] = args.train_type
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        if '-f4' in os.environ["RWKV_MY_TESTING"]:
            args.dim_ffn = int((args.n_embd * 4) // 32 * 32)
        else:
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    if args.data_type == "wds_img":
        args.run_name = f"v{args.my_img_version}-{args.my_img_size}-{args.my_img_bit}bit-{args.my_img_clip}x{args.my_img_clip_scale}"
        args.proj_dir = f"{args.proj_dir}-{args.run_name}"
    else:
        args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    if args.my_pile_stage > 0:
        magic_prime_bak = args.magic_prime

        if args.my_pile_shift < 0:
            args.my_pile_shift = 0

        if magic_prime_bak > 0:
            args.magic_prime = magic_prime_bak
        if args.my_qa_mask == 2:
            args.epoch_count = 2 * args.magic_prime // 40320
        else:
            args.epoch_count = args.magic_prime // 40320

        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320
        # if args.my_pile_stage == 2:
        #     assert args.lr_final == args.lr_init
        if args.my_pile_stage >= 2:  # find latest saved model
            list_p = []
            for p in os.listdir(args.proj_dir):
                if p.startswith("rwkv") and p.endswith(".pth"):
                    p = ((p.split("-"))[1].split("."))[0]
                    if p != "final":
                        if p == "init":
                            p = -1
                        else:
                            p = int(p)
                        list_p += [p]
            list_p.sort()
            max_p = list_p[-1]
            if len(list_p) > 1:
                args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if args.warmup_steps < 0:
                    if args.my_pile_stage == 2:
                        args.warmup_steps = 10
                    else:
                        args.warmup_steps = 30
            args.epoch_begin = max_p + 1

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-5 {args.precision} on {args.num_nodes}x{args.devices} {args.accelerator}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
   

    
    args.vocab_size = 9

    from src.model import RWKV
    model = RWKV(args)

    if len(args.load_model) == 0 or args.my_pile_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu")
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
    except:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.my_pile_stage >= 2:  # try again using another checkpoint
            max_p = args.my_pile_prev_p
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            args.epoch_begin = max_p + 1
            rank_zero_info(f"Trying {args.load_model}")
            load_dict = torch.load(args.load_model, map_location="cpu")

    state_file = f"{args.proj_dir}/rwkv-init-state.pth"
    if os.path.isfile(state_file):
        rank_zero_info(f"########## Loading State {state_file}... ##########")
        state_dict = torch.load(state_file, map_location="cpu")
        for k in state_dict:
            load_dict[k] = state_dict[k]

    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]
    model.load_state_dict(load_dict)
    model = model.to(dtype=torch.bfloat16)
    model = model.to('cuda').eval()
    print("Model loaded.")
    return model
def run_rnafold(sequence: str) -> str:
       
        process = subprocess.run(
            ["RNAfold", "--noPS"],
            input=sequence.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = process.stdout.decode().split('\n')
        return output[1].strip() if len(output) > 1 else ""
def evaluate_prediction(predicted_seq: str, target_struct: str) -> dict:
        predicted_struct = run_rnafold(predicted_seq)
    
        
        predicted_struct = predicted_struct.split()[0]
    
        match_count = sum(1 for a, b in zip(predicted_struct, target_struct) if a == b)
        correct_rate = match_count / len(target_struct)
        edit_dist = levenshtein(predicted_struct, target_struct)
        full_match = int(predicted_struct == target_struct)
        gc_count = predicted_seq.count('G') + predicted_seq.count('C')
        gc_content = gc_count / len(predicted_seq) if predicted_seq else 0.0
        return {
            "predicted_sequence": predicted_seq,
            "predicted_structure": predicted_struct,
            "target_structure": target_struct,
            "correct_rate": correct_rate,
            "edit_distance": edit_dist,
            "full_match": full_match,
            "gc_content": gc_content
        }
def levenshtein(a: str, b: str) -> int:
      
        n, m = len(a), len(b)
        if n > m:
            a, b = b, a
            n, m = m, n
        current = list(range(n + 1))
        for i in range(1, m + 1):
            previous, current = current, [i] + [0]*n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change += 1
                current[j] = min(add, delete, change)
        return current[n]
def compute_diversity(seqs):
    n = len(seqs)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(levenshtein(seqs[i], seqs[j]))
    return np.mean(dists)
def evaluate_diversity_on_sampled_structures(
    model,
    test_data,
    vocab,
    rna_tokens,
    ctx_len,
    num_structures=100,
    num_samples_per_structure=10,
    
 
):
    
    random.seed(42)
    torch.manual_seed(42)
    sampled_structures = random.sample(
        list({s["structure"] for s in test_data}), min(num_structures, len(test_data))
    )

    generated_sequences = defaultdict(list)

    with torch.no_grad():
        for structure in tqdm(sampled_structures, desc="Generating sequences"):
            prompt = structure + "\n"
            input_ids = torch.tensor([[vocab[c] for c in prompt]], dtype=torch.long, device='cuda')

            if input_ids.shape[1] > ctx_len:
                input_ids = input_ids[:, -ctx_len:]

            for _ in range(num_samples_per_structure):
                pred_seq = generate_rna_sequence(model, input_ids, rna_tokens, vocab, ctx_len,args.topk,args.temperature)
                generated_sequences[structure].append(pred_seq)

    diversity_scores = []

    for structure, seqs in generated_sequences.items():
        diversity = compute_diversity(seqs)
        diversity_scores.append((structure, diversity))

    avg_diversity = np.mean([d for _, d in diversity_scores])

    print(f"\nAverage Diversity over {len(diversity_scores)} structures: {avg_diversity:.2f}")


    return diversity_scores, avg_diversity
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gc_target",default=0.5,type=float )
    
    parser.add_argument("--gc_strength",default=1.0,type=float )
    parser.add_argument("--div_test", default=1,type=int,  help="Set 1 for diversity test")
    parser.add_argument("--topk",default=4,type=int )
    parser.add_argument("--temperature",default=0.00001 , type=float)
    parser.add_argument("--test_json", type=str, help="Path to test JSON file")
    parser.add_argument("--output_json", type=str,  help="Path to save result JSON")
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)
    parser.add_argument("--train_type", default="", type=str) # ""/"states"

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 20 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.95 if you see spikes
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float) # try 0.01 / 0.02 / 0.05 / 0.1
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1
    parser.add_argument("--weight_decay_final", default=-1, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float) # reduce it to 0.7 / 0.5 / 0.3 / 0.2 for problematic samples

    parser.add_argument("--my_pile_version", default=1, type=int)  # my special pile version
    parser.add_argument("--my_pile_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--my_pile_shift", default=-1, type=int)  # my special pile mode - text shift
    parser.add_argument("--my_pile_edecay", default=0, type=int)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    # parser.add_argument("--cuda_cleanup", default=0, type=int)  # extra cuda cleanup (sometimes helpful)

    parser.add_argument("--my_sample_len", default=0, type=int)
    parser.add_argument("--my_ffn_shift", default=1, type=int)
    parser.add_argument("--my_att_shift", default=1, type=int)
    parser.add_argument("--head_size_a", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_size_divisor", default=8, type=int)
    parser.add_argument("--my_pos_emb", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_qa_mask", default=0, type=int)
    parser.add_argument("--my_random_steps", default=0, type=int)
    parser.add_argument("--my_testing", default='x052', type=str)
    parser.add_argument("--my_exit", default=99999999, type=int)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    if pl.__version__[0]=='2':
        parser.add_argument("--accelerator", default="gpu", type=str)
        parser.add_argument("--strategy", default="auto", type=str)
        parser.add_argument("--devices", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int)
        parser.add_argument("--precision", default="fp16", type=str)
        parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    else:
        parser = Trainer.add_argparse_args(parser)
    
    args = parser.parse_args()
    model=load_model(args)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    results_path = args.output_json.replace(".json", "_details.jsonl")
    summary_path = args.output_json.replace(".json", "_summary.json")

    with open(args.test_json, "r") as f:
        test_data = json.load(f)

    completed_count = 0
    completed_structures = set()
    if os.path.exists(results_path):
        print(f"Found existing results: {results_path} - attempting to resume...")
        with open(results_path, "r") as fin:
            for line in fin:
                try:
                    data = json.loads(line)
                    completed_structures.add(data["target_structure"])
                    completed_count += 1
                except:
                    continue
        print(f"Resuming from sample {completed_count}/{len(test_data)}")

    total_correct = 0
    total_edit_distance = 0
    total_full_match = 0
    total_gc=0



    with open(results_path, "a") as fout:
        for idx, sample in enumerate(tqdm(test_data)):
            structure = sample["structure"]

            if structure in completed_structures:
                continue

            prompt = structure + "\n"
            input_ids = torch.tensor([[vocab[c] for c in prompt]], dtype=torch.long, device='cuda')

           
            if input_ids.shape[1] > args.ctx_len:
                print(f"[WARNING] Input length {input_ids.shape[1]} exceeds ctx_len {args.ctx_len}, truncating...")
                input_ids = input_ids[:, -args.ctx_len:]

            with torch.no_grad():
                pred_seq = generate_rna_sequence(model, input_ids, rna_tokens, vocab, id_to_token,args.ctx_len,args.topk,args.temperature,args.gc_target,args.gc_strength)
            result = evaluate_prediction(pred_seq, structure)
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            total_correct += result["correct_rate"]
            total_edit_distance += result["edit_distance"]
            total_full_match += result["full_match"]
            total_gc+=result["gc_content"]
            completed_structures.add(structure)
            done_so_far = len(completed_structures) 
            if done_so_far % 10 == 0 or done_so_far == len(test_data):
                avg_correct = total_correct / done_so_far
                avg_edit_dist = total_edit_distance / done_so_far
                full_match_ratio = total_full_match / done_so_far
                gc_ratio=total_gc/done_so_far
                print(f"[{done_so_far}/{len(test_data)}] Correct Rate: {avg_correct:.4f} | Edit Distance: {avg_edit_dist:.2f} | Full Match: {full_match_ratio:.4f},| GC Content: {gc_ratio:.4f}")

            
    summary = {
        "num_samples": len(completed_structures),
        "average_correct_rate": total_correct / len(completed_structures),
        "average_edit_distance": total_edit_distance / len(completed_structures),
        "full_match_accuracy": total_full_match / len(completed_structures),
        "gc_content":total_gc / len(completed_structures)
    }
    if(args.div_test==1):
        div_scores, avg_div = evaluate_diversity_on_sampled_structures(model=model,test_data=test_data,vocab=vocab,rna_tokens=rna_tokens,
    ctx_len=args.ctx_len,
    
)
        summary["avg_div"]=avg_div

    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation finished.")
    print(f"Summary saved to: {summary_path}")
    print(f"Details saved to: {results_path}")