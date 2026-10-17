"""
P-DPO training on the personalllm SEEN-users population -- near-identical
to prlhf/train_language_model_dpo.py's own main(), but:
  - model_class dispatch extended with "qwen" (UserQwen3ForCausalLM,
    Phase 0's own small addition, see user_qwen_model.py) instead of their
    hardcoded gptneo/gptj/llama.
  - dataset loading via prlhf_data.load_seen_dpo instead
    of their load_openai_comparisons/load_psoups_comparisons/
    load_prism_comparisons.
UserDPOTrainer/DPOConfig/LoraConfig themselves are reused UNMODIFIED from
the vendored prlhf/ package -- this script is a bridge, not a fork of
their training logic.

Parameter choices (see the approved plan's own parameter table): every
flag defaults to P-RLHF's OWN reported default EXCEPT --num-train-epochs
(1, not their 2) -- the only "training budget" knob, equalized across
every method in the comparison table. --beta/--alpha (P-DPO's individual/
generic loss trade-off, NOT this pipeline's own dataset alpha) both keep
their defaults (method-intrinsic, not a shared knob). --lora-target-modules
defaults to a Qwen3/Llama-correct list ["q_proj","k_proj","v_proj","o_proj"]
-- their own literal list only ever matches q/k/v on any of their 3
architectures anyway (out_proj/fc_in/fc_out/wte don't exist on Llama or
Qwen3, PEFT silently skips non-matching names).

Usage:
    python train_seen.py --user-model individual \\
        --train-dataset-path .../train.json \\
        --output-dir .../prlhf_individual_seen
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prlhf"))

import torch
import transformers
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer as _TRLDPOTrainer

from prlhf_data import load_seen_dpo
from user_dpo_trainer import UserDPOTrainer
from user_language_model import UserGPTNeoForCausalLM, UserGPTJForCausalLM, UserLlamaForCausalLM
from user_qwen_model import UserQwen3ForCausalLM
from user_gemma_model import UserGemma3ForCausalLM

# Compat shim, NOT a P-RLHF/trl bug fix -- an accidental method-NAME
# collision between two unrelated things: trl==0.10.1's own DPOTrainer.
# get_batch_samples(self, model, batch) (their eval-time sample-generation
# helper for --generate_during_eval, which this pipeline never enables --
# no eval_dataset at all) vs. a DIFFERENT transformers.Trainer.
# get_batch_samples(self, epoch_iterator, num_batches, device) (grad-
# accumulation batch prefetching inside _inner_training_loop, added to
# transformers AFTER trl 0.10.1 was written). _inner_training_loop calls
# the latter unconditionally regardless of generate_during_eval, so without
# this patch EVERY training run crashes with "get_batch_samples() takes 3
# positional arguments but 4 were given" the moment epoch 1 starts (found
# via job 328297). Restoring transformers' own batch-prefetch behavior is
# safe here since trl's own version of this method is genuinely dead code
# for this pipeline.
def _get_batch_samples_compat(self, epoch_iterator, num_batches, device=None):
    return transformers.Trainer.get_batch_samples(self, epoch_iterator, num_batches, device)


_TRLDPOTrainer.get_batch_samples = _get_batch_samples_compat

# Second compat shim, same class of issue: transformers' current
# Trainer.training_step calls self.compute_loss(model, inputs,
# num_items_in_batch=...) -- a kwarg added after UserDPOTrainer's own
# compute_loss override (user_dpo_trainer.py) was written against an
# older Trainer.compute_loss signature. UserDPOTrainer's own loss is
# already an alpha-weighted per-batch mean (see its own compute_loss
# body), so this kwarg (used elsewhere for gradient-accumulation loss
# scaling) is simply unused here, not silently miscounted -- found via
# job 328298's own traceback, one step past the get_batch_samples fix.
_orig_compute_loss = UserDPOTrainer.compute_loss


def _compute_loss_compat(self, model, inputs, return_outputs=False, **kwargs):
    return _orig_compute_loss(self, model, inputs, return_outputs=return_outputs)


UserDPOTrainer.compute_loss = _compute_loss_compat

# Third compat shim, same class of issue again: transformers' current
# Trainer._maybe_log_save_evaluate calls self.log(logs, start_time) -- a
# second positional arg (start_time) added after UserDPOTrainer's own
# log(self, logs) override (user_dpo_trainer.py) was written. That override
# only exists to merge in _stored_metrics before calling super().log(logs)
# -- start_time is just forwarded logging metadata, not something
# UserDPOTrainer's own body ever touches, so accepting and dropping it is
# safe. Found via job 328313's own traceback ("UserDPOTrainer.log() takes
# 2 positional arguments but 3 were given"), one step past the
# compute_loss fix -- training got all the way to the first logging_steps
# checkpoint (step 10) before hitting this.
_orig_log = UserDPOTrainer.log


def _log_compat(self, logs, *args, **kwargs):
    return _orig_log(self, logs)


UserDPOTrainer.log = _log_compat

MODEL_CLASSES = {
    "gptneo": UserGPTNeoForCausalLM,
    "gptj": UserGPTJForCausalLM,
    "llama": UserLlamaForCausalLM,
    "qwen": UserQwen3ForCausalLM,
    "gemma": UserGemma3ForCausalLM,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dataset-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--user-model", required=True, choices=["individual", "cluster"])
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--model-class", default="qwen", choices=list(MODEL_CLASSES.keys()))

    p.add_argument("--n-user-tokens", type=int, default=1)
    p.add_argument("--n-user-clusters", type=int, default=5)
    p.add_argument("--initialize-from-vocab", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    p.add_argument("--random-range", type=float, default=0.5)
    p.add_argument("--sep", default="||")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--beta", type=float, default=0.5, help="DPO temperature -- their own default, method-intrinsic.")
    p.add_argument("--alpha", type=float, default=0.5,
                    help="P-DPO's own individual/generic loss trade-off -- NOT this pipeline's dataset alpha.")
    p.add_argument("--max-prompt-length", type=int, default=512)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--lr-scheduler-type", default="cosine")
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--num-train-epochs", type=int, default=1,
                    help="1, NOT their default 2 -- the sole training-budget knob, equalized across "
                         "every method in the comparison table. See plan's own governing rule.")
    p.add_argument("--logging-steps", type=int, default=10)

    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    p.add_argument("--sanity-check", action="store_true", help="Train on the first 200 (duplicated) rows only.")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="personalllm")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-group", default=None,
                    help="Sets WANDB_RUN_GROUP so this run groups alongside run_trl_dpo.py's/"
                         "run_kadapter_dpo.py's own --wandb-group runs (e.g. 'prism/seen_users/dim1') "
                         "-- unlike those two, this script had no group tag at all before, so its own "
                         "runs never showed up when browsing wandb grouped by that field. See chat: "
                         "'would it be possible to integrate in the prism/seen_users/prlhf group?'.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f"prlhf_{args.user_model}_seen"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "sep_token", None) is None:
        tokenizer.add_special_tokens({"sep_token": args.sep})
    if tokenizer.bos_token is None:
        # Qwen has no BOS token at all (bos_token_id=None) -- trl's own
        # add_bos_token_if_needed (trainer/utils.py) only prepends one
        # `if bos_token_id is not None`, so for Qwen NOTHING gets prepended
        # to prompt_input_ids. P-RLHF's own code hardcodes "position 0 is
        # always the BOS token, skip it" in multiple places (_get_user_ids,
        # _cat_user_embedding_to_input_sep) -- without a real bos_token_id,
        # that slice eats the FIRST REAL CONTENT TOKEN instead (confirmed:
        # job 328300 decoded "USER: 229" as ": 229" -- "USER" itself got
        # sliced off). Aliasing bos_token to eos_token is the standard fix
        # for BOS-less tokenizers -- gives Qwen a real, distinct
        # bos_token_id so add_bos_token_if_needed actually prepends it.
        tokenizer.bos_token = tokenizer.eos_token

    # tokenizer built BEFORE dataset loading now (moved up from below) --
    # build_prompt() needs it to apply the chat template (see chat: "can we
    # change the convention to that of used in their PSOUPS experiment?!").
    print(f"=== Loading personalllm SEEN train set ({args.train_dataset_path}) ===", flush=True)
    train_dataset, user2idx, n_users = load_seen_dpo(
        args.train_dataset_path, sep=args.sep, n_user_tokens=args.n_user_tokens, tokenizer=tokenizer,
    )
    if args.sanity_check:
        train_dataset = train_dataset.select(range(min(200, len(train_dataset))))
    print(f"  {len(train_dataset)} rows (real + generic-user duplicates), {n_users} users", flush=True)
    with open(os.path.join(args.output_dir, "user2idx.json"), "w") as f:
        json.dump(user2idx, f)

    model_class = MODEL_CLASSES[args.model_class]
    common_kwargs = dict(
        pretrained_model_name_or_path=args.model_name, user_model_type=args.user_model, tokenizer=tokenizer,
        n_users=n_users, n_clusters=args.n_user_clusters, n_user_tokens=args.n_user_tokens, seed=args.seed,
        # add_generic_user MUST be True -- required (not a tunable choice), see module/plan docstring:
        # this is what makes the generic/index-0 embedding receive gradient from every real user's loss.
        add_generic_user=True,
        initialize_from_vocab=args.initialize_from_vocab, most_common_tokens=None,
        random_range=args.random_range, sep=args.sep, low_cpu_mem_usage=True,
    )
    print(f"=== Loading policy + ref model ({args.model_name}, class={args.model_class}, "
          f"user_model={args.user_model}) ===", flush=True)
    # bfloat16 (was float32 for the policy / float16 for ref) -- real OOM
    # fix, see chat: "could you check on the n10s job?" (job 330302/330307:
    # CUDA OOM at step 3/3125 on a 48GB card, 45.78GiB already in use).
    # torch_dtype=torch.float32 for the policy kept ~2x the memory a bf16
    # load would (fp32 "master weights" for mixed-precision training only
    # matter when the OPTIMIZER updates those weights directly -- but LoRA
    # freezes the entire base model here, only the adapter/embedding
    # modules (modules_to_save) ever get gradients, so there's no accuracy
    # reason to keep the frozen 3B-param base in fp32 at all). DPOConfig's
    # own bf16=True below already does autocast mixed-precision regardless
    # of the base weights' own dtype. This never surfaced on qwen/n3 (Qwen3-
    # 0.6B is 5x smaller, fp32 vs bf16 is a few GB either way) but Llama-
    # 3.2-3B pushed a 48GB card over the edge. ref_model's own float16 ->
    # bfloat16 too, for consistency (same memory footprint, better dynamic
    # range for this model family, matches every other method's own
    # convention in this pipeline -- run_trl_dpo.py/run_kadapter_dpo.py/
    # fspo's train.py all load bf16 directly).
    model = model_class.from_pretrained(is_reference=False, torch_dtype=torch.bfloat16, **common_kwargs)
    model.config.use_cache = False
    ref_model = model_class.from_pretrained(is_reference=True, torch_dtype=torch.bfloat16, **common_kwargs)

    if not args.no_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group

    run_name = args.wandb_run_name or output_path
    training_args = DPOConfig(
        beta=args.beta, max_prompt_length=args.max_prompt_length, max_length=args.max_length,
        truncation_mode="keep_start", force_use_ref_model=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs, logging_steps=args.logging_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing, learning_rate=args.learning_rate,
        output_dir=os.path.join(args.output_dir, "trainer_output"),
        lr_scheduler_type=args.lr_scheduler_type, warmup_steps=args.warmup_steps,
        bf16=True, remove_unused_columns=False,
        report_to=([] if args.no_wandb else ["wandb"]), run_name=run_name,
        save_strategy="no",
    )

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        modules_to_save=["user_embedding", "cluster_embedding", "user_weight"],
        bias="none", task_type="CAUSAL_LM",
    )

    print(f"=== Training via UserDPOTrainer (alpha={args.alpha}, beta={args.beta}, "
          f"lr={args.learning_rate}, epochs={args.num_train_epochs}) ===", flush=True)
    pdpo_trainer = UserDPOTrainer(
        alpha=args.alpha, sep=args.sep, model=model, ref_model=ref_model, args=training_args,
        train_dataset=train_dataset, tokenizer=tokenizer, peft_config=peft_config,
    )
    # bfloat16 -- see chat: "could you check on the n10s job?" -> "but the
    # lora adaptor weights are in fp32 in all these methods right?" -> "yes
    # please uniformize" (job 330302/330307/330318's own real crash:
    # RuntimeError: expected mat1 and mat2 to have the same dtype). PEFT's
    # own get_peft_model() (called internally by UserDPOTrainer.__init__,
    # a DPOTrainer subclass, when given peft_config) always creates lora_A/
    # lora_B AND every modules_to_save-wrapped module (user_embedding/
    # cluster_embedding/user_weight here) in fp32, independent of the base
    # model's own dtype -- unlike run_trl_dpo.py's/run_kadapter_dpo.py's own
    # identical situation (same fix, this session), THIS method's own hand-
    # written _cat_user_embedding_to_input_sep (prlhf/user_language_model.py)
    # does a raw torch.cat mixing that fp32 embedding output with the (now
    # bf16) hidden states -- torch.cat isn't autocast-reconciled the way
    # nn.Linear/matmul are, so the mismatch surfaces as a hard crash here
    # specifically, not just a silently-tolerated inconsistency. ref_model
    # needs no equivalent cast -- it's a bare model (is_reference=True,
    # already loaded bfloat16 above), never PEFT-wrapped at all.
    pdpo_trainer.model = pdpo_trainer.model.to(torch.bfloat16)
    train_result = pdpo_trainer.train()
    # timing_scope v2 (dpo_LRF/timing_scope.py): training time = the trainer's own train loop minus its eval passes -- model loading,
    # tokenisation and checkpoint saving EXCLUDED, same definition as DPO / DPO-LRF. (The launcher used to time the WHOLE script by shell
    # wall-clock, which included all of those.) Written next to the checkpoint; the launcher registers it.
    _train_seconds = train_result.metrics.get("train_runtime")
    if _train_seconds is not None:
        _train_seconds -= sum(e.get("eval_runtime", 0.0) for e in pdpo_trainer.state.log_history)
        import json as _json, socket as _socket
        with open(os.path.join(args.output_dir, "train_timing.json"), "w") as _f:
            _json.dump({"timing_scope": "v2_no_model_loading", "training_time_seconds": _train_seconds,
                        "hardware": f"{__import__('torch').cuda.get_device_name(0)} x{__import__('torch').cuda.device_count()} @ {_socket.gethostname()}"}, _f, indent=2)
        print(f"=== training compute time (excl. loading/eval): {_train_seconds:.0f}s ===", flush=True)

    final_ckpt_path = os.path.join(args.output_dir, "final_ckpt")
    pdpo_trainer.save_model(final_ckpt_path)
    print(f"=== Saved final checkpoint to {final_ckpt_path} ===", flush=True)
    print("[INFO] Script execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
