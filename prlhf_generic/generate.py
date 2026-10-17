"""
Generates {uid: {prompt: text}}-shaped output for a trained P-DPO
individual/cluster checkpoint (train_seen.py's own output) -- conforms to
the SAME JSON contract dpo_LRF/eval/generate.py's own output does (see
that script's own docstring: "any script producing {uid: {prompt:
generated_text}} in this exact shape is a valid input to
eval/downstream_eval.py"), so this checkpoint gets scored by the SAME
shared scoring script every other method uses, without any P-RLHF-
specific eval code.

Input population: a resolved dataset JSON built by (our main env's)
eval/dataset_builders/personalllm_test.py -- read directly as plain JSON
here, no dpo_LRF import (keeps env_prlhf fully decoupled from our main
env, same principle prlhf_data.py's own module docstring states).

User-id resolution: uses the checkpoint's own user2idx.json (written by
train_seen.py) to map each real user_id string (e.g. "user_82") to the
int id the embedding table was actually trained on. Any uid NOT in
user2idx.json (e.g. a genuinely unseen population) falls back to the
GENERIC user id 0 -- the correct, no-adaptation-needed behavior for
unseen users per the plan's own Phase 3 finding (confirmed directly in
user_model.py: IndividualUserModel.get_user_embeddings's own
"generic_user=True" path AND any prompt literally encoded with "USER: 0"
both route to the SAME index-0 embedding row through the normal forward
path) -- so this ONE script handles both seen and unseen populations
without a separate mode or script.

is_reference=False is REQUIRED when constructing the model for generation
-- confirmed directly in user_language_model.py's own
_cat_user_embedding_to_input_sep: when is_reference=True, the spliced
user_embedding is a ZERO-LENGTH tensor, silently disabling personalization
entirely. is_reference=True is only for DPO's own reference-model log-prob
computation during training, never for generation.

BOS-token handling: mirrors train_seen.py's own fix (Qwen has no native
BOS token -- job 328300's own traceback showed the FIRST REAL CONTENT
TOKEN getting sliced off instead of "USER: id" without it) -- tokenizer.
bos_token aliased to eos_token, then MANUALLY prepended to each encoded
prompt before generation (training went through TRL's own DPOTrainer
tokenization pipeline and its own add_bos_token_if_needed helper; this
script calls the tokenizer directly, so it replicates that prepend itself).

use_cache=False (both model.config and GenerationConfig) -- required, not
optional: _cat_user_embedding_to_input_sep re-parses "USER: {id}" out of
the FULL input_ids at every forward call; with KV-caching on, only the
newest token would be passed after the first generation step, and the
user identifier would be lost from the model's view entirely.

Usage:
    python generate.py \\
        --checkpoint-dir /mnt/scratch-dionysus/.../prlhf_runs/individual_seen \\
        --dataset-path /path/to/resolved_dataset.json \\
        --output-path /path/to/generations_alpha0.01.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import torch
from peft import PeftModel
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prlhf"))

from prlhf_data import build_prompt
from user_language_model import UserLlamaForCausalLM
from user_qwen_model import UserQwen3ForCausalLM
from user_gemma_model import UserGemma3ForCausalLM

# --model-class dispatch -- see chat: "you forgot PRLHF also" -- mirrors
# train_seen.py's own MODEL_CLASSES dict exactly (that script already had
# this; this one hardcoded UserQwen3ForCausalLM directly with no dispatch
# at all, a real gap for running generation against a Llama checkpoint).
MODEL_CLASSES = {"llama": UserLlamaForCausalLM, "qwen": UserQwen3ForCausalLM, "gemma": UserGemma3ForCausalLM}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True,
                    help="train_seen.py's own --output-dir -- expects <checkpoint-dir>/final_ckpt/ "
                         "(the PEFT adapter) and <checkpoint-dir>/user2idx.json.")
    p.add_argument("--dataset-path", required=True,
                    help="Resolved dataset JSON (from our main env's eval/dataset_builders/"
                         "personalllm_test.py) -- read as plain JSON, no dpo_LRF import.")
    p.add_argument("--output-path", required=True)

    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--model-class", default="qwen", choices=list(MODEL_CLASSES.keys()))
    p.add_argument("--user-model", default="individual", choices=["individual", "cluster"],
                    help="MUST match the checkpoint's own --user-model at training time.")
    p.add_argument("--n-user-tokens", type=int, default=1)
    p.add_argument("--n-user-clusters", type=int, default=5)
    p.add_argument("--sep", default="||")
    p.add_argument("--seed", type=int, default=123)
    # MUST match dpo_LRF/personalllm/gob_hparams.py's own GOB_MAX_NEW_TOKENS
    # (also 1024) -- every method's own downstream generation shares this
    # SAME response-length budget so the eval comparison stays apples-to-
    # apples (see chat: "can we make sure that the generation length 1024
    # is hardcoded across all methods to avoid silent-drift?"). Can't
    # import that constant directly -- this script's own module docstring
    # deliberately keeps env_prlhf decoupled from the dpo_LRF package (pure
    # stdlib + datasets/transformers/peft only) -- so this is a MANUALLY
    # kept-in-sync literal, not a shared import; if GOB_MAX_NEW_TOKENS ever
    # changes, update this line too.
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-prompt-length", type=int, default=512,
                   help="Prompt token cap, applied the way TRL's DPOTrainer applied it in training (train_seen.py: truncation_mode=keep_start, "
                        "after the BOS prepend) so generation sees prompts as they were seen in training. Same value as every other method's "
                        "prompt cap (SHARED_MAX_PROMPT_LENGTH). 0 disables.")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.dataset_path) as f:
        dataset = json.load(f)
    users = dataset["users"]

    user2idx_path = os.path.join(args.checkpoint_dir, "user2idx.json")
    with open(user2idx_path) as f:
        user2idx = json.load(f)
    n_users = len(user2idx)
    print(f"=== Loaded user2idx ({n_users} trained users) from {user2idx_path} ===", flush=True)

    n_fallback = sum(1 for uid in users if uid not in user2idx)
    if n_fallback:
        print(f"  {n_fallback}/{len(users)} dataset users NOT in this checkpoint's user2idx -- "
              f"falling back to the generic user (id=0) for those.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "sep_token", None) is None:
        tokenizer.add_special_tokens({"sep_token": args.sep})
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token  # see module docstring's own BOS-handling note

    print(f"=== Loading base model ({args.model_name}, user_model={args.user_model}, "
          f"model_class={args.model_class}) ===", flush=True)
    base_model = MODEL_CLASSES[args.model_class].from_pretrained(
        pretrained_model_name_or_path=args.model_name, user_model_type=args.user_model, tokenizer=tokenizer,
        n_users=n_users, n_clusters=args.n_user_clusters, n_user_tokens=args.n_user_tokens, seed=args.seed,
        add_generic_user=True, initialize_from_vocab=True, most_common_tokens=None,
        random_range=0.5, sep=args.sep, low_cpu_mem_usage=True,
        is_reference=False,  # REQUIRED -- see module docstring.
        # bfloat16 (was float32) -- see train_seen.py's own identical fix
        # (chat: "could you check on the n10s job?", job 330302/330307's
        # own CUDA OOM) -- this is a frozen base + PeftModel adapter at
        # generation time, no reason to pay fp32's memory here either.
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False

    final_ckpt_path = os.path.join(args.checkpoint_dir, "final_ckpt")
    print(f"=== Loading trained adapter from {final_ckpt_path} ===", flush=True)
    model = PeftModel.from_pretrained(base_model, final_ckpt_path, is_trainable=False)
    # bfloat16 -- see compute_test_accuracy.py's own identical fix (chat:
    # "could you check on the n10s job?" -> "yes please uniformize"). Same
    # gap: PeftModel.from_pretrained() loads the adapter checkpoint back in
    # whatever dtype it was SAVED in, independent of base_model's own
    # (already bf16) dtype -- base_model's own torch_dtype= above does
    # nothing for a separately-loaded adapter file.
    model = model.to(torch.bfloat16)
    model.eval()
    model.cuda()

    # Content hash of the loaded adapter weights -- see chat: "how to be more
    # robust for that in the future? can't we use parameter hashing?" (jobs
    # 328809/328810's own real bug: individual_seen_ut10/cluster3_seen_ut10's
    # cached generations_alpha0.01.json predated their retrained final_ckpt
    # by ~14-18h, but the old exists-check below resumed from them anyway --
    # scored the WRONG checkpoint's outputs). Same hashlib.sha256(...)
    # [:N]-truncated convention already used everywhere else in this
    # codebase for content-addressed caching (ref_log_prob_cache.py,
    # reward_score_cache.py, adapter_log_prob_cache.py) and for params_hash.txt
    # (run_trl_dpo.py) -- reused here, not a new pattern.
    checkpoint_hash = hashlib.sha256(
        open(os.path.join(final_ckpt_path, "adapter_model.safetensors"), "rb").read()
    ).hexdigest()[:16]

    # Always real (non-greedy) sampling -- see chat: "i think we should
    # remove the greedy option, and make sure we use their generation
    # process" (every branch of their own prlhf/generate.py sets
    # do_sample=True, none ever uses greedy).
    #
    # No temperature/top_p/top_k here (and no GenerationConfig object built
    # at all) -- see chat: "I think I would use the Qwen recommended
    # defaults everywhere... For any method that is not ours, we just use
    # the default sampling parameters for the model" -> "no this is not
    # what I want! I want to just pick the default parameters of the
    # model" (the model's own actual shipped defaults, NOT the model
    # card's "Best Practices" numbers hardcoded as our own literal
    # override -- and NOT their own paper's tldr/psoups recipe either,
    # top_k=0/top_p=1.0/no_repeat_ngram_size=3/repetition_penalty=1.15,
    # now dropped, still in git history if a literal reproduction run
    # ever needs it back). base_model is UserQwen3ForCausalLM.from_
    # pretrained(args.model_name) -- standard from_pretrained() loading
    # pulls in Qwen3-0.6B's own shipped generation_config.json (do_sample=
    # True, temperature=0.6, top_p=0.95, top_k=20); PeftModel wraps it
    # without replacing that attribute. Passing plain kwargs straight to
    # model.generate() below (no `generation_config=` argument) lets every
    # field this call doesn't set fall through to that real model default
    # -- building a fresh GenerationConfig(...) object instead would
    # silently replace it with the GenerationConfig CLASS's own defaults
    # (temperature=1.0, top_p=1.0, top_k=50), nothing to do with the model.
    generate_kwargs = dict(
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        do_sample=True, num_return_sequences=1, max_new_tokens=args.max_new_tokens,
        use_cache=False,  # REQUIRED -- see module docstring.
    )

    # Resumable: loaded from --output-path (if it already exists) before
    # decoding, flushed to disk after EVERY generated pair -- mirrors
    # dpo_LRF/personalllm/shared_prompt_eval.py's own product_of_experts_
    # decode resume_path convention (same {uid: {prompt: text}} shape), so
    # a job dying partway through a many-hour, unbatched (one pair at a
    # time, ~60-70s/pair observed) generation pass doesn't lose already-
    # generated answers -- just resubmit the same command and it picks up
    # where it left off.
    # checkpoint_hash sidecar validates the resume below -- if the adapter
    # weights changed since args.output_path was last written (a retrain
    # landed on the same --checkpoint-dir), the cached generations belong to
    # a DIFFERENT checkpoint and must not be resumed from. See checkpoint_hash's
    # own comment above for the real incident this fixes.
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    hash_path = args.output_path + ".checkpoint_hash"
    answers = {}
    if os.path.exists(args.output_path):
        stored_hash = open(hash_path).read().strip() if os.path.exists(hash_path) else None
        if stored_hash == checkpoint_hash:
            with open(args.output_path) as f:
                answers = json.load(f)
            n_done_already = sum(len(v) for v in answers.values())
            print(f"=== Resuming from {args.output_path} ({n_done_already} pairs already done, "
                  f"checkpoint_hash {checkpoint_hash} matches) ===", flush=True)
        else:
            print(f"=== {args.output_path} exists but its checkpoint_hash sidecar is "
                  f"missing/stale (stored={stored_hash}, current={checkpoint_hash}) -- the "
                  f"checkpoint changed since these generations were written. Ignoring cached "
                  f"generations, regenerating from scratch. ===", flush=True)
    with open(hash_path, "w") as f:
        f.write(checkpoint_hash)

    bos_id = tokenizer.bos_token_id
    total_pairs = sum(len(u["prompts"]) for u in users.values())
    done = sum(len(v) for v in answers.values())
    t0 = time.time()
    n_truncated = 0
    _done0 = done
    with torch.no_grad():
        for uid, u in users.items():
            int_id = user2idx.get(uid, 0)
            answers.setdefault(uid, {})
            for query in u["prompts"]:
                if query in answers[uid]:
                    continue
                text = build_prompt(int_id, args.sep, args.n_user_tokens, query, tokenizer)
                input_ids = tokenizer(text, return_tensors="pt").input_ids
                input_ids = torch.cat([torch.tensor([[bos_id]]), input_ids], dim=1)
                if args.max_prompt_length and input_ids.shape[1] > args.max_prompt_length:
                    input_ids = input_ids[:, :args.max_prompt_length]   # keep_start, as in training
                    n_truncated += 1
                input_ids = input_ids.to("cuda")

                output = model.generate(input_ids=input_ids, **generate_kwargs)
                new_tokens = output[0][input_ids.shape[1]:]
                answers[uid][query] = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                done += 1
                with open(args.output_path, "w") as f:
                    json.dump(answers, f, indent=2)
                if done % 20 == 0 or done == total_pairs:
                    print(f"  generated {done}/{total_pairs} ({time.time() - t0:.0f}s elapsed)", flush=True)

    print(f"=== {n_truncated} prompts truncated to {args.max_prompt_length} tokens (keep_start) ===", flush=True)
    print(f"wrote {args.output_path}", flush=True)

    # Timing sidecar, same {"inference_time_seconds": ...} shape/filename
    # dpo_LRF/eval/generate.py's own _write_timing writes -- downstream_eval.py
    # already looks for generation_timing_alpha{alpha}.json next to
    # --personalized-generations-path and folds it into the registry write
    # automatically (see chat: "will we get the inference time in the
    # registry?" -- it didn't before this, since this script is the
    # "completely separate external script that never writes one" its own
    # comment already called out). Measures ONLY this invocation's own
    # elapsed time (t0 above) -- on a resumed run that's the time to finish
    # whatever pairs were still missing, not the full job's cumulative time
    # across restarts, same limitation eval/generate.py's own timing has.
    with open(os.path.join(os.path.dirname(args.output_path),
                            f"generation_timing_alpha{dataset['metadata']['alpha']}.json"), "w") as f:
        import socket as _socket
        json.dump({"timing_scope": "v2_no_model_loading", "inference_time_seconds": time.time() - t0,
                   "hardware": f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()} @ {_socket.gethostname()}",
                   "n_generations": total_pairs, "n_generations_this_run": done - _done0}, f)

    print("[INFO] Script execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
