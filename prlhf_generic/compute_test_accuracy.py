"""
Test-set preference accuracy for a trained P-RLHF checkpoint (train_seen.py's
own output) -- P-RLHF analogue of dpo_LRF/kadapter_dpo/run_trl_dpo.py's own
end-of-training accuracy block (dim1's "accuracy" registry field), added
because neither train_seen.py nor generate.py ever computed/registered one
(see chat: "why don't I see any connection to our registry" -> "can we
rather use their code?").

Reuses UserDPOTrainer's OWN get_batch_loss_metrics (prlhf/user_dpo_trainer.py)
unmodified -- it already computes reward_accuracies = (chosen_rewards >
rejected_rewards) via TRL's own inherited concatenated_forward, split out
per real user AND pooled across all real (non-generic) users under the
"rewards/user_each_accuracies" key. This script only adds: building an
eval-only Dataset from the held-out test.json (NOT train.json), wiring a
minimal UserDPOTrainer around the ALREADY-TRAINED policy + a fresh reference
model, looping eval batches through get_batch_loss_metrics(train_eval="eval"),
and writing the pooled result to a JSON sidecar. No loss/forward-pass math
is reimplemented here.

Subsampling: test.json has 500 users x 1000 prompts = 500,000 rows -- unlike
dim1's own accuracy block, there is NO dedup opportunity here, because
P-RLHF's own prompt TEXT embeds the user id ("USER: {id}..."), so the same
underlying question produces a DIFFERENT prompt string per user -- every
row needs its own forward pass. --n-test-prompts-per-user caps this to a
tractable count (default 20/user = 10,000 rows) -- first N rows per user in
file order, not random (same deterministic-slicing convention as
fspo_personalllm/generate.py's own "first num_shots, no retrieval").

No generic-user (user_id=0) duplicate rows are added here (unlike
load_personalllm_seen_dpo's own training-time behavior) -- those exist only
to give the generic embedding gradient during training; including them here
would silently double-count and dilute a REAL per-user population accuracy
with the (unrelated) generic user's own accuracy.

Usage:
    python compute_test_accuracy.py \\
        --checkpoint-dir /path/to/prlhf_runs/individual_seen_ut10 \\
        --test-dataset-path /path/to/test.json \\
        --output-path /path/to/test_accuracy.json \\
        --user-model individual --n-user-tokens 10
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoTokenizer
from trl import DPOConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prlhf"))

from personalllm_data import build_prompt
from user_language_model import UserLlamaForCausalLM
from user_qwen_model import UserQwen3ForCausalLM
from user_gemma_model import UserGemma3ForCausalLM

# --model-class dispatch -- see generate.py's own identical fix, same chat
# ("you forgot PRLHF also").
MODEL_CLASSES = {"llama": UserLlamaForCausalLM, "qwen": UserQwen3ForCausalLM, "gemma": UserGemma3ForCausalLM}
from user_dpo_trainer import UserDPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True,
                    help="train_seen.py's own --output-dir -- expects <checkpoint-dir>/final_ckpt/ "
                         "(the PEFT adapter) and <checkpoint-dir>/user2idx.json.")
    p.add_argument("--test-dataset-path", required=True,
                    help="Raw test.json (user_id/prompt/chosen/rejected rows) -- the SAME held-out "
                         "split dim1/dim3's own accuracy block uses, NOT train.json.")
    p.add_argument("--output-path", required=True)

    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--model-class", default="qwen", choices=list(MODEL_CLASSES.keys()))
    p.add_argument("--user-model", default="individual", choices=["individual", "cluster"],
                    help="MUST match the checkpoint's own --user-model at training time.")
    p.add_argument("--n-user-tokens", type=int, default=1)
    p.add_argument("--n-user-clusters", type=int, default=5)
    p.add_argument("--sep", default="||")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--beta", type=float, default=0.5,
                    help="MUST match the checkpoint's own training --beta (train_seen.py's own "
                         "default is 0.5) -- doesn't affect accuracy itself (beta-invariant, see "
                         "DPOConfig construction below), kept correct for clarity.")
    p.add_argument("--n-test-prompts-per-user", type=int, default=20,
                    help="See module docstring -- caps the 500k-row raw test.json to a tractable "
                         "count (no dedup opportunity exists for P-RLHF, unlike dim1's own accuracy).")
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--max-prompt-length", type=int, default=512)
    # 16 -- untested guess, job 328736's own real OOM -- concatenated_forward's chosen+rejected
    # log_softmax over the full 151,936-token vocab at batch_size=16/max_length=1536 tried to
    # allocate 19.58GiB in one shot (47.53GB total capacity). 4 is a real, conservative starting
    # point (roughly matches the safe per-GPU load found for the SAME log_softmax-over-vocab
    # OOM pattern in FSPO's own IPO stage this session) -- not yet re-measured for real headroom.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--unseen-as-generic", action="store_true",
                   help="Score EVERY test user with the generic user id 0 instead of skipping users missing from "
                        "user2idx.json -- how unseen users are handled at generation time (no user token of their "
                        "own, they route through the generic/common embedding). Used for the UNSEEN population's "
                        "preference accuracy -- see chat: 'and DPO and P-DPO?'.")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    user2idx_path = os.path.join(args.checkpoint_dir, "user2idx.json")
    with open(user2idx_path) as f:
        user2idx = json.load(f)
    n_users = len(user2idx)
    print(f"=== Loaded user2idx ({n_users} trained users) from {user2idx_path} ===", flush=True)

    with open(args.test_dataset_path) as f:
        test_rows = json.load(f)
    by_user = defaultdict(list)
    for r in test_rows:
        # str() -- user2idx.json's own keys are JSON-loaded (always str), but
        # r["user_id"] is a JSON INT for Community Alignment (PRISM's own are
        # already str) -- an unconditional `in`/index against the raw value
        # silently skipped every CA test row (0 rows survived, then a bare
        # torch.cat([]) crash downstream) -- same int/str mismatch class as
        # eval/generate.py's own bon-branch bug found earlier this session.
        uid = str(r["user_id"])
        if args.unseen_as_generic or uid in user2idx:  # SEEN population only by default -- skip any test user this checkpoint never trained on.
            by_user[uid].append(r)
    n_skipped_users = len({r["user_id"] for r in test_rows}) - len(by_user)
    if n_skipped_users:
        print(f"  {n_skipped_users} test-set users not in user2idx (not this checkpoint's seen "
              f"population) -- skipped entirely.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "sep_token", None) is None:
        tokenizer.add_special_tokens({"sep_token": args.sep})
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token  # see prlhf_generic/generate.py's own module docstring.

    prompts, chosens, rejecteds = [], [], []
    for uid, rows in by_user.items():
        int_id = 0 if args.unseen_as_generic else user2idx[uid]   # 0 = the generic user
        for r in rows[:args.n_test_prompts_per_user]:  # first N, deterministic -- see module docstring.
            prompts.append(build_prompt(int_id, args.sep, args.n_user_tokens, r["prompt"], tokenizer))
            chosens.append(r["chosen"])
            rejecteds.append(r["rejected"])
    from datasets import Dataset
    eval_dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})
    print(f"=== Eval set: {len(eval_dataset)} rows, {len(by_user)} users, "
          f"up to {args.n_test_prompts_per_user} prompts/user ===", flush=True)

    common_kwargs = dict(
        pretrained_model_name_or_path=args.model_name, user_model_type=args.user_model, tokenizer=tokenizer,
        n_users=n_users, n_clusters=args.n_user_clusters, n_user_tokens=args.n_user_tokens, seed=args.seed,
        add_generic_user=True, initialize_from_vocab=True, most_common_tokens=None,
        random_range=0.5, sep=args.sep, low_cpu_mem_usage=True,
    )
    print(f"=== Loading policy (trained adapter) + reference (raw) models ({args.model_class}) ===", flush=True)
    model_cls = MODEL_CLASSES[args.model_class]
    # bfloat16 -- see train_seen.py's own identical fix (chat: "could you
    # check on the n10s job?", job 330302/330307's own CUDA OOM). Same
    # reasoning here: no gradient ever touches these base weights (policy
    # is a frozen PeftModel eval pass, ref_model is is_reference=True), so
    # there's no accuracy reason to pay fp32's ~2x memory over bf16.
    policy_base = model_cls.from_pretrained(
        is_reference=False, torch_dtype=torch.bfloat16, **common_kwargs)
    policy_base.config.use_cache = False
    final_ckpt_path = os.path.join(args.checkpoint_dir, "final_ckpt")
    policy = PeftModel.from_pretrained(policy_base, final_ckpt_path, is_trainable=False)
    # bfloat16 -- see chat: "could you check on the n10s job?" -> "yes
    # please uniformize" (job 330318's own real crash, RuntimeError:
    # expected mat1 and mat2 to have the same dtype -- found HERE, not in
    # train_seen.py: PeftModel.from_pretrained() loads the adapter_model.
    # safetensors checkpoint back in WHATEVER dtype it was SAVED in
    # (fp32, from a training run that predates train_seen.py's own post-
    # wrap cast) -- policy_base being bf16 above does nothing for an
    # adapter loaded from a separate file afterward. Casting the whole
    # PeftModel again here is the robust fix regardless of what dtype the
    # checkpoint on disk happens to carry.
    policy = policy.to(torch.bfloat16)

    # is_reference=True -- REQUIRED, same as generate.py's own module docstring: the spliced user
    # embedding becomes a zero-length tensor, i.e. the TRUE untrained reference behavior.
    ref_model = model_cls.from_pretrained(is_reference=True, torch_dtype=torch.bfloat16, **common_kwargs)

    # Minimal DPOConfig -- only used to construct a working UserDPOTrainer around the ALREADY-TRAINED
    # policy so its own concatenated_forward/get_batch_loss_metrics can be reused; .train() is never
    # called. train_dataset is required by TRL's DPOTrainer.__init__ but unused -- pass eval_dataset
    # again as a harmless placeholder.
    dpo_config = DPOConfig(
        output_dir="/tmp/prlhf_test_accuracy_scratch",
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        max_length=args.max_length, max_prompt_length=args.max_prompt_length,
        truncation_mode="keep_start", force_use_ref_model=True,
        # beta -- accuracy = mean(chosen_rewards > rejected_rewards) is provably invariant to beta
        # (both rewards get the SAME positive scalar multiplier, see trl's own dpo_trainer.py:1235-6
        # -- beta cancels out of the sign comparison), so this couldn't have changed the accuracy
        # number even at TRL's own un-set default (0.1). Set to match the checkpoint's REAL training
        # value (train_seen.py's own --beta default, 0.5) anyway, for correctness/clarity, not because
        # it changes the result.
        beta=args.beta,
        bf16=True, remove_unused_columns=False, report_to=[],
    )
    # train_dataset is a REQUIRED kwarg for TRL's DPOTrainer.__init__ but never used (.train() is
    # never called) -- DPOTrainer.__init__ eagerly tokenizes whatever it's given, so passing the
    # full eval_dataset again here would double the (real, measured: ~300 examples/sec) tokenization
    # cost for nothing. A 2-row dummy keeps __init__ happy at near-zero cost.
    dummy_train_dataset = eval_dataset.select(range(min(2, len(eval_dataset))))
    trainer = UserDPOTrainer(
        alpha=1.0, sep=args.sep, model=policy, ref_model=ref_model, args=dpo_config,
        train_dataset=dummy_train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, peft_config=None,
    )

    print("=== Running eval batches through UserDPOTrainer's own get_batch_loss_metrics ===", flush=True)
    trainer.model.eval()
    eval_dataloader = trainer.get_eval_dataloader()
    all_reward_accuracies = []
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            _, metrics = trainer.get_batch_loss_metrics(trainer.model, batch, train_eval="eval", return_mean=False)
            # "user_each_accuracies" deliberately EXCLUDES the generic user (id 0, see prlhf/user_dpo_trainer.py) -- so with
            # --unseen-as-generic (every row scored as user 0) the per-batch accuracies live under "user_00_accuracies".
            _acc_key = "eval_rewards/user_00_accuracies" if args.unseen_as_generic else "eval_rewards/user_each_accuracies"
            if _acc_key in metrics:
                all_reward_accuracies.append(metrics[_acc_key])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} eval batches done ({time.time() - t0:.0f}s elapsed)", flush=True)

    all_reward_accuracies = torch.cat(all_reward_accuracies)
    population_accuracy = all_reward_accuracies.mean().item()
    n_pairs = len(all_reward_accuracies)
    print(f"=== population_accuracy={population_accuracy:.4f} over {n_pairs} (user, test-prompt) pairs "
          f"({len(by_user)} users) ===", flush=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({
            "accuracy": population_accuracy,
            "n_pairs": n_pairs,
            "n_users": len(by_user),
            "n_test_prompts_per_user": args.n_test_prompts_per_user,
            "checkpoint_dir": args.checkpoint_dir,
        }, f, indent=2)
    print(f"wrote {args.output_path}", flush=True)
    print("[INFO] Script execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
