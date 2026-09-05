"""
Adapts our own build_dpo_dataset.py output (train.json: rows with
user_id/prompt/chosen/rejected, plain strings) into the HF Dataset
UserDPOTrainer expects: prompt column = f"USER: {int_id}" +
tokenizer.eos_token*n_user_tokens + sep + <chat-template-wrapped query>,
chosen/rejected unchanged.

Chat-template wrapping (see chat: "why are [P-RLHF's generations] so bad?"
thread -> "can we change the convention to that of used in their PSOUPS
experiment?!"): the ORIGINAL repo's own prlhf/utils.py does NOT use one
convention -- build_tldr_dataset_dpo_synthetic (GPT-J, a non-chat base
model) leaves the query as raw text after the identifier+sep, but
build_psoups_dataset_dpo (a real chat model) wraps it as f"<|user|>\\n
{prompt}\\n<|assistant|>\\n" AFTER the identifier+sep, at BOTH training
(build_psoups_dataset_dpo) and generation (build_psoups_prompts) time --
same for PRISM with Llama-3's own header tokens. Since Qwen3-0.6B is a
chat model (like the ones used for PSOUPS/PRISM), THAT is the convention
this port should have followed from the start, not TLDR's -- confirmed via
a real diagnostic (job 328746) that the chat-wrapped variant produces
direct, on-topic answers on the existing (un-fixed) checkpoint, vs. the
old raw format's "echoes another question before answering" pattern.

The identifier span (BOS -> sep) is UNTOUCHED by this change -- UserModel.
get_user_ids (prlhf/user_model.py) requires that exact "USER: {id}" text,
so the chat template only wraps what comes AFTER sep, never before/around
the identifier. Uses tokenizer.apply_chat_template(..., enable_thinking=
False) (real Qwen3 chat markers via the tokenizer itself, not a hardcoded
string) -- enable_thinking=False matches eval/generate.py's own call
exactly (the shared harness every other method already uses), so Qwen3's
own reasoning mode never activates here either.

Mirrors prlhf/utils.py's own build_tldr_dataset_dpo_synthetic, including
its CRITICAL "duplicate every real-user training row under user_id=0 too"
trick -- see prlhf/user_dpo_trainer.py's own UserDPOTrainer.compute_loss:
alpha splits the batch loss as alpha*mean(loss[user_id!=0]) +
(1-alpha)*mean(loss[user_id==0]) PER BATCH, so without generic-user
duplicate rows actually present in the training set, the (1-alpha) term
is always zero and the generic/index-0 embedding never receives any
gradient at all. This isn't a TLDR-specific quirk, it's required by
UserDPOTrainer's own architecture -- confirmed by reading that file
directly, not inferred.

Pure stdlib + `datasets` -- no dependency on the dpo_LRF package, so
env_prlhf never needs anything from our main env installed into it.
"""
import json
from typing import Optional


def assign_user_ids(user_ids_raw) -> dict:
    """Deterministic sorted user_id (string) -> int id (1..N) mapping --
    0 is reserved for the generic user (P-RLHF's own convention). Sorted,
    not insertion-order, so re-running this always produces the SAME
    mapping from the same input rows regardless of file order -- no
    randomness, since every method in the comparison must train/eval on
    identical data (see chat: "we really need to compare on exact same
    data")."""
    return {uid: i + 1 for i, uid in enumerate(sorted(user_ids_raw))}


_FIXED_DATE_STRING = "26 Jul 2024"  # see below.


def build_prompt(int_id: int, sep: str, n_user_tokens: int, query: str, tokenizer) -> str:
    # tokenizer.eos_token (was hardcoded "<|endoftext|>") -- real bug found
    # while investigating why n10/llama's own prlhf training crashed at
    # step 0 with ValueError: invalid literal for int() ... '23<|endoftext|>...'.
    # "<|endoftext|>" IS Qwen3's own eos_token (a real, registered special
    # token there, so UserModel.get_user_ids's own
    # tokenizer.batch_decode(..., skip_special_tokens=True) correctly
    # strips it) -- but Llama-3.2-3B-Instruct's tokenizer has no such
    # token at all (its own eos_token is "<|eot_id|>"), so the literal
    # string gets silently byte-pair-encoded as ordinary text instead of
    # one clean special token, survives skip_special_tokens, and breaks
    # the "USER: {id}" int() parse downstream. Deriving it from the
    # tokenizer itself is model-agnostic and a no-op for Qwen (same
    # value) -- see chat: "can we fix the previous problems with
    # dim1/refproblog/prlhf/fspo?".
    user_identifier = f"USER: {int_id}" + (tokenizer.eos_token * n_user_tokens)
    # See module docstring -- chat markers strictly AFTER sep, identifier span untouched.
    # date_string pinned -- see dpo_LRF/kadapter_dpo/run_kadapter_dpo.py's
    # own identical fix, same incident: Llama-3.x's own chat template
    # embeds "Today Date: <today>" via strftime_now() when date_string
    # isn't passed, making this function's output non-deterministic across
    # different calendar days -- immaterial for Qwen (no such template
    # field) but a real correctness risk for llama/n10 (this repo now
    # supports Llama via UserLlamaForCausalLM/--model-class llama).
    chat_query = tokenizer.apply_chat_template(
        [{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        date_string=_FIXED_DATE_STRING,
    )
    return user_identifier + sep + chat_query


def load_personalllm_seen_dpo(train_path: str, sep: str, n_user_tokens: int, tokenizer, user2idx: Optional[dict] = None):
    """Returns (train_dataset, user2idx, n_users) for the SEEN population.
    user2idx: {original user_id string: int id (1..n_users)} -- pass an
    existing mapping to reuse it (e.g. matching an eval-time mapping),
    otherwise a fresh deterministic one is built from this file's users.

    Every real row (user_id=i>0) ALSO gets a duplicate row re-prefixed
    with "USER: 0" (see module docstring) -- doubles train set size,
    required for the generic embedding to ever receive gradient.
    """
    from datasets import Dataset

    with open(train_path) as f:
        rows = json.load(f)

    if user2idx is None:
        user2idx = assign_user_ids({r["user_id"] for r in rows})
    n_users = len(user2idx)

    prompts, chosens, rejecteds = [], [], []
    for r in rows:
        int_id = user2idx[r["user_id"]]
        prompts.append(build_prompt(int_id, sep, n_user_tokens, r["prompt"], tokenizer))
        chosens.append(r["chosen"])
        rejecteds.append(r["rejected"])
        # Generic-user duplicate -- see module docstring.
        prompts.append(build_prompt(0, sep, n_user_tokens, r["prompt"], tokenizer))
        chosens.append(r["chosen"])
        rejecteds.append(r["rejected"])

    train_dataset = Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})
    return train_dataset, user2idx, n_users
