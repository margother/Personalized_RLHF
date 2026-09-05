"""
One-off diagnostic (NOT part of the pipeline) -- compares P-RLHF's own raw
build_prompt() format against a chat-template-after-sep variant, on a
handful of real prompts, using the ALREADY-TRAINED individual_seen_ut10
checkpoint (no retrain). See chat: "why are [P-RLHF's generations] so bad?"
-> traced to every other method (eval/generate.py) wrapping queries with
tokenizer.apply_chat_template, while P-RLHF's own build_prompt never does
-> "can't we reverse, and use the generation with their format?" -- this
script tests whether inserting the chat template strictly AFTER the sep
token (so UserModel.get_user_ids's own "USER: {id}" parsing, which requires
the BOS-to-sep span to decode to EXACTLY that text, still works) produces
better generations on the EXISTING checkpoint, before committing to a full
regeneration.
"""
import json
import sys
import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, GenerationConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prlhf"))
from personalllm_data import build_prompt
from user_qwen_model import UserQwen3ForCausalLM

CHECKPOINT_DIR = "/mnt/scratch-dionysus/margot/personalization/prlhf_runs/individual_seen_ut10"
MODEL_NAME = "Qwen/Qwen3-0.6B"
SEP = "||"
N_USER_TOKENS = 10

TEST_QUERIES = [
    ("user_10", "How do I clean faux suede?"),
    ("user_101", "What movie won the Oscar Best Picture award in 2020?"),
    ("user_104", "Do fairies exist?"),
]


def main():
    with open(os.path.join(CHECKPOINT_DIR, "user2idx.json")) as f:
        user2idx = json.load(f)
    n_users = len(user2idx)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "sep_token", None) is None:
        tokenizer.add_special_tokens({"sep_token": SEP})
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token

    base_model = UserQwen3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, user_model_type="individual", tokenizer=tokenizer,
        n_users=n_users, n_clusters=5, n_user_tokens=N_USER_TOKENS, seed=123,
        add_generic_user=True, initialize_from_vocab=True, most_common_tokens=None,
        random_range=0.5, sep=SEP, low_cpu_mem_usage=True,
        is_reference=False, torch_dtype=torch.float32,
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, os.path.join(CHECKPOINT_DIR, "final_ckpt"), is_trainable=False)
    model.eval()
    model.cuda()

    gen_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        do_sample=True, temperature=1.0, top_k=0, top_p=1.0, no_repeat_ngram_size=3,
        repetition_penalty=1.15, num_return_sequences=1, max_new_tokens=200,
        use_cache=False,
    )
    bos_id = tokenizer.bos_token_id

    # enable_thinking=False -- matches eval/generate.py's own chat_template call
    # (used by every other method: base/dim1/dim3/rag) exactly, see chat: "why
    # are [P-RLHF's generations] so bad?" thread. Without this, Qwen3 enters its
    # own reasoning mode (first test run showed all 3 examples still inside a
    # <think> block at 200 tokens, never reaching an answer) -- the template's
    # own conditional inserts an empty "<think>\n\n</think>\n\n" when this is
    # explicitly False, forcing a direct answer instead, same as every other
    # method already gets.
    chat_suffix = "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    with torch.no_grad():
        for uid, query in TEST_QUERIES:
            int_id = user2idx.get(uid, 0)

            # (A) OBSOLETE as a "raw format" comparison -- build_prompt() itself now
            # applies the chat template by default (see chat: "can we change the
            # convention to that of used in their PSOUPS experiment?!" -- this
            # diagnostic's own finding is what motivated that production change).
            # Kept only as a historical record of the before/after test; not part
            # of the pipeline. Passing tokenizer just to keep this callable.
            raw_text = build_prompt(int_id, SEP, N_USER_TOKENS, query, tokenizer)
            raw_ids = tokenizer(raw_text, return_tensors="pt").input_ids
            raw_ids = torch.cat([torch.tensor([[bos_id]]), raw_ids], dim=1).to("cuda")
            raw_out = model.generate(input_ids=raw_ids, generation_config=gen_config)
            raw_gen = tokenizer.decode(raw_out[0][raw_ids.shape[1]:], skip_special_tokens=True).strip()

            # (B) chat-template-after-sep -- identifier span (BOS->sep) UNCHANGED,
            # only what comes after sep gets the real Qwen3 chat wrapper.
            user_identifier = f"USER: {int_id}" + ("<|endoftext|>" * N_USER_TOKENS)
            chat_text = user_identifier + SEP + chat_suffix.format(query=query)
            chat_ids = tokenizer(chat_text, return_tensors="pt").input_ids
            chat_ids = torch.cat([torch.tensor([[bos_id]]), chat_ids], dim=1).to("cuda")
            chat_out = model.generate(input_ids=chat_ids, generation_config=gen_config)
            chat_gen = tokenizer.decode(chat_out[0][chat_ids.shape[1]:], skip_special_tokens=True).strip()

            print(f"\n=== {uid}: {query!r} ===")
            print(f"[RAW]  {raw_gen[:300]!r}")
            print(f"[CHAT] {chat_gen[:300]!r}")

    print("\n[INFO] Script execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
