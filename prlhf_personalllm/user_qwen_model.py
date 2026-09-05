"""
Adds Qwen3 support to P-RLHF's model_class dispatch (prlhf/train_language_
model_dpo.py, prlhf/generate.py) -- their own dispatch only knows gptneo/
gptj/llama (prlhf/user_language_model.py). No edits to any vendored file.

UserLlamaMixin (imported unmodified below) splices the per-user embedding
into inputs_embeds via self.model.embed_tokens / self.config.hidden_size --
exactly the attribute path Qwen3ForCausalLM also uses (both are decoder-
only causal LMs following the same `model.embed_tokens`/`model.layers`
convention in transformers). So this is a mechanical addition mirroring
UserLlamaForCausalLM's own one-line body, not a fork of the mixin logic.

Usage (mirrors prlhf/user_language_model.py's own UserLlamaForCausalLM):
    from user_qwen_model import UserQwen3ForCausalLM
    model = UserQwen3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path="Qwen/Qwen3-0.6B",
        user_model_type="individual", ...
    )
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prlhf"))

from transformers import Qwen3ForCausalLM
from user_language_model import UserLlamaMixin


class UserQwen3ForCausalLM(UserLlamaMixin, Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
