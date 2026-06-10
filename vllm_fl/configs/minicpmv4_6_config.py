# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-V 4.6 config bridge for vLLM plugin.

transformers 4.x does not recognise model_type ``minicpmv4_6``.
This config bridge lets vLLM load the HuggingFace checkpoint without
upgrading transformers.

The model uses Qwen3.5 (hybrid GDN+full-attention) as its LLM backbone
with model_type ``qwen3_5_text`` in the nested ``text_config``.
"""

from transformers import AutoConfig, PretrainedConfig


class MiniCPMV4_6TextConfig(PretrainedConfig):
    """Minimal config for the Qwen3.5 LLM backbone inside MiniCPM-V 4.6.

    Carries all fields from the ``text_config`` dict in config.json so that
    vLLM's hf_text_config attribute has the expected attributes
    (num_attention_heads, hidden_size, etc.).
    """

    model_type = "qwen3_5_text"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        num_hidden_layers: int = 24,
        intermediate_size: int = 3584,
        head_dim: int = 256,
        vocab_size: int = 248094,
        max_position_embeddings: int = 262144,
        rms_norm_eps: float = 1e-6,
        rope_parameters: dict | None = None,
        layer_types: list | None = None,
        full_attention_interval: int = 4,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 16,
        linear_value_head_dim: int = 128,
        attn_output_gate: bool = True,
        mamba_ssm_dtype: str = "float32",
        mlp_only_layers: list | None = None,
        mtp_num_hidden_layers: int = 1,
        mtp_use_dedicated_embeddings: bool = False,
        partial_rotary_factor: float = 0.25,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = rope_parameters or {}
        self.layer_types = layer_types or []
        self.full_attention_interval = full_attention_interval
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_value_head_dim = linear_value_head_dim
        self.attn_output_gate = attn_output_gate
        self.mamba_ssm_dtype = mamba_ssm_dtype
        self.mlp_only_layers = mlp_only_layers or []
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range


class MiniCPMV4_6Config(PretrainedConfig):
    """Top-level config for MiniCPM-V 4.6 (multimodal).

    Holds nested ``text_config`` and ``vision_config`` sub-objects and
    exposes ``get_text_config()`` so vLLM can extract LLM-level attributes.
    """

    model_type = "minicpmv4_6"

    def __init__(
        self,
        text_config: dict | None = None,
        vision_config: dict | None = None,
        # Vision encoder fields
        drop_vision_last_layer: bool = False,
        insert_layer_id: int = 6,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        **kwargs,
    ):
        # Build nested sub-configs BEFORE super().__init__() because
        # transformers 5.7+ calls validate_token_ids → get_text_config()
        # during __init__, which requires self.text_config to already exist.
        if isinstance(text_config, dict):
            self.text_config = MiniCPMV4_6TextConfig(**text_config)
        elif text_config is None:
            self.text_config = MiniCPMV4_6TextConfig()
        else:
            self.text_config = text_config

        if isinstance(vision_config, dict):
            self.vision_config = PretrainedConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = PretrainedConfig()
        else:
            self.vision_config = vision_config

        self.drop_vision_last_layer = drop_vision_last_layer
        self.insert_layer_id = insert_layer_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        """Return the LLM text config (used by vLLM for hf_text_config)."""
        return self.text_config
