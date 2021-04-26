import torch

class ChineseBERTBaseConfig:
    attention_dropout_prob = 0.1
    hidden_dropout_prob = 0.1
    hidden_size = 768
    num_attention_heads = 12
    num_hidden_layers = 12
    intermediate_size = 3072
    layer_norm_eps = 1e-05
    max_position_embeddings = 512
    vocab_size = 21128
    type_vocab_size = 2


class UnifiedRobertaBaseConfig:
    attention_dropout_prob = 0.1
    hidden_dropout_prob = 0.1
    hidden_size = 768
    num_attention_heads = 12
    num_hidden_layers = 12
    intermediate_size = 3072
    layer_norm_eps = 1e-05
    max_position_embeddings = 514
    # potentially remove it
    output_attentions = False
    output_hidden_states = False
    vocab_size = 50265
    padding_idx = 1
    type_vocab_size = 1
    padding_value = 1

class UnifiedGPT2MediumConfig:
    vocab_size = 50265
    n_positions = 1024
    n_ctx = 1024
    n_embd = 1024
    n_layer = 24
    n_head = 16
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = True
    padding_value = 1


class UnifiedGPT2SmallConfig:
    vocab_size = 50265
    n_positions = 1024
    n_ctx = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = False
    padding_value = 1

class UnifiedGPT2LargeConfig:
    vocab_size = 50265
    n_positions = 1024
    n_ctx = 1024
    n_embd = 1280
    n_layer = 36
    n_head = 20
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = True
    padding_value = 1
    
class UnifiedGPT2XLConfig:
    vocab_size = 50265
    n_positions = 1024
    n_ctx = 1024
    n_embd = 1600
    n_layer = 48
    n_head = 25
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = True
    padding_value = 1
    
class UnifiedGPT2DistillConfig:
    vocab_size = 50265
    n_positions = 1024
    n_ctx = 1024
    n_embd = 768
    n_layer = 6
    n_head = 12
    resid_pdrop = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    layer_norm_epsilon = 1e-5
    initializer_range = 0.02
    gradient_checkpointing = True
    padding_value = 1
