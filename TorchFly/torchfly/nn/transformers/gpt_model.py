import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math

# from ...utils.file_utils import gdrive_download
# from ..cuda import gpt_gelu as gelu
# assert installed
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
# from cudatest import GPT_GELU

# pylint:disable=no-member


@torch.jit.script
def gelu(x):
    """ GELU Activation Function
        math.sqrt(2 / math.pi) = 0.7978845608028654
    """
    return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3))))


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf, )
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        """
        This version is modified to support batch training
        mask needs to be precomputed
        """
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0

        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, mask):
        w = torch.matmul(q, k)
        w = w / math.sqrt(v.size(-1))
        # w = w * mask - 1e4 * (1 - mask)
        w.masked_fill_(~mask, -1e4)
        w = F.softmax(w, dim=-1)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            # (batch, head, head_features, seq_length)
            return x.permute(0, 2, 3, 1)
        else:
            # (batch, head, seq_length, head_features)
            return x.permute(0, 2, 1, 3)

    def forward(self, x, layer_past=None, mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            # transpose back cf below
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        # transpose to have same shapes for stacking
        present = torch.stack((key.transpose(-2, -1), value))

        a = self._attn(query, key, value, mask)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, mask=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, mask=mask)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2LMHead(nn.Module):
    """ Language Model Head for the transformer """
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2Model(nn.Module):
    """OpenAI GPT-2 model ("Language Models are Unsupervised Multitask Learners").
    """
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.config = config
        self.dropout = nn.Dropout(config.embd_pdrop)

        # word embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, position_ids=None, past=None, mask=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        # One-Hot
        if input_ids.dtype == torch.float32:
            input_shape = input_ids.shape[:-1]
            inputs_embeds = input_ids.matmul(self.wte.weight).unsqueeze(1)
        # Long Index
        else:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_ids.size(-1))
            inputs_embeds = self.wte(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # position embeddings
        position_ids = position_ids.view(-1, position_ids.size(-1))
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        presents = []

        for block, layer_past in zip(self.h, past):
            # added gradient checkpointing
            if self.gradient_checkpointing:
                hidden_states, present = torch.utils.checkpoint.checkpoint(block, hidden_states, layer_past, mask)
            else:
                hidden_states, present = block(hidden_states, layer_past, mask)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = position_ids.shape + (hidden_states.size(-1), )
        return hidden_states.view(*output_shape), presents


class GPT2SimpleLM(nn.Module):
    """OpenAI GPT-2 model with a Language Modeling head ("Language Models are Unsupervised Multitask Learners").
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_tied(self):
        """ Make sure we are sharing the embeddings
        """
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)

    def forward(self, input_ids, position_ids=None, past=None, mask=None):

        if past is None:
            past_length = input_ids.shape[1]
        else:
            # count self
            past_length = past[0].shape[3] + input_ids.shape[1]

        if mask is None:
            # print("mask is not provided")
            mask = torch.ones(input_ids.shape[0], past_length, dtype=torch.bool, device=input_ids.device)

        # Fast way to compute lower triangle attention mask
        # shape: (batch, num_head, key_length, query_length/seq_length)
        mask = mask.view(input_ids.shape[0], 1, 1, mask.shape[1]).repeat(1, self.config.n_head, mask.shape[1], 1)
        mask = mask & mask.permute(0, 1, 3, 2)
        mask = torch.tril(mask)
        mask = mask[:, :, -input_ids.shape[1]:, :]

        hidden_states, presents = self.transformer(input_ids, position_ids, past, mask)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits, presents
