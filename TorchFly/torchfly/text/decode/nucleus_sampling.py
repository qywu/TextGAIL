import torch
import torch.nn as nn
import torch.nn.functional as F

# def top_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
#     """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
#         Args:
#             logits: logits distribution shape (vocabulary size)
#             top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
#             top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
#                 whose total probability mass is greater than or equal to the threshold top_p.
#                 In practice, we select the highest probability tokens whose cumulative probability mass exceeds
#                 the threshold top_p.
#     """
#     # batch support!
#     if top_k > 0:
#         values, _ = torch.topk(logits, top_k)
#         min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
#         logits = torch.where(logits < min_values, 
#                              torch.ones_like(logits, dtype=logits.dtype) * -float('Inf'), 
#                              logits)
#     if top_p > 0.0:
#         # Compute cumulative probabilities of sorted tokens
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

#         # Remove tokens with cumulative probability above the threshold
#         sorted_indices_to_remove = cumulative_probabilities > top_p
#         # Shift the indices to the right to keep also the first token above the threshold
#         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         sorted_indices_to_remove[..., 0] = 0
        
#         sorted_logits = sorted_logits.masked_fill_(sorted_indices_to_remove, filter_value)
#         logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)
    
#     return logits

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-1e4, min_tokens_to_keep=1, is_log_probs=False):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        if is_log_probs:
            # log probs instead of logits
            cumulative_probs = torch.cumsum(torch.exp(sorted_logits), dim=-1)
        else:
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits