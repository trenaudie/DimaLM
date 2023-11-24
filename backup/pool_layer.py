import torch


def pool_logits(
    logits: torch.FloatTensor,
    sequence_lengths: torch.Tensor,
    pooler_type: str = "last",
    num_labels: int = 2,
) -> torch.FloatTensor:
    """
    Pool the logit tensors
    Args :
    sequence_lengths: torch.Tensor shape [batch_size] with position of the last valid token
    Returns:
    - logits pooled over the time dimension
    """
    batch_size = logits.size(0)
    seq_len = logits.size(1)
    print(f"inside pool_logits pooler_type {pooler_type}")
    if isinstance(sequence_lengths, int):
        sequence_lengths = torch.tensor([sequence_lengths])
    if pooler_type == "avg":
        # create mask of valid tokens then get mean
        # sequence_lengths is defined as above, with the position of the LAST VALID TOKEN
        #   - for ex. with -1 if the last valid token is valid
        sequence_lengths[sequence_lengths == -1] = seq_len - 1
        mask = torch.arange(logits.size(1), device=logits.device).expand(
            batch_size, logits.size(1)
        ) <= sequence_lengths.unsqueeze(1)
        logits_sum = torch.sum(
            logits * mask.unsqueeze(2).to(torch.int), dim=1
        )  # -> B,2
        pooled_logits = logits_sum / (sequence_lengths.unsqueeze(1).to(torch.float) + 1)
    elif pooler_type == "last":
        print(f"sequence_lengths in pool layer.py :  {sequence_lengths}")
        print(f"logits sample 0  {logits[0,63:, :]}")
        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]
        print(f"pooled_logits {pooled_logits.shape}")
        print(f"pooled_logits {pooled_logits}")
    elif pooler_type == "max_abs":
        # must obtain the logit for the maximum absolute value over the last two dims
        batch_size = logits.size(0)
        min_value = -1e8
        mask = torch.arange(logits.size(1), device=logits.device).expand(
            batch_size, logits.size(1)
        ) <= sequence_lengths.unsqueeze(1)
        mask[sequence_lengths == -1] = True
        logits[~mask] = min_value
        logitsabs = torch.abs(logits)
        # goal : get the value of logits corresponding to the max value of logitsabs, over dims 1 and 2

        # Get the indices of the maximum absolute values along dims 1 and 2
        max_indices_dim1 = torch.argmax(logitsabs.max(dim=2).values, dim=1)

        # Use the indices to gather the values from the original logits tensor
        pooled_logits = logits[
            torch.arange(batch_size).unsqueeze(1).expand(batch_size, num_labels),
            max_indices_dim1.unsqueeze(1).expand(batch_size, num_labels),
            torch.arange(num_labels).unsqueeze(0).expand(batch_size, num_labels),
        ]

    return pooled_logits


def pool_hidden_states(
    hidden_states: torch.FloatTensor,
    sequence_lengths: torch.Tensor,
    pooler_type: str = "none",
) -> torch.FloatTensor:
    if pooler_type == "none":
        return hidden_states
    batch_size, seq_len, hidden_size = hidden_states.size()
    if pooler_type == "avg":
        sequence_lengths[sequence_lengths == -1] = seq_len - 1
        mask = torch.arange(seq_len, device=hidden_states.device).expand(
            batch_size, seq_len
        ) <= sequence_lengths.unsqueeze(
            1
        )  # [B,T]
        hidden_states_sum = torch.sum(
            hidden_states * mask.unsqueeze(-1).to(torch.int), dim=1
        )  # -> B,2
        pooled_hs = hidden_states_sum / (sequence_lengths.unsqueeze(1) + 1)
        pooled_hs = pooled_hs.unsqueeze(1).expand(batch_size, seq_len, hidden_size)
    return pooled_hs
