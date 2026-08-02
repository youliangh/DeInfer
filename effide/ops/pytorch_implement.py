import torch


def torch_paged_copy(src: torch.Tensor, block_table: torch.Tensor, block_per_seq: torch.Tensor,
                     remapping_table: torch.Tensor, dst: torch.Tensor):
    for seq_idx, (block_tensor, remapping_tensor, num_blocks) in enumerate(zip(block_table, remapping_table, block_per_seq)):
        for block_idx, remapping_block_idx in zip(block_tensor[:num_blocks], remapping_tensor[:num_blocks]):
            dst[remapping_block_idx] = src[block_idx]
