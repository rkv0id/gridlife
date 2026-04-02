import torch


def compute_row_ranges(height: int, num_workers: int) -> list[tuple[int, int]]:
    """
    Compute (start_row, end_row) for each worker. end_row is exclusive.
    Remainder rows are distributed one each to the first workers.
    """
    base = height // num_workers
    remainder = height % num_workers
    ranges: list[tuple[int, int]] = []
    offset = 0
    for i in range(num_workers):
        h = base + (1 if i < remainder else 0)
        ranges.append((offset, offset + h))
        offset += h
    return ranges


def split_grid(grid: torch.Tensor, num_workers: int) -> list[torch.Tensor]:
    """
    Split (channels, height, width) grid into N strips along the row axis.
    Returns list of (channels, strip_height, width) tensors.
    """
    ranges = compute_row_ranges(grid.shape[1], num_workers)
    return [grid[:, start:end, :].clone() for start, end in ranges]


def merge_strips(strips: list[torch.Tensor]) -> torch.Tensor:
    """Reassemble strips into a full grid by stacking along the row axis."""
    return torch.cat(strips, dim=1)
