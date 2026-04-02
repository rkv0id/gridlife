import torch

from gridlife.engine.partition import compute_row_ranges, merge_strips, split_grid


class TestComputeRowRanges:
    def test_even_split(self) -> None:
        ranges = compute_row_ranges(100, 4)
        assert ranges == [(0, 25), (25, 50), (50, 75), (75, 100)]

    def test_uneven_split(self) -> None:
        ranges = compute_row_ranges(10, 3)
        # 10 / 3 = 3 remainder 1, first worker gets the extra row
        assert ranges == [(0, 4), (4, 7), (7, 10)]

    def test_more_remainder(self) -> None:
        ranges = compute_row_ranges(11, 4)
        # 11 / 4 = 2 remainder 3, first 3 workers get an extra row
        assert ranges == [(0, 3), (3, 6), (6, 9), (9, 11)]

    def test_single_worker(self) -> None:
        ranges = compute_row_ranges(64, 1)
        assert ranges == [(0, 64)]

    def test_covers_all_rows(self) -> None:
        for height in [7, 13, 64, 100, 257]:
            for n in range(1, min(height + 1, 10)):
                ranges = compute_row_ranges(height, n)
                assert ranges[0][0] == 0
                assert ranges[-1][1] == height
                for i in range(len(ranges) - 1):
                    assert ranges[i][1] == ranges[i + 1][0]


class TestSplitMerge:
    def test_roundtrip(self) -> None:
        grid = torch.randn(2, 64, 32)
        for n in [1, 2, 3, 4, 7]:
            strips = split_grid(grid, n)
            assert len(strips) == n
            merged = merge_strips(strips)
            assert torch.equal(merged, grid)

    def test_uneven_roundtrip(self) -> None:
        grid = torch.randn(1, 13, 8)
        strips = split_grid(grid, 5)
        heights = [s.shape[1] for s in strips]
        assert sum(heights) == 13
        merged = merge_strips(strips)
        assert torch.equal(merged, grid)
