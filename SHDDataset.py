import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import math


class SHDDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        train=True,
        t_bin_size=0.001,
        max_duration=1.37
    ):
        super().__init__()

        if t_bin_size <= 0:
            raise ValueError("t_bin_size must be positive")

        self.root = root
        self.transform = transform
        self.train = train
        self.t_bin_size = t_bin_size

        filename = "shd_train.h5" if train else "shd_test.h5"

        # Load the sparse spike events, not dense spike matrices.
        with h5py.File(root / filename, "r") as f:
            self.times = [
                np.asarray(t, dtype=np.float32)
                for t in f["spikes"]["times"]
            ]

            self.units = [
                np.asarray(u, dtype=np.int16)
                for u in f["spikes"]["units"]
            ]

            self.labels = torch.as_tensor(
                f["labels"][:],
                dtype=torch.long
            )

        self.num_neurons = 700
        self.tensor_time = math.floor(
            max_duration / t_bin_size
        ) + 1

        # Check that the configured duration covers the data.
        actual_max = max(
            (float(np.max(t)) for t in self.times if len(t)),
            default=0.0
        )

        if actual_max > max_duration:
            raise ValueError(
                f"Maximum spike time {actual_max:.6f}s "
                f"exceeds max_duration {max_duration:.6f}s"
            )

        print(
            f"SHD {'train' if train else 'test'}: "
            f"{len(self.labels)} samples, "
            f"{self.tensor_time} bins, "
            f"{self.num_neurons} neurons"
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        times = self.times[idx]
        units = self.units[idx]

        # Convert timestamps into integer bin indices.
        time_bins = np.floor(
            times / self.t_bin_size
        ).astype(np.int64)

        # Allocate ONLY the requested sample.
        spikes = torch.zeros(
            (self.tensor_time, self.num_neurons),
            dtype=torch.uint8
        )

        # Populate the binary spike matrix.
        if len(time_bins) > 0:
            spikes[
                torch.from_numpy(time_bins),
                torch.from_numpy(units.astype(np.int64))
            ] = 1

        if self.transform is not None:
            spikes = self.transform(spikes)

        return spikes, self.labels[idx]

    def shape(self):
        return (
            len(self),
            self.tensor_time,
            self.num_neurons
        )


if __name__ == "__main__":
    import pathlib

    home = pathlib.Path("~").expanduser()
    local_dir = home / "data" / "shd" / "SHD"

    ds = SHDDataset(local_dir, train=True)

    print("Dataset shape:", ds.shape())

    spikes, label = ds[0]

    print("Sample shape:", spikes.shape)
    print("Sample dtype:", spikes.dtype)
    print("Spike count:", torch.sum(spikes).item())
    print("Label:", label)