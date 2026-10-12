import numpy as np
import torch.utils.data as data
import torch

'''
Script used to generate sequence-matching data
'''

class SequenceDataset:
    def __init__(self, num_samples, width, delay=2, length=100, spike_threshold=0.1):
        self.num_samples = num_samples
        self.width = width
        self.delay = delay
        self.length = length
        self.spike_threshold = spike_threshold

        self.data = []

        self.initialise()

    def initialise(self):
        for i in range(self.num_samples):
            sample = np.random.rand(self.width, self.length)
            sample = np.where(sample > 1 - self.spike_threshold, 1, 0)

            target = sample.copy()
            # Shift over by delay
            target = np.roll(target, self.delay, axis=1)
            target[:, :self.delay] = 0

            self.data.append((sample, target))

    def __iter__(self):
        for sample, target in self.data:
            yield sample, target

    def __len__(self):
        return self.num_samples

class PatternDataset:
    def __init__(
        self,
        num_samples,
        width=5,
        length=100,
        spike_prob=0.05,
        pattern_neurons=(0, 1, 2),
        num_patterns=1,
    ):
        self.num_samples = num_samples
        self.width = width
        self.length = length
        self.spike_prob = spike_prob
        self.pattern_neurons = pattern_neurons
        self.num_patterns = num_patterns

        self.pattern_length = len(pattern_neurons)

        self.data = []

        self.initialise()

    def initialise(self):
        for _ in range(self.num_samples):

            # ----------------------------
            # Random background spikes
            # ----------------------------
            sample = (
                np.random.rand(self.width, self.length)
                < self.spike_prob
            ).astype(int)

            # One output neuron
            target = np.zeros((1, self.length), dtype=int)

            # ----------------------------
            # Insert patterns
            # ----------------------------
            for _ in range(self.num_patterns):

                # Need space for pattern + target spike
                max_start = (
                    self.length
                    - self.pattern_length
                    - 1
                )

                start = np.random.randint(
                    0,
                    max_start + 1
                )

                # Insert temporal pattern
                for offset, neuron in enumerate(
                    self.pattern_neurons
                ):
                    sample[
                        neuron,
                        start + offset
                    ] = 1

                # Teacher spike immediately after pattern
                target[
                    0,
                    start + self.pattern_length #- 1
                ] = 1

            self.data.append((sample, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

class ClusterPatternDataset:
    def __init__(
        self,
        num_samples=1000,
        noise_prob=0.05
    ):
        self.patterns = np.array([
            [1,1,1,0,0,0,0,0,0,0],
            [0,0,0,1,1,1,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,0],
        ])

        self.num_samples = num_samples
        self.noise_prob = noise_prob
        self.data = []

        self.initialise()

    def initialise(self):
        for _ in range(self.num_samples):

            label = np.random.randint(
                len(self.patterns)
            )

            sample = self.patterns[label].copy()

            noise = (
                np.random.rand(10)
                < self.noise_prob
            )

            sample = np.logical_xor(
                sample,
                noise
            ).astype(int)

            self.data.append(
                (sample, label)
            )

    def __iter__(self):
        return iter(self.data)

class SinWaveDS(data.Dataset):
    def __init__(self, channels, timesteps, wavelength=20, num_samples=1000):
        '''
        :param channels: number of channels in the generated data (i.e. width)
        :param timesteps: length of the data
        :param frequency: number of timesteps for one full sine wave cycle
        :param num_samples: number of samples to generate
        '''
        self.channels = channels
        self.timesteps = timesteps
        self.wavelength = wavelength
        self.num_samples = num_samples
        self.data = self.generate_data()

    def generate_data(self):
        data = torch.zeros((self.num_samples, self.channels, self.timesteps))

        for i in range(self.num_samples):
            phase_shift = torch.rand(1) * 2 * torch.pi

            for t in range(self.timesteps):
                angle = 2 * torch.pi * t / self.wavelength + phase_shift
                s = torch.sin(angle)
                spike_c = ((s + 1) / 2 * (self.channels - 1)).long()
                data[i, spike_c, t] = 1.0

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], 0

if __name__ == '__main__':
    ds = SequenceDataset(10, 5, length=20)

    for sample, target in ds.data:
        print(sample.shape, target.shape)

    print(ds.data[0][0])
    print(ds.data[0][1])