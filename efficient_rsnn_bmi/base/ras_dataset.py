import torch
from stork.datasets import SpikingDataset

class RasDataset(SpikingDataset):
    def __init__(
        self,
        dataset,
        nb_steps,
        nb_units,
        p_drop=0.0,
        p_insert=0.0,
        sigma_t=0.0,
        time_scale=1,
        data_augmentation=False,
        dtype=torch.long,
    ):
        """
        This converter provides an interface for standard Ras datasets to dense tensor format.

        Args:
            dataset: (data,labels) tuple where data is in RAS format
            p_drop: Probability of dropping a spike (default 0)
            p_insert: Probability of inserting a spurious spike in any time cell (default 0)
            sigma_t: Amplitude of time jitter added to each spike in bins (default 0)
            time_scale: Rescales the time-dimension (second dimension) of the dataset used to adjust to discrete time grid.
        """
        super().__init__(
            nb_steps,
            nb_units,
            p_drop=p_drop,
            p_insert=p_insert,
            sigma_t=sigma_t,
            time_scale=time_scale,
            data_augmentation=data_augmentation
        )

        data, labels = dataset

        if self.time_scale == 1:
            Xscaled = data
        else:
            Xscaled = []
            for times, units in data:
                times = self.time_scale * times
                idx = times < self.nb_steps
                Xscaled.append((times[idx], units[idx]))

        self.data = Xscaled
        self.labels = labels
        self.dtype = dtype

        if type(self.labels) == torch.Tensor:
            self.labels = labels.type(dtype=self.dtype)

    def __len__(self):
        "Returns the total number of samples in dataset"
        return len(self.data)

    def __getitem__(self, index):
        "Returns one sample of data"

        times, units = self.data[index]
        times, units = self.preprocess_events(times, units)

        times = times.long()

        X = torch.zeros((self.nb_steps, self.nb_units), dtype=self.dtype)
        X[times, units] = 1.0
        y = self.labels[index].type(dtype=self.dtype)

        return X, y