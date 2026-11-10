import h5py
import numpy as np
import torch

from datetime import datetime, timedelta
from torch.utils.data import Dataset


def load_taxibj(filename):
    # Load both the datapoints and the associated date of each point
    with h5py.File(filename, "r") as f:
        data = np.array(f["data"], dtype=np.float32)
        dates = np.array(f["date"])

    return data, dates


def parse_date(value):
    # Convert bytes to string
    date = str(int(value))

    # Date is of the format YYYYMMDDTT, with TT between 0-48
    # Convert the raw date to a datetime object
    day = datetime.strptime(date[:8], "%Y%m%d")

    # Add the time to the datetime object
    slot = int(date[8:])
    timestamp = day + timedelta(minutes=(slot - 1) * 30)

    return timestamp


class TaxiBJDataset(Dataset):

    def __init__(self, data, dates, t_params=(5, 3, 3)):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.t_params = t_params

        # Map each date to an actual datetime timestamp
        timestamps = [parse_date(d) for d in dates]

        # Create a mapping
        self.timestamp_to_index = {
            timestamp: i
            for i, timestamp in enumerate(timestamps)
        }

        # Store each sample. Each sample consists of a fixed number of:
        # - close points: timestamps before the current timestamp
        # - period points: days before the current day
        # - trend points: weeks before the current week
        self.samples = []

        lc, lp, lt = t_params

        # Generate valid training samples
        # The model needs three input tensors: closeness, period and trend
        for i, timestamp in enumerate(timestamps):

            # Only use observations between 06:00 and 23:59
            if not (6 <= timestamp.hour <= 23):
                continue

            # Fill the closeness tensor with x previous timestamps
            closeness = [
                timestamp - timedelta(minutes=30 * j)
                for j in range(lc, 0, -1)
            ]

            # Fill the period tensor with y previous days
            period = [
                timestamp - timedelta(days=j)
                for j in range(lp, 0, -1)
            ]

            # Fill the trend tensor with z previous weeks
            trend = [
                timestamp - timedelta(weeks=j)
                for j in range(lt, 0, -1)
            ]

            # Only combinations where no data is missing are allowed
            # According to the dataset Github, 7.2% of slots is missing
            required = closeness + period + trend

            if not all(
                t in self.timestamp_to_index
                for t in required
            ):
                continue

            # Map each timestamp in the tensor to an index

            close_indices = [
                self.timestamp_to_index[t]
                for t in closeness
            ]

            period_indices = [
                self.timestamp_to_index[t]
                for t in period
            ]

            trend_indices = [
                self.timestamp_to_index[t]
                for t in trend
            ]

            # Store samples, and give each sample an own index -> this is the current/target timestep
            self.samples.append((
                close_indices,
                period_indices,
                trend_indices,
                i
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Retrieve the index of each part of the training sample
        close_idx, period_idx, trend_idx, target_idx = self.samples[index]

        # Find the corresponding datapoint
        closeness = self.data[close_idx]
        period = self.data[period_idx]
        trend = self.data[trend_idx]
        # Find the corresponding label
        target = self.data[target_idx]

        # Convert [steps, channels, H, W]
        # into [channels * steps, H, W]
        closeness = closeness.flatten(0, 1)
        period = period.flatten(0, 1)
        trend = trend.flatten(0, 1)

        return closeness, period, trend, target
