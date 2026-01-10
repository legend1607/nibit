import numpy as np
from torch.utils.data import Dataset

class PathPlanDataset(Dataset):
    def __init__(self, dataset_filepath):
        data = np.load(dataset_filepath)

        self.pc = data['pc'].astype(np.float32)           # (N, 2000, 4)
        self.start_mask = data['start'].astype(np.float32) # (N, 2000)
        self.goal_mask  = data['goal'].astype(np.float32)  # (N, 2000)
        self.free_mask  = data['free'].astype(np.float32)  # (N, 2000)

        # Make labels robust: force to {0,1} int
        astar = data['astar']
        astar = (astar > 0.5).astype(np.int64)           # (N, 2000), robust binarize
        self.astar_mask = astar
        self.token = data['token']

        # Class frequency + inverse-frequency weights (PointNet-style)
        flat = self.astar_mask.reshape(-1)
        counts = np.bincount(flat, minlength=2).astype(np.float32)  # [count0, count1]
        freq = counts / (counts.sum() + 1e-12)

        # weight_c = (max(freq) / freq_c)^(1/3)
        labelweights = np.power((freq.max() + 1e-12) / (freq + 1e-12), 1.0 / 3.0).astype(np.float32)
        self.labelweights = labelweights

        print("Class counts:", counts, "freq:", freq, "loss weights:", self.labelweights)

    def __len__(self):
        return len(self.pc)

    def __getitem__(self, index):
        pc_xyz = self.pc[index].astype(np.float32)
        pc_xyz_raw = pc_xyz

        pc_features = np.stack(
            (self.start_mask[index], self.goal_mask[index], self.free_mask[index]),
            axis=-1,
        ).astype(np.float32)  # (2000, 3)

        pc_labels = self.astar_mask[index].astype(np.int64)  # (2000,)

        return pc_xyz_raw, pc_xyz, pc_features, pc_labels, self.token[index]
