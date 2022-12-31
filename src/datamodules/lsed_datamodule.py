from typing import Optional, Tuple, List
from lightgbm import train
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import TemporalData
import pandas as pd
import os

# there has some trouble with import temporaldataloader form torch_geometric.loader... so I copy it here
class TemporalDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges succesive events of a
    :class:`torch_geometric.data.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The :obj:`~torch_geometric.data.TemporalData`
            from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(self, data: TemporalData, batch_size: int = 1, **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'shuffle' in kwargs:
            del kwargs['shuffle']

        self.data = data
        self.events_per_batch = batch_size

        if kwargs.get('drop_last', False) and len(data) % batch_size != 0:
            arange = range(0, len(data) - batch_size, batch_size)
        else:
            arange = range(0, len(data), batch_size)

        super().__init__(arange, 1, shuffle=False, collate_fn=self, **kwargs)

    def __call__(self, arange: List[int]) -> TemporalData:
        return self.data[arange[0]:arange[0] + self.events_per_batch]
    

class LSED_datamodule(LightningDataModule):
    def __init__(self,
        data_dir: str = "data/",
        num_workers: int = 0,
        pin_memory: bool = True,
        snapshot: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.snapshot = snapshot
    
    def setup(self, stage: Optional[str] = None) -> None:
        data = pd.read_json(os.path.join(self.data_dir, "lsed.json"), orient="records")
        nodes = set(data["actor"].unique()) | set(data["recipient"].unique())
        node_id_map = {node: i for i, node in enumerate(nodes)}
        data.actor.replace(node_id_map, inplace=True)
        data.recipient.replace(node_id_map, inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d')
        data['timestamp'] = (data['timestamp'] - data['timestamp'].min()).dt.days
        ## 0-8112 is training data
        train_df, val_df = data.loc[:8110], data.loc[8114:]
        nodes_in_train = set(train_df["actor"].unique()) | set(train_df["recipient"].unique())
        val_df = val_df[val_df.actor.isin(nodes_in_train) & val_df.recipient.isin(nodes_in_train)]
        if self.snapshot:
            for i in range(10):
                train_df.loc[(int(i*0.1*978) <= train_df.timestamp) & (train_df.timestamp < int((i+1)*0.1*978)), 'timestamp'] = i
            train_df = train_df.drop_duplicates(subset=['actor', 'recipient', 'timestamp'], keep='first')
            train_df = train_df.reset_index()
        self.train_batch_size = len(train_df)
        val_df = val_df.drop_duplicates(subset=['actor', 'recipient'])
        train_src, train_dst, train_t = train_df.actor.to_list(), train_df.recipient.to_list(), train_df.timestamp.to_list()
        val_src, val_dst, val_t = val_df.actor.to_list(), val_df.recipient.to_list(), val_df.timestamp.to_list()
        
        self.train_dataset = TemporalData(src=torch.tensor(train_src).long(), dst=torch.tensor(train_dst).long(), t=torch.tensor(train_t).float())
        self.val_dataset = TemporalData(src=torch.tensor(val_src).long(), dst=torch.tensor(val_dst).long(), t=torch.tensor(val_t).float())
    
    def train_dataloader(self):
        return TemporalDataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def val_dataloader(self):
        return TemporalDataLoader(self.val_dataset, batch_size=128, num_workers=self.num_workers, pin_memory=self.pin_memory)
        
        
