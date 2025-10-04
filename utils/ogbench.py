import os
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from flax.core.frozen_dict import FrozenDict

DEFAULT_DATASET_DIR = '~/.ogbench/data'
DATASET_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'

def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result


def load_dataset(dataset_path, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False, add_info=False):
    """Load OGBench dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        add_info: Whether to add observation information ('qpos', 'qvel', and 'button_states') to the dataset.

    Returns:
        Dictionary containing the dataset. The dictionary contains the following keys: 'observations', 'actions',
        'terminals', and 'next_observations' (if `compact_dataset` is False) or 'valids' (if `compact_dataset` is True).
        If `add_info` is True, the dictionary may also contain additional keys for observation information.
    """
    file = np.load(dataset_path)

    # Example:
    # Assume each trajectory has length 4, and (s0, a0, s1), (s1, a1, s2), (s2, a2, s3), (s3, a3, s4) are transition
    # tuples. Note that (s4, a4, s0) is *not* a valid transition tuple, and a4 does not have a corresponding next state.
    # At this point, `dataset` loaded from the file has the following structure:
    #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
    # -------------------------------------------------------------
    # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # 'terminals'   : [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  1, ...]
    dataset = dict()
    for k in ['observations', 'actions', 'terminals']:
        if k == 'observations':
            dtype = ob_dtype
        elif k == 'actions':
            dtype = action_dtype
        else:
            dtype = np.float32
        dataset[k] = file[k][...].astype(dtype, copy=False)

    if add_info:
        # Read observation information.
        info_keys = []
        for k in ['qpos', 'qvel', 'button_states']:
            if k in file:
                dataset[k] = file[k][...]
                info_keys.append(k)

    # Compact dataset: We need to invalidate the last state of each trajectory so that we can safely get
    # `next_observations[t]` by using `observations[t + 1]`.
    # Our goal is to have the following structure:
    #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
    # -------------------------------------------------------------
    # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # 'terminals'   : [ 0,  0,  0,  1,  1,  0,  0,  0,  1,  1, ...]
    # 'valids'      : [ 1,  1,  1,  1,  0,  1,  1,  1,  1,  0, ...]

    dataset['valids'] = 1.0 - dataset['terminals']
    new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
    dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)


    return dataset


def download_datasets(dataset_names, dataset_dir=DEFAULT_DATASET_DIR):
    """Download OGBench datasets.

    Args:
        dataset_names: List of dataset names to download.
        dataset_dir: Directory to save the datasets.
    """
    # Make dataset directory.
    dataset_dir = os.path.expanduser(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download datasets.
    dataset_file_names = []
    for dataset_name in dataset_names:
        dataset_file_names.append(f'{dataset_name}.npz')
        dataset_file_names.append(f'{dataset_name}-val.npz')
    for dataset_file_name in dataset_file_names:
        dataset_file_path = os.path.join(dataset_dir, dataset_file_name)
        if not os.path.exists(dataset_file_path):
            dataset_url = f'{DATASET_URL}/{dataset_file_name}'
            print('Downloading dataset from:', dataset_url)
            response = urllib.request.urlopen(dataset_url)
            tmp_dataset_file_path = f'{dataset_file_path}.tmp'
            with tqdm.wrapattr(
                open(tmp_dataset_file_path, 'wb'),
                'write',
                miniters=1,
                desc=dataset_url.split('/')[-1],
                total=getattr(response, 'length', None),
            ) as file:
                for chunk in response:
                    file.write(chunk)
            os.rename(tmp_dataset_file_path, dataset_file_path)


def make_datasets(
    dataset_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    dataset_path=None,
    compact_dataset=False,
    add_info=False,
    **env_kwargs,
):
    """Make OGBench environment and load datasets.

    Args:
        dataset_name: Dataset name.
        dataset_dir: Directory to save/load the datasets.
        dataset_path: (Optional) Path to the dataset file.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        env_only: Whether to return only the environment.
        dataset_only: Whether to return only the datasets.
        cur_env: Current environment (only used when `dataset_only` is True).
        add_info: Whether to add observation information ('qpos', 'qvel', and 'button_states') to the datasets.
        **env_kwargs: Keyword arguments to pass to the environment.
    """
    # Make environment.
    splits = dataset_name.split('-')
    dataset_add_info = add_info
   
    # Load datasets.
    if dataset_path is None:
        dataset_dir = os.path.expanduser(dataset_dir)
        download_datasets([dataset_name], dataset_dir)
        train_dataset_path = os.path.join(dataset_dir, f'{dataset_name}.npz')
        val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.npz')
    else:
        train_dataset_path = dataset_path
        val_dataset_path = dataset_path.replace('.npz', '-val.npz')

    ob_dtype = np.uint8
    action_dtype = np.float32
    train_dataset = load_dataset(
        train_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
    )
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
    )

    for k in ['qpos', 'qvel', 'button_states']:
        if k in train_dataset:
            del train_dataset[k]
        if k in val_dataset:
            del val_dataset[k]

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)

    return train_dataset, val_dataset