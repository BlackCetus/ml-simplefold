#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import json
import pickle
import random
import numpy as np
from pathlib import Path
from dataclasses import asdict
from typing import Optional

from torch import Tensor
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from boltz_data_pipeline import const
from boltz_data_pipeline.tokenize.tokenizer import Tokenizer
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.filter.dynamic.filter import DynamicFilter
from boltz_data_pipeline.types import Manifest, Record
from utils.datamodule_utils import (
    Dataset,
    DatasetConfig,
    load_input,
    collate,
    extract_sequence_from_tokens,
)

# The training data was built from OpenFold's pdb_mmcif.zip (the full PDB), which
# includes DNA/RNA and protein-nucleic entries (e.g. the PDB *d series). SimpleFold's
# process_mmcif.py hardcodes every chain as protein (gemmi PeptideL), so mol_type is
# always PROTEIN and the tokenizer's non-protein-chain skip never fires; nucleotide
# residues leak into this protein-only pipeline and crash extract_sequence_from_tokens.
# res_type is reliable, so detect nucleic-acid residues by token id and skip the sample.
_NUCLEOTIDE_TOKEN_IDS = np.array(
    [const.token_ids[t] for t in ("A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN")]
)


"""
We assume the training data is stored in the following structure:
- target_dir_for_dataset_A/
    - structures/
        - {record_id}.npz
- tokenized_dir_for_dataset_A/
    - tokens/
        - {record_id}.pkl
    - records/
        - {record_id}.json
    - manifest.json
- target_dir_for_dataset_B/
    - ...
- tokenized_dir_for_dataset_B/
    - ...
- ...
"""


class SimpleFoldTrainingDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        datasets: list[Dataset],
        symmetries: dict,
        max_atoms: int,
        max_tokens: int,
        pad_to_max_atoms: bool = False,
        pad_to_max_tokens: bool = False,
        atoms_per_window_queries: int = 32,
        min_dist: float = 2.0,
        max_dist: float = 22.0,
        num_bins: int = 64,
        return_symmetries: Optional[bool] = False,
        rotation_augment_ref_pos: Optional[bool] = False,
        rotation_augment_coords: Optional[bool] = True,
        **kwargs: dict[str, any],
    ) -> None:
        """Initialize the training dataset."""
        super().__init__()
        self.datasets = datasets
        self.symmetries = symmetries
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms
        self.pad_to_max_tokens = pad_to_max_tokens
        self.pad_to_max_atoms = pad_to_max_atoms
        self.atoms_per_window_queries = atoms_per_window_queries
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.num_bins = num_bins
        self.return_symmetries = return_symmetries
        self.rotation_augment_ref_pos = rotation_augment_ref_pos
        self.rotation_augment_coords = rotation_augment_coords
        self.samples = []
        self.num_samples = 0
        for dataset_idx, dataset in enumerate(datasets):
            if dataset.cluster is None:
                records = dataset.manifest.records
                self.samples.extend(
                    [(dataset_idx, record.id) for record in records]
                )
                self.num_samples += len(records)
            else:
                data_cluster = json.load(open(dataset.cluster, "r"))
                for k, v in data_cluster.items():
                    # one cluster is considered as one sample in enumerating
                    self.samples.append((dataset_idx, v["members"]))
                self.num_samples += len(data_cluster)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Get an item from the dataset.

        Parameters
        ----------
        idx : int
            The data index.

        Returns
        -------
        dict[str, Tensor]
            The sampled data features.

        """
        # Pick a random dataset
        dataset_idx, record_ids = self.samples[idx]
        dataset = self.datasets[dataset_idx]

        if isinstance(record_ids, list):
            # Pick a random record from the cluster
            record_id = random.choice(record_ids)
        else:
            # Use the record_id directly
            record_id = record_ids

        # load record + tokenized (from a webdataset archive if present, else loose files)
        record, tokenized = self._load_sample(dataset, record_id)
        if tokenized is None:
            return self.__getitem__(random.randint(0, self.num_samples - 1))

        max_num_tokens = len(tokenized.tokens)
        if max_num_tokens == 0:
            print(f"No tokens in {record.id}. Skipping.")
            return self.__getitem__(random.randint(0, self.num_samples - 1))

        # Skip nucleic-acid structures that leak in via the always-PROTEIN mol_type
        # (see _NUCLEOTIDE_TOKEN_IDS above). This is a protein-only model.
        if np.isin(tokenized.tokens["res_type"], _NUCLEOTIDE_TOKEN_IDS).any():
            return self.__getitem__(random.randint(0, self.num_samples - 1))

        # Compute crop
        try:
            max_atoms = self.max_atoms
            max_tokens = self.max_tokens

            if self.max_tokens is not None:
                tokenized = dataset.cropper.crop(
                    tokenized,
                    max_atoms=max_atoms,
                    max_tokens=max_tokens,
                    random=np.random,
                )
        except Exception as e:
            print(f"Cropper failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(random.randint(0, self.num_samples - 1))

        sequence = extract_sequence_from_tokens(tokenized)

        # Check if there are tokens
        if len(tokenized.tokens) == 0:
            msg = "No tokens in cropped structure."
            raise ValueError(msg)

        # Compute features
        try:
            features = dataset.featurizer.process(
                tokenized,
                max_atoms=max_atoms if self.pad_to_max_atoms else None,
                max_tokens=max_tokens if self.pad_to_max_tokens else None,
                symmetries=self.symmetries,
                atoms_per_window_queries=self.atoms_per_window_queries,
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                num_bins=self.num_bins,
                compute_symmetries=self.return_symmetries,
                rotation_augment_ref_pos=self.rotation_augment_ref_pos,
                rotation_augment_coords=self.rotation_augment_coords,
            )

            features["aa_seq"] = sequence
            features['record'] = asdict(record)
            features['max_num_tokens'] = torch.tensor(max_num_tokens, dtype=torch.long)
            features['cropped_num_tokens'] = torch.tensor(len(tokenized.tokens), dtype=torch.long)

        except Exception as e:
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")
            return self.__getitem__(random.randint(0, self.num_samples - 1))

        return features

    def _archive_for(self, dataset):
        """Return an ArchiveReader if dataset.tokenized_dir is a webdataset archive
        (contains index.json), else None. Cached per tokenized_dir, built lazily so
        each DataLoader worker opens its own tar handles after fork."""
        cache = getattr(self, "_archive_cache", None)
        if cache is None:
            cache = self._archive_cache = {}
        key = str(dataset.tokenized_dir)
        if key not in cache:
            if (Path(dataset.tokenized_dir) / "index.json").exists():
                from datasets.archive import ArchiveReader
                cache[key] = ArchiveReader(dataset.tokenized_dir)
            else:
                cache[key] = None
        return cache[key]

    def _load_sample(self, dataset, record_id):
        """Load (Record, Tokenized) for one id. Returns (None, None) on failure so
        the caller resamples. Reads from a webdataset archive when present."""
        reader = self._archive_for(dataset)
        if reader is not None:
            try:
                record = Record(**reader.record(record_id.lower()))
                tokenized = reader.tokenized(record.id)
            except Exception as e:
                print(f"Archive load failed for {record_id}: {e}")
                return None, None
            return record, tokenized

        # loose-file layout (original behavior)
        record = Record(**json.load(
            open(os.path.join(dataset.tokenized_dir, "records", f"{record_id.lower()}.json"), "r")
        ))
        tokenized_path = os.path.join(dataset.tokenized_dir, "tokens", f"{record.id}.pkl")
        try:
            with open(tokenized_path, "rb") as f:
                tokenized = pickle.load(f)
        except:
            try:
                input_data = load_input(record, dataset.target_dir)
                tokenized = dataset.tokenizer.tokenize(input_data)
            except:
                print(f"Failed tokenize {record.id}")
                return None, None
        return record, tokenized

    def __len__(self) -> int:
        return self.num_samples


class SimpleFoldTrainingDataModule(LightningDataModule):
    """DataModule for SimpleFold Training."""

    def __init__(
        self,
        datasets: list[DatasetConfig],
        tokenizer: Tokenizer,
        featurizer: BoltzFeaturizer,
        max_atoms: int,
        max_tokens: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        symmetries: str,
        atoms_per_window_queries: int,
        min_dist: float,
        max_dist: float,
        num_bins: int,
        filters: Optional[list[DynamicFilter]] = None,
        pad_to_max_tokens: bool = False,
        pad_to_max_atoms: bool = False,
        return_train_symmetries: bool = False,
        rotation_augment_ref_pos: Optional[bool] = False,
        rotation_augment_coords: Optional[bool] = False,
    ):

        super().__init__()

        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size_per_device_test = 1

        # Load datasets
        train: list[Dataset] = []
        val: list[Dataset] = []

        for data_config in datasets:
            # Load manifest
            if data_config.manifest_path is not None:
                path = Path(data_config.manifest_path)
            else:
                path = Path(data_config.target_dir) / "manifest.json"
            manifest: Manifest = Manifest.load(path)

            train_records = manifest.records

            if data_config.record_list is not None:
                with Path(data_config.record_list).open("r") as f:
                    record_list = {x.lower() for x in f.read().splitlines()}
                train_records = [
                    record for record in train_records if record.id.lower() in record_list
                ]

            # Filter training records
            if filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in filters)
                ]

            # Filter training records
            if data_config.filters is not None:
                train_records = [
                    record
                    for record in train_records
                    if all(f.filter(record) for f in data_config.filters)
                ]

            # Create train dataset
            train_manifest = Manifest(train_records)
            train.append(
                Dataset(
                    data_config.tokenized_dir,
                    Path(data_config.target_dir),
                    data_config.esm_dir,
                    data_config.connectivity_dir,
                    data_config.cheap_dir,
                    train_manifest,
                    data_config.cropper,
                    tokenizer,
                    featurizer,
                    cluster=data_config.cluster,
                )
            )

            val_records = train_records[-16:]
            val_manifest = Manifest(val_records)
            val.append(
                Dataset(
                    data_config.tokenized_dir,
                    Path(data_config.target_dir),
                    data_config.esm_dir,
                    data_config.connectivity_dir,
                    data_config.cheap_dir,
                    val_manifest,
                    data_config.cropper,
                    tokenizer,
                    featurizer,
                )
            )

        # Print dataset sizes
        for dataset in train:
            dataset: Dataset
            print(f"Training dataset size: {len(dataset.manifest.records)}")

        # Create wrapper datasets
        self._train_set = SimpleFoldTrainingDataset(
            datasets=train,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            pad_to_max_atoms=pad_to_max_atoms,
            pad_to_max_tokens=pad_to_max_tokens,
            symmetries=symmetries,
            atoms_per_window_queries=atoms_per_window_queries,
            min_dist=min_dist,
            max_dist=max_dist,
            num_bins=num_bins,
            return_symmetries=return_train_symmetries,
            rotation_augment_ref_pos=rotation_augment_ref_pos,
            rotation_augment_coords=rotation_augment_coords,
        )
        # this is a dummy validation set, as we disable validation in training
        self._val_set = SimpleFoldTrainingDataset(
            datasets=val,
            max_atoms=max_atoms,
            max_tokens=max_tokens,
            pad_to_max_atoms=pad_to_max_atoms,
            pad_to_max_tokens=pad_to_max_tokens,
            symmetries=symmetries,
            atoms_per_window_queries=atoms_per_window_queries,
            min_dist=min_dist,
            max_dist=max_dist,
            num_bins=num_bins,
            return_symmetries=True,
            rotation_augment_ref_pos=False,
            rotation_augment_coords=False,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Run the setup for the DataModule.

        Parameters
        ----------
        stage : str, optional
            The stage, one of 'fit', 'validate', 'test'.

        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.batch_size // self.trainer.world_size
            )

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        return DataLoader(
            self._train_set,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader.

        Returns
        -------
        DataLoader
            The validation dataloader.

        """
        return DataLoader(
            self._val_set,
            batch_size=self.batch_size_per_device_test,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )