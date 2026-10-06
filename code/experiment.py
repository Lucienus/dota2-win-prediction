"""Shared loading and evaluation; corrected preparation lives in prepare_raw.py."""


from __future__ import annotations


import argparse


import copy


import hashlib


import json


import platform


import random


import time


from pathlib import Path


import numpy as np


import torch


from torch.utils.data import DataLoader, Dataset


from data_protocol import ROOT, SCHEMA_VERSION, load_manifest, read_json, write_json, fingerprint


from feature_extractor import MatchFeatureExtractor, fit_purchase_prior, feature_contract


from models import DotaMultiModalPredictor, VARIANTS


from evaluation import probability_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class PrefixDataset(Dataset):
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {key: data[key].copy() for key in data.files}

    def __len__(self):
        return len(self.arrays["outcome"])

    def __getitem__(self, index):
        return {key: torch.as_tensor(value[index]) for key, value in self.arrays.items()}


def collate_prefixes(samples):
    batch = {key: torch.stack([s[key] for s in samples]) for key in samples[0] if key != "sequence"}
    sequences = [sample["sequence"] for sample in samples]
    batch["lengths"] = torch.tensor([len(s) for s in sequences])
    batch["sequence"] = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return batch


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    labels, probabilities, ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            out = model(to_device(batch, device))
            labels.extend(batch["outcome"].numpy().tolist())
            ids.extend(batch["match_id"].numpy().tolist())
            probabilities.extend(out["logits"].sigmoid().cpu().numpy().tolist())
    return np.array(labels), np.array(probabilities), np.array(ids)


def environment():
    from runtime_identity import collect_environment
    return collect_environment()


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def open_run(folder, specification, resume=False):
    """Only resume identical data, code, runtime and hyperparameters."""
    specification = json.loads(json.dumps(specification))
    if folder.exists():
        if not resume:
            raise FileExistsError(f"Run exists: {folder}. Use --resume for the identical configuration.")
        saved = folder / "run_config.json"
        if not saved.exists():
            raise ValueError(f"Cannot resume an unversioned run: {folder}. Use a new output directory.")
        from runtime_identity import assert_same
        assert_same(read_json(saved),specification,str(folder))
        if (folder / "result.json").exists():
            result=read_json(folder / "result.json")
            assert_same(result.get('specification',{}),specification,str(folder/'result.json'))
            return result
    else:
        folder.mkdir(parents=True)
        write_json(folder / "run_config.json", specification)
    return None


def validate_progress(progress,specification):
    """Bind an epoch checkpoint to the run configuration before restoring state."""
    if progress.get('specification_sha256')!=fingerprint(specification):
        raise ValueError('Progress checkpoint belongs to a different or legacy run. Use a new output directory.')


def save_predictions(path, dataset, labels, probabilities, ids):
    if not np.array_equal(ids, dataset.arrays["match_id"]):
        raise ValueError("Prediction order differs from test match order")
    np.savez_compressed(path, match_id=ids, labels=labels, probabilities=probabilities,
                        patch=dataset.arrays["patch"], start_time=dataset.arrays["start_time"],
                        gold_difference=dataset.arrays["sequence"][:, -1, 0])


def baseline_features(dataset, simple=False):
    arrays = dataset.arrays
    n = len(dataset)
    if simple:
        return arrays["sequence"][:, -1, :2]
    # Static hero one-hot avoids imposing ordinal distances on categorical hero IDs.
    ids = arrays["heroes"]
    if np.any(ids >= 512):
        raise ValueError("Hero ID vocabulary needs extending")
    heroes = np.eye(512, dtype=np.float32)[ids].reshape(n, -1)
    return np.concatenate((arrays["nodes"].reshape(n, -1), arrays["sequence"].reshape(n, -1),
                           arrays["edges"].reshape(n, -1), heroes), axis=1)

