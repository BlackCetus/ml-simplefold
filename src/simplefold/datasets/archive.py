#
# For licensing see accompanying LICENSE file.
#
"""Random-access reader for the webdataset-style archive produced by
data/afesm_stage.py --pack webdataset:

    <root>/shard-00000.tar ...   members: {id}.structure.npz / {id}.record.json / {id}.tokens.pkl
    <root>/index.json            {id: "shard-00000.tar"}
    <root>/manifest.json         [record, ...]

One inode per shard instead of three files per target, while still supporting
lookup by id (which the AFESM cluster sampler needs). Each DataLoader worker
process builds its own reader (and its own tar file handles) lazily.
"""
import io
import json
import pickle
import tarfile
import threading
from pathlib import Path

import numpy as np


class ArchiveReader:
    def __init__(self, root):
        self.root = Path(root)
        with open(self.root / "index.json") as fh:
            self.index = json.load(fh)          # id -> shard filename
        self._tars = {}                         # shard -> open TarFile
        self._lock = threading.Lock()

    def __contains__(self, rid):
        return rid in self.index

    def __len__(self):
        return len(self.index)

    def _read(self, rid, ext):
        shard = self.index[rid]
        with self._lock:
            tar = self._tars.get(shard)
            if tar is None:
                tar = self._tars[shard] = tarfile.open(self.root / shard)
            member = tar.extractfile(f"{rid}{ext}")
            if member is None:
                raise KeyError(f"{rid}{ext} not in {shard}")
            return member.read()

    def record(self, rid):
        return json.loads(self._read(rid, ".record.json"))

    def tokenized(self, rid):
        return pickle.loads(self._read(rid, ".tokens.pkl"))

    def structure(self, rid):
        return np.load(io.BytesIO(self._read(rid, ".structure.npz")))
