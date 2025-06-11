import argparse
from glob import glob
from pathlib import Path
import os

import awkward as ak
import h5py
import numpy as np
import uproot as ur

def awkward3D_to_padded(a: ak.Array, max_len: int | None = None) -> np.ndarray:
    if max_len is None:
        max_len = int(ak.max(ak.num(a)))
    np_arr = ak.pad_none(a, target=max_len, clip=True)
    np_arr = ak.to_numpy(np_arr)
    np_arr = np.array(np_arr)
    return np.nan_to_num(np_arr, 0.0)

def init_dataset(file: h5py.File, group: str, name: str, data: np.ndarray, chunksize: int = 4000) -> None:
    if name not in file[group]:
        file[group].create_dataset(
            name,
            shape=(0,) + data.shape[1:],
            maxshape=(None,) + data.shape[1:],
            chunks=(chunksize,) + data.shape[1:],
            dtype=data.dtype,
        )

def extend_dataset(file: h5py.File, group: str, name: str, data: np.ndarray) -> None:
    n = len(data)
    file[group][name].resize((file[group][name].shape[0] + n), axis=0)
    file[group][name][-n:] = data

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", type=str)
    args.add_argument("--output_dir", type=str)
    args.add_argument("--output_name", type=str, default="combined.h5")
    args.add_argument("--tree", type=str, default="Events")
    args.add_argument("--chunksize", type=int, default=4000)
    return args.parse_args()

def main():
    args = parse_args()
    outName = Path(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    outFile = h5py.File(outName, "w")
    outFile.create_group("even")
    outFile.create_group("odd")

    files = sorted(glob(str(Path(args.input_dir) / "*.root")))
    if "NUFLOWS_SINGLE_FILE" in os.environ:
        files = [Path(args.input_dir) / os.environ["NUFLOWS_SINGLE_FILE"]]

    print(f"Found {len(files)} files")

    for i, file in enumerate(files):
        print(f"Processing {file.name} ({i+1}/{len(files)})")
        try:
            tree = ur.open(file)[args.tree]

            evt_info = tree.arrays(["event"], library="np")

            lep = ak.concatenate(
                [tree["Muon_pt"].array()[..., np.newaxis],
                 tree["Muon_eta"].array()[..., np.newaxis],
                 tree["Muon_phi"].array()[..., np.newaxis],
                 tree["Muon_mass"].array()[..., np.newaxis]],
                axis=-1
            )
            lep = awkward3D_to_padded(lep, 2).astype(np.float32)
            jet = ak.concatenate(
                [tree["Jet_pt"].array()[..., np.newaxis],
                 tree["Jet_eta"].array()[..., np.newaxis],
                 tree["Jet_phi"].array()[..., np.newaxis],
                 tree["Jet_mass"].array()[..., np.newaxis]],
                axis=-1
            )
            jet = awkward3D_to_padded(jet, 10).astype(np.float32)
            jet[..., 3] = np.clip(jet[..., 3], 0, None)
            jet[..., 0] = np.clip(jet[..., 0], 0, None)
            met = ak.concatenate(
                [tree["MET_pt"].array()[..., np.newaxis],
                 tree["MET_phi"].array()[..., np.newaxis]],
                axis=-1
            )
            met = awkward3D_to_padded(met).astype(np.float32)
            if i == 0:
                for split in ["even", "odd"]:
                    init_dataset(outFile, split, "lep", lep)
                    init_dataset(outFile, split, "jet", jet)
                    init_dataset(outFile, split, "met", met)

            for split in ["even", "odd"]:
                save = evt_info["event"] % 2 == (split == "odd")
                extend_dataset(outFile, split, "lep", lep[save])
                extend_dataset(outFile, split, "jet", jet[save])
                extend_dataset(outFile, split, "met", met[save])

        except Exception as e:
            print(f"Failed to process {file.name}: {e}")

    outFile.close()

if __name__ == "__main__":
    main()

