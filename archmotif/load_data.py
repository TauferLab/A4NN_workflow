from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import torch
from ast import literal_eval


def load_training_data_CIFAR(root_path):
    """
    Load all training.h5 files from arch_* directories, decode genomes,
    and return a single dataframe.

    Returns columns:
    [
        'arch_id',
        'arch',
        'epoch',
        'epoch_time',
        'genome',
        'architecture',
        'train_accs',
        'val_accs'
    ]
    """

    def phase_dencode(phase_bit_string):
        n = int(np.sqrt(2 * len(phase_bit_string) - 7 / 4) - 1 / 2)

        genome = []
        for i in range(n):
            operator = []
            for j in range(i + 1):
                operator.append(
                    phase_bit_string[int(i * (i + 1) / 2 + j)]
                )
            genome.append(operator)

        genome.append([phase_bit_string[-1]])
        return genome

    def convert(bit_string, n_phases=3):
        bit_string = np.asarray(bit_string)

        assert bit_string.shape[0] % n_phases == 0

        phase_length = bit_string.shape[0] // n_phases

        genome = []
        for i in range(0, bit_string.shape[0], phase_length):
            genome.append(bit_string[i:i + phase_length].tolist())

        return genome

    def decode(genome):
        genotype = []
        for gene in genome:
            genotype.append(phase_dencode(gene))
        return genotype

    all_rows = []

    root = Path(root_path)

    arch_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith("arch_")]
    )

    for arch_id, arch_dir in enumerate(arch_dirs):
        training_file = arch_dir / "training.h5"

        if not training_file.exists():
            continue
        flops_file = arch_dir / "flops.csv"
        flops_value = None
        flops_value = None
        if flops_file.exists():
            flops_value = float(flops_file.read_text().strip())
        # if flops_file.exists():
        #     flops_df = pd.read_csv(flops_file, header=None)
        #     flops_value = flops_df.iloc[0, 0]
        with h5py.File(training_file, "r") as f:

            n_rows = len(f["epoch"])

            for i in range(n_rows):

                genome = f["genome"][i]

                all_rows.append({
                    "arch": arch_id,         # arch_1 -> 0, arch_2 -> 1, ...
                    # "arch": arch_dir.name,
                    "epoch": f["epoch"][i],
                    "epoch_time": f["epoch_time"][i],
                    "genome": genome,
                    "architecture": decode(genome),
                    "train_accs": f["train_accs"][i],
                    "val_accs": f["val_accs"][i],
                    "flops": flops_value,
                })
    return pd.DataFrame(all_rows)


def read_model(path, device):

    """
    Returns the pytorch model object

    :param path: the path to the .pt model file
    :param device: either CPU or GPU
    """
    importer = torch.package.PackageImporter(path)

    if importer.file_structure().has_file("nsganet"): # check if serialized directory exists
        mod = importer.import_module('nsganet.models')
    else:
        mod = importer.import_module('models')

    arch = path.split('/')[-1].split('_')[1]
    epoch =  path.split('/')[-1].split('_')[3].split('.')[0].strip()

    package_name = f"arch_{arch}_epoch_{epoch}"
    resource_name = f"arch_{arch}_epoch_{epoch}.pkl"

    loaded_model = importer.load_pickle(package_name, resource_name, map_location=torch.device(device))

    assert torch.package.is_from_package(mod)
    assert torch.package.is_from_package(loaded_model)
    return loaded_model

    # loaded_model = read_model('/media/volume/a4nn-ap-volume/PD_Data/models/gpu1/no_early_termination/classification-search-penguin-pd_exv100_1gpu_25e_1e14-macro-20230328-061437/arch_71/arch_71_epoch_0.pt', device='cpu') 
    # print(loaded_model)




def phase_dencode(phase_bit_string):
    n = int(np.sqrt(2 * len(phase_bit_string) - 7 / 4) - 1 / 2)

    genome = []

    for i in range(n):
        operator = []

        for j in range(i + 1):
            operator.append(
                phase_bit_string[int(i * (i + 1) / 2 + j)]
            )

        genome.append(operator)

    genome.append([phase_bit_string[-1]])

    return genome


def convert(bit_string, n_phases=3):
    bit_string = np.asarray(bit_string)

    assert bit_string.shape[0] % n_phases == 0

    phase_length = bit_string.shape[0] // n_phases

    genome = []

    for i in range(0, bit_string.shape[0], phase_length):
        genome.append(bit_string[i:i + phase_length].tolist())

    return genome


def decode(genome):
    genotype = []

    for gene in genome:
        genotype.append(phase_dencode(gene))

    return genotype


def read_file(file):
    df = pd.read_csv(file)
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
    return df


def read_files(files):
    """
    Read all csv files, add metadata columns,
    and create a decoded architecture column from genome.
    """

    dfs = []

    intensities = {
        "14": "low",
        "15": "med",
        "16": "high"
    }

    for i, file in enumerate(files):

        print(f"File {i+1}: ", end="")

        df = read_file(file)

        category = Path(file).name.split("_", 1)[0]

        df["run"] = i
        df["type"] = intensities[category]
        df["id"] = (
            df["type"].astype(str)
            + "-"
            + df["arch"].astype(str)
        )

        # genome stored as string in CSV
        df["architecture"] = df["genome"].apply(
            lambda g: decode(
                    np.asarray(literal_eval(g))
            )
        )

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)