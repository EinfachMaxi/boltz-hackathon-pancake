import os
import random
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional
from rdkit import Chem
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import json
import click
import torch
from pytorch_lightning import Trainer#, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
from boltz.data.types import Connection

from boltz.data import const
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record,StructureV2
from boltz.data.write.writer import BoltzWriter
from boltz.model.models import Boltz2
import numpy as np
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import json






# ------------------------------
# URLs for Boltz2
# ------------------------------
MOLS_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
MODEL_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"



@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class BoltzDiffusionParams:
    """Diffusion process hyperparameters for Boltz2 (v2-compatible)."""

    # ======================================================
    # 🔹 Diffusion / noise schedule parameters
    # ======================================================
    gamma_0: float = 0.607               # Base gamma at t=0
    gamma_min: float = 1.109             # Minimum gamma
    noise_scale: float = 0.901           # Scale for noise distribution
    rho: float = 7.8                     # Power for sigma interpolation (controls noise curvature)
    step_scale: float = 1.58             # Multiplier for diffusion step size

    # ======================================================
    # 🔹 Sigma range / variance parameters
    # ======================================================
    sigma_min: float = 0.0004            # Minimum diffusion noise
    sigma_max: float = 160.0             # Maximum diffusion noise
    sigma_data: float = 16.0             # Data variance baseline

    # ======================================================
    # 🔹 Log-normal sampling parameters
    # ======================================================
    P_mean: float = -1.2                 # Mean of log-sigma sampling
    P_std: float = 1.5                   # Std of log-sigma sampling

    # ======================================================
    # 🔹 Conditioning and diffusion options
    # ======================================================
    coordinate_augmentation: bool = True          # Apply random rotations/noise to coordinates
    alignment_reverse_diff: bool = True           # Reverse diffusion in aligned coordinate space
    synchronize_sigmas: bool = True               # Keep same sigma across diffusion samples


   


# --- Kabsch RMSD helper ---
def kabsch_rmsd(P, Q):
    """Compute RMSD between two Nx3 coordinate sets using Kabsch alignment."""
    P = np.asarray(P)
    Q = np.asarray(Q)
    assert P.shape == Q.shape
    # Center both
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    # Covariance and SVD
    C = np.dot(np.transpose(P_cent), Q_cent)
    V, S, W = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, W)))
    U = np.dot(V, np.dot(np.diag([1, 1, d]), W))
    P_rot = np.dot(P_cent, U)
    return np.sqrt(((P_rot - Q_cent) ** 2).sum() / len(P))

# --- your get_coords function ---
def get_coords(tmp_id, idx):
    cif_file = f"outputs_prediction/boltz2_results_inputs_prediction/predictions/{tmp_id}/{tmp_id}_model_{idx}.cif"
    mmcif_dict = MMCIF2Dict(cif_file)
    x = mmcif_dict["_atom_site.Cartn_x"]
    y = mmcif_dict["_atom_site.Cartn_y"]
    z = mmcif_dict["_atom_site.Cartn_z"]
    atom_names = mmcif_dict["_atom_site.label_atom_id"]
    c1_coords = [
        (float(x[i]), float(y[i]), float(z[i]))
        for i, atom in enumerate(atom_names)
        if atom == "C1'"
    ]
    conf_file = f"outputs_prediction/boltz2_results_inputs_prediction/predictions/{tmp_id}/confidence_{tmp_id}_model_{idx}.json"
    with open(conf_file, "r") as f:
        conf_data = json.load(f)
    conf = conf_data.get("confidence_score", -1)
    return np.array(c1_coords), conf




@rank_zero_only
def download(cache: Path) -> None:
    """Download Boltz-2 resources (molecule library + model weights)."""
    mols_tar = cache / "mols.tar"
    if not mols_tar.exists():
        click.echo(f"📦 Downloading molecule library to {mols_tar}")
        urllib.request.urlretrieve(MOLS_URL, str(mols_tar))
        click.echo("✅ mols.tar downloaded successfully.")
    else:
        click.echo("✅ mols.tar already exists in cache.")

    model = cache / "boltz2_conf.ckpt"
    if not model.exists():
        click.echo(f"📥 Downloading Boltz-2 model weights to {model}")
        urllib.request.urlretrieve(MODEL_URL, str(model))
        click.echo("✅ boltz2_conf.ckpt downloaded successfully.")
    else:
        click.echo("✅ boltz2_conf.ckpt already exists in cache.")




def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data




@rank_zero_only
def process_inputs(
    data: list[Path],
    out_dir: Path,
    mol_dir: Path,
    use_msa_server: bool = False,
    msa_server_url: str = "",
    msa_pairing_strategy: str = "",
    max_msa_seqs: int = 4096,
) -> None:
    """
    Boltz-2 compatible preprocessing for antibody–antigen complexes.
    Reads YAML or FASTA, loads molecule templates, skips MSA generation if already present,
    and prepares ready-to-run folders for inference.
    """

    click.echo("🚀 Starting Boltz-2 preprocessing")
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = out_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 🔹 Handle existing manifest safely
    # -------------------------------------------------------------
    manifest_path = processed_dir / "manifest.json"
    existing_records = []

    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
        existing_ids = {r.id for r in manifest.records}
        data = [d for d in data if d.stem not in existing_ids]
        existing_records = manifest.records
        if not data:
            click.echo("✅ All inputs already processed — manifest is up to date.")
            return
        click.echo(f"⚙️ Found {len(data)} new files to process.")
    else:
        click.echo("🆕 No manifest found — starting fresh.")

    # -------------------------------------------------------------
    # 🔹 Setup subdirectories
    # -------------------------------------------------------------
    dirs = {
        "msa": out_dir / "msa",
        "structures": processed_dir / "structures",
        "processed_msa": processed_dir / "msa",
        "predictions": out_dir / "predictions",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 🔹 Load molecule templates (Boltz2 CCD replacement)
    # -------------------------------------------------------------
    mols_root = Path(mol_dir)
    if (mols_root / "mols").exists():
        mols_root = mols_root / "mols"

    click.echo(f"📦 Loading molecule templates from {mols_root}")
    ccd = {}
    for mol_path in mols_root.glob("*.pkl"):
        try:
            with open(mol_path, "rb") as f:
                mol = pickle.load(f)
            if not isinstance(mol, Chem.Mol):
                if isinstance(mol, (bytes, str)):
                    mol = Chem.MolFromMolBlock(
                        mol.decode() if isinstance(mol, bytes) else mol
                    )
            if mol:
                ccd[mol_path.stem.upper()] = mol
        except Exception as e:
            print(f"⚠️ Skipping molecule {mol_path.name}: {e}")

    if not ccd:
        raise RuntimeError(f"No valid molecules found in {mols_root}")

    # -------------------------------------------------------------
    # 🔹 Main preprocessing loop
    # -------------------------------------------------------------
    new_records = []
    for path in tqdm(data, desc="Preprocessing YAML/FASTA files"):
        try:
            # 1️⃣ Parse YAML or FASTA
            if path.suffix in (".yml", ".yaml"):
                target = parse_yaml(path, ccd=ccd, mol_dir=mols_root, boltz2=True)
            elif path.suffix in (".fa", ".fas", ".fasta"):
                target = parse_fasta(path, ccd=ccd, mol_dir=mols_root, boltz2=True)
            else:
                raise RuntimeError(f"Unsupported file format: {path}")

            target_id = target.record.id
            click.echo(f"🧬 Processing target: {target_id}")

            # 2️⃣ Handle existing MSA paths (normalize filenames)
            msas = sorted({c.msa_id for c in target.record.chains if c.msa_id})
            msa_id_map = {}

            for msa_path in msas:
                msa_path = Path(msa_path)
                if not msa_path.exists():
                    raise FileNotFoundError(f"Missing MSA file: {msa_path}")

                # ✅ Normalize MSA name to avoid nested folders
                msa_name = msa_path.stem  # e.g. "8YX1_0"
                processed = dirs["processed_msa"] / f"{msa_name}.npz"
                msa_id_map[str(msa_path)] = msa_name

                if not processed.exists():
                    if msa_path.suffix == ".a3m":
                        msa = parse_a3m(msa_path, taxonomy=None, max_seqs=max_msa_seqs)
                    elif msa_path.suffix == ".csv":
                        msa = parse_csv(msa_path, max_seqs=max_msa_seqs)
                    else:
                        raise RuntimeError(f"Unsupported MSA type: {msa_path}")
                    msa.dump(processed)

            # ✅ Relink and normalize MSA IDs
            for c in target.record.chains:
                if c.msa_id in msa_id_map:
                    c.msa_id = msa_id_map[c.msa_id]
                elif str(c.msa_id) in msa_id_map:
                    c.msa_id = msa_id_map[str(c.msa_id)]
                # Final cleanup → keep only filename stem (e.g., "8V52_0")
                if isinstance(c.msa_id, (str, Path)):
                    c.msa_id = Path(c.msa_id).stem

            # 3️⃣ Save structure arrays
            struct_path = dirs["structures"] / f"{target_id}.npz"
            target.structure.dump(struct_path)

            # 4️⃣ Post-fix arrays (ensure contiguity and required keys)
            try:
                npz = np.load(struct_path, allow_pickle=True)
                arrays = {}
                expected = [
                    "atoms", "bonds", "residues", "chains",
                    "interfaces", "mask", "coords", "ensemble", "pocket"
                ]
                for key in expected:
                    arr = npz[key] if key in npz else np.empty((0,))
                    if hasattr(arr, "flags") and not arr.flags["C_CONTIGUOUS"]:
                        arr = np.ascontiguousarray(arr)
                    arrays[key] = arr
                structure = StructureV2(**arrays)
                structure.dump(struct_path)
            except Exception as e:
                print(f"⚠️ Post-fix skipped for {target_id}: {e}")

            new_records.append(target.record)

        except Exception as e:
            print(f"❌ Failed to process {path.name}: {e}")

    # -------------------------------------------------------------
    # 🔹 Save manifest
    # -------------------------------------------------------------
    all_records = existing_records + new_records
    Manifest(all_records).dump(manifest_path)
    click.echo(f"✅ Completed preprocessing → {len(new_records)} new targets added")








# ------------------------------
# predict()
# ------------------------------
def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
):
    if accelerator == "cpu":
        click.echo("⚠️ Running on CPU — slow")

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    if seed is not None:
        seed_everything(seed)

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()

    # ✅ Do NOT append data.stem ("input") → keeps consistent path with preprocessing
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Download Boltz2 resources
    download(cache)

    # ✅ Handle molecule library (mols.tar instead of CCD)
    mols_tar = cache / "mols.tar"
    mols_dir = cache / "mols"
    if not mols_dir.exists():
        import tarfile
        click.echo(f"📂 Extracting {mols_tar} → {mols_dir}")
        with tarfile.open(mols_tar, "r") as tar:
            tar.extractall(mols_dir)
        click.echo("✅ Extracted mols.tar successfully.")
    
    # ✅ Gather YAMLs for preprocessing
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    # ✅ Preprocessing (MSA, YAML)
    process_inputs(
        data=data,
        out_dir=out_dir,
        mol_dir=mols_dir,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    processed_dir = out_dir / "processed"
    print(f"🔍 Using processed MSA dir: {processed_dir / 'msa'}")

    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    mols_dir = "boltz-2/mols/mols"  # ✅ adjust if your mols dir is inside boltz-2
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mols_dir,
        num_workers=num_workers,
    )

    if checkpoint is None:
        checkpoint = cache / "boltz2_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }

    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale
    params_dict = asdict(diffusion_params)
    
    # ✅ remove obsolete keys
    for k in ["use_inference_model_cache", "alignment_reverse_diff"]:
        params_dict.pop(k, None)

    steering_args = {
        "fk_steering": True,
        "fk_weight": 0.4,
        "directional_guidance": True,
        "steering_strength": 0.3,
        "steering_sigma_threshold": 0.1,
        "physical_guidance_update": False,
        "contact_guidance_update": False,
        "num_particles": 1,
        "fk_resampling_interval": 1,
        "fk_lambda": 0.8,
    }

    # ✅ Load model
    model_module: Boltz2 = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=False,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        steering_args=steering_args,
        ema=False,
    )
    print(type(model_module.pairformer_module.layers[0].attention))
    model_module.eval()

    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
        boltz2=True,
    )

    # ✅ Patch DataModule
    orig_collate = getattr(data_module, "collate_fn", None)
    def collate_with_method_feature(batch):
        feats = orig_collate(batch) if orig_collate else batch
        if isinstance(feats, dict) and "method_feature" not in feats:
            ref_key = "s" if "s" in feats else next(iter(feats))
            ref_tensor = feats[ref_key]
            feats["method_feature"] = torch.zeros(
                (ref_tensor.shape[0], 1),
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )
        return feats
    data_module.collate_fn = collate_with_method_feature

    # ✅ Trainer
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy="auto",
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # ✅ Predict
    trainer.predict(model_module, datamodule=data_module, return_predictions=False)



# ------------------------------
# Main
# ------------------------------
# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    predict(
        # ✅ Input folder containing YAMLs
        data="hackathon_data/intermediate_files/abag_public/input",

        # ✅ Output folder for Boltz-2 predictions
        out_dir="boltz-2",

        # ✅ Local Boltz-2 cache for models and mols
        cache="boltz-2",

        # ✅ Optional local checkpoint (or auto-download if missing)
        checkpoint="boltz-2/boltz2_conf.ckpt",

        # ✅ Compute configuration
        devices=1,
        accelerator="gpu",
        diffusion_samples=5,         # use 5 for balanced speed/accuracy
        recycling_steps=3,
        sampling_steps=200,
        step_scale=1.638,

        # ✅ Output configuration
        output_format="pdb",         # generate PDBs (or "mmcif" if preferred)
        write_full_pae=True,
        write_full_pde=True,

        # ✅ Preprocessing options
        use_msa_server=False,        # skip MSA generation (MSAs already in YAML)
        msa_server_url="",           # not needed
        msa_pairing_strategy="greedy",

        # ✅ Misc
        num_workers=4,
        override=True,
        seed=42,
    )

    


