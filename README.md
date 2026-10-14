# CoTaReS

**CoTaReS** (Count-Tanimoto Reference-Residual Solver) is the `count_kernel-v2` molecular activity regressor. This repository contains only its model (`model.py`) and fixed-protocol training runner (`train.py`). It does not contain datasets, pretrained checkpoints, per-compound predictions, baseline implementations, or ablation scripts.

The model combines a learned count-fingerprint baseline with a local, regularized solve over eight labeled training references. Reference selection uses binary overlap of un-hashed radius-two Morgan environment identifiers; the prediction uses a query-weighted count-Tanimoto Gram matrix and each reference's measured-minus-baseline residual.

## Training

Install Python with PyTorch, RDKit, NumPy, SciPy, and scikit-learn. Obtain the corrected [MoleculeACE v3.0.0 release](https://github.com/molML/MoleculeACE/releases/tag/v3.0.0) separately. Supply one CSV per task named `<dataset>.csv` with `smiles`, `split` (`train`/`test`), and either `exp_mean [nM]` or `y [pEC50/pKi]`. The runner reads **training rows only**; it neither reads test activities nor computes published test metrics.

```bash
python train.py --data /path/to/moleculeace-v3-csvs --output /path/to/new-output --datasets CHEMBL2034_Ki --seeds 0 1 2 3 4
```

For every seed, the runner uses the immutable official train split, a 10% validation split with split seed 0, at most 100 epochs, batch size 128, AdamW learning rate 1e-4 and weight decay 1e-5, patience 15, and validation overall RMSE for checkpoint selection. It writes model checkpoints, a split/reference-index manifest, configuration, and history **only to the chosen local output directory**. It refuses to overwrite an existing nonempty run directory.

The public model is a trimmed implementation of the archived `count_kernel-v2` forward path. In a fixed-seed compatibility check, its active weights, forward values, and one optimizer step matched the archived implementation exactly; the training loader produced identical training features, labels, split indices, and reference indices for `CHEMBL2034_Ki`. This is a code-level check, **not** a fresh reproduction of all published runs. The paper's numerical results come from separately preserved original training archives. This repository does not include the source data or the original run artifacts, so it is not a self-contained replication package.
