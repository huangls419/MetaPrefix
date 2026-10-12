# MetaPrefix

MetaPrefix: Two-Stage Meta-Learning for Anticancer Peptide Generation under Data Imbalance.

## Requirements

```bash
pip install torch transformers datasets peft higher tensorboard numpy tqdm
```

## ProtGPT2

Put the official ProtGPT2 model, tokenizer, and weights in the 'MetaPrefix/ProtGPT2' folder: https://drive.google.com/drive/u/2/folders/1XQDxrd7oMlIOZ-veH69IL8HoH0u09rUj or https://huggingface.co/nferruz/ProtGPT2/tree/main

## PEFT Patch

Use the runtime patch:

```python
import tools.patch_peft
```

Or copy `peft_patch/prefix_tuning.py` to `site-packages/peft/tuners/prefix_tuning.py`.

## Data Format

CSV with `Sequence` as the first column and cancer-type labels as `0/1` columns.

Place files at:

```text
data/maml/train_dataset.csv
data/maml/test_dataset.csv
```

## Stage 1: Multi-Cancer SBS-MAML

```bash
python stage1_sbs_maml.py
```

Output:

```text
cache/maml/stage1/prefix_SBSMAML_E5_LR1e-06_BS8_ML50_VT10
```

## Stage 2: Target-Cancer FOMAML

Edit `stage2_fomaml.py`:

```python
self.stage1_model_path = "cache/maml/stage1/prefix_SBSMAML_E5_LR1e-06_BS8_ML50_VT10"
self.target_cancer = "lung"
self.target_column = self.target_cancer
```

Run:

```bash
python stage2_fomaml.py
```

## Generate Candidate ACPs

```bash
python generate.py \
  --peft_model_path cache/maml/stage2/lung/lung_epoch92_loss5.4465 \
  --output_path generated/lung.npy \
  --num_sequences 10000
```

## Citation

If you use this code or our model for your publication, please cite the original paper