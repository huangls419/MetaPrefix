# MetaPrefix

MetaPrefix trains a prefix encoder on a frozen ProtGPT2 backbone for anticancer peptide generation. The repository provides prepared group-disjoint data splits, training scripts, and generation code.

Run all commands from the project root.

---

## Project structure

```text
MetaPrefix/
├── data/
│   ├── train.csv              # training data
│   ├── valid.csv              # validation data
│   ├── test.csv               # test data
│   └── zero_shot.csv          # held-out zero-shot tasks
├── peft_overlay/              # MetaPrefix prefix encoder implementation
├── scripts/
│   ├── 00_check_environment.py
│   ├── 01_prepare_meta.py
│   ├── 02_stage1_sbs_maml.py
│   ├── 03_stage2_fomaml.py
│   ├── 04_generate.py
│   └── common.py
├── ProtGPT2/                  # put the downloaded model here
├── tools/                     # legacy files; not used by current scripts
└── requirements.txt
```

---

## 1. Install dependencies

Python 3.9+ and PyTorch 2.1+ are required. A CUDA GPU is recommended.

Install PyTorch for your CUDA version first, for example:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```powershell
pip install -r requirements.txt
```

---

## 2. Add the ProtGPT2 model

Download the official ProtGPT2 model, tokenizer, and weights and place all files in `MetaPrefix/ProtGPT2/`.

Download sources:

- Google Drive: <https://drive.google.com/drive/u/2/folders/1XQDxrd7oMlIOZ-veH69IL8HoH0u09rUj>
- Hugging Face: <https://huggingface.co/nferruz/ProtGPT2/tree/main>

Or download directly with Transformers:

```powershell
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; m=AutoModelForCausalLM.from_pretrained('nferruz/ProtGPT2'); t=AutoTokenizer.from_pretrained('nferruz/ProtGPT2'); m.save_pretrained('ProtGPT2'); t.save_pretrained('ProtGPT2')"
```

The folder must contain the complete model configuration, weights, and tokenizer files.

Check the environment and model setup:

```powershell
python scripts\00_check_environment.py
```

---

## 3. Prepare meta-training data

```powershell
python scripts\01_prepare_meta.py --out data
```

This creates:

```text
data/meta_support.csv
data/meta_query.csv
data/meta_roles.csv
data/meta_task_manifest.csv
```

---

## 4. Stage 1 training

```powershell
python scripts\02_stage1_sbs_maml.py --data-dir data --model ProtGPT2 --out models\stage1\best
```

Main outputs:

```text
models/stage1/best/
models/stage1/stage1_history.csv
models/stage1/stage1_summary.json
```

If GPU memory is insufficient, reduce the batch size:

```powershell
python scripts\02_stage1_sbs_maml.py --data-dir data --model ProtGPT2 --out models\stage1\best --batch-size 4
```

---

## 5. Stage 2 target adaptation

Replace `"Breast cancer"` with an eligible cancer task from `data/meta_task_manifest.csv` where `Stage1_status` is `include`.

```powershell
python scripts\03_stage2_fomaml.py --target "Breast cancer" --data-dir data --model ProtGPT2 --stage1 models\stage1\best
```

Main outputs:

```text
models/stage2/Breast_cancer/meta_init/
models/stage2/Breast_cancer/target_adapter/
models/stage2/Breast_cancer/stage2_history.csv
models/stage2/Breast_cancer/stage2_summary.json
```

The path uses the target name with spaces replaced by underscores.

---

## 6. Generate peptides

Target-specific generation:

```powershell
python scripts\04_generate.py --adapter models\stage2\Breast_cancer\target_adapter --target "Breast cancer" --n 500 --out output\generated_breast.csv
```

Zero-shot generation uses the Stage 1 adapter directly:

```powershell
python scripts\04_generate.py --adapter models\stage1\best --target "Osteosarcoma" --n 500 --out output\zero_shot_osteosarcoma.csv
```

Optional generation from a known N-terminal sequence:

```powershell
python scripts\04_generate.py --adapter models\stage2\Breast_cancer\target_adapter --target "Breast cancer" --prompt "KLLLK" --n 200 --out output\breast_conditioned.csv
```

Generated sequences are written to the CSV path specified by `--out`.

---

## Full command sequence

```powershell
python scripts\00_check_environment.py
python scripts\01_prepare_meta.py --out data
python scripts\02_stage1_sbs_maml.py --data-dir data --model ProtGPT2 --out models\stage1\best
python scripts\03_stage2_fomaml.py --target "cancer" --data-dir data --model ProtGPT2 --stage1 models\stage1\best
python scripts\03_generate.py --adapter models\stage2\Breast_cancer\target_adapter --target "Breast cancer" --n 500 --out output\generated_breast.csv
```
