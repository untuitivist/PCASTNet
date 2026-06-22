# PCASTNet

Physics-Constrained Adaptive Style Transfer Network for sample generation in
cross-machine small-sample fault diagnosis.

This repository is the reset-local demo project for the CWRU -> BJTU workflow.
It keeps the experiment runnable from one Python entrypoint, with fixed CWT
datasets, explicit data counts, stage-wise outputs, and logs under one run
directory.

## Method overview

PCASTNet addresses the case where a monitored machine has only a few labelled
fault samples, while a reference machine has more available data. Instead of
generating samples from random noise, the method transfers diagnostic fault
content from the reference machine to the monitored machine style.

The paper-level interpretation is:

- content: fault-discriminative spectral structures that should be preserved
  across machines;
- style: machine-specific time-frequency appearance caused by different
  structures, sensors, loads, and speeds;
- physical consistency: generated CWT samples should preserve the band-energy
  distribution of the monitored machine.

The pipeline uses CWT spectrograms and contains:

1. a VGG-based spectral feature encoder initialized from the AdaIN VGG weights;
2. an Adaptive Style Normalization module for style-content fusion;
3. a decoder that reconstructs target-style CWT samples;
4. a multi-objective loss:

```text
L = 1 * L_content + 10 * L_style + 1 * L_energy
```

The energy term constrains frequency-band energy and is the physics-aware part
of the demo.

## Demo protocol

The included config reproduces the CWRU -> BJTU small-sample setting:

| Role | Dataset | Count | Usage |
| --- | --- | ---: | --- |
| reference/content train | CWRU | 500 | content source for generation |
| monitored/style train | BJTU | 50 | encoder training, style transfer, classifier real samples |
| monitored/style valid | BJTU | 100 | validation |
| monitored/style test | BJTU | 500 | final evaluation |

Encoder training uses only the 50 BJTU monitored training samples. It creates an
in-memory 8:2 train/validation split and does not write split datasets to disk.

## Repository layout

```text
reset/
  configs/
    experiments/
      cwru_bjtu.json
  data/
    datasets/
      machines/
        CWRU/cwts/
        BJTU/cwts/
  demo.py
  src/
    pcastnet/
    models/
    STC.py
    data_loader.py
    function.py
    sampler.py
    stc_*.py
  experiments/
  requirements.txt
  pyproject.toml
```

Generated samples and trained models are run outputs under `experiments/`; they
are not fixed assets under `data/`.

## Install

Use the existing Python environment if available. Otherwise install the project
editable:

```powershell
python -m pip install -e .
```

## AdaIN VGG weights

The encoder is initialized from the normalized VGG used by AdaIN. The expected
local path is:

```text
src/models/vgg/vgg.pth
```

Download command:

```bash
mkdir -p src/models/vgg
wget -c https://s3.amazonaws.com/xunhuang-public/adain/vgg_normalised.t7 -O src/models/vgg/vgg.pth
```

Although the upstream file name ends with `.t7`, this project stores it as
`vgg.pth` because the code loads it as the initial VGG encoder checkpoint.

## Validate assets

```powershell
python demo.py --dry-run --config configs/experiments/cwru_bjtu.json
```

The dry run checks:

- CWRU and BJTU CWT dataset paths;
- fixed image counts;
- AdaIN VGG checkpoint path;
- configured output paths;
- encoder in-memory split policy.

## Run the full demo

Use `demo.py` as the only demo entrypoint:

```powershell
python demo.py --stage all --config configs/experiments/cwru_bjtu.json
```

Available stages:

```powershell
python demo.py --stage encoder --config configs/experiments/cwru_bjtu.json
python demo.py --stage style-transfer --config configs/experiments/cwru_bjtu.json
python demo.py --stage generate --config configs/experiments/cwru_bjtu.json
python demo.py --stage classifier --config configs/experiments/cwru_bjtu.json
```

Long runs write stage output to `train.log` inside the run directory.

## Output structure

A normal full run creates:

```text
experiments/pcastnet_cwru_bjtu_<timestamp>/
  config.json
  effective_config.json
  meta.json
  status.json
  encoder_manifest.json
  pipeline_manifest.json
  train.log
  flow1_CWRU-BJTU_encoder_c/
    *_encoder.pth.tar
    *_classifier.pth.tar
    *_confusion_matrix_*.png
  flow2_CWRU-BJTU_st/
    *_decoder.pth.tar
    *_adailn.pth.tar
    *_encoder.pth.tar
    *_iter_*.png
  generated/
    train/
    dataset_info.json
  downstream_cnn_CWRU-BJTU_c/
    *_confusion_matrix_*.png
    *_confusion_matrix_*.npy
    *_predictions_*.npz
    *_iter_*.pth.tar
```

The encoder is not copied to a second canonical checkpoint. The encoder stage
writes `encoder_manifest.json` with `selected_encoder_path`, and downstream
stages read that selected training artifact directly.

## Metrics

The classifier stage saves:

- confusion matrix image: `*_confusion_matrix_*.png`;
- raw confusion matrix: `*_confusion_matrix_*.npy`;
- labels, predictions, probabilities, accuracy, F1, and AUC:
  `*_predictions_*.npz`.

AUC is computed as macro one-vs-rest ROC-AUC from softmax probabilities.

## Notes

- The demo is intentionally fixed to the paper-scale CWRU -> BJTU setting.
- `content-valid` and `content-test` are not used.
- `data/` should contain only the fixed CWT assets required by the demo.
- Do not create physical split datasets for encoder training; the split is
  runtime-only.

## License

See [LICENSE.md](LICENSE.md) before publishing or redistributing the code and
assets.
