# PCASTNet: Physics-Constrained Adaptive Style Transfer Network

Official PyTorch implementation of  
**"PCASTNet: A Physics-Constrained Adaptive Style Transfer Network for Sample Generation in Cross-Machine Small-Sample Fault Diagnosis"**. (Code **coming soon**.)

PCASTNet targets **cross-machine small-sample fault diagnosis**, where the monitored machine has very limited fault data while auxiliary machines provide abundant samples. The framework decouples **fault content** (from a reference machine) and **machine style** (from the monitored machine) via style transfer, and introduces **physics-aware constraints** to ensure realistic and physically consistent synthetic samples. 

<img width="3616" height="2284" alt="image" src="https://github.com/user-attachments/assets/1dce1ec0-31b3-4bd9-91b2-55e1ebeb7133" />


---

## Overview

PCASTNet is a **physics-constrained adaptive style transfer network** designed for vibration-based fault diagnosis. It operates in the **time–frequency domain** (CWT spectrograms) and aims to:

- Generate realistic fault samples for the monitored machine under **small-sample** conditions.
- Preserve **fault-discriminative content** while aligning with the **statistical style** of the monitored machine.
- Enforce **band energy preservation** to keep the generated signals physically plausible.

The framework is built upon:

- A **spectral feature encoder** (VGG19 backbone).
- An **Adaptive Style Normalization (AdaSN)** module that blends Instance Norm and Layer Norm with a learnable coefficient.
- A **decoder** that reconstructs style-transferred spectrograms.
- A **multi-objective loss** combining content fidelity, style consistency, and energy preservation.

> 🔔 **Note:** This repository is currently a placeholder. The full training & inference code will be released soon.

---

## Key Features (Planned)

- **Style–Content Separation**
  - Content: fault-specific semantics from a reference machine.
  - Style: machine- and condition-specific statistics from the monitored machine.

- **Adaptive Style Normalization (AdaSN)**
  - Learnable channel-wise interpolation between IN and LN.
  - Supports flexible style alignment across machines and speeds.

- **Band Energy Preservation Constraint**
  - Aligns band-wise energy distribution of generated samples with the target machine.
  - Encourages physically consistent synthetic spectrograms in the frequency domain.

- **Cross-Machine Small-Sample Setting**
  - Designed for scenarios with extremely limited monitored data.
  - Suitable for *CWRU → BJTU-RAO*, *CWRU → HUSTbearing* style transfer and similar tasks.

---

## Method Architecture (High-Level)

Planned implementation pipeline:

1. **Data Preparation**
   - Raw vibration signals → CWT spectrograms (e.g., using complex Morlet wavelet).
   - Resize and colormap to obtain RGB-like inputs for CNN backbones.

2. **Spectral Feature Encoder**
   - Pretrained VGG19 truncated at an intermediate layer (e.g., before ReLU4_1).
   - Frozen during PCASTNet training to extract stable spectral features.

3. **AdaSN Module**
   - Takes content features (reference machine) and style features (monitored machine).
   - Produces adaptively normalized features with machine-specific statistics.

4. **Feature Reconstruction Decoder**
   - Symmetric to the encoder.
   - Reconstructs high-fidelity spectrograms without additional normalization layers.

5. **Multi-Loss Training**
   - Content loss: preserves diagnostic semantics in latent space.
   - Style loss: matches multi-layer feature statistics of style images.
   - Band energy loss: aligns frequency-domain energy distributions.

> A detailed architecture diagram and training script will be added after the full code release.

---

## Installation (Coming Soon)

The full installation instructions will be provided once the code is released.

Planned environment (subject to change):

- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA (optional, for GPU acceleration)
- Additional dependencies listed in `requirements.txt` (coming soon)

```bash
# Placeholder
git clone https://github.com/your-username/PCASTNet.git
cd PCASTNet

# Coming soon:
# pip install -r requirements.txt
````

---

## Quick Start (Coming Soon)

We plan to provide:

1. **Preprocessing scripts**

   * Convert raw vibration signals to CWT spectrograms.
   * Example configurations for CWRU, HUSTbearing, BJTU-RAO, etc.

2. **Training script**

   * Train PCASTNet to generate style-aligned synthetic samples.
   * Support for different cross-machine scenarios.

3. **Evaluation script**

   * Train a simple CNN classifier on:

     * Real monitored samples.
     * Real + generated samples (augmented).
   * Report accuracy, F1, AUC, and confusion matrices.

Example commands (will be updated):

```bash
# Train PCASTNet (placeholder)
python train_pcastnet.py --config configs/cwru_to_bjtu.yaml

# Generate samples
python generate_samples.py --output_dir outputs/cwru_to_bjtu/

# Evaluate with augmented data
python train_classifier.py --data_dir outputs/cwru_to_bjtu/
```

---

## Datasets

PCASTNet is evaluated on multiple cross-machine bearing fault scenarios, including (but not limited to):

* **CWRU** (reference machine)
* **BJTU-RAO** (monitored machine)
* **HUSTbearing** (monitored machine)

Each scenario uses:

* Reference machine samples as **content**.
* Monitored machine small-sample data as **style**.
* Generated samples to augment the monitored machine training set.

Dataset preparation guides will be added later.

---

## Citation

If you find this work helpful, please consider citing the following paper:

```bibtex
@article{hu2025pcastnet,
  title   = {PCASTNet: A Physics-Constrained Adaptive Style Transfer Network for Sample Generation in Cross-Machine Small-Sample Fault Diagnosis},
  author  = {Hu, Xiaoxi and Li, Junyi and Huang, Yuhan and Zhang, Xinyu and Wang, Hengjun and Wang, Huan and He, Yiming},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  year    = {2025}
}
```

(We will adjust the BibTeX once the final publication details are confirmed.)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations, please feel free to contact:

- Contact: untuitivist@163.com

More details and documentation are **coming soon**.

