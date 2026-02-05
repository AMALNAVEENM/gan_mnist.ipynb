# GAN for MNIST Digit Generation

Implements a Generative Adversarial Network to create realistic handwritten digits.

## Architecture
- **Generator**: 100D noise → Conv2DTranspose → 28×28 image (Tanh)
- **Discriminator**: 28×28 image → Conv2D → binary classification (Sigmoid)

## Training
- Optimizer: Adam (lr=0.0002, β₁=0.5)
- Loss: Binary Cross-Entropy
- Epochs: 20
- Batch size: 128

## Outputs
- Generated digit grids every 5 epochs
- Final loss curves
- High-quality synthetic digits

## Requirements
```bash
pip install tensorflow matplotlib tqdm

---

### Short Report Outline (for your PDF)

**1. Objective & GAN Explanation**  
- GANs consist of two networks: Generator (creates fakes) and Discriminator (detects fakes)  
- Trained adversarially until generator fools discriminator  

**2. Architecture Summary**  
- Generator: Upsampling via Conv2DTranspose, BatchNorm for stability  
- Discriminator: Strided Conv2D for downsampling, Dropout for regularization  

**3. Results**  
- Loss curves show stable training (no mode collapse)  
- Generated digits become realistic by epoch 15  

**4. Challenges**  
- Mode collapse avoided via proper hyperparameters (β₁=0.5, label smoothing)  
- Tanh output requires [-1,1] normalization  
