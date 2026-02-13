# 🎨 GAN on CIFAR-10 using PyTorch

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A clean, educational implementation of Generative Adversarial Networks for image synthesis**

[Features](#-features) • [Installation](#️-installation) • [Usage](#-usage) • [Results](#-results) • [Architecture](#-architecture)

</div>

---

## 📖 Overview

This repository provides a **beginner-friendly implementation** of a Generative Adversarial Network (GAN) trained on the CIFAR-10 dataset. The project demonstrates how two neural networks—a Generator and a Discriminator—compete in a zero-sum game to produce realistic 32×32 color images.

### 🎯 Key Highlights

- ✅ **Production-ready code** with clear documentation
- ✅ **Step-by-step training visualization** 
- ✅ **Modular architecture** for easy experimentation
- ✅ **GPU acceleration** support
- ✅ **Comprehensive logging** and checkpointing

### 🎓 Perfect For

- Deep learning students and researchers
- Interview preparation and portfolio projects
- Understanding generative modeling fundamentals
- Academic coursework and presentations

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| **Auto Dataset Download** | Automatically fetches CIFAR-10 via `torchvision` |
| **Real-time Monitoring** | Track loss curves and generated samples during training |
| **Flexible Configuration** | Easy hyperparameter tuning via config file |
| **Checkpoint System** | Save and resume training from any epoch |
| **Mixed Precision** | Optional AMP support for faster training |

---

## 🧠 Dataset Information

**CIFAR-10** is a well-established computer vision dataset containing:

- **60,000** color images (32×32 pixels)
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **50,000** training images + **10,000** test images
- Automatically downloaded on first run

---

## 🛠️ Tech Stack

```
Core Framework    : PyTorch 2.0+
Data Pipeline     : Torchvision, NumPy
Visualization     : Matplotlib, TensorBoard (optional)
Environment       : Python 3.10+
```

---

## 📁 Project Structure

```
gan-cifar10-pytorch/
│
├── 📂 data/                    # CIFAR-10 dataset (auto-downloaded)
├── 📂 checkpoints/             # Saved model weights
├── 📂 results/
│   ├── generated_samples/      # Generated images per epoch
│   └── loss_curves/            # Training loss plots
│
├── 📄 gan_cifar10.py           # Main training script
├── 📄 models.py                # Generator & Discriminator architectures
├── 📄 utils.py                 # Helper functions
├── 📄 config.py                # Hyperparameter configuration
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md                # You are here!
└── 📄 LICENSE                  # MIT License
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/gan-cifar10-pytorch.git
cd gan-cifar10-pytorch
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Or install manually:**
```bash
pip install torch torchvision matplotlib numpy tqdm
```

---

## ▶️ Usage

### Basic Training
```bash
python gan_cifar10.py
```

### Custom Configuration
```bash
python gan_cifar10.py --epochs 100 --batch_size 128 --lr 0.0002
```

### Resume from Checkpoint
```bash
python gan_cifar10.py --resume checkpoints/gan_epoch_50.pth
```

### Available Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 0.0002 | Learning rate for both networks |
| `--latent_dim` | 100 | Dimension of latent noise vector |
| `--device` | cuda | Device to use (cuda/cpu) |
| `--save_interval` | 5 | Epochs between saving checkpoints |

---

## 🖼️ Results

### Training Progress

The Generator learns to produce increasingly realistic images over time:

| Epoch 1 | Epoch 4 | Epoch 7 | Epoch 10 |
|---------|----------|----------|-----------|
| ![](generated_images/generated_epoch_1.png) | ![](generated_images/generated_epoch_4.png) | ![](generated_images/generated_epoch_7.png) | ![](generated_images/generated_epoch_10.png) |


---

## 🏗️ Architecture

### Generator Network

```
Input: Random Noise Vector (100-dim)
    ↓
Linear Layer (100 → 4096)
    ↓
Reshape to (256, 4, 4)
    ↓
ConvTranspose2d + BatchNorm + ReLU (256 → 128)
    ↓
ConvTranspose2d + BatchNorm + ReLU (128 → 64)
    ↓
ConvTranspose2d + Tanh (64 → 3)
    ↓
Output: RGB Image (3, 32, 32)
```

**Total Parameters**: ~1.2M

### Discriminator Network

```
Input: RGB Image (3, 32, 32)
    ↓
Conv2d + LeakyReLU + Dropout (3 → 64)
    ↓
Conv2d + BatchNorm + LeakyReLU + Dropout (64 → 128)
    ↓
Conv2d + BatchNorm + LeakyReLU + Dropout (128 → 256)
    ↓
Flatten + Linear Layer
    ↓
Output: Probability (Real/Fake)
```

**Total Parameters**: ~850K

---

## 📊 Training Details

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rate, good for GANs |
| **Learning Rate** | 0.0002 | Standard for DCGAN architecture |
| **Beta1** | 0.5 | Momentum for Adam (lower for stability) |
| **Beta2** | 0.999 | Second momentum term |
| **Loss Function** | Binary Cross Entropy | Standard GAN objective |
| **Batch Size** | 64 | Balance between speed and stability |
| **Latent Dimension** | 100 | Size of random noise input |

### Training Tips

- **Mode Collapse**: If Generator produces same images, reduce learning rate
- **Discriminator Too Strong**: Balance training by updating Generator more frequently
- **Poor Quality**: Increase training epochs or adjust architecture depth
- **Instability**: Try label smoothing (0.9 instead of 1.0 for real images)

---

## 🚀 Future Enhancements

- [ ] Implement **DCGAN** (Deep Convolutional GAN) architecture
- [ ] Add **WGAN-GP** for improved training stability
- [ ] Conditional GAN for class-specific generation
- [ ] FID score calculation for quality metrics
- [ ] TensorBoard integration for better visualization
- [ ] Progressive growing for higher resolution outputs
- [ ] Mixed precision training for 2x speedup
- [ ] Multi-GPU support via DataParallel

---

## 📚 Learning Outcomes

After completing this project, you will understand:

✅ **GAN Architecture**: How Generator and Discriminator interact  
✅ **Adversarial Training**: The min-max game between networks  
✅ **PyTorch Fundamentals**: Model building, training loops, optimization  
✅ **Image Processing**: Normalization, transformations, tensor operations  
✅ **Training Dynamics**: Loss interpretation, convergence patterns  
✅ **Debugging GANs**: Common failure modes and solutions  

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Ian Goodfellow et al.** - [Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661)
- **Radford et al.** - [DCGAN Paper (2015)](https://arxiv.org/abs/1511.06434)
- **PyTorch Team** - [Official Documentation](https://pytorch.org/docs/)
- **CIFAR-10** - [Dataset by Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 👨‍💻 Author

<div align="center">

**Roshal Dsouza**

Computer Science Student | Full Stack & AI Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/roshal-dsouza-571910228/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/roshaldsouza)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=for-the-badge&logo=gmail)](mailto:your.roshalds789@gmail.com)

</div>

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ and PyTorch**

</div>
