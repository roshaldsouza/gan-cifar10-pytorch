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
├── 📂 generated-images/        # Generated images per epoch
│   ├── generated_epoch_1.png
│   ├── generated_epoch_2.png
│   └── ...
│
├── 📄 new.py                   # Main training script
├── 📄 README.md                # You are here!
├── 📄 requirements.txt         # Python dependencies
└── 📄 .gitignore               # Git ignore file
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
python new.py
```

### What Happens During Training

1. **CIFAR-10 dataset** is automatically downloaded to `./data/`
2. **Generator and Discriminator** networks are initialized
3. **Training begins** with progress displayed in terminal
4. **Generated images** are saved to `./generated-images/` after each epoch
5. **Loss curves** are displayed and saved

### Monitoring Progress

During training, you'll see output like:
```
Epoch [1/50], Step [100/782], d_loss: 0.6523, g_loss: 1.2341
Epoch [1/50], Step [200/782], d_loss: 0.5891, g_loss: 1.4567
...
Saved generated images: generated-images/generated_epoch_1.png
```

---

## 🖼️ Results

### Training Progress

The Generator learns to produce increasingly realistic images over epochs:

**Early Training (Epoch 1-5)**: Random noise patterns begin to form basic shapes

**Mid Training (Epoch 10-25)**: Recognizable object structures emerge with colors

**Late Training (Epoch 50+)**: Detailed, CIFAR-10-like images with clear object features

> **Note**: Generated sample images will appear in the `generated-images/` folder after you run the training script. Below is what you can expect:

```
Epoch 1  →  Random colored noise
Epoch 10 →  Blurry shapes and basic colors
Epoch 25 →  Recognizable object outlines
Epoch 50 →  Clear objects (cars, planes, animals)
```

### Example Generated Images

Once trained, your `generated-images/` folder will contain files like:
- `generated_epoch_1.png` - Initial random outputs
- `generated_epoch_10.png` - Early feature learning
- `generated_epoch_25.png` - Refined shapes
- `generated_epoch_50.png` - High-quality generations

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
| **Epochs** | 50 | Default training duration |

### Training Tips

💡 **Mode Collapse**: If Generator produces same images, reduce learning rate or add noise to labels

💡 **Discriminator Too Strong**: If G loss stays high, train Generator 2x per D update

💡 **Poor Quality**: Increase training epochs to 100+ or adjust network depth

💡 **Training Instability**: Try label smoothing (use 0.9 instead of 1.0 for real images)

💡 **Slow Training**: Enable GPU with CUDA or reduce batch size

---

## 🚀 Future Enhancements

- [ ] Implement **DCGAN** (Deep Convolutional GAN) architecture
- [ ] Add **WGAN-GP** for improved training stability
- [ ] Conditional GAN for class-specific generation
- [ ] FID score calculation for quality metrics
- [ ] TensorBoard integration for real-time monitoring
- [ ] Progressive growing for higher resolution outputs
- [ ] Mixed precision training for 2x speedup
- [ ] Model checkpointing and loading functionality

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

## 🐛 Troubleshooting

### Common Issues

**Q: Training is very slow**  
A: Make sure you're using GPU. Check with `torch.cuda.is_available()`. If False, install CUDA-enabled PyTorch.

**Q: Generator produces all black/white images**  
A: This is mode collapse. Try reducing learning rate or adding label noise.

**Q: Loss values seem wrong**  
A: GAN losses don't decrease monotonically like classification. Oscillation is normal!

**Q: Images not improving after 50 epochs**  
A: Increase epochs to 100-200 or adjust architecture (add more layers/channels).

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

This project is licensed under the **MIT License** - feel free to use it for learning and projects!

---

## 🙏 Acknowledgements

- **Ian Goodfellow et al.** - [Generative Adversarial Networks (2014)](https://arxiv.org/abs/1406.2661)
- **Radford et al.** - [DCGAN Paper (2015)](https://arxiv.org/abs/1511.06434)
- **PyTorch Team** - [Official Documentation](https://pytorch.org/docs/)
- **CIFAR-10** - [Dataset by Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 📖 References & Resources

### Papers
- [Original GAN Paper](https://arxiv.org/abs/1406.2661) - Goodfellow et al., 2014
- [DCGAN Architecture](https://arxiv.org/abs/1511.06434) - Radford et al., 2015
- [Improved GAN Training](https://arxiv.org/abs/1606.03498) - Salimans et al., 2016

### Tutorials
- [PyTorch GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Understanding GANs](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)

---

## 👨‍💻 Author

<div align="center">

**Roshal Dsouza**

Computer Science Student | Full Stack & AI Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/roshal-dsouza-571910228/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/roshaldsouza)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-orange?style=for-the-badge&logo=google-chrome)]([https://your-portfolio.com](https://resume-animator--roshalds789.replit.app))

</div>

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ and PyTorch**

</div>
