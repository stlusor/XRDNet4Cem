# XRDNet4Cem: Physics-informed deep learning for automated phase identification of cementitious materials

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.8%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
  <img src="https://img.shields.io/badge/Platform-Windows%7CLinux%7CMacOS-lightgrey" alt="Platform">
</p>

## 🎯 Overview

**XRDNet4Cem** is an advanced deep learning framework designed for automated, high-throughput phase identification in cementitious materials from X-ray diffraction (XRD) patterns. By integrating physical crystallographic principles with state-of-the-art neural networks, this tool revolutionizes traditional search-match workflows, providing rapid and reliable phase analysis for both research and industrial applications.

## ✨ Key Features

### 🚀 **One-Click Execution**
- **XRDNet4Cem.bat** for Windows users - download, extract, and run
- No Python installation or coding knowledge required
- Automatic dependency management

### 🧠 **Intelligent Architecture**
- **Physics-informed neural networks** that understand crystallographic rules
- **Continuous Wavelet Transform (CWT)** processing for superior peak resolution
- **Multi-scale feature fusion** for handling variable crystallinity
- **Attention mechanisms** focusing on diagnostic diffraction features

### 📊 **Superior Performance**
- **>95% accuracy** for major cement phases
- **<2 seconds** per sample analysis
- **±2 wt.% precision** for quantitative estimates
- **130+ supported phases** covering OPC, CSA, CAC systems

### 🛠️ **User-Friendly Interface**
- **Graphical User Interface (GUI)** for intuitive operation
- **Batch processing** for high-throughput analysis
- **Multiple export formats** (CSV, Excel)
- **Interactive visualization** tools

## 🚀 Quick Start

### For Windows Users (Recommended)
1. **Download** the latest release from [Releases page](https://github.com/yourusername/XRDNet4Cem/releases)
2. **Extract** `XRDNet4Cem_v1.0.zip` to any folder
3. **Double-click** `XRDNet4Cem.bat`
4. **Start analyzing** your XRD data!

### For Developers & Researchers
```bash
# Clone the repository
git clone https://github.com/stlusor/XRDNet4Cem.git
cd XRDNet4Cem

# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py
```

## 📁 Project Structure

```
XRDNet4Cem/
├── XRDNet4Cem.bat              # Windows launcher
├── src/                        # Source code
│   ├── models/                 # Neural network architectures
│   ├── preprocessing/          # CWT and data processing
│   ├── simulation/             # Synthetic data generation
│   ├── gui/                    # Graphical interface
│   └── utils/                  # Helper functions
├── trained_models/             # Pre-trained weights
├── phase_library/              # 130+ cement phases (.cif)
├── examples/                   # Sample datasets
├── docs/                       # Documentation
└── tests/                      # Unit tests
```

## 🔬 Applications

### 🏭 **Industrial Use Cases**
- **Quality control** in cement manufacturing
- **Raw material characterization**
- **Production troubleshooting**
- **R&D formulation screening**

### 🔬 **Research Applications**
- **Hydration kinetics** monitoring
- **Carbonation studies** and durability assessment
- **Sustainable material** development
- **Phase evolution** tracking
- **Comparative studies** of different binders

## 📊 Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| ASCII | .xy, .xye, .txt | Multi-column data |
| CSV | .csv | Comma-separated values |

## 🏗️ Supported Phases

### Core Cement Phases (130 Total)
- **Calcium Silicates**: C3S (alite polymorphs), C2S (belite variants)
- **Calcium Aluminates**: C3A, CA, C12A7
- **Ferrites**: C4AF, C2F, C6AF2
- **Hydration Products**: AFt, AFm, Friedel's salt, Portlandite
- **Sulfates & Carbonates**: Gypsum, anhydrite, calcite, vaterite
- **Supplementary Materials**: Fly ash phases, slag minerals

### Regular Updates
- **Monthly**: Bug fixes and performance improvements
- **Quarterly**: New phase additions
- **Bi-annually**: Major feature updates

## 🤝 Community Contributions

### 📥 **We Need Your Data!**
Help us build the world's most comprehensive cement XRD database by contributing:

#### **Share Labeled XRD Data**
- Cement pastes at different hydration ages
- Clinker samples with known compositions
- Hydration products with quantitative analysis
- Experimental XRD patterns with metadata

#### **Expand Phase Library**
Missing a phase? Submit:
- **Phase name** and chemical formula
- **Standard .cif file** (PDF card)
- **Reference patterns** if available
- **Relevant publications**

### 🔄 **How to Contribute**
1. **Data Submission**: Open a GitHub Issue with "[Data]" prefix
2. **Phase Addition**: Submit .cif files with "[Phase]" prefix
3. **Bug Reports**: Use "[Bug]" prefix with detailed description
4. **Feature Requests**: Use "[Feature]" prefix with use case

All contributors will be acknowledged in our documentation!


## 🛠️ System Requirements

### Minimum
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **CPU**: 4-core processor
- **RAM**: 8 GB
- **Storage**: 2 GB free space

### Recommended
- **OS**: Windows 11, Ubuntu 20.04+
- **CPU**: 8-core processor
- **RAM**: 16 GB
- **GPU**: NVIDIA with 4+ GB VRAM (CUDA support)
- **Storage**: 5 GB SSD

## 🆘 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| "Python not found" | The .bat file will auto-install Miniconda |
| Memory errors | Reduce batch size in Settings |
| Slow performance | Enable GPU acceleration if available |
| File format issues | Use the built-in format converter |
| Missing phases | Submit .cif file via GitHub Issue |

### Getting Help
- **Documentation**: Complete user guide in `/docs/`
- **Examples**: Sample workflows in `/examples/`
- **Issues**: GitHub Issues for bug reports
- **Email**: xrdnet4cem.support@polyu.edu.hk

## 📝 Citation

If you use XRDNet4Cem in your research, please cite:

```bibtex
@article{lu2025xrdnet4cem,
  title={Physics-informed deep learning for automated phase identification 
         of cementitious materials: A high-throughput pre-screening framework},
  author={Lu, Jia-Hao and Ding, Siqi and Zhang, Yangyang},
  journal={-},
  volume={-},
  pages={-},
  year={-},
  publisher={-}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Funding**: Supported by The Hong Kong Polytechnic University (P0051104)
- **Contributors**: All community members sharing data and phases
- **Testing Partners**: Industrial partners for real-world validation
- **Open Source Community**: For all the amazing tools we build upon

## 📧 Contact

**Research Team:**
- Jia-Hao Lu (jia-hao.lu@connect.polyu.hk)
- Siqi Ding (siqi.ding@polyu.edu.hk) - Corresponding
- Yangyang Zhang (zhangyangyang@ysu.edu.cn) - Corresponding


**Repository:** https://github.com/stlusor/XRDNet4Cem

---

<p align="center">
  <em>Accelerating cement materials discovery through intelligent automation</em>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/yourusername/XRDNet4Cem?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/yourusername/XRDNet4Cem?style=social" alt="GitHub forks">
  <img src="https://img.shields.io/github/issues/yourusername/XRDNet4Cem" alt="GitHub issues">
</p>
