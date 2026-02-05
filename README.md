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
- **130 supported phases** covering OPC, CSA, CAC systems

### 🛠️ **User-Friendly Interface**
- **Graphical User Interface (GUI)** for intuitive operation
- **Batch processing** for high-throughput analysis
- **Multiple export formats** (CSV, Excel)
- **Interactive visualization** tools

## 🚀 Quick Start

### For Windows Users
1. **Download** the latest release from [Releases page](https://github.com/stlusor/XRDNet4Cem/releases)
2. **Extract** `XRDNet4Cem_v1.0.zip` to any folder
3. **Double-click** `XRDNet4Cem.bat`
4. **Start analyzing** your XRD data!

## ▶️ Instructions
**Step 1: Data preprocessing module**
    **Import your XRD data file (.txt, .csv, .xye). The raw diffraction pattern will be displayed. Click the processing buttons in sequence: Resample to standardize the angle range, Subtract Background to remove noise, and Normalize the intensity. Finally, save the processed data or send it to the AI module.**
<img width="1446" height="792" alt="image" src="https://github.com/user-attachments/assets/8c842420-ed41-4ce9-9b87-f16761daae5d" />

**Step 2: AI analysis module**
    **Configure the settings: select your computing device (GPU/CPU) and load the AI model file (.pth). Add the preprocessed data to the queue and click Start Analysis. The software will intelligently identify the phases and display the fitting results, stacked phase contributions, a quantitative bar chart, and a detailed results table.**

<img width="1435" height="838" alt="image" src="https://github.com/user-attachments/assets/e87ba7df-4fcd-4177-9b58-b18418bf0abb" />

**Step 3: GSAS-II refinement module**
    **Set the path to your local GSAS-II installation. Load the XRD data file and the corresponding CIF files. Select the phases identified by the AI module to include in the refinement. Click Start Refinement. The module will run the Rietveld refinement and display the fitted pattern, quantitative phase results, and a statistical table with refined structural parameters.**

<img width="1291" height="753" alt="image" src="https://github.com/user-attachments/assets/a05569d8-5cf6-4119-af34-d8baba9357ad" />


## 📁 Project Structure

```
XRDNet4Cem/
├── XRDNet4Gem.bat                    # Main launcher - CLICK THIS!
├── python.exe                        # Embedded Python 3.13
├── Software_Rievied.py              # Main application script
├── Model.py                         # Deep learning model
├── Model/                           # Pre-trained neural networks
├── CIF/                             # Phase library (.cif files)
├── Phase_names.csv                  # Phase database
├── INST_XRY.PRM                     # Instrument parameters
├── ExperimentRawData/               # Sample XRD data
├── GSAS-II/                         # Rietveld refinement interface
├── GSAS_Output/                     # Analysis results
└── CementXRD/                       # Additional resources
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

# 🤝 Community Contributions

## 📊 **Help Build Cement XRD Database!**

We believe that collaborative science drives the fastest progress. Your contributions of high-quality labeled XRD data will help make XRDNet4Cem more accurate, robust, and comprehensive for the entire cement research community.

---

### **📁 How to Submit Your Data**

We've designed a simple, organized submission system with three dedicated folders:

```
New XRD data collection/
├── Labelized XRD data/                          # Folder for XRD pattern files
│   ├── OPC_7days_hydration.txt
│   ├── CSA_clinker.csv
│   └── Blended_cement_28d.xrd
├── Label/                        # Folder for quantitative analysis labels
│   ├── OPC_7days_hydration_label.csv
│   ├── CSA_clinker_label.csv
│   └── Blended_cement_28d_label.csv
└── New phase/                        # Folder for crystallographic information
    ├── New_Alite_Polymorph.cif
    ├── Doped_Yeelimite.cif
    └── Special_AFm_Phase.cif
```

---

### **📝 Detailed Submission Guidelines**

#### **1. XRD Data Folder (`Labelized XRD data/`)**
**What to include:**
- Raw XRD patterns in any standard format (.txt, .csv, .xrd, .raw, .xy, .dat)
- Recommended 2θ range: 5-70° for cement analysis
- Preferred step size: ≤0.02° for better resolution

**File naming convention:**
```
Material_System_Property_Condition.extension
Example: OPC_wc0.4_28d_hydration_25C.txt
```

**Required data format for .txt/.csv:**
```
Angle,Intensity
5.00,125
5.02,128
5.04,132
...
```

#### **2. Label Folder (`Label/`)**
**Each label file should correspond to an XRD file with the same base name.**

**Required CSV format: (It is recommented to follow the Label(Example).csv as a reference)**
```csv
Phase_Name,Weight_Percent
C3S,45.2
C2S,25.8
C3A,8.5
C4AF,9.3
Gypsum,5.2
Calcite,3.5
Amorphous,2.5
```

#### **3. Phase Folder (`New phase/`)**
**For any new or unusual phases in your samples:**
- Provide standard .cif files (Crystallographic Information Files)
- Sources: ICSD, COD, or your own refined structures
- Name files descriptively: `Ca3SiO5_Monoclinic.cif`, `Yeelimite_Sr-doped.cif`

**If .cif is unavailable:**
- Provide PDF card number (e.g., PDF 00-042-0551)
- Or provide unit cell parameters in a text file


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
| File format issues | Use the built-in format converter |
| Missing phases | Submit .cif file via GitHub Issue |


## 📝 Citation

If you use XRDNet4Cem in your research, please cite:

```bibtex
@article{luxrdnet4cem,
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
- Jia-Hao Lu (jia-hao.lu@connect.polyu.hk) - First author
- Siqi Ding (siqi.ding@polyu.edu.hk) - Corresponding author
- Yangyang Zhang (zhangyangyang@ysu.edu.cn) - Corresponding author


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
