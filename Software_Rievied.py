import sys
import os
import shutil
import glob
import copy
import time
import json
import traceback
import io
import contextlib
import importlib.util
import ctypes
from pathlib import Path


# =========================================================================
# 🚀 核心修复：防止 Anaconda DLL 劫持 (DLL Hell Fix)
# =========================================================================
def force_load_qt_dlls():
    # 1. 确定基准路径
    if getattr(sys, 'frozen', False):
        # 打包后的路径 (sys._MEIPASS 或 exe 所在目录)
        base_dir = Path(sys.executable).parent
    else:
        # 开发环境路径
        base_dir = Path(__file__).parent
        # 如果是开发环境，强制指向 pip 安装的路径
        # 请根据你的实际情况确认这个路径是否存在，如果不存在请修改
        dev_qt_bin = Path(r"C:\Software\Anaconda\envs\pytorch313\Lib\site-packages\PyQt6\Qt6\bin")
        if dev_qt_bin.exists():
            os.add_dll_directory(str(dev_qt_bin))
            base_dir = dev_qt_bin  # 临时借用变量

    # 2. 定义关键 DLL 列表 (顺序很重要: Core -> Gui -> Widgets)
    dlls_to_load = ["Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll"]

    # 3. 搜索路径：优先看 exe 同级目录，其次看 PyQt6/Qt6/bin 子目录
    search_paths = [
        base_dir,
        base_dir / "PyQt6" / "Qt6" / "bin",
    ]

    print(f"🔍 Searching for Qt DLLs in: {base_dir}")

    for dll_name in dlls_to_load:
        loaded = False
        for search_path in search_paths:
            dll_path = search_path / dll_name
            if dll_path.exists():
                try:
                    # 【核心大招】显式加载，锁定版本！
                    ctypes.CDLL(str(dll_path))
                    print(f"✅ Successfully pre-loaded: {dll_path}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load {dll_path}: {e}")

        if not loaded:
            print(f"❌ Warning: Could not find/load {dll_name}")


# 执行强制加载
force_load_qt_dlls()

# =========================================================================
# 设置环境变量
# =========================================================================
os.environ["QT_API"] = "PyQt6"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================================
# 现在开始导入库
# =========================================================================
# 1. 先导入 PyQt6 核心
import PyQt6
# 截图中的第 59 行就在这里：
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QListWidget, QProgressBar, QTabWidget, QMessageBox,
                             QComboBox, QGroupBox, QTableWidget, QTableWidgetItem,
                             QSpinBox, QDoubleSpinBox, QDialog, QTextEdit, QGridLayout,
                             QHeaderView, QCheckBox, QListWidgetItem, QProgressDialog,
                             QFormLayout, QLineEdit, QMenu, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QTextCursor

# 2. 再导入 Matplotlib 并配置后端
import matplotlib
matplotlib.use('qtagg')  # 强制指定 Qt6 后端
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# 设置 Matplotlib 字体
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 3. 科学计算库
import numpy as np
import pandas as pd
import torch
import pywt
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

# 4. 其他库
from pymatgen.io.cif import CifParser
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import subprocess

# 5. 模型导入 (带保护)
try:
    from Model import XRD_CNN_CWT
except ImportError:
    print("⚠️ Warning: Model.py not found or failed to import. AI Analysis features may fail.")
    # 定义一个空类防止报错
    class XRD_CNN_CWT:
        pass


def auto_configure_gsas(gsas_root):
    """
    自动扫描并配置 GSAS-II（从根目录开始）
    返回: (success, G2sc, config_info)
    """


    if not os.path.exists(gsas_root):
        return False, None, {"error": f"Path does not exist: {gsas_root}"}

    config_info = {
        'root': gsas_root,
        'module_path': None,
        'binary_path': None,
        'found_binaries': [],
        'search_details': []
    }

    # ============================================================
    # Step 1: 查找 GSASIIscriptable.py（排除 backcompat）
    # ============================================================
    config_info['search_details'].append("🔍 Searching for GSASIIscriptable.py...")

    scriptable_files = glob.glob(
        os.path.join(gsas_root, "**", "GSASIIscriptable.py"),
        recursive=True
    )

    if not scriptable_files:
        config_info['error'] = "GSASIIscriptable.py not found"
        config_info['search_details'].append("❌ GSASIIscriptable.py not found")
        return False, None, config_info

    # ✅ 优先选择非 backcompat 的版本
    module_path = None
    for sf in scriptable_files:
        if 'backcompat' not in sf.lower():
            module_path = os.path.dirname(sf)
            config_info['search_details'].append(f"✅ Found module at: {module_path}")
            break

    if not module_path:
        module_path = os.path.dirname(scriptable_files[0])
        config_info['search_details'].append(f"⚠️  Using: {module_path}")

    config_info['module_path'] = module_path

    # ============================================================
    # Step 2: 查找二进制目录
    # ============================================================
    config_info['search_details'].append("🔍 Searching for binary files...")

    # 查找 GSAS-II 专用的二进制文件
    binary_patterns = [
        os.path.join(gsas_root, "**", "GSASII-bin", "**", "*.pyd"),
        os.path.join(gsas_root, "**", "GSASII-bin", "**", "*.so"),
    ]

    binary_files = []
    for pattern in binary_patterns:
        binary_files.extend(glob.glob(pattern, recursive=True))

    if binary_files:
        # 使用第一个找到的二进制文件的目录
        binary_path = os.path.dirname(binary_files[0])
        config_info['binary_path'] = binary_path
        config_info['search_details'].append(f"✅ Found binaries at: {binary_path}")

        # 统计二进制文件
        try:
            for f in os.listdir(binary_path):
                if f.endswith(('.pyd', '.so', '.dll')):
                    config_info['found_binaries'].append(f)
            config_info['search_details'].append(f"📊 Found {len(config_info['found_binaries'])} binary files")
        except:
            pass
    else:
        config_info['search_details'].append("⚠️  No GSAS-II binary files found")

    # ============================================================
    # Step 3: 配置环境
    # ============================================================
    config_info['search_details'].append("⚙️  Configuring environment...")

    # 清理旧的 GSAS-II 相关路径
    paths_to_remove = [p for p in sys.path if 'GSAS' in p or 'gsas' in p.lower()]
    for p in paths_to_remove:
        sys.path.remove(p)
        config_info['search_details'].append(f"🗑️  Removed old path: {os.path.basename(p)}")

    # ✅ 添加模块路径（必须在最前面）
    if module_path not in sys.path:
        sys.path.insert(0, module_path)
        config_info['search_details'].append(f"✅ Added to sys.path: {module_path}")

    # ✅ 添加父目录（重要！解决相对导入问题）
    parent_path = os.path.dirname(module_path)
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)
        config_info['search_details'].append(f"✅ Added parent to sys.path: {parent_path}")

    # ✅ 添加二进制路径
    if config_info['binary_path']:
        bin_path = config_info['binary_path']
        if bin_path not in sys.path:
            sys.path.insert(0, bin_path)
            config_info['search_details'].append(f"✅ Added to sys.path: {bin_path}")

        # 添加到系统 PATH
        if bin_path not in os.environ.get('PATH', ''):
            os.environ['PATH'] = bin_path + os.pathsep + os.environ.get('PATH', '')
            config_info['search_details'].append(f"✅ Added to PATH: {bin_path}")

    # ============================================================
    # Step 4: 导入模块（使用多种方法）
    # ============================================================
    config_info['search_details'].append("🔧 Attempting to import GSAS-II...")

    # 清除旧模块
    modules_to_clear = [mod for mod in list(sys.modules.keys()) if 'GSASII' in mod or 'gsasii' in mod.lower()]
    for mod in modules_to_clear:
        del sys.modules[mod]

    if modules_to_clear:
        config_info['search_details'].append(f"🗑️  Cleared {len(modules_to_clear)} old GSAS modules")

    # 保存当前工作目录
    original_cwd = os.getcwd()

    try:
        # ✅ 切换到模块目录
        os.chdir(module_path)
        config_info['search_details'].append(f"📂 Changed working directory to module path")



        # 静默导入
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):

            # ✅ 方法 1：尝试标准导入
            try:
                # 先导入 GSASIIpath
                try:
                    import GSASIIpath
                    try:
                        GSASIIpath.SetBinaryPath(quiet=True)
                        config_info['search_details'].append("✅ GSASIIpath.SetBinaryPath(quiet=True)")
                    except TypeError:
                        GSASIIpath.SetBinaryPath()
                        config_info['search_details'].append("✅ GSASIIpath.SetBinaryPath()")
                except Exception as e:
                    config_info['search_details'].append(f"⚠️  GSASIIpath: {str(e)[:40]}")

                # 导入主模块
                import GSASIIscriptable as G2sc
                config_info['search_details'].append("✅ GSASIIscriptable imported (standard method)")

                os.chdir(original_cwd)
                return True, G2sc, config_info

            except Exception as e1:
                config_info['search_details'].append(f"⚠️  Standard import failed: {str(e1)[:50]}")

                # ✅ 方法 2：使用 importlib 手动加载
                try:
                    config_info['search_details'].append("🔧 Trying manual import with importlib...")

                    # 手动加载 GSASIIpath
                    try:
                        path_file = os.path.join(module_path, "GSASIIpath.py")
                        if os.path.exists(path_file):
                            spec = importlib.util.spec_from_file_location("GSASIIpath", path_file)
                            GSASIIpath = importlib.util.module_from_spec(spec)
                            sys.modules['GSASIIpath'] = GSASIIpath
                            spec.loader.exec_module(GSASIIpath)

                            try:
                                GSASIIpath.SetBinaryPath(quiet=True)
                            except:
                                try:
                                    GSASIIpath.SetBinaryPath()
                                except:
                                    pass

                            config_info['search_details'].append("✅ GSASIIpath loaded manually")
                    except Exception as e2:
                        config_info['search_details'].append(f"⚠️  GSASIIpath manual load: {str(e2)[:40]}")

                    # 手动加载 GSASIIscriptable
                    scriptable_file = os.path.join(module_path, "GSASIIscriptable.py")
                    spec = importlib.util.spec_from_file_location("GSASIIscriptable", scriptable_file)
                    G2sc = importlib.util.module_from_spec(spec)

                    # ✅ 关键：设置 __package__ 避免相对导入错误
                    G2sc.__package__ = 'GSASII'

                    sys.modules['GSASIIscriptable'] = G2sc
                    spec.loader.exec_module(G2sc)

                    config_info['search_details'].append("✅ GSASIIscriptable loaded manually (workaround)")

                    os.chdir(original_cwd)
                    return True, G2sc, config_info

                except Exception as e2:
                    config_info['search_details'].append(f"❌ Manual import also failed: {str(e2)[:50]}")
                    raise e2

    except Exception as e:
        # 确保恢复工作目录
        try:
            os.chdir(original_cwd)
        except:
            pass

        config_info['error'] = str(e)
        config_info['search_details'].append(f"❌ All import methods failed")

        # 详细错误诊断

        error_trace = traceback.format_exc()

        if "relative import" in str(e).lower():
            config_info['search_details'].append("⚠️  Issue: Python package structure problem")
            config_info['search_details'].append("💡 This is a known GSAS-II import issue")
        elif "numpy" in str(e).lower():
            config_info['search_details'].append("⚠️  Issue: NumPy version incompatibility")
        elif "dll" in str(e).lower() or "pyd" in str(e).lower():
            config_info['search_details'].append("⚠️  Issue: Binary file loading problem")

        return False, None, config_info

def baseline_als(y, lam=10000, p=0.001, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def create_cwt_image(xrd_data, scales=32, wavelet='morl'):
    xrd_normalized = (xrd_data - np.min(xrd_data)) / (np.max(xrd_data) - np.min(xrd_data) + 1e-10)
    scales_values = np.logspace(0.1, 1.5, num=scales)
    coefficients, frequencies = pywt.cwt(xrd_normalized, scales_values, wavelet)
    cwt_image = np.abs(coefficients)
    cwt_image = (cwt_image - np.min(cwt_image)) / (np.max(cwt_image) - np.min(cwt_image) + 1e-10)
    return cwt_image


def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'same') / window


class XRDFormatDialog(QDialog):
    """XRD 数据格式配置对话框"""

    def __init__(self, xrd_file, parent=None):
        super().__init__(parent)
        self.xrd_file = xrd_file
        self.preview_data = []

        self.setWindowTitle("Configure XRD Data Format")
        self.setModal(True)
        self.resize(700, 500)

        self.init_ui()
        self.load_preview()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # ============================================================
        # 标题和说明
        # ============================================================
        title_label = QLabel("📊 Configure XRD Data Format")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        layout.addWidget(title_label)

        info_label = QLabel(
            "Please specify the data format of your XRD file.\n"
            "Preview the first few lines below to identify columns."
        )
        info_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(info_label)

        # ============================================================
        # 文件预览
        # ============================================================
        preview_group = QGroupBox("File Preview (First 20 lines)")
        preview_layout = QVBoxLayout()

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setFont(QFont("Courier New", 9))
        self.preview_text.setMaximumHeight(200)
        preview_layout.addWidget(self.preview_text)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # ============================================================
        # 格式配置
        # ============================================================
        config_group = QGroupBox("Data Format Configuration")
        config_layout = QFormLayout()

        # 跳过行数
        self.skip_rows_spin = QSpinBox()
        self.skip_rows_spin.setRange(0, 100)
        self.skip_rows_spin.setValue(0)
        self.skip_rows_spin.setToolTip("Number of header lines to skip")
        self.skip_rows_spin.valueChanged.connect(self.update_preview_highlight)
        config_layout.addRow("Skip Rows:", self.skip_rows_spin)

        # 分隔符
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems(["Whitespace (auto)", "Tab", "Comma", "Semicolon"])
        self.delimiter_combo.setCurrentIndex(0)
        config_layout.addRow("Delimiter:", self.delimiter_combo)

        # 2θ 列
        self.twotheta_spin = QSpinBox()
        self.twotheta_spin.setRange(1, 20)
        self.twotheta_spin.setValue(1)
        self.twotheta_spin.setToolTip("Column number for 2θ values (1-based)")
        config_layout.addRow("2θ Column:", self.twotheta_spin)

        # Intensity 列
        self.intensity_spin = QSpinBox()
        self.intensity_spin.setRange(1, 20)
        self.intensity_spin.setValue(2)
        self.intensity_spin.setToolTip("Column number for Intensity values (1-based)")
        config_layout.addRow("Intensity Column:", self.intensity_spin)

        # ESD 列（可选）
        self.esd_check = QCheckBox("Use ESD column")
        self.esd_check.setChecked(False)
        self.esd_check.toggled.connect(self.toggle_esd)
        config_layout.addRow("", self.esd_check)

        self.esd_spin = QSpinBox()
        self.esd_spin.setRange(1, 20)
        self.esd_spin.setValue(3)
        self.esd_spin.setEnabled(False)
        self.esd_spin.setToolTip("Column number for ESD (error) values (1-based)")
        config_layout.addRow("ESD Column:", self.esd_spin)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # ============================================================
        # 数据验证预览
        # ============================================================
        validation_group = QGroupBox("Data Validation")
        validation_layout = QVBoxLayout()

        self.validation_label = QLabel("Click 'Validate' to check data format")
        self.validation_label.setStyleSheet("color: #666;")
        validation_layout.addWidget(self.validation_label)

        self.validate_btn = QPushButton("🔍 Validate Format")
        self.validate_btn.clicked.connect(self.validate_format)
        validation_layout.addWidget(self.validate_btn)

        validation_group.setLayout(validation_layout)
        layout.addWidget(validation_group)

        # ============================================================
        # 按钮
        # ============================================================
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_btn = QPushButton("✓ OK")
        self.ok_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px 20px;")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)  # 需要先验证
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("✗ Cancel")
        self.cancel_btn.setStyleSheet("padding: 8px 20px;")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def load_preview(self):
        """加载文件预览"""
        try:
            with open(self.xrd_file, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    lines.append(f"{i + 1:3d}: {line.rstrip()}")

                self.preview_text.setPlainText('\n'.join(lines))

        except Exception as e:
            self.preview_text.setPlainText(f"Error loading file: {str(e)}")

    def update_preview_highlight(self):
        """更新预览中跳过行的高亮"""
        skip_rows = self.skip_rows_spin.value()

        try:
            with open(self.xrd_file, 'r', encoding='utf-8') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 20:
                        break

                    if i < skip_rows:
                        # 跳过的行显示为灰色
                        lines.append(f"<span style='color: #999;'>{i + 1:3d}: {line.rstrip()}</span>")
                    else:
                        lines.append(f"{i + 1:3d}: {line.rstrip()}")

                self.preview_text.setHtml('<pre>' + '\n'.join(lines) + '</pre>')

        except Exception as e:
            pass

    def toggle_esd(self, checked):
        """切换 ESD 列启用状态"""
        self.esd_spin.setEnabled(checked)

    def get_delimiter(self):
        """获取分隔符"""
        delimiter_map = {
            "Whitespace (auto)": None,  # None 表示使用 split()
            "Tab": "\t",
            "Comma": ",",
            "Semicolon": ";"
        }
        return delimiter_map[self.delimiter_combo.currentText()]

    def validate_format(self):
        """验证数据格式"""
        try:
            skip_rows = self.skip_rows_spin.value()
            twotheta_col = self.twotheta_spin.value() - 1  # 转为 0-based
            intensity_col = self.intensity_spin.value() - 1
            use_esd = self.esd_check.isChecked()
            esd_col = self.esd_spin.value() - 1 if use_esd else None
            delimiter = self.get_delimiter()

            # 读取数据
            valid_count = 0
            error_count = 0
            data_preview = []

            with open(self.xrd_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    # 跳过指定行数
                    if i < skip_rows:
                        continue

                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith(';'):
                        continue

                    # 分割数据
                    if delimiter is None:
                        parts = line.split()
                    else:
                        parts = line.split(delimiter)

                    # 验证列数
                    max_col = max(twotheta_col, intensity_col)
                    if esd_col is not None:
                        max_col = max(max_col, esd_col)

                    if len(parts) <= max_col:
                        error_count += 1
                        if error_count <= 3:
                            data_preview.append(f"❌ Line {i + 1}: Not enough columns ({len(parts)} < {max_col + 1})")
                        continue

                    # 验证数据类型
                    try:
                        two_theta = float(parts[twotheta_col])
                        intensity = float(parts[intensity_col])

                        if use_esd and esd_col is not None:
                            esd = float(parts[esd_col])

                        valid_count += 1

                        if valid_count <= 3:
                            data_preview.append(
                                f"✓ Line {i + 1}: 2θ={two_theta:.3f}°, I={intensity:.1f}"
                            )

                    except (ValueError, IndexError) as e:
                        error_count += 1
                        if error_count <= 3:
                            data_preview.append(f"❌ Line {i + 1}: Cannot parse as numbers")

                    # 只检查前 100 行
                    if valid_count + error_count >= 100:
                        break

            # 显示验证结果
            if valid_count == 0:
                self.validation_label.setText(
                    f"❌ Validation Failed\n\n"
                    f"No valid data found!\n"
                    f"Errors: {error_count}\n\n"
                    f"Preview:\n" + "\n".join(data_preview)
                )
                self.validation_label.setStyleSheet("color: #f44336;")
                self.ok_btn.setEnabled(False)

            elif error_count > valid_count * 0.1:  # 超过 10% 错误
                self.validation_label.setText(
                    f"⚠️ Validation Warning\n\n"
                    f"Valid rows: {valid_count}\n"
                    f"Error rows: {error_count}\n\n"
                    f"Preview:\n" + "\n".join(data_preview) + "\n\n"
                                                              f"Continue anyway?"
                )
                self.validation_label.setStyleSheet("color: #FF9800;")
                self.ok_btn.setEnabled(True)

            else:
                self.validation_label.setText(
                    f"✅ Validation Successful\n\n"
                    f"Valid data rows: {valid_count}\n"
                    f"Error rows: {error_count}\n\n"
                    f"Preview:\n" + "\n".join(data_preview) + "\n\n"
                                                              f"Format looks good!"
                )
                self.validation_label.setStyleSheet("color: #4CAF50;")
                self.ok_btn.setEnabled(True)

        except Exception as e:
            self.validation_label.setText(f"❌ Validation Error:\n\n{str(e)}")
            self.validation_label.setStyleSheet("color: #f44336;")
            self.ok_btn.setEnabled(False)

    def get_config(self):
        """获取用户配置"""
        return {
            'skip_rows': self.skip_rows_spin.value(),
            'twotheta_col': self.twotheta_spin.value() - 1,  # 转为 0-based
            'intensity_col': self.intensity_spin.value() - 1,
            'use_esd': self.esd_check.isChecked(),
            'esd_col': self.esd_spin.value() - 1 if self.esd_check.isChecked() else None,
            'delimiter': self.get_delimiter()
        }



# Custom Import Dialog
# =========================================================
class DataImportDialog(QDialog):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Import Settings: {os.path.basename(filepath)}")
        self.resize(800, 600)
        self.filepath = filepath
        self.parsed_x = None
        self.parsed_y = None
        self.init_ui()
        self.load_preview()

    def init_ui(self):
        """初始化 XRD 数据导入对话框界面"""

        # 主布局
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # ============================================================
        # 1. 文件预览区域
        # ============================================================
        preview_label = QLabel("<b>📄 File Content Preview (First 50 lines):</b>")
        layout.addWidget(preview_label)

        self.txt_preview = QTextEdit()
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setFont(QFont("Courier New", 9))
        self.txt_preview.setMaximumHeight(200)  # 限制高度
        self.txt_preview.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.txt_preview)

        # ============================================================
        # 2. 解析设置组
        # ============================================================
        gb_settings = QGroupBox("⚙️ Parsing Settings")
        gb_settings.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)

        form = QGridLayout()
        form.setSpacing(10)
        form.setContentsMargins(10, 15, 10, 10)

        # Skip Rows 设置
        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(0, 1000)
        self.spin_skip.setValue(0)
        self.spin_skip.setToolTip("Number of header lines to skip")
        self.spin_skip.setMinimumWidth(80)
        self.spin_skip.valueChanged.connect(self.try_parse_preview)

        # Delimiter 设置
        self.combo_sep = QComboBox()
        self.combo_sep.addItems(["Auto", "Space/Tab", "Comma (,)", "Semicolon (;)", "Tab only"])
        self.combo_sep.setToolTip("Column delimiter/separator")
        self.combo_sep.setMinimumWidth(120)
        self.combo_sep.currentIndexChanged.connect(self.try_parse_preview)

        # Column X 设置 (2Theta)
        self.spin_col_x = QSpinBox()
        self.spin_col_x.setRange(0, 50)
        self.spin_col_x.setValue(0)
        self.spin_col_x.setToolTip("Column index for 2Theta values (0-based)")
        self.spin_col_x.setMinimumWidth(80)
        self.spin_col_x.valueChanged.connect(self.try_parse_preview)

        # Column Y 设置 (Intensity)
        self.spin_col_y = QSpinBox()
        self.spin_col_y.setRange(0, 50)
        self.spin_col_y.setValue(1)
        self.spin_col_y.setToolTip("Column index for Intensity values (0-based)")
        self.spin_col_y.setMinimumWidth(80)
        self.spin_col_y.valueChanged.connect(self.try_parse_preview)

        # 添加到表单布局
        # 第一行
        form.addWidget(QLabel("Skip Rows:"), 0, 0, Qt.AlignmentFlag.AlignRight)
        form.addWidget(self.spin_skip, 0, 1)
        form.addWidget(QLabel("Delimiter:"), 0, 2, Qt.AlignmentFlag.AlignRight)
        form.addWidget(self.combo_sep, 0, 3)

        # 第二行
        form.addWidget(QLabel("Column 2θ (X):"), 1, 0, Qt.AlignmentFlag.AlignRight)
        form.addWidget(self.spin_col_x, 1, 1)
        form.addWidget(QLabel("Column Intensity (Y):"), 1, 2, Qt.AlignmentFlag.AlignRight)
        form.addWidget(self.spin_col_y, 1, 3)

        # 第三行 - 测试按钮
        btn_preview_parse = QPushButton("🔄 Test Parse & Update Plot")
        btn_preview_parse.setToolTip("Parse the file with current settings and update the plot")
        btn_preview_parse.clicked.connect(self.try_parse_preview)
        btn_preview_parse.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        form.addWidget(btn_preview_parse, 2, 0, 1, 4)

        # 添加数据统计标签
        self.lbl_data_info = QLabel("📊 No data parsed yet")
        self.lbl_data_info.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        form.addWidget(self.lbl_data_info, 3, 0, 1, 4)

        gb_settings.setLayout(form)
        layout.addWidget(gb_settings)

        # ============================================================
        # 3. 图表预览区域
        # ============================================================
        plot_label = QLabel("<b>📈 Data Preview Plot:</b>")
        layout.addWidget(plot_label)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('2θ (degrees)', fontsize=10)
        self.ax.set_ylabel('Intensity (a.u.)', fontsize=10)
        self.ax.set_title('XRD Pattern Preview', fontsize=11, fontweight='bold')
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # 设置画布样式
        self.canvas.setStyleSheet("""
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 4px;
        """)

        layout.addWidget(self.canvas)

        # ============================================================
        # 4. 底部按钮
        # ============================================================
        btns = QHBoxLayout()
        btns.setSpacing(10)

        # 取消按钮
        btn_cancel = QPushButton("❌ Cancel")
        btn_cancel.setMinimumWidth(100)
        btn_cancel.setMinimumHeight(35)
        btn_cancel.clicked.connect(self.reject)
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

        # 导入按钮
        btn_ok = QPushButton("✅ Import Data")
        btn_ok.setMinimumWidth(100)
        btn_ok.setMinimumHeight(35)
        btn_ok.clicked.connect(self.accept_data)
        btn_ok.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)

        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        layout.addLayout(btns)



    def load_preview(self):
        try:
            with open(self.filepath, 'r') as f:
                head = [next(f) for _ in range(50)]
            self.txt_preview.setText("".join(head))
            for i, line in enumerate(head):
                if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '-'):
                    self.spin_skip.setValue(i)
                    break
            self.try_parse_preview()
        except Exception as e:
            self.txt_preview.setText(f"Error: {e}")

    def get_df(self):
        sep_map = {"Space/Tab": r'\s+', "Comma (,)": ',', "Semicolon (;)": ';'}
        sep = sep_map.get(self.combo_sep.currentText(), None)
        return pd.read_csv(self.filepath, sep=sep, skiprows=self.spin_skip.value(), header=None, engine='python')

    def try_parse_preview(self):
        try:
            df = self.get_df()
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            x_idx, y_idx = self.spin_col_x.value(), self.spin_col_y.value()
            if x_idx < df.shape[1] and y_idx < df.shape[1]:
                x, y = df.iloc[:, x_idx].values, df.iloc[:, y_idx].values
                self.ax.clear()
                self.ax.plot(x, y, 'b-')
                self.canvas.draw()
                return x, y
            return None, None
        except:
            return None, None

    def accept_data(self):
        x, y = self.try_parse_preview()
        if x is not None:
            self.parsed_x = x
            self.parsed_y = y
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Parse failed.")

        # =========================================================



class AnalysisWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, input_data, model_path, phase_names_path, ref_dir, device_type='cuda'):
        super().__init__()
        self.input_data = input_data
        self.model_path = model_path
        self.phase_names_path = phase_names_path
        self.ref_dir = ref_dir

        # 定义需要合并的主要物相
        self.main_phases_to_merge = {
            'C3S': ['C3Smono_nishi', 'C3Smono_torre', 'C3Srhom_jeffrey', 'C3Srhom_nishi', 'C3Striclinic_belov'],
            'C2S': ['C2Salpha\'H-m', 'C2Salpha\'L-m', 'C2Salpha_mumme', 'C2Sbeta_berliner', 'C2Sbeta_jost',
                    'C2Sbeta_mumme', 'C2Sgam_mumme'],
            'C3A': ['C3Acub', 'C3Amonoclinic', 'C3Anacub', 'C3Anaorth'],
            'C4AF': ['C4AF-trans', 'C4AF_colville']
        }

        if device_type == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def merge_phases_before_selection(self, all_phases, all_weights):
        """
        先合并相似相，然后选择权重最高的前6个
        """
        merged_data = {}

        # 第一遍：合并主要相
        for phase_name, weight in zip(all_phases, all_weights):
            phase_name_str = str(phase_name)
            merged = False

            for main_phase, variants in self.main_phases_to_merge.items():
                if phase_name_str in variants:
                    if main_phase not in merged_data:
                        merged_data[main_phase] = {
                            'weight': weight,
                            'original_names': [phase_name_str],
                            'is_merged': True
                        }
                    else:
                        merged_data[main_phase]['weight'] += weight
                        merged_data[main_phase]['original_names'].append(phase_name_str)
                    merged = True
                    break

            # 如果不需要合并，保持独立
            if not merged:
                merged_data[phase_name_str] = {
                    'weight': weight,
                    'original_names': [phase_name_str],
                    'is_merged': False
                }

        # 转换为结果列表并排序
        result = []
        for phase_name, data in merged_data.items():
            if data['is_merged'] and len(data['original_names']) > 1:
                display_name = f"{phase_name} ({len(data['original_names'])} variants)"
            else:
                display_name = phase_name

            result.append((display_name, data['weight'], data['original_names']))

        # 按权重降序排序
        result.sort(key=lambda x: x[1], reverse=True)

        # 返回前6个
        return result[:6]

    def robust_read_xye(self, filepath):
        try:
            data = []
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.replace(',', ' ').replace(';', ' ').split()

                try:
                    nums = [float(p) for p in parts]
                    if len(nums) >= 2:
                        data.append(nums[:2])
                except ValueError:
                    continue

            if len(data) > 5:
                arr = np.array(data)
                return arr[:, 0], arr[:, 1]
            else:
                return None, None
        except Exception as e:
            return None, None

    def find_ref_file(self, phase_name):
        p_str = str(phase_name).strip()

        direct_path = os.path.join(self.ref_dir, f"{p_str}.xye")
        if os.path.exists(direct_path):
            return direct_path

        direct_path_txt = os.path.join(self.ref_dir, f"{p_str}.txt")
        if os.path.exists(direct_path_txt):
            return direct_path_txt

        try:
            files = os.listdir(self.ref_dir)

            for f in files:
                if not (f.endswith('.xye') or f.endswith('.txt')):
                    continue
                f_base = os.path.splitext(f)[0]

                if p_str.lower() in f_base.lower():
                    return os.path.join(self.ref_dir, f)

        except Exception as e:
            self.log_signal.emit(f"Dir Error: {e}")

        return None

    def run(self):
        try:
            self.log_signal.emit(f"Initializing on: {self.device}")

            if not os.path.exists(self.phase_names_path):
                raise FileNotFoundError(f"Phase names file not found: {self.phase_names_path}")
            phase_names = pd.read_csv(self.phase_names_path, header=None).values.flatten()

            if 'XRD_CNN_CWT' not in globals():
                raise ImportError("Model class XRD_CNN_CWT not found.")
            model = XRD_CNN_CWT().to(self.device)

            with torch.no_grad():
                _ = model(torch.randn(1, 1, 32, 3251).to(self.device))

            try:
                state = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state if isinstance(state, dict) else state.state_dict())
            except Exception as e:
                self.log_signal.emit(f"Load Model Error: {str(e)}")
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.eval()
            self.log_signal.emit("Model loaded.")

            results_data = {}
            target_x = np.linspace(5.0, 70.0, 3251)

            is_file_list = isinstance(self.input_data, list)
            items = self.input_data if is_file_list else self.input_data.items()
            total = len(items)

            for idx, item in enumerate(items):
                fname = "Unknown"
                try:
                    if is_file_list:
                        fpath = item
                        fname = os.path.basename(fpath)
                        x_raw, y_raw = self.robust_read_xye(fpath)
                        if x_raw is None:
                            raise ValueError(f"Cannot read {fname}")
                    else:
                        fname, data_dict = item
                        x_raw, y_raw = data_dict['x'], data_dict['y']

                    f_int = interp1d(x_raw, y_raw, kind='linear', fill_value='extrapolate')
                    y_interp = np.maximum(f_int(target_x), 0)
                    y_max = np.max(y_interp)
                    if y_max <= 0:
                        y_max = 1e-10
                    y_norm = y_interp / y_max

                    cwt_img = create_cwt_image(y_norm, scales=32, wavelet='morl')
                    inp = torch.tensor(cwt_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        preds = model(inp).cpu().numpy().flatten()

                    sorted_indices = np.argsort(preds)[::-1]
                    all_phases = phase_names[sorted_indices]
                    all_weights = preds[sorted_indices]

                    # ========== 关键修改：先合并再选择前6个 ==========
                    merged_results = self.merge_phases_before_selection(all_phases, all_weights)

                    # 提取合并后的相名称和权重
                    top_phases_merged = [item[0] for item in merged_results]
                    top_weights_merged = [item[1] for item in merged_results]
                    original_names_list = [item[2] for item in merged_results]

                    weight_sum = np.sum(top_weights_merged)
                    if weight_sum == 0:
                        weight_sum = 1e-10
                    top_weights_norm = top_weights_merged / weight_sum

                    calc_pattern = np.zeros_like(target_x)
                    phase_curves = []

                    # 使用合并后的相进行参考文件查找和计算
                    for merged_phase_name, w, original_names in zip(top_phases_merged, top_weights_norm,
                                                                    original_names_list):
                        # 尝试使用第一个原始相名称查找参考文件
                        first_original_name = original_names[0] if original_names else merged_phase_name
                        p_file = self.find_ref_file(first_original_name)

                        if p_file:
                            self.log_signal.emit(f"✅ Match: {merged_phase_name} -> {os.path.basename(p_file)}")
                            px, py = self.robust_read_xye(p_file)

                            if px is not None:
                                try:
                                    py_max_ref = np.max(py)
                                    if py_max_ref > 0:
                                        py_norm_ref = py / py_max_ref
                                    else:
                                        py_norm_ref = py

                                    f_p = interp1d(px, py_norm_ref, kind='linear', fill_value='extrapolate',
                                                   bounds_error=False)
                                    interp_vals = f_p(target_x)
                                    interp_vals[np.isnan(interp_vals)] = 0

                                    p_val = np.maximum(interp_vals, 0) * w

                                    calc_pattern += p_val
                                    phase_curves.append((merged_phase_name, p_val))
                                except Exception as e:
                                    self.log_signal.emit(f"⚠️ Curve calculation failed for {merged_phase_name}: {e}")
                        else:
                            self.log_signal.emit(
                                f"⚠️ Ref Not Found: {merged_phase_name} (tried: {first_original_name})")

                    diff = y_norm - calc_pattern
                    diff_smooth = moving_average(diff, window=5)

                    results_data[fname] = {
                        'angles': target_x,
                        'exp_norm': y_norm,
                        'calc_pattern': calc_pattern,
                        'diff_pattern': diff_smooth,
                        'phase_curves': phase_curves,
                        'top_phases': top_phases_merged,
                        'top_weights': top_weights_norm,
                        'raw_weights': top_weights_merged,
                        'original_names_list': original_names_list
                    }

                except Exception as e:
                    self.log_signal.emit(f"❌ Error {fname}: {str(e)}")

                self.progress_signal.emit(int((idx + 1) / total * 100))

            self.finished_signal.emit(results_data)
        except Exception as e:
            self.error_signal.emit(str(e))

class GSASIIHelper:
    """GSAS-II 多版本兼容性辅助类"""

    @staticmethod
    def create_project(G2sc, output_path):
        """
        兼容多个版本的 GSAS-II 创建项目

        参数:
            G2sc: GSASIIscriptable 模块
            output_path: 输出的 .gpx 文件完整路径

        返回: (success: bool, gpx_object, method_name: str, error_msg: str)
        """
        # 确保输出目录存在
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except Exception as e:
            return False, None, None, f"Cannot create output directory: {str(e)}"

        # 如果文件已存在，删除它
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception as e:
                return False, None, None, f"Cannot remove existing file: {str(e)}"

        # ✅ 尝试多种创建方法
        methods = [
            # 方法1: 带文件路径和 newgpx 参数
            ("G2Project(gpxfile=path, newgpx=True)",
             lambda: G2sc.G2Project(gpxfile=output_path, newgpx=True)),

            # 方法2: 只带文件路径
            ("G2Project(newgpx=path)",
             lambda: G2sc.G2Project(newgpx=output_path)),

            # 方法3: 只带 newgpx=True，然后设置文件名
            ("G2Project(newgpx=True) + filename",
             lambda: G2sc.G2Project(newgpx=True)),
        ]

        # 检查是否有其他创建方法
        if hasattr(G2sc, 'NewG2Project'):
            methods.append(
                ("NewG2Project(path)",
                 lambda: G2sc.NewG2Project(output_path))
            )

        last_error = "All creation methods failed"

        for method_name, method_func in methods:
            try:
                gpx = method_func()

                # ✅ 验证返回对象
                if gpx is None:
                    last_error = f"{method_name}: returned None"
                    continue

                # ✅ 检查类型
                if isinstance(gpx, (str, bytes)):
                    last_error = f"{method_name}: returned {type(gpx).__name__} instead of object"
                    continue

                # ✅ 验证有必要的方法
                required_methods = ['add_phase', 'add_powder_histogram', 'save']
                missing = [m for m in required_methods if not hasattr(gpx, m)]

                if missing:
                    last_error = f"{method_name}: missing methods {missing}"
                    continue

                # ✅ 设置文件名（如果需要）
                if not hasattr(gpx, 'filename') or gpx.filename is None:
                    gpx.filename = output_path
                elif gpx.filename != output_path:
                    # 如果文件名不匹配，也设置为正确的路径
                    gpx.filename = output_path

                # ✅ 确保 data 属性存在
                if not hasattr(gpx, 'data'):
                    gpx.data = {}

                # ✅ 成功！
                return True, gpx, method_name, None

            except TypeError as e:
                last_error = f"{method_name}: TypeError - {str(e)}"
                continue
            except AttributeError as e:
                last_error = f"{method_name}: AttributeError - {str(e)}"
                continue
            except Exception as e:
                last_error = f"{method_name}: {type(e).__name__} - {str(e)}"
                continue

        return False, None, None, last_error

    @staticmethod
    def save_project(gpx, filepath):
        """
        兼容多个版本的保存方法
        返回: (success: bool, method_name: str, error_msg: str)
        """
        # 确保目录存在
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        except Exception as e:
            return False, None, f"Cannot create directory: {str(e)}"

        # 确保文件名设置正确
        if hasattr(gpx, 'filename'):
            gpx.filename = filepath

        # 尝试各种保存方法
        save_methods = [
            ("save(filepath)", lambda: gpx.save(filepath)),
            ("save()", lambda: gpx.save()),
        ]

        last_error = "No save method succeeded"

        for method_name, method_func in save_methods:
            try:
                result = method_func()

                # 验证文件是否被创建
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    return True, method_name, None
                else:
                    last_error = f"{method_name}: file not created or empty"

            except TypeError as e:
                last_error = f"{method_name}: TypeError - {str(e)}"
                continue
            except Exception as e:
                last_error = f"{method_name}: {type(e).__name__} - {str(e)}"
                continue

        return False, None, last_error


# ============================================================
# Refinement Worker 类（完整版）
# ============================================================

# =========================================================
# 改进的 Refinement Worker（基于您成功的独立脚本）
# =========================================================

# =========================================================
# 基于成功独立脚本的 Refinement Worker
# =========================================================

# =========================================================
# 完全基于成功独立脚本的 Refinement Worker
# =========================================================




class RefinementWorker(QThread):
    """Refinement thread (integrated with standalone script logic)"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    plot_data_signal = pyqtSignal(object, object, object, object, float)
    detailed_data_signal = pyqtSignal(dict)  # ← 只添加这一行

    def __init__(self, xrd_file, phase_names, cif_dir, output_dir, gsas_path, refinement_range=(7.0, 65.0)):
        super().__init__()
        self.xrd_file = xrd_file
        self.phase_names = phase_names
        self.cif_dir = cif_dir
        self.output_dir = output_dir
        self.gsas_path = gsas_path
        self.refinement_range = refinement_range
        self._is_running = True

    def create_publication_quality_plot(self, hist, phases, output_path):
        """
        创建期刊质量的Rietveld精修图
        """
        try:
            # 设置期刊风格的绘图参数
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'Arial',
                'mathtext.fontset': 'stix',
                'axes.linewidth': 1.2,
                'lines.linewidth': 1.5,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight'
            })

            # 创建图形和子图
            fig, (ax_main, ax_residual) = plt.subplots(2, 1, figsize=(10, 8),
                                                       gridspec_kw={'height_ratios': [3, 1]},
                                                       sharex=True)

            # 获取数据
            x_data = hist.getdata('X')
            y_obs = hist.getdata('Yobs')
            y_calc = hist.getdata('Ycalc')
            y_bkg = hist.getdata('Background')

            # 1. 主图：观测值、计算值、背景
            ax_main.plot(x_data, y_obs, 'k.', markersize=2, alpha=0.7, label='Observed')
            ax_main.plot(x_data, y_calc, 'r-', linewidth=1.2, label='Calculated')
            ax_main.plot(x_data, y_bkg, 'g--', linewidth=1, alpha=0.7, label='Background')

            # 2. 绘制各相的衍射峰（晶面指标）
            colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray']
            for i, phase in enumerate(phases):
                if i >= len(colors):
                    break

                phase_name = phase.data['General']['Name']
                try:
                    # 获取该相的反射数据
                    refl_dict = hist.reflections()
                    if phase_name in refl_dict:
                        reflist = refl_dict[phase_name].get('RefList', [])

                        for refl in reflist:
                            if len(refl) > 4:
                                d = float(refl[4])
                                wavelength = 1.5406  # Cu Kα
                                if d > 0:
                                    sin_theta = wavelength / (2 * d)
                                    if abs(sin_theta) <= 1:
                                        two_theta = np.degrees(2 * np.arcsin(sin_theta))
                                        hkl = f"({refl[0]:.0f}{refl[1]:.0f}{refl[2]:.0f})"

                                        # 在计算曲线上找到对应的强度
                                        idx = np.argmin(np.abs(x_data - two_theta))
                                        if idx < len(y_calc):
                                            intensity = y_calc[idx]

                                            # 绘制垂直线
                                            ax_main.vlines(two_theta, y_bkg[idx], intensity,
                                                           colors=colors[i], linewidth=1, alpha=0.6)

                                            # 添加晶面指标（选择性标注，避免重叠）
                                            if intensity > np.max(y_obs) * 0.1:  # 只标注强峰
                                                ax_main.text(two_theta, intensity + np.max(y_obs) * 0.02,
                                                             hkl, fontsize=8, color=colors[i],
                                                             ha='center', va='bottom', rotation=90)

                except Exception as e:
                    self.log(f"⚠️  Failed to plot peaks for {phase_name}: {e}")

            # 3. 残差图
            residuals = y_obs - y_calc
            ax_residual.plot(x_data, residuals, 'k-', linewidth=1, alpha=0.8)
            ax_residual.axhline(y=0, color='r', linestyle='-', linewidth=1, alpha=0.5)

            # 设置坐标轴标签
            ax_residual.set_xlabel('2θ (degrees)', fontsize=14, fontweight='bold')
            ax_main.set_ylabel('Intensity (a.u.)', fontsize=14, fontweight='bold')
            ax_residual.set_ylabel('Difference', fontsize=12, fontweight='bold')

            # 设置标题和图例
            rwp = hist.get_wR() or 0
            ax_main.set_title(f'Rietveld Refinement Pattern (Rwp = {rwp:.2f}%)',
                              fontsize=16, fontweight='bold', pad=20)

            ax_main.legend(loc='upper right', frameon=True, fancybox=True,
                           shadow=True, fontsize=10)

            # 设置网格
            ax_main.grid(True, alpha=0.3, linestyle='--')
            ax_residual.grid(True, alpha=0.3, linestyle='--')

            # 设置x轴范围（使用精修范围）
            limits = hist.data['Limits'][1]
            if limits and len(limits) >= 2:
                x_min, x_max = limits[0], limits[1]
                ax_main.set_xlim(x_min, x_max)

            # 自动调整y轴范围
            y_max_obs = np.max(y_obs)
            ax_main.set_ylim(-0.05 * y_max_obs, 1.2 * y_max_obs)

            # 残差图y轴范围
            residual_max = np.max(np.abs(residuals))
            ax_residual.set_ylim(-1.5 * residual_max, 1.5 * residual_max)

            # 调整布局
            plt.tight_layout()

            # 保存图像
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

            # 【修复建议】: 显式清理，防止内存泄漏或线程冲突
            plt.close(fig)
            plt.clf()

            self.log(f"✅ Publication-quality plot saved: {output_path}")
            return True

        except Exception as e:
            self.log(f"❌ Failed to create publication plot: {e}")
            # 确保出错也关闭
            try:
                plt.close('all')
            except:
                pass
            return False
    # 【新增】停止方法
    def stop(self):
        """外部调用此方法来请求中断"""
        self._is_running = False

    # 【新增】检查辅助函数
    def check_stop(self):
        """在每一步检查是否需要停止"""
        if not self._is_running:
            self.log("⛔ Refinement stopped by user.")
            return True
        return False
    def log(self, message):
        """Send log message"""
        self.log_signal.emit(message)

    def create_instrument_file(self, output_dir):
        """Use instrument parameter file from root directory"""

        # Fixed path (according to your project structure)
        inst_file = r"INST_XRY.PRM"

        # Check if file exists
        if not os.path.exists(inst_file):
            error_msg = (
                f"Instrument parameter file not found!\n\n"
                f"Path: {inst_file}\n\n"
                f"Please run standalone script Rievield.py first to generate this file"
            )
            self.log(f"❌ {error_msg}")
            self.error_signal.emit(error_msg)
            return None

        self.log(f"✅ Using instrument parameters: INST_XRY.PRM")
        self.log(f"   Path: {inst_file}")

        return inst_file

    def convert_xye_to_gsas(self, input_file, output_file):
        """XYE → GSAS format conversion"""
        self.log(f"🔄 Converting data format: {os.path.basename(input_file)}")

        if not os.path.exists(input_file):
            self.log(f"❌ Input file not found: {input_file}")
            return False, None

        data = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(('#', '!', 'Angle')):
                        continue

                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            angle = float(parts[0])
                            intensity = float(parts[1])
                            esd = np.sqrt(max(intensity, 1.0))
                            data.append([angle, intensity, esd])
                        except ValueError:
                            continue
        except Exception as e:
            self.log(f"❌ Failed to read file: {e}")
            return False, None

        if len(data) == 0:
            self.log("❌ No valid data found")
            return False, None

        data = np.array(data)
        data_min = data[0, 0]
        data_max = data[-1, 0]
        step = data[1, 0] - data[0, 0] if len(data) > 1 else 0.02

        self.log(f"   ✅ Successfully read {len(data)} data points")
        self.log(f"   2θ range: {data_min:.3f}° - {data_max:.3f}°")

        # Write GSAS format
        try:
            with open(output_file, 'w') as f:
                f.write("COMM XRD data converted from XYE format\n")
                f.write(f"COMM Angle range: {data_min:.3f} to {data_max:.3f} degrees\n")

                # BANK line (critical: format must be correct)
                f.write(f"BANK 1 {len(data)} {len(data)} CONST {data_min * 100.0:.2f} {step:.6f} 0.000000 0 FXYE\n")

                # Data lines
                for row in data:
                    f.write(f"  {row[0] * 100.0:12.2f}{row[1]:16.4f}{row[2]:16.4f}\n")

            self.log(f"   ✅ Conversion successful")
            return True, (data_min, data_max)

        except Exception as e:
            self.log(f"   ❌ Write failed: {e}")
            return False, None

    def run(self):
        """Main execution function"""
        try:
            self.log("=" * 70)
            self.log("🔬 Starting GSAS-II Rietveld Refinement")
            self.log("=" * 70)

            if self.check_stop(): return  # 🛑 检查点

            # Step 1: Configure GSAS-II environment
            self.progress_signal.emit(10)
            self.log("🔧 Setting up GSAS-II environment...")

            success, G2sc, config_info = auto_configure_gsas(self.gsas_path)
            if not success:
                self.error_signal.emit("GSAS-II configuration failed")
                return
            if self.check_stop(): return  # 🛑 检查点

            self.log("✅ GSAS-II loaded successfully")

            # Step 2: Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Step 3: Create instrument parameter file
            self.progress_signal.emit(20)
            inst_file = self.create_instrument_file(self.output_dir)
            if not inst_file:
                self.error_signal.emit("Instrument parameter file creation failed")
                return

            # Step 4: Check input files
            self.log("\n📋 Checking input files...")
            self.log(f"✅ XRD file: {os.path.basename(self.xrd_file)}")
            self.log(f"✅ CIF directory: {self.cif_dir}")

            # Step 5: Convert XRD data format
            self.progress_signal.emit(30)
            self.log("\n🔄 Converting XRD data format...")

            xrd_file_gsas = os.path.join(self.output_dir, "temp_data.gsas")
            success, data_range = self.convert_xye_to_gsas(self.xrd_file, xrd_file_gsas)

            if not success:
                self.error_signal.emit("Data format conversion failed")
                return
            if self.check_stop(): return  # 🛑 检查点
            # Step 6: Start refinement - COMPLETELY REWRITTEN
            self.progress_signal.emit(40)
            self.log("\n🔬 Starting Rietveld refinement")

            project_file = os.path.join(self.output_dir, "refinement.gpx")
            self.log(f"📦 Creating project: {os.path.basename(project_file)}")

            gpx = G2sc.G2Project(newgpx=project_file)

            # Step 7: Add diffraction data & Set Limits
            self.progress_signal.emit(50)
            self.log("📊 Adding powder diffraction data...")

            try:
                hist = gpx.add_powder_histogram(xrd_file_gsas, inst_file, fmthint='GSAS powder')
                self.log(f"   ✅ {hist.name}")

                # 设置精修范围
                x_data = hist.getdata('X')
                y_obs = hist.getdata('Yobs')
                data_min = float(np.min(x_data))
                data_max = float(np.max(x_data))

                safe_min = max(7.0, data_min + 0.5)
                safe_max = min(65.0, data_max - 0.5)

                if safe_min >= safe_max:
                    safe_min = data_min + 0.5
                    safe_max = data_max - 0.5

                self.log(f"   Data range: {data_min:.2f}° - {data_max:.2f}°")
                self.log(f"   🎯 Refinement limits set to: {safe_min:.2f}° - {safe_max:.2f}°")

                hist.set_refinements({'Limits': [safe_min, safe_max]})


            except Exception as e:
                self.error_signal.emit(f"Failed to add diffraction data: {e}")
                return



            # ============================================================
            # Step 8: 加载物相 & 数据清洗 (Robust Loading)
            # ============================================================
            self.progress_signal.emit(60)
            self.log("\n🔬 Adding phases & Sanitizing Input Data...")

            added_phases = []

            for phase_name in self.phase_names:
                cif_file = os.path.join(self.cif_dir, phase_name + ".cif")
                if not os.path.exists(cif_file): continue
                try:
                    phase = gpx.add_phase(cif_file, phasename=phase_name, histograms=[hist], fmthint='CIF')

                    # --- 关键修复：处理各向异性原子 ---
                    for atom in phase.atoms():
                        try:
                            # 检查原子热参数类型 ('I'=Isotropic, 'A'=Anisotropic)
                            # 这里的 API 调用取决于 GSAS 版本，我们用 try-except 确保稳健
                            if hasattr(atom, 'adp_flag'):
                                if atom.adp_flag == 'A':
                                    atom.adp_flag = 'I'  # 强制转为各向同性
                                    atom.uiso = 0.025  # 赋予默认值

                            # 再次检查数值范围，防止过大过小
                            if atom.uiso < 0.001 or atom.uiso > 0.1:
                                atom.uiso = 0.025
                        except:
                            # 如果上述操作失败，强制赋值以防万一
                            try:
                                atom.uiso = 0.025
                            except:
                                pass

                    # 初始设置：只开 Scale
                    phase.set_refinements({'Cell': False})
                    phase.set_HAP_refinements({'Scale': True, 'Mustrain': {'refine': False}, 'Size': {'refine': False}})

                    added_phases.append(phase)
                    self.log(f"   ✅ Loaded: {phase_name}")
                except Exception as e:
                    self.log(f"   ❌ Error loading {phase_name}: {e}")

            if not added_phases:
                self.error_signal.emit("No phases loaded")
                return
            if self.check_stop(): return  # 🛑 检查点

            # ============================================================
            # 🚀 GENERAL SCIENTIFIC REFINEMENT (Updated)
            # ============================================================
            self.progress_signal.emit(70)
            self.log("\n🚀 Starting Refinement Strategy")

            # 0. Init Controls
            try:
                gpx.set_Controls('cycles', 8)
                gpx.data['Controls']['min dM/M'] = 0.0
            except:
                pass

            hist.data['Sample Parameters']['Shift'] = [0.0, False]
            hist.data['Sample Parameters']['Zero'] = [0.0, False]

            # Init Scale
            try:
                total_int = np.sum(hist.getdata('Yobs'))
                init_s = total_int / (5000.0 * max(1, len(added_phases)))
                for p in added_phases: p.HAPvalue('Scale', init_s, targethistlist=[hist])
            except:
                pass

            # ------------------------------------------------------------
            # 1️⃣ Stage 1: Background & Zero
            # ------------------------------------------------------------
            if self.check_stop(): return  # 🛑 检查点
            self.log("\n   1️⃣ Stage 1: Background & Zero")
            try:
                gpx.do_refinements([{'set': {'Background': {'no. coeffs': 12, 'refine': True},
                                             'Sample Parameters': ['Zero', 'Scale']}}])
                self.log(f"      Rwp = {hist.get_wR():.3f}%")
            except:
                pass

            # ------------------------------------------------------------
            # 2️⃣ Stage 2: Unit Cell (按含量排序)
            # ------------------------------------------------------------
            if self.check_stop(): return  # 🛑 检查点
            self.log("\n   2️⃣ Stage 2: Unit Cell Refinement")

            # 计算含量并排序
            p_scales = [(p.HAPvalue('Scale', targethistlist=[hist]), p) for p in added_phases]
            p_scales.sort(key=lambda x: x[0], reverse=True)
            tot_s = sum([x[0] for x in p_scales]) if p_scales else 1.0

            for s, p in p_scales:
                if (s / tot_s) > 0.01:  # >1% 开启晶胞
                    p.set_refinements({'Cell': True})

            gpx.do_refinements([{}])
            self.log(f"      Rwp = {hist.get_wR():.3f}%")

            # ------------------------------------------------------------
            # 3️⃣ Stage 3: General Texture Sweep (通用织构扫描)
            # ------------------------------------------------------------
            if self.check_stop(): return  # 🛑 检查点
            self.log("\n   3️⃣ Stage 3: Texture & Morphology (The 'Multiple MD' effect)")


            # 重新计算含量
            scales = [p.HAPvalue('Scale', targethistlist=[hist]) for p in added_phases]
            tot_s = sum(scales)
            p_indices = sorted(range(len(added_phases)), key=lambda k: scales[k], reverse=True)

            for i in p_indices:
                p = added_phases[i]
                frac = scales[i] / tot_s

                # 只有主相 (>10%) 值得做这种高级修正
                if frac > 0.10:
                    self.log(f"      Analyzing: {p.name} ({frac * 100:.1f}%)")

                    hap_data = p.data['Histograms'][hist.name]
                    state_best = copy.deepcopy(hap_data['Pref.Ori.'])
                    rwp_best = hist.get_wR()

                    # --- Step A: March-Dollase [001] (平板模型) ---
                    hap_data['Pref.Ori.'] = ['MD', 0.9, True, [0, 0, 1]]
                    try:
                        gpx.do_refinements([{}])
                        rwp_md = hist.get_wR()
                        if rwp_md < rwp_best - 0.2:  # 有显著提升
                            self.log(f"         ✅ MD [001] Accepted (Rwp: {rwp_md:.3f}%)")
                            rwp_best = rwp_md
                            state_best = copy.deepcopy(hap_data['Pref.Ori.'])
                        else:
                            hap_data['Pref.Ori.'] = state_best
                            gpx.do_refinements([{}])
                    except:
                        pass

                    # --- Step B: 各向异性晶粒尺寸 (Anisotropic Size) ---
                    # 物理意义：晶体是薄片，厚度方向(001)的衍射峰比侧面(100)更宽、更矮。
                    # 这能修正 18度(001) 和 34度(101) 的相对高度差！
                    self.log(f"         -> Testing Anisotropic Size (Plate-like)...")

                    # 保存微观结构设置
                    size_state = copy.deepcopy(hap_data['Size'])

                    # 设置为单轴各向异性 (Uniaxial), 轴 [0 0 1]
                    p.set_HAP_refinements({
                        'Size': {'type': 'uniaxial', 'refine': True, 'direction': [0, 0, 1]}
                    })

                    try:
                        gpx.do_refinements([{}])
                        rwp_size = hist.get_wR()

                        if rwp_size < rwp_best - 0.1:
                            self.log(f"         ✅ Anisotropic Size Accepted (Rwp: {rwp_size:.3f}%)")
                            rwp_best = rwp_size
                            # 此时不再回滚，保留这个设置
                        else:
                            self.log(f"         -> Size effect small, reverting.")
                            hap_data['Size'] = size_state
                            gpx.do_refinements([{}])
                    except:
                        hap_data['Size'] = size_state

                    # --- Step C: 球谐函数 (Spherical Harmonics) ---
                    # 这就是您想要的“多个 MD”效果。它允许晶体向任意方向取向。
                    # 只有当 MD 已经起效时，我们才尝试升级到 SH，作为终极手段。
                    if hap_data['Pref.Ori.'][0] == 'MD':
                        self.log(f"         -> Upgrading to Spherical Harmonics (Order 4)...")
                        hap_data['Pref.Ori.'] = ['SH', 4, True, []]

                        try:
                            gpx.do_refinements([{}])
                            rwp_sh = hist.get_wR()

                            if rwp_sh < rwp_best - 0.15:
                                self.log(f"         🏆 SH (Multi-Texture) Winner! (Rwp: {rwp_sh:.3f}%)")
                                # 成功！
                            else:
                                self.log(f"         -> SH didn't help enough, sticking to MD.")
                                hap_data['Pref.Ori.'] = state_best  # 回滚到 MD
                                gpx.do_refinements([{}])
                        except:
                            hap_data['Pref.Ori.'] = state_best
                            gpx.do_refinements([{}])

            # ------------------------------------------------------------
            # 4️⃣ Stage 4: Instrument & Profile
            # ------------------------------------------------------------
            if self.check_stop(): return  # 🛑 检查点
            self.log("\n   4️⃣ Stage 4: Instrument & Profile")

            inst_dict = {'set': {'Instrument Parameters': ['U', 'W', 'X', 'SH/L']}}

            for s, p in p_scales:
                # 只对主相 (>5%) 修峰宽，防止发散
                if (s / tot_s) > 0.05:
                    p.set_HAP_refinements({
                        'Mustrain': {'type': 'isotropic', 'refine': True},
                        'Size': {'type': 'isotropic', 'refine': True}
                    })



            gpx.do_refinements([inst_dict])
            self.log(f"      Final Rwp = {hist.get_wR():.3f}%")
            final_rwp = hist.get_wR()

            gpx.save(project_file)

            # 从这里开始继续您原来的代码结构...




            # ============================================================
            # Step 13 & 14: 终极数据提取 (积分归属法 + 暴力参数提取)
            # ============================================================
            self.progress_signal.emit(95)
            self.log("\n📊 Extracting final results (Saving ALL to JSON)...")

            # 1. 重新加载 GPX
            try:
                gpx_final = G2sc.G2Project(project_file)
                hist_final = gpx_final.histograms()[0]
            except:
                hist_final = hist

            def get_val(obj, default=0.0):
                try:
                    if obj is None: return default
                    if isinstance(obj, (list, tuple, np.ndarray)):
                        if len(obj) > 0: return float(obj[0])
                        return default
                    return float(obj)
                except:
                    return default

            # 2. 提取统计
            rwp = get_val(hist_final.residuals.get('wR', 0.0))
            gof = get_val(hist_final.residuals.get('GOF', 0.0))
            chi2 = get_val(hist_final.residuals.get('Chi2', hist_final.residuals.get('chisq', 0.0)))

            # 兜底统计
            x = hist_final.getdata('X')
            y_o = hist_final.getdata('Yobs')
            y_c = hist_final.getdata('Ycalc')
            y_b = hist_final.getdata('Background')  # 确保获取背景

            if gof == 0.0:
                try:
                    w = hist_final.getdata('Yweight')
                    if w is None: w = np.where(y_o > 0, 1.0 / y_o, 1.0)
                    limits = hist_final.data['Limits'][1]
                    mask = (x >= limits[0]) & (x <= limits[1])
                    chi2 = np.sum(w[mask] * (y_o[mask] - y_c[mask]) ** 2)
                    N = np.sum(mask)
                    P = 20 + len(gpx.phases()) * 7
                    gof = np.sqrt(chi2 / max(1, N - P))
                except:
                    pass

            # 3. 提取参数
            inst = hist_final.InstrumentParameters
            samp = hist_final.SampleParameters
            U = get_val(inst.get('U', 0.0))
            V = get_val(inst.get('V', 0.0))
            W = get_val(inst.get('W', 0.0))
            inst_params = {
                'U': get_val(inst.get('U')), 'V': get_val(inst.get('V')), 'W': get_val(inst.get('W')),
                'X': get_val(inst.get('X')), 'Y': get_val(inst.get('Y')), 'Zero': get_val(samp.get('Zero'))
            }

            # 4. 【DEBUG】背景参数提取
            bg_params = {'type': 'unknown', 'coeffs': []}
            try:
                bg_raw = hist_final.data.get('Background', [])

                if bg_raw and len(bg_raw) > 0:
                    # 1. 获取第一个列表
                    bg_main_list = bg_raw[0]

                    if isinstance(bg_main_list, list):
                        # 类型在第 0 位
                        bg_params['type'] = str(bg_main_list[0])

                        # 系数从第 3 位开始 (跳过 type, flag, count)
                        # 例如: ['cheb', 'True', '12', 7.8, -6.2, ...]
                        if len(bg_main_list) > 3:
                            # 直接切片提取，并确保转为 float
                            bg_params['coeffs'] = [get_val(x) for x in bg_main_list[3:]]

                    self.log(f"   ✅ Bkg Extracted: {bg_params['type']} ({len(bg_params['coeffs'])} terms)")
            except Exception as e:
                self.log(f"   ⚠️ Bkg extraction error: {e}")

            # 4. 提取相信息
            phases_data = []
            refl_dict = hist_final.reflections()
            scales_list = []
            try:
                for p in gpx.phases(): scales_list.append(p.HAPvalue('Scale', targethistlist=[hist_final]))
            except:
                scales_list = []
            total_scale = sum(scales_list) if sum(scales_list) > 0 else 1.0

            for i, p in enumerate(gpx.phases()):
                try:
                    s = scales_list[i] if i < len(scales_list) else 0
                    c = p.data['General']['Cell']
                    cell = {'a': get_val(c[1]), 'b': get_val(c[2]), 'c': get_val(c[3]),
                            'alpha': get_val(c[4]), 'beta': get_val(c[5]), 'gamma': get_val(c[6]),
                            'volume': get_val(c[7])}

                    # === 计算该相的物理峰高 ===
                    top_peaks = []
                    if p.name in refl_dict:
                        refs = refl_dict[p.name].get('RefList', [])

                        candidates = []
                        for row in refs:
                            try:
                                pos = float(row[5])  # 2Theta
                                if 7.0 <= pos <= 65.0:
                                    # 1. 计算纯积分强度 (Area) = F^2 * Icorr
                                    # Col 9: Fcalc^2, Col 11: Icorr
                                    # 如果没有 Col 11，说明可能是旧格式，回退到 Col 7 (Icalc)
                                    if len(row) > 11:
                                        f_sq = float(row[9])
                                        i_corr = float(row[11])
                                        area = f_sq * i_corr
                                    elif len(row) > 7:
                                        area = float(row[7])
                                    else:
                                        area = float(row[6])  # Iobs

                                    # 2. 计算峰宽 (FWHM) -> Cagliotti
                                    # FWHM^2 = U*tan^2 + V*tan + W
                                    rad = np.radians(pos / 2.0)
                                    tan_theta = np.tan(rad)
                                    sig_sq = U * (tan_theta ** 2) + V * tan_theta + W

                                    # 保护：防止负数开根号
                                    if sig_sq < 1e-6: sig_sq = 1e-6
                                    width = np.sqrt(sig_sq)

                                    # 3. 物理峰高 = 面积 / 宽度
                                    # 乘以 scale (s) 是为了在多相之间比较时公平，但在单相内部排序不乘也没事
                                    # 这里我们乘上，方便后续可能的扩展
                                    phys_height = (area / width) * s

                                    h, k, l = int(row[0]), int(row[1]), int(row[2])

                                    candidates.append({
                                        'pos': pos,
                                        'hkl': f"({h}{k}{l})",
                                        'height': phys_height  # 依据这个排序
                                    })
                            except:
                                pass

                        # 按【物理峰高】降序排序
                        candidates.sort(key=lambda x: x['height'], reverse=True)

                        # 取前 4 个
                        top_peaks = candidates[:6]

                    phases_data.append({
                        'name': p.name, 'scale': s, 'percentage': s / total_scale * 100, 'cell': cell,
                        'top_peaks': top_peaks  # 这里存的是该相自己的“山头”
                    })
                except Exception as e:
                    self.log(f"   ⚠️ Phase Error: {e}")

            # 5. 构建最终字典 (包含 plot_data)
            detailed_data = {
                'phases': phases_data,
                'data_limits': [safe_min, safe_max],
                'statistics': {
                    'Rwp': rwp, 'chi2': chi2, 'GOF': gof,
                    'inst': inst_params, 'bkg': bg_params
                },
                'plot_data': {
                    'x': x, 'y_obs': y_o, 'y_calc': y_c, 'y_bkg': y_b
                }
            }

            # --- 6. 写入 JSON ---
            json_path = os.path.join(self.output_dir, "results.json")

            def np_encoder(object):
                if isinstance(object, np.generic):
                    return object.item()
                elif isinstance(object, np.ndarray):
                    return object.tolist()
                raise TypeError

            with open(json_path, 'w') as f:
                json.dump(detailed_data, f, default=np_encoder, indent=4)

            self.log(f"   💾 Results saved to: {os.path.basename(json_path)}")

            # 7. 发送信号
            self.detailed_data_signal.emit(detailed_data)
            self.plot_data_signal.emit(x, y_o, y_c, y_b, rwp)  # 兼容旧逻辑

            self.log(f"\n✨ Final Rwp: {rwp:.3f}%, GOF: {gof:.3f}")
            self.finished_signal.emit({'project_file': project_file, 'Rwp': rwp, 'chi2': chi2})


        except Exception as e:
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.log(f"❌ Refinement process failed: {error_msg}")
            self.error_signal.emit(error_msg)







# Main Window
# =========================================================
class XRDApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepXRD Pro - Full Analysis Suite")
        self.resize(1400, 900)

        # =========================================================================
        # 1. 获取动态基准路径 (Base Directory)
        # =========================================================================
        # 无论用户把文件夹放在哪，这一行都能找到 Software_Rievied.py 所在的文件夹
        if getattr(sys, 'frozen', False):
            # 如果是打包成的 exe
            self.base_dir = os.path.dirname(sys.executable)
        else:
            # 如果是脚本运行 (绿色版环境)
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        print(f"📂 Current Software Location: {self.base_dir}")

        # =========================================================================
        # 2. 定义相对路径 (Relative Paths)
        # =========================================================================
        # 这里假设你的文件都放在软件根目录，或者特定的子文件夹里

        # A. 模型文件 (尝试在根目录找，也在子文件夹找)
        model_filename = 'best_xrd_cnn_cwt_msff_fpn_ca_model_20251116.pth'

        # 定义搜索顺序：先找根目录，再找 models 文件夹
        possible_model_paths = [
            os.path.join(self.base_dir, model_filename),  # 根目录
            os.path.join(self.base_dir, 'Model', model_filename),  # models 子目录
        ]

        # 智能查找：使用第一个存在的路径
        self.model_path = next((p for p in possible_model_paths if os.path.exists(p)), possible_model_paths[0])

        # B. 其他资源文件 (使用 os.path.join 拼接，这就是相对路径的写法)
        self.phase_names_path = os.path.join(self.base_dir, 'Phase_names.csv')
        self.ref_dir = os.path.join(self.base_dir, 'CementXRD')
        self.cif_dir = os.path.join(self.base_dir, 'CIF')

        # =========================================================================
        # 3. 调试信息与自检
        # =========================================================================
        print(f"🔍 Checking Model: {self.model_path} -> {'✅ Found' if os.path.exists(self.model_path) else '❌ Missing'}")
        print(
            f"🔍 Checking PhaseCSV: {self.phase_names_path} -> {'✅ Found' if os.path.exists(self.phase_names_path) else '❌ Missing'}")
        print(f"🔍 Checking CIF Dir: {self.cif_dir} -> {'✅ Found' if os.path.exists(self.cif_dir) else '❌ Missing'}")

        # 如果关键文件缺失，弹窗警告 (防止发给别人时文件丢了)
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "Missing File",
                                f"Model file not found!\nPlease ensure '{model_filename}' is in the software folder.")

        # ✅ GSAS-II 路径（支持用户自定义）
        self.gsas_path = None
        self.auto_detect_gsas_path()

        self.data_pool = {}
        self.analysis_results = {}
        self.tab2_input_data = []
        self.raw_data_for_refinement = {}

        self.init_ui()

    def check_required_files(self):
        """检查必要的资源文件是否存在"""
        required_files = [
            self.model_path,
            self.phase_names_path
        ]

        required_dirs = [
            self.ref_dir,
            self.cif_dir
        ]

        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_files.append(dir_path + " (directory)")

        if missing_files:
            error_msg = "Missing required files/directories:\n" + "\n".join(missing_files)
            print(f"❌ {error_msg}")
            # 在打包版本中显示错误对话框
            if getattr(sys, 'frozen', False):
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(None, "Missing Files", error_msg)

    def auto_detect_gsas_path(self):
        """自动检测 GSAS-II 安装路径 (优先检测软件内部集成的 GSAS-II)"""

        print(f"🔍 Looking for GSAS-II relative to: {self.base_dir}")

        # =========================================================
        # 1. 定义搜索列表 (优先级从高到低)
        # =========================================================
        possible_paths = [
            # 优先级 1: 软件根目录下的 GSAS-II (相对路径)
            # 这样你把 GSAS-II 文件夹复制进来，发给别人就能直接用
            os.path.join(self.base_dir, "GSAS-II"),
            os.path.join(self.base_dir, "GSASII"),

            # 优先级 2: 上一级目录 (开发时有时候会把库放在外面)
            os.path.join(os.path.dirname(self.base_dir), "GSAS-II"),

            # 优先级 3: 常见的系统绝对路径 (兜底方案)
            r"C:\GSAS-II",
            r"D:\GSAS-II",
            r"C:\Program Files\GSAS-II",
            os.path.join(os.path.expanduser("~"), "GSAS-II"),
        ]

        # =========================================================
        # 2. 开始遍历检测
        # =========================================================
        for path in possible_paths:
            # 只有路径存在才尝试加载
            if os.path.exists(path):
                print(f"   Checking: {path} ... ", end="")
                try:
                    # 尝试配置
                    success, G2sc, config_info = auto_configure_gsas(path)

                    if success:
                        print("✅ Success!")
                        self.gsas_path = path

                        # 这是一个好习惯：提示用户用的是内部版还是系统版
                        if self.base_dir in os.path.abspath(path):
                            print("   (Using Embedded/Portable GSAS-II)")
                        else:
                            print("   (Using System GSAS-II)")

                        return
                    else:
                        print("❌ Found but configuration failed.")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
            else:
                # 路径不存在，跳过
                pass

        # 如果循环结束还没return，说明没找到
        print("⚠️ GSAS-II not auto-detected. Please set path manually.")
        self.gsas_path = None

    def diagnose_gsas_path(self):
        """诊断 GSAS-II 路径问题"""
        if not self.gsas_path or not os.path.exists(self.gsas_path):
            return "❌ Path does not exist"

        report = []
        report.append("=" * 70)
        report.append("🔍 GSAS-II Path Diagnostic Report")
        report.append("=" * 70)
        report.append(f"\n📁 Root Path: {self.gsas_path}\n")

        # 检查目录结构
        report.append("📂 Directory Structure:")
        try:
            subdirs = [d for d in os.listdir(self.gsas_path) if os.path.isdir(os.path.join(self.gsas_path, d))]
            for subdir in subdirs[:10]:
                report.append(f"   • {subdir}/")
            if len(subdirs) > 10:
                report.append(f"   ... and {len(subdirs) - 10} more directories")
        except Exception as e:
            report.append(f"   ❌ Error reading directory: {e}")


        report.append("\n🔍 Searching for key files:")

        # GSASIIscriptable.py
        scriptable = glob.glob(os.path.join(self.gsas_path, "**", "GSASIIscriptable.py"), recursive=True)
        if scriptable:
            report.append(f"   ✅ GSASIIscriptable.py found at:")
            for f in scriptable:
                rel_path = os.path.relpath(f, self.gsas_path)
                report.append(f"      • {rel_path}")
        else:
            report.append("   ❌ GSASIIscriptable.py NOT FOUND")

        # GSASIIpath.py
        pathfile = glob.glob(os.path.join(self.gsas_path, "**", "GSASIIpath.py"), recursive=True)
        if pathfile:
            report.append(f"   ✅ GSASIIpath.py found at:")
            for f in pathfile:
                rel_path = os.path.relpath(f, self.gsas_path)
                report.append(f"      • {rel_path}")
        else:
            report.append("   ⚠️  GSASIIpath.py not found")

        # Binary files
        binaries = glob.glob(os.path.join(self.gsas_path, "**", "*.pyd"), recursive=True)
        binaries += glob.glob(os.path.join(self.gsas_path, "**", "*.so"), recursive=True)

        if binaries:
            report.append(f"\n   ✅ Found {len(binaries)} binary files:")
            # 按目录分组
            bin_dirs = {}
            for b in binaries:
                dir_name = os.path.dirname(b)
                if dir_name not in bin_dirs:
                    bin_dirs[dir_name] = []
                bin_dirs[dir_name].append(os.path.basename(b))

            for dir_name, files in bin_dirs.items():
                rel_dir = os.path.relpath(dir_name, self.gsas_path)
                report.append(f"      📁 {rel_dir}/")
                for f in files[:3]:
                    report.append(f"         • {f}")
                if len(files) > 3:
                    report.append(f"         ... and {len(files) - 3} more")
        else:
            report.append("   ⚠️  No binary files (.pyd or .so) found")

        report.append("\n" + "=" * 70)

        return "\n".join(report)
    def init_ui(self):
        self.main_tabs = QTabWidget()
        self.setCentralWidget(self.main_tabs)

        self.tab_process = QWidget()
        self.setup_process_ui()
        self.main_tabs.addTab(self.tab_process, "1. Data Preprocessing")

        self.tab_analysis = QWidget()
        self.setup_analysis_ui()
        self.main_tabs.addTab(self.tab_analysis, "2. AI Analysis")

        self.tab_refinement = QWidget()
        self.setup_refinement_ui()
        self.main_tabs.addTab(self.tab_refinement, "3. GSAS-II Refinement")

    def setup_process_ui(self):
        layout = QHBoxLayout(self.tab_process)
        left_panel = QWidget()
        left_panel.setFixedWidth(360)
        left_layout = QVBoxLayout(left_panel)

        gb_file = QGroupBox("1. Import")
        v_file = QVBoxLayout()
        self.lst_raw = QListWidget()
        self.lst_raw.itemClicked.connect(self.update_process_plot)
        btn_add = QPushButton("Add Files (Custom Import)")
        btn_add.clicked.connect(self.add_raw_files_custom)
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setStyleSheet("background-color: #ffcdd2; color: #b71c1c;") # Light red style
        self.btn_remove.clicked.connect(self.remove_raw_file)
        btn_reset = QPushButton("Reset Selected")
        btn_reset.clicked.connect(self.reset_current_file)
        v_file.addWidget(btn_add)
        v_file.addWidget(self.btn_remove)
        v_file.addWidget(self.lst_raw)
        v_file.addWidget(btn_reset)
        gb_file.setLayout(v_file)
        left_layout.addWidget(gb_file)

        gb_steps = QGroupBox("2. Process")
        v_steps = QVBoxLayout()
        self.btn_step1 = QPushButton("Step 1: Resample (5-70°)")
        self.btn_step1.clicked.connect(self.do_step1_length)
        self.btn_step2 = QPushButton("Step 2: Subtract Background")
        self.btn_step2.clicked.connect(self.do_step2_bg)
        self.btn_step3 = QPushButton("Step 3: Normalize")
        self.btn_step3.clicked.connect(self.do_step3_norm)

        h_param = QHBoxLayout()
        h_param.addWidget(QLabel("Lam:"))
        self.spin_lam = QSpinBox()
        self.spin_lam.setRange(1000, 100000)
        self.spin_lam.setValue(10000)
        h_param.addWidget(self.spin_lam)
        h_param.addWidget(QLabel("P:"))
        self.spin_p = QDoubleSpinBox()
        self.spin_p.setRange(0.0001, 0.1)
        self.spin_p.setValue(0.001)
        self.spin_p.setDecimals(4)
        h_param.addWidget(self.spin_p)
        v_steps.addWidget(self.btn_step1)
        v_steps.addLayout(h_param)
        v_steps.addWidget(self.btn_step2)
        v_steps.addWidget(self.btn_step3)
        gb_steps.setLayout(v_steps)
        left_layout.addWidget(gb_steps)

        gb_save = QGroupBox("3. Output")
        v_save = QVBoxLayout()
        self.btn_save = QPushButton("Save Processed")
        self.btn_save.clicked.connect(self.save_processed_data)
        self.btn_send = QPushButton("Send to AI Analysis ->")
        self.btn_send.setStyleSheet("background:#2196F3;color:white;font-weight:bold;")
        self.btn_send.clicked.connect(self.send_to_analysis)
        v_save.addWidget(self.btn_save)
        v_save.addWidget(self.btn_send)
        gb_save.setLayout(v_save)
        left_layout.addWidget(gb_save)

        layout.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.fig_proc = Figure(figsize=(8, 6))
        self.cv_proc = FigureCanvas(self.fig_proc)
        self.tb_proc = NavigationToolbar(self.cv_proc, self)
        self.ax_proc = self.fig_proc.add_subplot(111)
        right_layout.addWidget(self.tb_proc)
        right_layout.addWidget(self.cv_proc)
        layout.addWidget(right_panel)

    def remove_raw_file(self):
        """Removes the currently selected file from the list and memory."""
        row = self.lst_raw.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Selection Error", "Please select a file to remove.")
            return

        # Get the filename
        item = self.lst_raw.item(row)
        fname = item.text()

        # Optional: Confirmation Dialog
        reply = QMessageBox.question(self, "Confirm Remove",
                                     f"Are you sure you want to remove '{fname}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.No:
            return

        # 1. Remove from UI List
        self.lst_raw.takeItem(row)

        # 2. Remove from Data Pool (Used for plotting and AI)
        if fname in self.data_pool:
            del self.data_pool[fname]

        # 3. Remove from Refinement Data (Used for GSAS)
        if hasattr(self, 'raw_data_for_refinement') and fname in self.raw_data_for_refinement:
            del self.raw_data_for_refinement[fname]

        # 4. Clear or Update Plot
        self.ax_proc.clear()
        self.ax_proc.grid(True, alpha=0.3)
        self.cv_proc.draw()

        # If there are other items left, select the adjacent one
        if self.lst_raw.count() > 0:
            new_row = min(row, self.lst_raw.count() - 1)
            self.lst_raw.setCurrentRow(new_row)
            self.update_process_plot()
        else:
            # If list is empty, clear plot completely
            self.ax_proc.set_xlabel('2θ (degrees)')
            self.ax_proc.set_ylabel('Intensity')
            self.cv_proc.draw()
    def add_raw_files_custom(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Select Files", self.base_dir, "Data (*.csv *.txt *.xye *.dat)")
        for f in fs:
            dlg = DataImportDialog(f, self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                fname = os.path.basename(f)
                self.raw_data_for_refinement[fname] = {
                    'filepath': f,
                    'x_raw': dlg.parsed_x.copy(),
                    'y_raw': dlg.parsed_y.copy()
                }
                self.data_pool[fname] = {
                    'x_raw': dlg.parsed_x,
                    'y_raw': dlg.parsed_y,
                    'x': dlg.parsed_x.copy(),
                    'y': dlg.parsed_y.copy(),
                    'bg': None
                }
                self.lst_raw.addItem(fname)
        if self.lst_raw.count() > 0:
            self.lst_raw.setCurrentRow(self.lst_raw.count() - 1)
            self.update_process_plot()

    def reset_current_file(self):
        item = self.lst_raw.currentItem()
        if not item:
            return
        d = self.data_pool[item.text()]
        d['x'] = d['x_raw'].copy()
        d['y'] = d['y_raw'].copy()
        d['bg'] = None
        self.update_process_plot()
    def update_process_plot(self):
        item = self.lst_raw.currentItem()
        if not item:
            self.ax_proc.clear()
            self.cv_proc.draw()
            return
        d = self.data_pool[item.text()]
        self.ax_proc.clear()
        scale = 1.0 / (np.max(d['y_raw']) + 1e-10) if np.max(d['y_raw']) > 100 else 1.0
        self.ax_proc.plot(d['x_raw'], d['y_raw'] * scale, color='lightgray', label='Raw', lw=2)
        self.ax_proc.plot(d['x'], d['y'], color='#2196F3', label='Current', lw=1.5)
        if d['bg'] is not None:
            self.ax_proc.plot(d['x'], d['bg'], color='#FF9800', ls='--', label='Background')
        self.ax_proc.legend()
        self.ax_proc.set_xlabel('2θ (degrees)')
        self.ax_proc.set_ylabel('Intensity')
        self.cv_proc.draw()

    def do_step1_length(self):
        tx = np.linspace(5.0, 70.0, 3251)
        for k, d in self.data_pool.items():
            f = interp1d(d['x'], d['y'], kind='linear', fill_value='extrapolate')
            ny = np.maximum(f(tx), 0)
            orig_min, orig_max = d['x'].min(), d['x'].max()
            ny[tx < orig_min] = 0
            ny[tx > orig_max] = 0
            d['x'] = tx
            d['y'] = ny
            d['bg'] = None
        self.update_process_plot()
        QMessageBox.information(self, "OK", "Resampled to 5-70° (3251 pts)")

    def do_step2_bg(self):
        lam, p = self.spin_lam.value(), self.spin_p.value()
        for k, d in self.data_pool.items():
            bg = baseline_als(d['y'], lam=lam, p=p)
            d['bg'] = bg
            d['y'] = np.maximum(d['y'] - bg, 0)
        self.update_process_plot()
        QMessageBox.information(self, "OK", "Background Subtracted")

    def do_step3_norm(self):
        for k, d in self.data_pool.items():
            m = np.max(d['y'])
            if m > 0:
                d['y'] /= m
            if d['bg'] is not None:
                d['bg'] /= m
        self.update_process_plot()
        QMessageBox.information(self, "OK", "Normalized")

    def save_processed_data(self):
        d = QFileDialog.getExistingDirectory(self, "Select Dir")
        if not d:
            return
        for k, v in self.data_pool.items():
            pd.DataFrame({'2Theta': v['x'], 'Intensity': v['y']}).to_csv(
                os.path.join(d, k + "_processed.csv"), index=False, header=True)
        QMessageBox.information(self, "OK", "Saved")

    def send_to_analysis(self):
        if not self.data_pool:
            return
        self.tab2_input_data = copy.deepcopy(self.data_pool)
        self.list_files_ana.clear()
        for k in self.tab2_input_data:
            self.list_files_ana.addItem(f"[Mem] {k}")
        self.main_tabs.setCurrentIndex(1)

    def setup_analysis_ui(self):
        layout = QHBoxLayout(self.tab_analysis)
        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)

        # --- 1. Config Group Box ---
        gc = QGroupBox("1. Config")
        lc = QVBoxLayout()

        h_dev = QHBoxLayout()
        h_dev.addWidget(QLabel("Device Mode:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["Auto / CUDA (GPU)", "Force CPU"])
        h_dev.addWidget(self.combo_device)
        lc.addLayout(h_dev)

        btn_m = QPushButton("Model")
        btn_m.clicked.connect(self.select_model)
        lc.addWidget(btn_m)
        self.lbl_m = QLabel(os.path.basename(self.model_path))
        self.lbl_m.setStyleSheet("color:gray; font-size:9pt;")
        lc.addWidget(self.lbl_m)

        btn_p = QPushButton("Phase Names")
        btn_p.clicked.connect(self.select_phase)
        lc.addWidget(btn_p)
        self.lbl_p = QLabel(os.path.basename(self.phase_names_path))
        self.lbl_p.setStyleSheet("color:gray; font-size:9pt;")
        lc.addWidget(self.lbl_p)

        btn_r = QPushButton("Ref Dir")
        btn_r.clicked.connect(self.select_ref)
        lc.addWidget(btn_r)
        self.lbl_r = QLabel(self.ref_dir)
        self.lbl_r.setStyleSheet("color:gray; font-size:9pt;")
        self.lbl_r.setWordWrap(True)
        lc.addWidget(self.lbl_r)

        gc.setLayout(lc)
        left_layout.addWidget(gc)

        # --- 2. Queue Group Box ---
        gf = QGroupBox("2. Queue")
        lf = QVBoxLayout()
        self.list_files_ana = QListWidget()
        btn_add = QPushButton("Add Files")
        btn_add.clicked.connect(self.add_ana_files)
        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self.clear_ana_files)
        lf.addWidget(btn_add)
        lf.addWidget(self.list_files_ana)
        lf.addWidget(btn_clr)
        gf.setLayout(lf)
        left_layout.addWidget(gf)

        # --- 3. Control Group Box ---
        gr = QGroupBox("3. Control")
        lr = QVBoxLayout()
        self.btn_start = QPushButton("Start Analysis")
        self.btn_start.setStyleSheet("background:#2196F3;color:white;font-weight:bold;height:40px")
        self.btn_start.clicked.connect(self.start_analysis)
        self.pbar = QProgressBar()
        self.log_w = QListWidget()
        lr.addWidget(self.btn_start)
        lr.addWidget(self.pbar)
        lr.addWidget(self.log_w)
        gr.setLayout(lr)
        left_layout.addWidget(gr)

        # --- 4. NEW: Configuration Group Box (CIF Directory) ---
        # 这个组框将被添加到 Control 组框之后
        gconf = QGroupBox("4. Configuration")  # 重新编号
        lconf = QVBoxLayout()

        h_cif_dir = QHBoxLayout()
        h_cif_dir.addWidget(QLabel("CIF Dir:"))
        # 使用 QLabel 显示当前路径
        self.lbl_analysis_cif_dir = QLabel(self.cif_dir)  # 初始化显示
        self.lbl_analysis_cif_dir.setStyleSheet("color:gray; font-size:9pt;")
        self.lbl_analysis_cif_dir.setWordWrap(True)  # 允许换行显示长路径
        h_cif_dir.addWidget(self.lbl_analysis_cif_dir, 1)  # 1 表示该标签可以拉伸
        # 添加 Browse 按钮
        btn_cif_dir = QPushButton("Browse...")
        btn_cif_dir.clicked.connect(self.select_analysis_cif_dir)  # 连接到新的槽函数
        h_cif_dir.addWidget(btn_cif_dir)

        lconf.addLayout(h_cif_dir)
        gconf.setLayout(lconf)
        left_layout.addWidget(gconf)

        # 重要：将 left_layout 的末端添加一个弹性空间，使组框靠上对齐
        left_layout.addStretch()

        layout.addWidget(left_panel)

        # --- Right Panel (Results and Visualization) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        h_res = QHBoxLayout()
        h_res.addWidget(QLabel("Result:"))
        self.combo_res = QComboBox()
        self.combo_res.currentIndexChanged.connect(self.display_result)
        h_res.addWidget(self.combo_res, 1)
        btn_exp = QPushButton("Export CSV")
        btn_exp.clicked.connect(self.export_csv)
        h_res.addWidget(btn_exp)

        btn_send_refine = QPushButton("Send to Refinement ->")
        btn_send_refine.setStyleSheet("background:#4CAF50;color:white;font-weight:bold;")
        btn_send_refine.clicked.connect(self.send_to_refinement)
        h_res.addWidget(btn_send_refine)

        right_layout.addLayout(h_res)

        self.tabs_viz = QTabWidget()
        self.fig1 = Figure()
        self.cv1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)
        self.tabs_viz.addTab(self.cv1, "Fit Analysis")
        self.fig2 = Figure()
        self.cv2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)
        self.tabs_viz.addTab(self.cv2, "Stacked")
        self.fig3 = Figure()
        self.cv3 = FigureCanvas(self.fig3)
        self.ax3 = self.fig3.add_subplot(111)
        self.tabs_viz.addTab(self.cv3, "Bar Chart")
        self.table_res = QTableWidget()
        self.table_res.setColumnCount(3)
        self.table_res.setHorizontalHeaderLabels(["Phase", "Weight", "Raw"])
        self.tabs_viz.addTab(self.table_res, "Table")



        right_layout.addWidget(self.tabs_viz)
        layout.addWidget(right_panel)

    def log(self, m):
        self.log_w.addItem(m)
        self.log_w.scrollToBottom()

    def select_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Model", "", "*.pth")
        if f:
            self.model_path = f
            self.lbl_m.setText(os.path.basename(f))

    def select_phase(self):
        f, _ = QFileDialog.getOpenFileName(self, "Phase", "", "*.csv")
        if f:
            self.phase_names_path = f
            self.lbl_p.setText(os.path.basename(f))

    def select_ref(self):
        d = QFileDialog.getExistingDirectory(self, "Ref Dir")
        if d:
            self.ref_dir = d
            self.lbl_r.setText(d)

    def add_ana_files(self):
        fs, _ = QFileDialog.getOpenFileNames(self, "Files", "", "Data (*.xye *.txt *.csv)")
        if not fs:
            return
        if isinstance(self.tab2_input_data, dict):
            self.tab2_input_data = []
            self.list_files_ana.clear()
        for f in fs:
            self.tab2_input_data.append(f)
            self.list_files_ana.addItem(os.path.basename(f))

    def clear_ana_files(self):
        self.tab2_input_data = []
        self.list_files_ana.clear()

    def start_analysis(self):
        if not self.tab2_input_data:
            return

        mode_idx = self.combo_device.currentIndex()
        dev_mode = 'cuda' if mode_idx == 0 else 'cpu'

        self.worker = AnalysisWorker(
            self.tab2_input_data,
            self.model_path,
            self.phase_names_path,
            self.ref_dir,
            device_type=dev_mode
        )
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.pbar.setValue)
        self.worker.finished_signal.connect(self.on_ana_finished)
        self.worker.error_signal.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.btn_start.setEnabled(False)
        self.pbar.setValue(0)
        self.worker.start()

    def on_ana_finished(self, res):
        self.analysis_results = res
        self.btn_start.setEnabled(True)
        self.log("✅ Analysis Finished")
        self.combo_res.clear()
        self.combo_res.addItems(list(res.keys()))
        self.display_result()

    # ... 在您的类定义中 ...

    def select_analysis_cif_dir(self):
        """槽函数：用于 AI Analysis 标签页中选择 CIF 文件夹"""
        d = QFileDialog.getExistingDirectory(self, "Select CIF Directory for Analysis")
        if d:
            # 更新类属性 cif_dir
            self.cif_dir = d
            # 更新 QLabel 显示
            self.lbl_analysis_cif_dir.setText(d)
            # 可选：设置工具提示以显示完整路径（当路径过长被截断时有用）
            self.lbl_analysis_cif_dir.setToolTip(d)
            # 可选：在日志中记录
            self.log(f"📁 CIF directory for analysis updated to: {d}")
    def display_result(self):
        k = self.combo_res.currentText()
        if k not in self.analysis_results:
            return
        d = self.analysis_results[k]

        self.ax1.clear()
        self.ax1.plot(d['angles'], d['exp_norm'], 'k-', lw=1.5, label='Experimental')
        self.ax1.plot(d['angles'], d['calc_pattern'], 'r--', lw=1.5, label='Calculated')
        offset = -0.2
        self.ax1.plot(d['angles'], d['diff_pattern'] + offset, 'b-', lw=1, label='Difference')
        self.ax1.axhline(y=offset, color='gray', lw=0.5, alpha=0.5)
        self.ax1.set_xlabel('2θ (degrees)')
        self.ax1.set_ylabel('Intensity (normalized)')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        self.cv1.draw()

        self.ax2.clear()
        self.ax2.plot(d['angles'], d['exp_norm'] + 1.2, 'k-', lw=2, label='Exp')
        colors = cm.tab10(np.linspace(0, 1, len(d['top_phases'])))
        for i, (p, curve) in enumerate(d['phase_curves']):
            off = 1.0 - i * 0.25
            self.ax2.plot(d['angles'], curve + off, color=colors[i], lw=1.2)
            self.ax2.text(d['angles'][0], off + 0.05, f"{p} ({d['top_weights'][i]:.3f})",
                          color=colors[i], fontsize=9)
        self.ax2.set_yticks([])
        self.ax2.set_xlabel('2θ (degrees)')
        self.cv2.draw()


        self.ax3.clear()
        bars = self.ax3.bar(d['top_phases'], d['top_weights'], color=colors)
        self.ax3.tick_params(axis='x', rotation=30)
        self.ax3.set_ylabel('Normalized Weight')
        for bar in bars:
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width() / 2, height,
                          f"{height:.3f}", ha='center', va='bottom', fontsize=9)
        self.cv3.draw()

        self.table_res.setRowCount(len(d['top_phases']))
        for i, (p, w, r) in enumerate(zip(d['top_phases'], d['top_weights'], d['raw_weights'])):
            self.table_res.setItem(i, 0, QTableWidgetItem(str(p)))
            self.table_res.setItem(i, 1, QTableWidgetItem(f"{w:.4f}"))
            self.table_res.setItem(i, 2, QTableWidgetItem(f"{r:.4f}"))

    def export_csv(self):
        if not self.analysis_results:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save", "Results.csv", "CSV (*.csv)")
        if f:
            out = []
            for k, v in self.analysis_results.items():
                for p, w in zip(v['top_phases'], v['top_weights']):
                    out.append({'File': k, 'Phase': p, 'Weight': w})
            pd.DataFrame(out).to_csv(f, index=False)
            QMessageBox.information(self, "OK", "Exported")

    def send_to_refinement(self):
        """将 AI 分析结果发送到精修界面 (支持多晶型展开)"""
        current_file = self.combo_res.currentText()

        if not current_file or current_file not in self.analysis_results:
            QMessageBox.warning(self, "Warning", "Please select a valid analysis result first!")
            return

        result_data = self.analysis_results[current_file]

        # ✅ 获取 AI 识别的相
        phases = result_data['top_phases'][:6]
        weights = result_data['top_weights'][:6]

        if len(phases) == 0:
            QMessageBox.warning(self, "Warning", "No phases identified in this result!")
            return

        # ✅ 填充到精修界面的相列表
        self.list_phases_refine.clear()

        # =================================================================
        # 1. 定义多晶型映射 (Polymorph Mapping)
        #    键(Key): AI 识别出的基础相名称
        #    值(Value): 该相对应的所有 CIF 文件名列表 (按常见程度排序)
        # =================================================================
        polymorph_map = {
            'C3S': [
                'C3Smono_nishi',  # 最常见 (M3)
                'C3Striclinic_belov',  # 三斜 (T1)
                'C3Srhom_nishi'  # 菱方 (R)
            ],
            'C2S': [
                'C2Sbeta_mumme',  # 贝塔 (Beta) - 最常见
                'C2Salpha_mumme',  # 阿尔法 (Alpha)
                'C2Sgam_mumme'  # 伽马 (Gamma)
            ],
            'C3A': [
                'C3Acub',  # 立方
                'C3Amonoclinic'  # 单斜 (Orthorhombic/Monoclinic)
            ]
        }

        # 2. 定义普通映射 (Simple Mapping)
        simple_mapping = {
            'C4AF': 'C4AF_colville',
            'Portlandite': 'Portlandite',
            # 如果有其他相，可以在这里补充，例如 'Gypsum': 'Gypsum_...'
        }

        for i, (phase, weight) in enumerate(zip(phases, weights)):
            # 清理相名称 (去除 AI 可能输出的 " (5 variants)" 等后缀)
            base_phase = phase.split(' (')[0].strip()

            # -------------------------------------------------------
            # 情况 A: 是需要展开的多晶型相 (如 C3S, C2S)
            # -------------------------------------------------------
            if base_phase in polymorph_map:
                variants = polymorph_map[base_phase]

                # 遍历该相的所有变体，全部加到列表中
                for idx, variant_cif in enumerate(variants):
                    # 显示名称：例如 "C3Smono_nishi (C3S)"
                    display_text = f"{variant_cif} ({base_phase})"

                    item = QListWidgetItem(display_text)
                    item.setData(Qt.ItemDataRole.UserRole, variant_cif)  # 存储真实的 CIF 文件名
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

                    # 智能勾选逻辑：
                    # 如果 AI 认为该相存在 (weight > 0.05)，
                    # 我们只默认勾选列表中的第一个变体 (通常是最常见的，如 C3S mono)，
                    # 其他变体加入列表但不勾选，方便用户自己切换。
                    if weight > 0.05 and idx == 0:
                        item.setCheckState(Qt.CheckState.Checked)
                    else:
                        item.setCheckState(Qt.CheckState.Unchecked)

                    self.list_phases_refine.addItem(item)

            # -------------------------------------------------------
            # 情况 B: 是普通单相
            # -------------------------------------------------------
            else:
                # 获取 CIF 文件名，如果没有映射则使用原名
                cif_name = simple_mapping.get(base_phase, base_phase)

                item = QListWidgetItem(cif_name)
                item.setData(Qt.ItemDataRole.UserRole, cif_name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

                if weight > 0.05:
                    item.setCheckState(Qt.CheckState.Checked)
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)

                self.list_phases_refine.addItem(item)

        # ✅ 获取原始 XRD 文件路径
        if current_file in self.raw_data_for_refinement:
            raw_file = self.raw_data_for_refinement[current_file]['filepath']
            self.current_xrd_file = raw_file
            self.lbl_xrd_file.setText(os.path.basename(raw_file))
            self.lbl_xrd_file.setToolTip(raw_file)
        else:
            self.current_xrd_file = None
            self.lbl_xrd_file.setText(f"⚠️ {current_file} (need to re-select)")

        # ✅ 切换到精修标签页
        self.main_tabs.setCurrentIndex(2)

        # ✅ 显示成功消息
        QMessageBox.information(
            self,
            "Success",
            f"✅ Transferred to Refinement Tab!\n\n"
            f"Polymorphs for C3S/C2S/C3A have been expanded.\n"
            f"Default variants are checked based on AI probability.\n"
            f"You can manually check/uncheck specific polymorphs in the list."
        )
    def toggle_phases(self, checked):
        """Select/deselect all phases"""
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.list_phases_refine.count()):
            item = self.list_phases_refine.item(i)
            item.setCheckState(state)

    def setup_refinement_ui(self):
        """Setup refinement interface with comprehensive visualization"""
        layout = QHBoxLayout(self.tab_refinement)

        # ========== 左侧面板 ==========
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # 1. GSAS-II Configuration
        gb_gsas = QGroupBox("1. GSAS-II Configuration")
        v_gsas = QVBoxLayout()

        h_path = QHBoxLayout()
        h_path.addWidget(QLabel("GSAS Path:"))
        self.txt_gsas_path = QLineEdit()
        self.txt_gsas_path.setText(self.gsas_path if self.gsas_path else "")
        self.txt_gsas_path.setReadOnly(True)
        h_path.addWidget(self.txt_gsas_path, 1)
        btn_browse_gsas = QPushButton("Browse")
        btn_browse_gsas.clicked.connect(self.select_gsas_path)
        h_path.addWidget(btn_browse_gsas)
        v_gsas.addLayout(h_path)

        btn_test = QPushButton("Test Connection")
        btn_test.clicked.connect(self.test_gsas_connection)
        v_gsas.addWidget(btn_test)

        self.lbl_gsas_status = QLabel("Status: Not configured")
        v_gsas.addWidget(self.lbl_gsas_status)

        gb_gsas.setLayout(v_gsas)
        left_layout.addWidget(gb_gsas)

        # 2. Files Configuration
        gb_files = QGroupBox("2. Files")
        v_files = QVBoxLayout()

        h_xrd = QHBoxLayout()
        h_xrd.addWidget(QLabel("XRD File:"))
        self.lbl_xrd_file = QLabel("(None)")
        h_xrd.addWidget(self.lbl_xrd_file, 1)
        btn_xrd = QPushButton("Browse")
        btn_xrd.clicked.connect(self.select_xrd_for_refinement)
        h_xrd.addWidget(btn_xrd)
        v_files.addLayout(h_xrd)

        h_cif = QHBoxLayout()
        h_cif.addWidget(QLabel("CIF Dir:"))
        self.lbl_cif_dir = QLabel(self.cif_dir)
        h_cif.addWidget(self.lbl_cif_dir, 1)
        btn_cif = QPushButton("Browse")
        btn_cif.clicked.connect(self.select_cif_dir)
        h_cif.addWidget(btn_cif)
        v_files.addLayout(h_cif)

        gb_files.setLayout(v_files)
        left_layout.addWidget(gb_files)

        # 3. Phase Selection
        gb_phases = QGroupBox("3. Select Phases")
        v_phases = QVBoxLayout()

        h_ctrl = QHBoxLayout()
        btn_select = QPushButton("Select All")
        btn_select.clicked.connect(lambda: self.toggle_phases(True))
        btn_deselect = QPushButton("Deselect All")
        btn_deselect.clicked.connect(lambda: self.toggle_phases(False))
        h_ctrl.addWidget(btn_select)
        h_ctrl.addWidget(btn_deselect)
        h_ctrl.addStretch()
        v_phases.addLayout(h_ctrl)

        self.list_phases_refine = QListWidget()
        v_phases.addWidget(self.list_phases_refine)

        gb_phases.setLayout(v_phases)
        left_layout.addWidget(gb_phases)

        # 4. Control
        gb_control = QGroupBox("4. Control")
        v_control = QVBoxLayout()

        # --- 修改部分：添加水平布局放两个按钮 ---
        h_btns = QHBoxLayout()

        self.btn_start_refine = QPushButton("Start Refinement")
        self.btn_start_refine.clicked.connect(self.start_refinement)
        self.btn_start_refine.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")

        self.btn_stop_refine = QPushButton("Stop")
        self.btn_stop_refine.clicked.connect(self.stop_refinement)
        self.btn_stop_refine.setEnabled(False)  # 初始禁用
        self.btn_stop_refine.setStyleSheet("""
            QPushButton { background-color: #F44336; color: white; font-weight: bold; padding: 5px; }
            QPushButton:disabled { background-color: #e57373; color: #dddddd; }
        """)

        h_btns.addWidget(self.btn_start_refine)
        h_btns.addWidget(self.btn_stop_refine)
        v_control.addLayout(h_btns)
        # -------------------------------------

        self.pbar_refine = QProgressBar()
        v_control.addWidget(self.pbar_refine)

        # Log control buttons
        h_log_buttons = QHBoxLayout()
        btn_copy_log = QPushButton("Copy Log")
        btn_copy_log.clicked.connect(self.copy_refinement_log)
        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.clicked.connect(self.clear_refinement_log)
        h_log_buttons.addWidget(btn_copy_log)
        h_log_buttons.addWidget(btn_clear_log)
        h_log_buttons.addStretch()
        v_control.addLayout(h_log_buttons)

        # Simple log area
        self.log_refine = QListWidget()
        self.log_refine.setMaximumHeight(200)
        v_control.addWidget(self.log_refine)

        gb_control.setLayout(v_control)
        left_layout.addWidget(gb_control)

        layout.addWidget(left_panel)

        # ========== 右侧面板（多标签页） ==========
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 创建标签页控件
        self.tab_results = QTabWidget()

        # Tab 1: Main Pattern
        self.create_pattern_tab()

        # Tab 2: Phase Analysis
        self.create_phase_analysis_tab()

        # Tab 3: Statistics
        self.create_statistics_tab()


        right_layout.addWidget(self.tab_results)

        layout.addWidget(right_panel)

        # Initialize state
        self.current_xrd_file = None



    def create_pattern_tab(self):
        """Tab 1: Main refinement pattern - 双窗格布局"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 创建图形（不预先创建子图，让绘图函数来创建）
        self.fig_pattern = Figure(figsize=(10, 8))
        self.canvas_pattern = FigureCanvas(self.fig_pattern)

        # 初始占位符
        ax = self.fig_pattern.add_subplot(111)
        ax.text(0.5, 0.5, 'Run refinement to see publication-quality pattern',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, style='italic', color='gray')
        ax.axis('off')

        layout.addWidget(self.canvas_pattern)

        # 工具栏
        toolbar = NavigationToolbar(self.canvas_pattern, tab)
        layout.addWidget(toolbar)

        self.tab_results.addTab(tab, "📈 Pattern")

        # 初始绘制
        self.canvas_pattern.draw()
        return tab

    def stop_refinement(self):
        """中断精修进程"""
        if hasattr(self, 'refine_worker') and self.refine_worker.isRunning():
            # 1. 发送停止信号给 Worker
            self.refine_worker.stop()

            # 2. UI 反馈
            self.btn_stop_refine.setEnabled(False)  # 防止重复点击
            self.btn_stop_refine.setText("Stopping...")
            self.log_refinement_message("⚠️ Stop signal sent. Waiting for current step to finish...")

            # 注意：我们不在这里恢复 "Start" 按钮
            # 我们等待 Worker 线程真正结束后，通过 QThread 的 finished 信号来恢复界面

    def create_phase_analysis_tab(self):
        """Tab 2: Phase scale factors and composition"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Split into two plots
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: Bar chart
        widget_top = QWidget()
        layout_top = QVBoxLayout(widget_top)
        self.fig_scales = Figure(figsize=(8, 4))
        self.canvas_scales = FigureCanvas(self.fig_scales)
        self.ax_scales = self.fig_scales.add_subplot(111)
        layout_top.addWidget(self.canvas_scales)
        splitter.addWidget(widget_top)

        # Bottom: Pie chart
        widget_bottom = QWidget()
        layout_bottom = QVBoxLayout(widget_bottom)
        self.fig_pie = Figure(figsize=(8, 4))
        self.canvas_pie = FigureCanvas(self.fig_pie)
        self.ax_pie = self.fig_pie.add_subplot(111)
        layout_bottom.addWidget(self.canvas_pie)
        splitter.addWidget(widget_bottom)

        layout.addWidget(splitter)

        self.tab_results.addTab(tab, "🔬 Phases")

    def create_statistics_tab(self):
        """Tab 3: 重写的统计信息界面"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- 1. 顶部指标栏 ---
        gb_metrics = QGroupBox("Refinement Metrics")
        gb_metrics.setMaximumHeight(80)
        metrics_layout = QHBoxLayout()

        # 定义样式
        label_style = "font-size: 14pt; font-weight: bold; color: #2196F3;"

        self.lbl_rwp = QLabel("Rwp: --")
        self.lbl_rwp.setStyleSheet(label_style)
        self.lbl_chi2 = QLabel("χ²: --")
        self.lbl_chi2.setStyleSheet(label_style)
        self.lbl_gof = QLabel("GOF: --")
        self.lbl_gof.setStyleSheet(label_style)

        metrics_layout.addWidget(self.lbl_rwp)
        metrics_layout.addStretch()
        metrics_layout.addWidget(self.lbl_chi2)
        metrics_layout.addStretch()
        metrics_layout.addWidget(self.lbl_gof)
        gb_metrics.setLayout(metrics_layout)
        layout.addWidget(gb_metrics)

        # --- 2. 中间：晶胞参数表格 ---
        gb_phase = QGroupBox("Phase & Unit Cell Parameters")
        phase_layout = QVBoxLayout()
        self.table_phase_details = QTableWidget()
        self.table_phase_details.setColumnCount(9)
        self.table_phase_details.setHorizontalHeaderLabels([
            "Phase", "Scale", "a (Å)", "b (Å)", "c (Å)",
            "α (°)", "β (°)", "γ (°)", "Vol (Å³)"
        ])
        self.table_phase_details.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        phase_layout.addWidget(self.table_phase_details)
        gb_phase.setLayout(phase_layout)
        layout.addWidget(gb_phase, 2)  # 占比 2

        # --- 3. 底部：仪器与背景参数 ---
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # 仪器参数表格
        gb_inst = QGroupBox("Instrument Parameters")
        inst_layout = QVBoxLayout()
        self.table_inst = QTableWidget()
        self.table_inst.setColumnCount(2)
        self.table_inst.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.table_inst.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        inst_layout.addWidget(self.table_inst)
        gb_inst.setLayout(inst_layout)

        # 背景系数表格
        gb_bkg = QGroupBox("Background Coefficients")
        bkg_layout = QVBoxLayout()
        self.table_bkg = QTableWidget()
        self.table_bkg.setColumnCount(2)
        self.table_bkg.setHorizontalHeaderLabels(["Index", "Value"])
        self.table_bkg.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bkg_layout.addWidget(self.table_bkg)
        gb_bkg.setLayout(bkg_layout)

        bottom_layout.addWidget(gb_inst)
        bottom_layout.addWidget(gb_bkg)
        layout.addWidget(bottom_widget, 2)  # 占比 2

        self.tab_results.addTab(tab, "📊 Statistics")

    def copy_refinement_log(self):
        """Copy refinement log to clipboard"""
        try:
            if self.log_refine.count() == 0:
                QMessageBox.information(self, "Copy Log", "Log is empty.")
                return

            # Collect all log content
            log_content = []
            for i in range(self.log_refine.count()):
                log_content.append(self.log_refine.item(i).text())

            # Add header and timestamp
            full_log = f"=== GSAS-II REFINEMENT LOG ===\n"
            full_log += f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            full_log += f"XRD File: {getattr(self, 'current_xrd_file', 'Unknown')}\n"
            full_log += f"GSAS Path: {self.gsas_path or 'Not set'}\n"
            full_log += "=" * 50 + "\n\n"

            # Add log content
            full_log += "\n".join(log_content)

            # Copy to clipboard
            QApplication.clipboard().setText(full_log)

            # Show success message
            QMessageBox.information(self, "Copy Log",
                                    f"Copied {self.log_refine.count()} log entries to clipboard.")

            # Log this action
            self.log_refinement_message("Log copied to clipboard")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy log: {str(e)}")

    def clear_refinement_log(self):
        """Clear refinement log"""
        reply = QMessageBox.question(self, "Clear Log",
                                     "Are you sure you want to clear the log?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.log_refine.clear()
            self.log_refinement_message("Log cleared")

    def log_refinement_message(self, message):
        """Simple log recording"""
        self.log_refine.addItem(f"{time.strftime('%H:%M:%S')} - {message}")
        self.log_refine.scrollToBottom()

    def handle_refinement_error(self, error_message):
        """Handle errors during refinement process"""
        # Log error
        self.log_refinement_message(f"ERROR: {error_message}")

        # Restore interface state
        self.btn_start_refine.setEnabled(True)
        self.btn_start_refine.setText("Start Refinement")

        # Show error dialog with suggestion to copy log
        reply = QMessageBox.critical(
            self,
            "Refinement Error",
            f"Refinement failed with error:\n\n{error_message}\n\n"
            f"You can use 'Copy Log' button to copy the complete log for troubleshooting.",
            QMessageBox.StandardButton.Ok
        )

    def start_refinement(self):
        """Start refinement"""
        try:
            # Validation
            if not self.gsas_path:
                QMessageBox.warning(self, "Error", "Please set GSAS-II path")
                return

            if not self.current_xrd_file:
                QMessageBox.warning(self, "Error", "Please select XRD file")
                return

            # Get selected phases
            selected_phases = []
            for i in range(self.list_phases_refine.count()):
                item = self.list_phases_refine.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    phase_name = item.text().split('(')[0].strip()
                    selected_phases.append(phase_name)

            if not selected_phases:
                QMessageBox.warning(self, "Error", "Please select at least one phase")
                return

            # Prepare output directory
            output_dir = os.path.join(self.base_dir, "GSAS_Output")

            # Create worker
            self.refine_worker = RefinementWorker(
                self.current_xrd_file,
                selected_phases,
                self.cif_dir,
                output_dir,
                self.gsas_path,
                refinement_range=(7.0, 65.0)
            )

            # Connect signals
            self.refine_worker.log_signal.connect(self.log_refinement_message)
            self.refine_worker.progress_signal.connect(self.pbar_refine.setValue)
            self.refine_worker.finished_signal.connect(self.on_refinement_finished)
            self.refine_worker.error_signal.connect(self.handle_refinement_error)
            self.refine_worker.plot_data_signal.connect(self.plot_refinement_results)
            self.refine_worker.detailed_data_signal.connect(self.update_all_tabs)
            self.refine_worker.finished.connect(self.on_worker_stopped)

            # Initialize interface
            self.btn_start_refine.setEnabled(False)
            self.btn_stop_refine.setEnabled(True)     # 启用停止按钮
            self.btn_stop_refine.setText("Stop")      # 重置文字
            self.pbar_refine.setValue(0)
            self.log_refine.clear()

            # Start
            self.refine_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start: {str(e)}")
            # 如果启动失败，确保按钮状态正确
            self.btn_start_refine.setEnabled(True)
            self.btn_stop_refine.setEnabled(False)

    def on_worker_stopped(self):
        """当 Worker 线程彻底停止运行时调用（无论是完成、出错还是被中断）"""
        self.btn_start_refine.setEnabled(True)
        self.btn_stop_refine.setEnabled(False)
        self.btn_stop_refine.setText("Stop")
        self.log_refinement_message("ℹ️ Process terminated.")

    def plot_refinement_results(self, x_obs=None, y_obs=None, y_calc=None, y_bkg=None, rwp=0.0):
        """绘制精修图谱 (动态范围版)"""
        try:


            # 1. 加载数据
            json_path = os.path.join(self.base_dir, "GSAS_Output", "results.json")

            # 默认范围
            view_min, view_max = 7.0, 65.0

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)

                    # 【新增】读取数据范围
                    if 'data_limits' in json_data:
                        limits = json_data['data_limits']
                        view_min = float(limits[0])
                        view_max = float(limits[1])

                    if 'plot_data' in json_data:
                        pd = json_data['plot_data']
                        x_obs = np.array(pd['x'])
                        y_obs = np.array(pd['y_obs'])
                        y_calc = np.array(pd['y_calc'])
                        y_bkg = np.array(pd['y_bkg'])
                        rwp = json_data['statistics'].get('Rwp', 0.0)
                        self.detailed_data = json_data

            if x_obs is None: return

            # 2. 绘图初始化
            self.fig_pattern.clear()
            gs = self.fig_pattern.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax_main = self.fig_pattern.add_subplot(gs[0])
            ax_residual = self.fig_pattern.add_subplot(gs[1], sharex=ax_main)

            # 绘制曲线
            ax_main.plot(x_obs, y_obs, 'k.', markersize=3, alpha=0.5, label='Observed')
            ax_main.plot(x_obs, y_calc, 'r-', linewidth=1.0, label='Calculated', alpha=0.8)
            ax_main.plot(x_obs, y_bkg, 'g--', linewidth=1.0, label='Background', alpha=0.6)
            diff = y_obs - y_calc
            ax_residual.plot(x_obs, diff, 'b-', linewidth=0.8)
            ax_residual.axhline(0, color='r', linestyle='--', linewidth=0.5)

            # 3. 标记 Top 8 峰 (基于动态范围)
            if hasattr(self, 'detailed_data') and self.detailed_data:
                phases = self.detailed_data.get('phases', [])
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                phase_colors = {p['name']: colors[i % len(colors)] for i, p in enumerate(phases)}


                # --- Step 1: 检索 Y_calc 的前 8 个峰记录位置 ---
                # 阈值：峰高 > 全局最大值的 3%
                global_max = np.max(y_calc)
                peaks_idx, props = find_peaks(y_calc, height=global_max * 0.03, distance=20)

                # 按峰高排序
                sorted_indices = np.argsort(props['peak_heights'])[::-1]
                top_8_indices = sorted_indices[:8]

                # 得到 8 个可见峰的 [位置, 高度]
                visible_peaks = []
                for idx in top_8_indices:
                    px = x_obs[peaks_idx[idx]]
                    py = props['peak_heights'][idx]
                    if view_min <= px <= view_max:
                        visible_peaks.append({'pos': px, 'y': py})

                # --- Step 2: 寻找每个相各自的前 4 强度 ---
                # 这一步已经在 Worker 里做完了，数据存在 phases[i]['top_peaks'] 里
                # 里面存的是 {pos, hkl, height}

                # --- Step 3: 比较与归属 ---
                occupied_positions = []

                # 遍历图上的 8 个大峰
                for v_peak in visible_peaks:
                    v_pos = v_peak['pos']
                    v_y = v_peak['y']

                    best_match_phase = None
                    best_match_hkl = None
                    min_dist = 0.35  # 允许 0.35 度的误差
                    max_theory_height = -1.0  # 如果有多个相匹配，选理论贡献最大的

                    # 遍历所有相的 Top 4 理论峰
                    for phase in phases:
                        p_name = phase['name']
                        theory_peaks = phase.get('top_peaks', [])

                        for t_peak in theory_peaks:
                            t_pos = t_peak['pos']
                            t_height = t_peak.get('height', 0)  # 理论高度

                            dist = abs(v_pos - t_pos)
                            if dist < min_dist:
                                # 找到匹配！
                                # 如果这个位置同时匹配了两个相，选理论高度更高的那个
                                if t_height > max_theory_height:
                                    max_theory_height = t_height
                                    min_dist = dist  # 锁定这个更优解
                                    best_match_phase = p_name
                                    best_match_hkl = t_peak['hkl']

                    # 如果找到了归属，就画上去
                    if best_match_phase:
                        color = phase_colors.get(best_match_phase, 'black')
                        short_name = best_match_phase[:4]

                        # 堆叠避让
                        level = 0
                        cols = [lvl for p, lvl in occupied_positions if abs(v_pos - p) < 1.5]
                        if cols: level = max(cols) + 1
                        occupied_positions.append((v_pos, level))

                        y_txt = v_y + (level * global_max * 0.12) + (global_max * 0.05)

                        ax_main.vlines(v_pos, v_y, y_txt, colors=color, linestyle=':', alpha=0.8)
                        ax_main.text(v_pos, y_txt, f"{short_name}\n{best_match_hkl}",
                                     color=color, fontsize=9, fontweight='bold',
                                     ha='center', va='bottom', rotation=90,
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0))

            if hasattr(self, 'detailed_data') and self.detailed_data:
                phases = self.detailed_data.get('phases', [])
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                phase_colors = {}
                for i, p in enumerate(phases):
                    phase_colors[p['name']] = colors[i % len(colors)]

                # 仅在视图范围内寻找最大值
                mask_view = (x_obs >= view_min) & (x_obs <= view_max)
                if np.any(mask_view):
                    global_max_y = np.max(y_obs[mask_view])
                else:
                    global_max_y = np.max(y_obs)

                # 寻峰
                # 1. 计算步长 (假设数据是等步长的)
                if len(x_obs) > 1:
                    step_size = x_obs[1] - x_obs[0]
                else:
                    step_size = 0.01  # 默认兜底

                # 2. 将 0.5 度转换为点数
                min_deg_dist = 1
                distance_points = int(min_deg_dist / step_size)

                # 确保至少为 1
                distance_points = max(1, distance_points)

                # 3. 寻峰 (应用计算出的距离限制)
                # 阈值：峰高 > 全局最大值的 2%
                peaks_idx, props = find_peaks(y_calc, height=global_max_y * 0.02, distance=distance_points)

                # 4. 按峰高排序，取前 8 个
                peak_heights = props['peak_heights']
                sorted_indices = np.argsort(peak_heights)[::-1]
                top_8_indices = sorted_indices[:10]

                occupied_positions = []

                for idx in top_8_indices:
                    peak_x = x_obs[peaks_idx[idx]]
                    peak_y = peak_heights[idx]

                    # 【关键】只标记视图范围内的峰
                    if peak_x < view_min or peak_x > view_max: continue

                    # 反查归属
                    best_phase = None
                    best_hkl = None
                    min_distance = 0.25
                    max_int_found = -1

                    for phase in phases:
                        all_refs = phase.get('all_refs', [])
                        for ref in all_refs:
                            if abs(ref['pos'] - peak_x) < 0.25:
                                # 优先匹配距离近的，距离差不多时匹配强度大的(如果有强度数据的话)
                                # 这里简化为距离最近优先
                                d = abs(ref['pos'] - peak_x)
                                if d < min_distance:
                                    min_distance = d
                                    best_phase = phase['name']
                                    best_hkl = ref['hkl']

                    if best_phase:
                        color = phase_colors.get(best_phase, 'black')
                        short_name = best_phase[:4]

                        # 堆叠避让
                        level = 0
                        cols = [lvl for p, lvl in occupied_positions if abs(peak_x - p) < 1.5]
                        if cols: level = max(cols) + 1
                        occupied_positions.append((peak_x, level))

                        # 坐标
                        y_offset = (level * global_max_y * 0.12) + (global_max_y * 0.05)
                        text_y = peak_y + y_offset

                        ax_main.vlines(peak_x, peak_y, text_y, colors=color, linestyle=':', alpha=0.8)
                        ax_main.text(peak_x, text_y, f"{short_name}\n{best_hkl}",
                                     color=color, fontsize=9, fontweight='bold',
                                     ha='center', va='bottom', rotation=90,
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0))

            # 4. 轴设置
            ax_main.set_title(f'Rietveld Refinement (Rwp = {rwp:.2f}%)', fontsize=12, fontweight='bold')
            handles, labels = ax_main.get_legend_handles_labels()
            if handles: ax_main.legend(loc='upper right', fontsize=9)

            ax_main.tick_params(labelbottom=False)
            ax_residual.set_xlabel('2θ (degrees)', fontsize=10, fontweight='bold')
            ax_residual.set_ylabel('Diff', fontsize=10)
            ax_main.set_ylabel('Intensity', fontsize=10, fontweight='bold')

            # 【关键】应用 JSON 中的范围
            ax_main.set_xlim(view_min, view_max)

            # 动态 Y 轴
            max_level = max([lvl for p, lvl in occupied_positions]) if occupied_positions else 0
            ax_main.set_ylim(top=global_max_y * (1.15 + max_level * 0.15))

            if np.any(mask_view):
                diff_max = np.max(np.abs(diff[mask_view]))
                ax_residual.set_ylim(-1.2 * diff_max, 1.2 * diff_max)

            self.canvas_pattern.draw()

            # 保存
            self.last_x_obs = x_obs
            self.last_y_calc = y_calc
            self.final_rwp = rwp

        except Exception as e:
            print(f"Plot error: {e}")
            traceback.print_exc()

    def on_refinement_finished(self, results):
        """Refinement completed - Robust Handling"""
        try:

            # Store results
            self.refinement_results = results
            self.final_rwp = results.get('Rwp', 0.0)


            # 获取 detailed data
            if hasattr(self.refine_worker, 'detailed_data'):
                self.detailed_data = self.refine_worker.detailed_data
            else:
                # Fallback
                self.detailed_data = {
                    'phases': results.get('phases', []),
                    'statistics': {'Rwp': self.final_rwp, 'chi2': results.get('chi2', 0)}
                }

            # 调试信息
            self.log_refinement_message(f"✨ Final Rwp: {self.final_rwp:.3f}%")


            # 更新所有标签页
            self.log_refinement_message("📊 Updating all analysis tabs...")
            self.update_all_tabs(self.detailed_data)

            # 显示总结弹窗
            self.show_refinement_summary(results)

        except Exception as e:
            error_msg = f"Error in completion handler: {str(e)}"
            self.log_refinement_message(f"❌ {error_msg}")
            print(traceback.format_exc())  # 打印到控制台方便调试


    def add_peak_markers(self, ax):
        """添加峰标记和晶面指标"""
        try:
            if not hasattr(self, 'detailed_data'):
                return

            phases = self.detailed_data.get('phases', [])
            colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray']
            x = self.last_x_obs
            y_calc = self.last_y_calc

            for i, phase in enumerate(phases):
                if i >= len(colors):
                    break

                phase_name = phase['name']
                peak_positions = phase.get('peak_positions', [])

                # 只标记前几个主要峰避免拥挤
                major_peaks = peak_positions[:8]

                for two_theta in major_peaks:
                    idx = np.argmin(np.abs(x - two_theta))
                    if idx < len(y_calc):
                        intensity = y_calc[idx]

                        # 只标记强度足够大的峰
                        if intensity > np.max(y_calc) * 0.05:
                            ax.vlines(two_theta, 0, intensity,
                                      colors=colors[i], linewidth=0.8, alpha=0.5,
                                      linestyle=':')

                            # 选择性标注晶面指标（避免重叠）
                            if intensity > np.max(y_calc) * 0.15:
                                ax.text(two_theta, intensity + np.max(y_calc) * 0.02,
                                        f'{phase_name[:3]}', fontsize=6, color=colors[i],
                                        ha='center', va='bottom', rotation=90, alpha=0.7)

        except Exception as e:
            self.log_refinement_message(f"⚠️  Peak marking error: {e}")

    def show_refinement_summary(self, results):
        """显示详细的精修结果摘要"""
        try:
            # 创建详细的结果消息
            phases = results.get('phases', [])

            summary_msg = f"""
    🎉 REFINEMENT COMPLETED SUCCESSFULLY

    📊 Quality Metrics:
       • Rwp = {results['Rwp']:.3f}%
       • χ² = {results['chi2']:.3f}
       • GOF = {np.sqrt(results['chi2']):.3f}

    🔬 Phase Composition:
    """

            # 添加各相信息
            for phase in phases:
                percentage = phase.get('percentage', 0)
                scale = phase.get('scale', 0)
                cell = phase.get('cell', {})

                summary_msg += f"   • {phase['name']}: {percentage:.1f}% (Scale: {scale:.2e})\n"

                # 添加晶胞参数（如果可用）
                if cell:
                    summary_msg += f"     a={cell.get('a', 0):.3f}Å, b={cell.get('b', 0):.3f}Å, c={cell.get('c', 0):.3f}Å\n"

            summary_msg += f"\n💾 Results saved in: {os.path.join(self.base_dir, 'GSAS_Output')}"

            # 显示消息框
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Refinement Completed")
            msg_box.setText(summary_msg.strip())
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)

            # 添加复制按钮
            copy_button = msg_box.addButton("Copy Summary", QMessageBox.ButtonRole.ActionRole)
            msg_box.exec()

            if msg_box.clickedButton() == copy_button:
                QApplication.clipboard().setText(summary_msg.strip())
                self.log_refinement_message("📋 Refinement summary copied to clipboard")

        except Exception as e:
            # 回退到简单消息
            QMessageBox.information(self, "Completed",
                                    f"Refinement finished!\n\nRwp = {results['Rwp']:.2f}%\nχ² = {results['chi2']:.2f}")
    def select_gsas_path(self):
        """Select GSAS-II directory"""
        d = QFileDialog.getExistingDirectory(self, "Select GSAS-II Directory")
        if d:
            self.gsas_path = d
            self.txt_gsas_path.setText(d)

    def select_xrd_for_refinement(self):
        """Select XRD file for refinement"""
        f, _ = QFileDialog.getOpenFileName(self, "Select XRD File", "",
                                           "XRD Files (*.xye *.xy *.txt);;All Files (*.*)")
        if f:
            self.current_xrd_file = f
            self.lbl_xrd_file.setText(os.path.basename(f))

    def select_cif_dir(self):
        """Select CIF directory"""
        d = QFileDialog.getExistingDirectory(self, "Select CIF Directory")
        if d:
            self.cif_dir = d
            self.lbl_cif_dir.setText(d)

    def test_gsas_connection(self):
        """Test GSAS-II connection"""
        if not self.gsas_path:
            QMessageBox.warning(self, "Warning", "Please set GSAS-II path first")
            return

        try:
            success, G2sc, config_info = auto_configure_gsas(self.gsas_path)
            if success:
                self.lbl_gsas_status.setText("Status: Connected ✓")
                self.lbl_gsas_status.setStyleSheet("color: green;")
                QMessageBox.information(self, "Success", "GSAS-II connection successful!")
            else:
                self.lbl_gsas_status.setText("Status: Failed ✗")
                self.lbl_gsas_status.setStyleSheet("color: red;")
                QMessageBox.warning(self, "Failed", "Failed to connect to GSAS-II")
        except Exception as e:
            self.lbl_gsas_status.setText("Status: Error ✗")
            self.lbl_gsas_status.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Error", f"Connection error: {str(e)}")

    def show_format_config_dialog(self, xrd_file):
        """Show format configuration dialog"""
        try:
            dialog = XRDFormatDialog(xrd_file, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                config = dialog.get_config()
                self.refine_worker.format_config = config
                self.refine_worker.format_dialog_result = True
                self.log_refinement_message("Format configured successfully")
            else:
                self.refine_worker.format_dialog_result = False
                self.log_refinement_message("Format configuration cancelled")
        except Exception as e:
            self.log_refinement_message(f"Format dialog error: {str(e)}")

    def update_all_tabs(self, detailed_data):
        """更新所有标签页的可视化"""
        try:
            self.log_refinement_message("📊 Updating visualizations...")

            # 保存数据供后续使用
            self.detailed_data = detailed_data

            # 更新 Phase Analysis 标签页
            if hasattr(self, 'ax_scales'):
                self.update_phase_charts(detailed_data)

            # 更新 Statistics 标签页
            if hasattr(self, 'lbl_rwp'):
                self.update_statistics_display(detailed_data)

            self.log_refinement_message("✅ Visualizations updated")

        except Exception as e:
            self.log_refinement_message(f"⚠️  Visualization error: {e}")

    def update_phase_charts(self, data=None):
        """更新相分析图表 (从 JSON 读取)"""
        try:
            # --- 强制从文件加载 ---
            json_path = os.path.join(self.base_dir, "GSAS_Output", "results.json")

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
            elif data is None:
                self.log_refinement_message("⚠️ No results.json found for charts")
                return

            phases = data.get('phases', [])
            if len(phases) == 0:
                self.log_refinement_message("⚠️ No phase data in JSON")
                return

            phase_names = [p['name'] for p in phases]
            # 注意：这里直接用 scale，不需要再归一化，因为 percentage 已经是归一化的
            percentages = [p.get('percentage', 0.0) for p in phases]
            scales = [p.get('scale', 0.0) for p in phases]

            if sum(scales) == 0:
                self.ax_scales.clear()
                self.ax_scales.text(0.5, 0.5, 'No Scale Data', ha='center')
                self.canvas_scales.draw()
                return

            # 柱状图
            self.ax_scales.clear()
            colors = plt.cm.Set3(np.linspace(0, 1, len(phase_names)))
            bars = self.ax_scales.bar(range(len(phase_names)), percentages,
                                      color=colors, edgecolor='black', linewidth=1.2)

            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                self.ax_scales.text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{pct:.1f}%', ha='center', va='bottom',
                                    fontsize=9, fontweight='bold')

            self.ax_scales.set_xticks(range(len(phase_names)))
            self.ax_scales.set_xticklabels(phase_names, rotation=45, ha='right', fontsize=9)
            self.ax_scales.set_ylabel('Relative Amount (%)', fontsize=10, fontweight='bold')
            self.ax_scales.set_title('Phase Quantities', fontsize=11, fontweight='bold')
            self.ax_scales.grid(True, axis='y', alpha=0.3)
            self.fig_scales.tight_layout()
            self.canvas_scales.draw()

            # 饼图
            self.ax_pie.clear()
            self.ax_pie.pie(percentages, labels=phase_names, autopct='%1.1f%%',
                            startangle=90, colors=colors, explode=[0.05] * len(phase_names))
            self.ax_pie.set_title('Phase Composition', fontsize=11, fontweight='bold')
            self.fig_pie.tight_layout()
            self.canvas_pie.draw()

        except Exception as e:
            self.log_refinement_message(f"Phase chart error: {e}")

    def update_statistics_display(self, data=None):
        """更新统计信息 (强制读取 JSON 文件版)"""

        # 定义 JSON 路径 (和 Worker 里保存的路径一致)
        output_dir = os.path.join(self.base_dir, "GSAS_Output")
        json_path = os.path.join(output_dir, "results.json")

        # 尝试从文件读取数据
        final_data = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    final_data = json.load(f)
                print("✅ UI successfully loaded results.json")
            except Exception as e:
                print(f"❌ Failed to load JSON: {e}")
                final_data = data  # 降级使用信号传来的数据
        else:
            final_data = data

        if not final_data or 'statistics' not in final_data:
            return

        try:
            stats = final_data['statistics']
            phases = final_data.get('phases', [])

            def fmt(val, decimals=4):
                try:
                    return f"{float(val):.{decimals}f}"
                except:
                    return "0.0000"

            # 1. 统计指标 (直接显示，不做任何计算)
            rwp = float(stats.get('Rwp', 0.0))
            chi2 = float(stats.get('chi2', 0.0))
            gof = float(stats.get('GOF', 0.0))

            self.lbl_rwp.setText(f"Rwp: {rwp:.3f}%")
            self.lbl_chi2.setText(f"χ²: {chi2:.2f}")
            self.lbl_gof.setText(f"GOF: {gof:.3f}")

            color = "#4CAF50" if (0 < rwp < 15) else "#F44336"
            self.lbl_rwp.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {color};")

            # 2. 晶胞表格
            self.table_phase_details.setRowCount(len(phases))
            for i, p in enumerate(phases):
                cell = p.get('cell', {})
                name = str(p.get('name', 'Unknown'))
                self.table_phase_details.setItem(i, 0, QTableWidgetItem(name))
                self.table_phase_details.setItem(i, 1, QTableWidgetItem(fmt(p.get('scale', 0))))
                self.table_phase_details.setItem(i, 2, QTableWidgetItem(fmt(cell.get('a', 0))))
                self.table_phase_details.setItem(i, 3, QTableWidgetItem(fmt(cell.get('b', 0))))
                self.table_phase_details.setItem(i, 4, QTableWidgetItem(fmt(cell.get('c', 0))))
                self.table_phase_details.setItem(i, 5, QTableWidgetItem(fmt(cell.get('alpha', 90), 2)))
                self.table_phase_details.setItem(i, 6, QTableWidgetItem(fmt(cell.get('beta', 90), 2)))
                self.table_phase_details.setItem(i, 7, QTableWidgetItem(fmt(cell.get('gamma', 90), 2)))
                self.table_phase_details.setItem(i, 8, QTableWidgetItem(fmt(cell.get('volume', 0), 2)))

            # 3. 仪器参数
            if hasattr(self, 'table_inst'):
                inst = stats.get('inst', {})
                keys = ['U', 'V', 'W', 'X', 'Y', 'Zero']
                self.table_inst.setRowCount(len(keys))
                for r, k in enumerate(keys):
                    val = inst.get(k, 0.0)
                    self.table_inst.setItem(r, 0, QTableWidgetItem(k))
                    self.table_inst.setItem(r, 1, QTableWidgetItem(fmt(val, 6)))

            # 4. 背景参数
            if hasattr(self, 'table_bkg'):
                bkg = stats.get('bkg', {})
                coeffs = bkg.get('coeffs', [])
                self.table_bkg.setRowCount(len(coeffs))
                for r, val in enumerate(coeffs):
                    self.table_bkg.setItem(r, 0, QTableWidgetItem(f"Coeff {r + 1}"))
                    self.table_bkg.setItem(r, 1, QTableWidgetItem(fmt(val, 3)))

            # 更新详细数据供绘图使用
            self.detailed_data = final_data

        except Exception as e:
            print(f"Error updating stats UI: {e}")
            traceback.print_exc()


# ... (你的 XRDApp 类代码结束) ...

if __name__ == '__main__':
    try:
        # 1. 初始化 Application
        app = QApplication(sys.argv)
        app.setFont(QFont("Segoe UI", 9))

        # 2. 启动主窗口
        win = XRDApp()
        win.show()

        # 3. 进入事件循环
        sys.exit(app.exec())

    except Exception as e:
        # =================================================
        # 崩溃捕获：让窗口停住，显示错误信息
        # =================================================
        error_msg = traceback.format_exc()
        print("\n" + "!" * 60)
        print("CRITICAL ERROR: The application crashed!")
        print("!" * 60)
        print(error_msg)
        print("!" * 60 + "\n")

        # 如果是打包环境，尝试弹窗显示错误（因为没有控制台）
        if getattr(sys, 'frozen', False):
            try:
                # 尝试创建一个临时的 QApplication 来显示弹窗
                if not QApplication.instance():
                    app = QApplication(sys.argv)
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("Application Error")
                msg.setText("A critical error occurred and the application must close.")
                msg.setDetailedText(error_msg)
                msg.exec()
            except:
                # 如果弹窗也失败，最后一道防线：写日志文件
                with open("crash_log.txt", "w") as f:
                    f.write(error_msg)

        # 暂停控制台，等待用户按键
        input("Press Enter to exit...")
        sys.exit(1)