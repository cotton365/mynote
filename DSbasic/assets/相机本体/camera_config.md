"""
相机颜色配置加载与管理模块
=======================

从 color_config.json 加载 HSV 阈值、预处理参数、形态学参数，提供统一的配置访问接口。

函数列表:
    load_config() -> dict
        加载配置文件（带缓存），文件不存在时返回默认值
    get_hsv_thresholds() -> dict
        获取 HSV 红/绿/蓝阈值
    get_morphology_params() -> dict
        获取形态学参数（开闭运算核大小、最小面积）
    get_config() -> dict
        获取完整配置
"""

import json
import os

CONFIG_FILE = "color_config.json"

_config = None


def load_config() -> dict:
    """加载 color_config.json，带模块级缓存。文件不存在则返回默认配置。"""
    global _config
    if _config is not None:
        return _config

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            _config = json.load(f)
        print(f"[Camera] 已加载配置: {config_path}")
    else:
        print(f"[Camera] 警告: 配置文件不存在，使用默认值")
        _config = _get_default_config()
    return _config


def _get_default_config() -> dict:
    """返回内置默认配置（HSV红/绿/蓝阈值 + 预处理 + 形态学参数）。"""
    return {
        "camera": {
            "index": 0,
            "width": 1280,
            "height": 720,
            "fps": 30,
            "auto_exposure": 1,
            "exposure": 0,
            "auto_wb": 0,
            "wb_temperature": 4600,
            "brightness": 128,
            "contrast": 128,
            "saturation": 160,
            "gain": 0,
            "sharpness": 128,
            "gamma": 100,
        },
        "preprocessing": {
            "clahe_clip": 2.5,
            "clahe_tile": 8,
            "blur_ksize": 5,
            "wb_correction": False,
            "wb_bgr_gain": [1.0, 1.0, 1.0],
        },
        "hsv": {
            "red1":  {"h_lo": 0,   "h_hi": 10,  "s_lo": 120, "s_hi": 255, "v_lo": 70, "v_hi": 255},
            "red2":  {"h_lo": 170, "h_hi": 180, "s_lo": 120, "s_hi": 255, "v_lo": 70, "v_hi": 255},
            "green": {"h_lo": 40,  "h_hi": 80,  "s_lo": 70,  "s_hi": 255, "v_lo": 70, "v_hi": 255},
            "blue":  {"h_lo": 100, "h_hi": 140, "s_lo": 150, "s_hi": 255, "v_lo": 50, "v_hi": 255},
        },
        "morphology": {
            "open_ksize": 5,
            "close_ksize": 9,
            "min_area": 500,
        }
    }


def get_hsv_thresholds() -> dict:
    """返回 HSV 阈值配置: {"red1": {...}, "red2": {...}, "green": {...}, "blue": {...}}"""
    cfg = load_config()
    return cfg["hsv"]


def get_morphology_params() -> dict:
    """返回形态学参数: {"open_ksize": int, "close_ksize": int, "min_area": int}"""
    cfg = load_config()
    return cfg["morphology"]


def get_config() -> dict:
    """返回完整配置字典。"""
    return load_config()
