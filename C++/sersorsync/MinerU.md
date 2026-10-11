**Magic-PDF**（现已升级重构为 **MinerU** 项目的核心 PDF 解析引擎）主要用于将复杂的 PDF 文件（包含公式、表格、图片和排版）高质量地转换为 Markdown 格式。
核心命令为 `magic-pdf`。以下是常用的指令及使用方法：
### 一、 核心基础指令

#### 1. 基础解析（最常用）

直接将本地 PDF 文件转换为 Markdown：

  

Bash

```
magic-pdf -p example.pdf -o ./output
```

- **`-p` / `--path`**: 指定输入的 PDF 文件路径（或包含 PDF 的文件夹路径）。
    
      
    
- **`-o` / `--output-dir`**: 指定转换后结果的输出目录。
    
      
    

#### 2. 指定处理设备 (CPU / CUDA)

如果拥有 NVIDIA 显卡并配置了 GPU 环境，可以通过 `-d` 指定推理设备，以大幅提升解析速度：

  

Bash

```
# 使用 GPU (CUDA) 加速
magic-pdf -p example.pdf -o ./output -d cuda

# 强制使用 CPU 处理
magic-pdf -p example.pdf -o ./output -d cpu
```

#### 3. 指定处理模式 (Model / OCR / TXT)

Magic-PDF 支持选择底层解析策略：

  

Bash

```
# 自动/模型模式（默认，结合版面分析与公式识别）
magic-pdf -p example.pdf -o ./output -m auto

# 强制使用 OCR 模式（适合扫描件、纯图片 PDF 或未内嵌文本的文档）
magic-pdf -p example.pdf -o ./output -m ocr

# 纯文本抽取模式（适合无格式要求的纯文字版 PDF）
magic-pdf -p example.pdf -o ./output -m txt
```

### 二、 批量处理与多线程指令

#### 1. 批量解析文件夹内所有 PDF

将 `-p` 参数后接文件夹路径，程序会自动扫描并处理其中的所有 PDF：

  

Bash

```
magic-pdf -p /path/to/pdf_folder -o ./output
```

#### 2. 多进程/多线程并发处理

处理大量文件时，可通过 `-t` 或 `--threads` 参数指定并行线程数以加快处理进度：

  

Bash

```
magic-pdf -p /path/to/pdf_folder -o ./output -t 4
```

### 三、 输出内容与目录结构

运行指令后，在指定的 `-o` 输出目录下会按文件名生成对应文件夹，包含以下文件：

  

|**输出文件名**|**说明**|
|---|---|
|**`auto/xxx.md`**|最终转换好的 Markdown 文件|
|**`auto/images/`**|自动提取的 PDF 中的插图、表格截图以及识别出的公式图片|
|**`auto/xxx_layout.pdf`**|标注了版面分析（Layout）框选结果的 PDF 调试文件|
|**`auto/xxx_spans.json`**|包含位置排版和结构信息的中间 JSON 文件|

### 四、 配置文件检查与更新

Magic-PDF 依赖模型权重文件。首次使用或更新环境时，可以通过相关辅助指令进行环境校验：

  

Bash

```
# 检查当前配置文件及模型路径是否正确
magic-pdf --version
```

> **提示**：在使用 `magic-pdf` 指令前，请确保已按照官方指引完成 `magic-pdf.json` 配置文件的生成（通常位于用户主目录下 `~/magic-pdf.json`），并正确下载了所需的模型权重（Layout, MFD, MFR 等）。