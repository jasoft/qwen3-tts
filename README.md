# qwen-tts-fast

把 Qwen3-TTS 模型常驻在本机内存，用命令行快速播放文本语音。

不提供 OpenAI API，不启动 FastAPI 服务。第一次调用加载模型，后续调用复用后台 daemon，避免重复加载。

## 环境要求

- macOS Apple Silicon
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- 可用的本机音频输出设备

## 安装

**方式一：uv tool install（推荐，任意目录可用）**

```bash
uv tool install /Users/weiwang/Projects/qwen3-tts
```

安装后可在任意目录直接运行：

```bash
qwen-tts "你好呀，壮壮"
```

**方式二：uv sync（项目目录内运行）**

```bash
uv sync
uv run qwen-tts "你好呀，壮壮"
```

## 使用

基本播放：

```bash
qwen-tts "你好呀，壮壮，你想要我带你玩吗？" --speaker Serena
```

第一次运行会下载并加载模型，耗时较长。模型加载完成后 daemon 留在后台，后续调用直接复用内存中的模型并流式播放。

### 常用选项

查看 daemon 状态：

```bash
qwen-tts --status
```

停止 daemon：

```bash
qwen-tts --stop
```

空闲超时自动卸载（类似 LM Studio 行为）：

```bash
qwen-tts "你好" --idle-timeout 1800
```

daemon 在空闲 1800 秒（30 分钟）后自动退出并释放模型内存。每次请求会重置空闲计时器。设为 0（默认）则禁用自动卸载。

从 stdin 读取文本：

```bash
echo "这是一段要播放的文字" | qwen-tts -
```

指定语言和情绪指令：

```bash
qwen-tts "I'm ready." --language English --speaker Serena
qwen-tts "今天真开心。" --instruct "用开心、轻快的语气说"
```

调整流式分块间隔（秒）：

```bash
qwen-tts "测试流式播放" --stream-interval 0.3
```

## 支持的语言

| 语言 | 代码 |
|------|------|
| 中文 | zh |
| 英文 | en |
| 日文 | ja |
| 韩文 | ko |
| 德文 | de |
| 法文 | fr |
| 俄文 | ru |
| 葡文 | pt |
| 西文 | es |
| 意文 | it |

未指定 `--language` 时，脚本根据文本是否包含 CJK 字符自动选择中文或英文。

## 默认模型

```
mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit
```

临时更换模型：

```bash
qwen-tts "你好" --model mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit
```

## 项目结构

- `qwen_tts_fast.py` — 全部逻辑：CLI、后台 daemon、MLX 模型加载和音频播放
- `pyproject.toml` — 项目依赖与 `qwen-tts` 命令入口
- `~/.cache/qwen-tts-fast/daemon.log` — daemon 日志（运行时生成）
- `~/.cache/qwen-tts-fast/daemon.pid` — daemon 进程号（运行时生成）

## 工作原理

1. 首次调用时，脚本在后台启动一个 HTTP daemon（端口 8765），加载 MLX 版 Qwen3-TTS 模型
2. 后续调用通过 HTTP 向 daemon 发送文本，daemon 流式生成音频 PCM 数据
3. 客户端收到 PCM 分块后直接用 `sounddevice` 播放，实现低延迟语音输出
