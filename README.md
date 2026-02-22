# cn-sentence-embed

中文句向量与语义相似度计算项目，基于本地 ModelScope 句向量模型实现，支持 Python API 与 CLI 两种使用方式。

## 项目亮点

- 本地推理：无需外部 API，适合离线或内网场景
- 模块化架构：配置、运行时、业务、示例、测试职责清晰
- 兼容旧调用：保留 `compute_similarity(...)` 返回 `(embeddings, scores)` 的习惯用法
- 快速回归：`test/test_client.py` 基于 mock，不依赖真实模型加载

## 架构说明

项目已将原先集中在 `test/test.py` 的单文件实现拆分为可维护结构：

- 配置层：集中管理模型路径、设备、序列长度、静默模式
- 运行时层：统一处理警告和日志抑制
- 业务层：`SentenceEmbeddingClient` 提供稳定 API
- 示例层：独立演示脚本，避免与测试混用
- 测试层：补充单元测试，降低回归风险

## 项目结构

```text
cn-sentence-embed/
├── cn_sentence_embed/
│   ├── __init__.py
│   ├── __main__.py
│   ├── client.py
│   ├── config.py
│   ├── runtime.py
│   └── types.py
├── examples/
│   └── demo.py
├── models/
│   └── nlp_gte_sentence-embedding_chinese-base/
├── test/
│   ├── test.py
│   └── test_client.py
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

不激活虚拟环境时，可直接使用 `.venv/bin/python` 执行下面命令。

## 快速开始

### 1. Python API

```python
from cn_sentence_embed import SentenceEmbeddingClient

client = SentenceEmbeddingClient(device="cpu")
embeddings, scores = client.compute_similarity(
    ["吃完海鲜可以喝牛奶吗?"],
    [
        "吃了海鲜后不能再喝牛奶，建议间隔数小时。",
        "早晨喝牛奶要看个人体质。"
    ]
)

print(embeddings.shape)
print(scores)
```

### 2. 运行示例脚本

```bash
.venv/bin/python examples/demo.py
```

### 3. 命令行模式

计算相似度：

```bash
.venv/bin/python -m cn_sentence_embed similarity \
  --source "吃完海鲜可以喝牛奶吗?" \
  --compare "不建议立即喝牛奶" "建议间隔后再饮用"
```

仅编码：

```bash
.venv/bin/python -m cn_sentence_embed encode --text "你好" "今天天气不错"
```

## 核心 API

- `SentenceEmbeddingClient.similarity(source_sentences, sentences_to_compare)`
  - 返回 `SimilarityResult`，包含 `text_embedding` 与 `scores`
- `SentenceEmbeddingClient.compute_similarity(...)`
  - 兼容旧调用方式，返回 `(embeddings, scores)`
- `SentenceEmbeddingClient.batch_similarity(source_list, compare_list)`
  - 批量计算多个 source 的相似度
- `SentenceEmbeddingClient.encode(sentences)`
  - 仅获取句向量

## 测试

运行单元测试：

```bash
.venv/bin/python -m unittest discover -s test
```

> `test/test_client.py` 使用 mock，不依赖真实模型加载；适合快速回归。

## 迁移说明

旧版调用主要在 `test/test.py`。重构后：

- 推荐业务代码直接依赖 `cn_sentence_embed` 包
- `test/test.py` 仍可作为手工验证入口
- 如需切换模型目录，使用 `model_path` 参数或 CLI 的 `--model-path`

## 联系方式与社群

欢迎通过以下渠道获取项目动态、技术分享与服务支持：

- 公众号：扫码关注「微信公众号」
- 服务号：扫码关注「微信服务号」
- 粉丝群：扫码加入「技术粉丝群」

<table>
  <tr>
    <td align="center"><strong>微信公众号</strong></td>
    <td align="center"><strong>微信服务号</strong></td>
    <td align="center"><strong>粉丝群</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="assets/qrcode/微信公众号.jpeg" alt="微信公众号二维码" width="220" /></td>
    <td align="center"><img src="assets/qrcode/微信服务号.jpeg" alt="微信服务号二维码" width="220" /></td>
    <td align="center"><img src="assets/qrcode/粉丝群.jpeg" alt="粉丝群二维码" width="220" /></td>
  </tr>
</table>

> 若二维码失效，请在仓库提 Issue 联系维护者更新。
