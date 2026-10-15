# MPMQA RAG 架构图（零安装交付）

本目录提供一套**同源生成**的架构图，覆盖
`切片 → 入库 → 模型部署 → 后端 → 前端 → 跑批评估` 全流程。
全部文件都是纯文本，**不需要在本机安装 draw.io / Node / Java / 浏览器 / 中文字体**。

| 文件 | 用途 | 怎么用 |
| --- | --- | --- |
| `rag-architecture.svg` | 矢量图，终极清晰 | VS Code 直接打开预览；或拖进 Windows 浏览器看 |
| `rag-architecture.drawio` | 可编辑源文件 | [app.diagrams.net](https://app.diagrams.net) → Open Existing Diagram → 选它；改完任意导出 PNG/PDF/SVG |
| `rag-architecture.html` | 看图 + 一键导出 PNG | 拖进 Windows 浏览器（Chrome/Edge）双击打开，右上选倍率 → 「下载 PNG」 |
| `gen_architecture.py` | 生成器（唯一数据源） | `python3 gen_architecture.py` 一次重写上面 3 个文件 |

> 三个文件来自同一份版式数据（`gen_architecture.py` 里的 `BANDS / BOXES / ARROWS / NOTES`），
> 坐标完全一致，改图只需改生成器再重跑，不会出现「SVG 和 drawio 版本不一致」。

---

## 图的内容

版式：**左侧三横带（离线 / 在线 / 评估） + 右侧一竖列（模型部署）**，1960 × 1580。

| 阶段带 | 关键信息 |
| --- | --- |
| ① 离线阶段 · 切片 + 入库 | `data/docs/{DS}/*.docx` → `docx_parser.py`（按 Heading 层级切分、标题/图片边界 flush、抽图落盘）→ `docs/{DS}_json/` → `indexer.py`（batch=32 + `EmbeddingClient`）→ `ChromaDB`（`collection={DS}_docs`，HNSW cosine） |
| ② 在线服务 | `Streamlit :8501/:8502`（session_id、历史渲染、模式切换）→ `FastAPI :8002/:8003` → `① intent_chain` → `② rewrite_chain` → `③ Retriever`（vector / bm25 / hybrid-RRF(k=60)，top_k=10）→ `④ rag_chain` → `⑤ 后处理`（`normalize_ref_position` / `dedup_refs` / `strip_refs`）→ 回前端 |
| ③ 评估 | `data/test/{DS}/*.xlsx` → `eval_ragas.py`（批量 POST /ask）→ Ragas 四项指标 → `ragas_results/{DS}/{doc}_{mode}_ragas_result.xlsx` |
| ④ 模型部署（右列） | `deploy_all.sh` → `deploy_chat.sh`（`:8000` Qwen3.5-4B-AWQ，gpu-util 0.65）+ `deploy_emb.sh`（`:8001` Qwen3-Embedding-0.6B，gpu-util 0.25）；另附端口速查 / 日志目录 / 重建索引命令 |

图中同时标出了 **`SessionManager` 是纯内存**、**图片只存相对文件名（前端按 `--image-base` 复原）**、
**`[ref:uid]` 闭环（LLM 出 ref → 归一 → 去重 → 前端内联）** 等实现要点。

---

## 导出 PNG 的三条路（按推荐顺序）

### 路线 A：HTML 一键导出（最快，零依赖）

1. 在 Windows 资源管理器里进入
   `\\wsl$\Ubuntu\home\tfeng\projects\vllm\main\docs\diagrams\`
   （或直接在 WSL 里 `explorer.exe .`）
2. 双击 `rag-architecture.html`
3. 右上角选 `2x（3920 px）`，点「**下载 PNG**」

> 原理：页面内嵌了 SVG，用 `<canvas>` + `drawImage` 栅格化后 `toBlob` 下载，
> 全程在浏览器本地完成，不联网。
>
> ⚠️ 若弹出的 PNG 里**文字缺失或字体不对**，说明该浏览器的 canvas 渲染 SVG 时
> 没能解析到系统字体 —— 换路线 B，或者直接把浏览器缩放调到 150% 后截图。

### 路线 B：draw.io 导出（最稳，任意分辨率）

1. 打开 <https://app.diagrams.net/>（或 Windows 上已装的 draw.io 桌面版）
2. `Open Existing Diagram` → 选 `rag-architecture.drawio`
   （提示「此文件由其他应用生成」直接确认即可）
3. `File → Export as → PNG…`，勾选 **Zoom: 200%**、**Transparent Background 取消**、`Crop` 不勾
4. 想改图就直接拖拽节点/连线，改完 `File → Save As` 覆盖回同一个 `.drawio`

### 路线 C：浏览器直接开 SVG

`rag-architecture.svg` 是纯手工矢量图，**在浏览器里 `Ctrl+P` 打印成 PDF**
即可得到无损矢量 PDF；VS Code 里也能直接预览（需装内置 SVG 预览或在 Markdown 里引用）。

---

## 字体说明

WSL 的 Linux 侧**一个中文字体都没有**（`fc-list :lang=zh` 返回 0 行），
所以图里的 `font-family` 写成优先命中的字体链：

```
'Microsoft YaHei','PingFang SC','Noto Sans CJK SC','Source Han Sans SC',SimHei,sans-serif
```

你在 Windows 侧渲染时，第一项 **微软雅黑** 必然存在（`C:\Windows\Fonts\msyh.ttc`），
所以显示正常。只有在 Linux 侧用没有配字体的工具渲染才会变方块 —— 这也是
本方案走「Windows 浏览器渲染」而不是「本机装机渲染」的原因。

---

## 改图 / 重新生成

```bash
cd ~/projects/vllm/main/docs/diagrams

python3 gen_architecture.py --check   # 只做自检，不写文件
python3 gen_architecture.py           # 自检 + 重写 svg / drawio / html
```

生成器只依赖 Python 标准库（`argparse / os / sys / xml.sax.saxutils / xml.etree.ElementTree`）。

### 自检都查什么

本机没有渲染器，肉眼看不到图，所以把「画图容易犯的错」都写成了程序化检查：

| 检查项 | 说明 |
| --- | --- |
| 元素越界 | 所有方框 / 阶段带必须落在 1960×1580 画布内 |
| 文字溢出 | 按微软雅黑字宽（汉字 1.0em、拉丁 0.58em）估算每行文字宽度，超过框宽 −16px 就报警 |
| 方框重叠 | 两两相交检测（容器框 `api` 除外） |
| 连线穿框 | 逐段检查正交折线是否从方框内部穿过 |
| 标签压框 | 连线标签的白底矩形是否盖到方框上 |
| 注解越界 / 压框 | 文字注解是否超画布或被方框遮挡 |
| XML 良构 | 生成的 SVG / drawio 用 `ElementTree` 解析一遍 |

当前状态：**全部通过**（最宽文字占框宽 82.5%，最高占框高 81.7%，余量充足）。

改图流程：编辑 `gen_architecture.py` 里「版式数据」那段（每个元素一行，含 x/y/w/h 与文字）→
跑 `--check` 看有没有新报警 → 去掉 `--check` 重新生成。

---

## 与他处的关系

- 文字版说明：[`../architecture.md`](../architecture.md)（ASCII 图 / Mermaid / 目录树 / 端口表 / 启动顺序）
- 相关脚本：`../../deploy/*.sh`（部署）、`../../ingestion/*.py`（切片入库）、
  `../../backend/*.py`（后端）、`../../frontend/app.py`（前端）、`../../eval/*`（评估）
