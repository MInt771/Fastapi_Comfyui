# AI 素材生成工厂 (FastAPI + ComfyUI + LangChain)

基于 FastAPI 封装的 AIGC 素材生成服务：接收文案与风格，调用 ComfyUI 工作流批量生成图片，返回图片链接与 MD5，并支持异步任务队列与记录入库。  
可选地接入 **LangChain + 大语言模型** 对提示词进行智能优化，更贴近真实业务中的 Prompt 工程场景。

## 技术栈

- **Python 3.10+** / FastAPI / Pydantic  
- **LangChain + LLM**：对文案与风格进行提示词优化（可选）  
- **ComfyUI**：文生图工作流（Stable Diffusion / SD3 等）  
- **SQLite + SQLModel**：生成记录与任务状态持久化  
- **httpx**：异步调用 ComfyUI API  

## 环境准备

1. **Python**：建议 3.10+，使用 conda 或 venv 创建虚拟环境。  
2. **ComfyUI**：本地已安装并可正常启动，默认 API 地址 `http://127.0.0.1:8188`。  
3. **工作流模板**：在 ComfyUI 中搭好文生图工作流（含 `CLIPTextEncode`、`SaveImage` 等），使用 **Save (API)** 导出为 JSON，保存到项目根目录，命名为 `comfyui_workflow_api.json`。  

## 安装与运行

```bash
# 克隆仓库
git clone https://github.com/MInt771/Fastapi_Comfyui.git
cd Fastapi_Comfyui

# 创建虚拟环境（任选其一）
python -m venv .venv
.venv\Scripts\activate   # Windows

# 或 conda
conda create -n ai_factory python=3.10 -y
conda activate ai_factory

# 安装依赖
pip install -r requirements.txt
```

**启动顺序：**

1. 先启动 ComfyUI（保证 `http://127.0.0.1:8188` 可访问）。  
2. 再启动本服务（默认端口 8001，避免与 ComfyUI 的 8188 冲突）：  

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

浏览器访问：

- 接口文档：http://127.0.0.1:8001/docs  
- 健康检查：http://127.0.0.1:8001/health  

## 接口说明

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/generate` | 同步生成：传入文案、风格、数量，阻塞直到生成完成，返回图片列表与 MD5。 |
| POST | `/generate_collab` | 多模型协同生成：先用两个 LLM 协同产出最终提示词，再调用 ComfyUI 生成图片。 |
| POST | `/llm/prompt_preview` | 仅调用 LangChain + 单一 LLM 返回优化后的英文提示词，不触发图片生成，可用于 Prompt 调试。 |
| POST | `/llm/prompt_collab` | 多模型协同：使用两个不同的 LLM 完成“基础提示词 + 审稿优化”两阶段生成。 |
| POST | `/tasks` | 异步任务：提交生成请求，立即返回 `task_id`，由后台 worker 排队执行。 |
| GET  | `/tasks/{task_id}` | 查询任务状态与结果（成功时含图片链接与 MD5）。 |
| GET  | `/tasks` | 列出最近任务，支持 `?status=PENDING` 等筛选。 |
| GET  | `/records` | 查看最近生成记录（文案、风格、图片 URL、MD5、耗时）。 |
| GET  | `/health` | 健康检查。 |

### 请求示例

**同步生成** `POST /generate`：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "赛博朋克插画",
  "count": 2,
  "use_llm": true
}
```

- 当 `use_llm = false`（默认）时：直接将 `style + text` 拼接为提示词后注入 ComfyUI 工作流；  
- 当 `use_llm = true` 时：先通过 **LangChain + LLM** 将中文文案与风格转为一段更细致的英文提示词，再注入工作流，更贴近真实业务中的 Prompt 优化需求。

**异步任务** `POST /tasks`：支持三种模式，返回 `{"task_id": 1}`，再用 `GET /tasks/1` 轮询结果。

- 纯 ComfyUI（默认）：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "赛博朋克插画",
  "count": 2
}
```

- 单模型 LLM 模式（等价于 `/generate` + `use_llm=true`）：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "赛博朋克插画",
  "count": 2,
  "use_llm": true,
  "llm_name": "deepseek"
}
```

- 多模型协同模式（等价于 `/generate_collab`）：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "像素风插画",
  "count": 2,
  "use_collab": true,
  "planner_llm_name": "deepseek",
  "reviewer_llm_name": "qwen"
}
```

**LLM 提示词预览** `POST /llm/prompt_preview`：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "像素风插画"
}
```

返回示例：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "像素风插画",
  "optimized_prompt": "a pixel art style illustration of a cat floating in outer space, ... "
}
```

**多模型协同 Prompt** `POST /llm/prompt_collab`：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "像素风插画",
  "planner_llm_name": "deepseek",
  "reviewer_llm_name": "qwen"
}
```

返回示例（简化）：

```json
{
  "text": "...",
  "style": "...",
  "planner_llm_name": "deepseek",
  "reviewer_llm_name": "qwen",
  "base_prompt": "first english prompt ...",
  "final_prompt": "reviewed and enhanced english prompt ..."
}
```

其中：
- `base_prompt` 由 `planner_llm_name` 指定的模型生成；
- `final_prompt` 由 `reviewer_llm_name` 指定的模型在 `base_prompt` 上进一步优化，可作为最终喂给 ComfyUI 的提示词。

**多模型协同直接出图** `POST /generate_collab`：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "像素风插画",
  "count": 2,
  "planner_llm_name": "deepseek",
  "reviewer_llm_name": "qwen"
}
```

该接口内部会：

- 调用 `/llm/prompt_collab` 对应的链生成 `base_prompt` 和 `final_prompt`；  
- 使用 `final_prompt` 注入 ComfyUI 工作流批量生成图片；  
- 将最终使用的提示词、图片 URL、MD5 等信息写入数据库，方便后续效果复盘。

## 环境变量（可选）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `COMFYUI_API_BASE` | `http://127.0.0.1:8188` | ComfyUI API 地址。 |
| `COMFYUI_WORKFLOW_PATH` | `comfyui_workflow_api.json` | 工作流 JSON 路径。 |
| `OPENAI_API_KEY` / `LLM_API_KEY` | - | LangChain 使用的大模型 API Key（支持 OpenAI 协议兼容的国内模型，如千问、DeepSeek 等），可在 `llm_config.json` 中覆盖。 |
| `OPENAI_BASE_URL` / `LLM_API_BASE` | - | OpenAI 或兼容接口的 Base URL，例如千问/DeepSeek 自家的 OpenAI 协议地址，可在 `llm_config.json` 中覆盖。 |
| `LLM_MODEL_NAME` | `gpt-4.1-mini` | LangChain 使用的模型名称，可根据实际模型调整。 |
| `LLM_TEMPERATURE` | `0.7` | LLM 采样温度。 |

## 多模型 LLM 配置

项目支持通过 JSON 文件管理多个大模型供应商（如 DeepSeek、千问），并在请求中按名称切换：

1. 复制示例配置：

```bash
cp llm_config.example.json llm_config.json
```

2. 根据实际情况修改 `llm_config.json` 中的 `api_base`、`api_key`、`model` 等字段：

```json
{
  "default": "deepseek",
  "providers": {
    "deepseek": { "...": "..." },
    "qwen": { "...": "..." }
  }
}
```

3. 在请求体中通过 `llm_name` 指定使用的模型，例如：

```json
{
  "text": "一只在太空漂浮的猫",
  "style": "像素风插画",
  "use_llm": true,
  "llm_name": "deepseek"
}
```

如果未指定 `llm_name`，则使用 `llm_config.json` 中的 `default`，若文件不存在则退回到环境变量配置。

## 项目结构

```
.
├── main.py                    # FastAPI 应用、接口、ComfyUI 调用、后台 worker
├── db.py                      # SQLModel 表（GenerationRecord、GenerationTask）
├── requirements.txt
├── comfyui_workflow_api.json   # 需自行从 ComfyUI 导出
├── data.db                    # SQLite（自动创建，已加入 .gitignore）
└── README.md
```

## 说明

- 首次运行前请确保 `comfyui_workflow_api.json` 存在且工作流中包含 **SaveImage** 节点，否则无法从 ComfyUI 的 history 中获取图片。  
- 数据库文件 `data.db` 仅用于本地，未提交到仓库。  

## License

MIT
