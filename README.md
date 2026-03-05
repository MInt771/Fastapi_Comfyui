# AI 素材生成工厂 (FastAPI + ComfyUI)

基于 FastAPI 封装的 AIGC 素材生成服务：接收文案与风格，调用 ComfyUI 工作流批量生成图片，返回图片链接与 MD5，并支持异步任务队列与记录入库。

## 技术栈

- **Python 3.10+** / FastAPI / Pydantic  
- **ComfyUI**：文生图工作流（Stable Diffusion 等）  
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
  "count": 2
}
```

**异步任务** `POST /tasks`：同上请求体，返回 `{"task_id": 1}`，再用 `GET /tasks/1` 轮询结果。

## 环境变量（可选）

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `COMFYUI_API_BASE` | `http://127.0.0.1:8188` | ComfyUI API 地址。 |
| `COMFYUI_WORKFLOW_PATH` | `comfyui_workflow_api.json` | 工作流 JSON 路径。 |

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
