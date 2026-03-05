from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import time
from datetime import datetime

import httpx
from sqlmodel import Session, select

from db import GenerationRecord, GenerationTask, TaskStatus, engine, init_db
from llm_prompt import generate_image_prompt, generate_image_prompt_collab


COMFYUI_API_BASE = os.getenv("COMFYUI_API_BASE", "http://127.0.0.1:8000")
WORKFLOW_PATH = os.getenv("COMFYUI_WORKFLOW_PATH", "comfyui_workflow_api.json")


class GenerateRequest(BaseModel):
    text: str = Field(..., description="文案内容")
    style: str = Field(..., description="风格描述，例如 二次元插画/电商海报")
    count: int = Field(1, ge=1, le=8, description="需要生成的图片数量（1-8）")
    use_llm: bool = Field(
        default=False,
        description="是否使用 LangChain + 大模型对提示词进行智能优化",
    )
    llm_name: Optional[str] = Field(
        default=None,
        description="可选：指定在 llm_config.json 中配置的模型名称，例如 deepseek/qwen",
    )


class GeneratedImage(BaseModel):
    image_url: str
    md5: str


class GenerateResponse(BaseModel):
    images: List[GeneratedImage]


app = FastAPI(title="AI 素材生成工厂 Demo")


class PromptPreviewRequest(BaseModel):
    text: str = Field(..., description="原始中文文案")
    style: str = Field(..., description="风格描述，例如 二次元插画/电商海报")
    llm_name: Optional[str] = Field(
        default=None,
        description="可选：指定在 llm_config.json 中配置的模型名称，例如 deepseek/qwen",
    )


class PromptCollabRequest(BaseModel):
    text: str = Field(..., description="原始中文文案")
    style: str = Field(..., description="风格描述，例如 二次元插画/电商海报")
    planner_llm_name: Optional[str] = Field(
        default=None,
        description="第一阶段生成基础提示词的模型名称（llm_config.json 中的 key）",
    )
    reviewer_llm_name: Optional[str] = Field(
        default=None,
        description="第二阶段审稿/优化提示词的模型名称（llm_config.json 中的 key，可为空，默认与 planner 相同）",
    )


class GenerateCollabRequest(BaseModel):
    text: str = Field(..., description="原始中文文案")
    style: str = Field(..., description="风格描述，例如 二次元插画/电商海报")
    count: int = Field(1, ge=1, le=8, description="需要生成的图片数量（1-8）")
    planner_llm_name: Optional[str] = Field(
        default=None,
        description="第一阶段生成基础提示词的模型名称（llm_config.json 中的 key）",
    )
    reviewer_llm_name: Optional[str] = Field(
        default=None,
        description="第二阶段审稿/优化提示词的模型名称（llm_config.json 中的 key，可为空，默认与 planner 相同）",
    )

"""
1. 初始化数据库
2. 加载工作流模板
3. 注入提示词到工作流
4. 调用 ComfyUI 生成图片
5. 入库
6. 返回图片信息
"""
#@app.on_event("startup")：
#FastAPI 启动时会调用 on_startup。
#里面的 init_db() 会创建数据库表（如果还不存在）。
# 控制 worker 是否退出（用于优雅关闭，本 demo 暂不实现）
_worker_running = True


async def _worker_loop() -> None:
    """后台 worker：轮询 PENDING 任务，逐个执行生成并更新状态。"""
    while _worker_running:
        try:
            with Session(engine) as session:
                stmt = (
                    select(GenerationTask)
                    .where(GenerationTask.status == TaskStatus.PENDING)
                    .order_by(GenerationTask.id.asc())
                    .limit(1)
                )
                task = session.exec(stmt).first()
                if task is None:
                    await asyncio.sleep(2.0)
                    continue

                # 标记为运行中
                task.status = TaskStatus.RUNNING
                task.updated_at = datetime.utcnow()
                session.add(task)
                session.commit()
                session.refresh(task)

            # 1) 根据任务配置决定提示词来源：直接拼接 / 单模型 LLM / 多模型协同
            try:
                if getattr(task, "use_collab", False):
                    collab_result = await generate_image_prompt_collab(
                        text=task.text,
                        style=task.style,
                        planner_llm_name=getattr(task, "planner_llm_name", None),
                        reviewer_llm_name=getattr(task, "reviewer_llm_name", None),
                    )
                    prompt_text = collab_result.get("final_prompt") or collab_result.get(
                        "base_prompt"
                    )
                    if not prompt_text:
                        raise RuntimeError("协同模式下未生成有效提示词")
                elif getattr(task, "use_llm", False):
                    prompt_text = await generate_image_prompt(
                        text=task.text,
                        style=task.style,
                        llm_name=getattr(task, "llm_name", None),
                    )
                else:
                    prompt_text = f"{task.style}, {task.text}".strip(", ")

                start_ts = time.perf_counter()

                # 2) 调用 ComfyUI 出图
                workflow = _load_workflow_template()
                workflow = _inject_prompt_into_workflow(workflow, prompt_text, task.count)
                results = await _call_comfyui(workflow, task.count)
                duration_ms = int((time.perf_counter() - start_ts) * 1000)

                # 3) 写入记录与更新任务状态
                with Session(engine) as session:
                    record = GenerationRecord(
                        text=task.text,
                        style=task.style,
                        prompt_text=prompt_text,
                        image_urls=json.dumps([url for url, _ in results], ensure_ascii=False),
                        md5_list=json.dumps([md5 for _, md5 in results]),
                        duration_ms=duration_ms,
                    )
                    session.add(record)
                    session.commit()
                    session.refresh(record)

                    t = session.get(GenerationTask, task.id)
                    if t:
                        t.status = TaskStatus.SUCCESS
                        t.record_id = record.id
                        t.error_message = None
                        t.updated_at = datetime.utcnow()
                        session.add(t)
                        session.commit()
            except Exception as e:
                with Session(engine) as session:
                    t = session.get(GenerationTask, task.id)
                    if t:
                        t.status = TaskStatus.FAILED
                        t.error_message = str(e)
                        t.updated_at = datetime.utcnow()
                        session.add(t)
                        session.commit()
        except Exception:
            await asyncio.sleep(2.0)
        else:
            await asyncio.sleep(0.5)


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    asyncio.create_task(_worker_loop())

"""
加载工作流模板：
1. 检查工作流模板文件是否存在
2. 如果存在，则加载工作流模板
3. 如果不存在，则抛出 FileNotFoundError
"""
def _load_workflow_template() -> Dict[str, Any]:
    if not os.path.exists(WORKFLOW_PATH):
        raise FileNotFoundError(
            f"找不到工作流模板文件: {WORKFLOW_PATH}，"
            f"请先从 ComfyUI 导出 API 工作流为 JSON 并保存到该路径。"
        )
    import json

    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _inject_prompt_into_workflow(
    workflow: Dict[str, Any], prompt_text: str, count: int
) -> Dict[str, Any]:
    """
    一个非常简单的“泛用”逻辑：
    - 遍历所有节点
    - 如果节点的 inputs 里有 text 字段，并且是字符串，就替换为我们的 prompt_text
    - 如果有 batch_size 字段且为整数，就改成 count（一次性批量出多张）
    """
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and isinstance(inputs.get("text"), str):
            inputs["text"] = prompt_text
        # 高性能多图：把 batch_size 设为 count
        if isinstance(inputs, dict) and isinstance(inputs.get("batch_size"), int):
            inputs["batch_size"] = count
        node["inputs"] = inputs
    return workflow


async def _call_comfyui(prompt: Dict[str, Any], num_images: int) -> List[Tuple[str, str]]:
    """
    调用 ComfyUI:
    1. POST /prompt -> 拿到 prompt_id
    2. 轮询 GET /history/{prompt_id} -> 找到输出图片信息
    3. 拼出 /view?filename=... 的图片 URL
    4. 下载图片并计算 MD5
    """
    # 整体等待时间放宽到 5 分钟，避免大模型/大图生成超时
    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1. 提交工作流
        resp = await client.post(f"{COMFYUI_API_BASE}/prompt", json={"prompt": prompt})
        if resp.status_code != 200:
            raise RuntimeError(f"提交到 ComfyUI 失败，状态码 {resp.status_code}，响应：{resp.text}")
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("ComfyUI 返回结果中没有 prompt_id")

        # 2. 简单轮询 history
        history_url = f"{COMFYUI_API_BASE}/history/{prompt_id}"
        image_infos: List[Dict[str, Any]] = []
        start_wait = time.perf_counter()
        # 最长等待 300 秒，防止长时间生成被误判为失败
        while True:
            h_resp = await client.get(history_url)
            if h_resp.status_code != 200:
                raise RuntimeError(f"查询 ComfyUI history 失败，状态码 {h_resp.status_code}")
            h = h_resp.json()
            # history 结构：{ prompt_id: { "outputs": { node_id: { "images": [...] } } } }
            if prompt_id not in h:
                if time.perf_counter() - start_wait > 300:
                    raise RuntimeError("等待 ComfyUI 结果超时（>300s）")
                await asyncio.sleep(1.0)
                continue
            entry = h[prompt_id]
            outputs = entry.get("outputs", {})
            for node_id, node_output in outputs.items():
                images = node_output.get("images") or []
                for img in images:
                    image_infos.append(img)
            if image_infos:
                break

            if time.perf_counter() - start_wait > 300:
                raise RuntimeError("等待 ComfyUI 结果超时（>300s）")
            await asyncio.sleep(1.0)

        if not image_infos:
            raise RuntimeError("在 ComfyUI history 中没有找到生成的图片信息")

        # 最多返回 num_images 张，多余的丢弃
        selected = image_infos[:num_images]
        results: List[Tuple[str, str]] = []

        for info in selected:
            filename = info["filename"]
            subfolder = info.get("subfolder", "")
            image_url = (
                f"{COMFYUI_API_BASE}/view?filename={filename}"
                f"&subfolder={subfolder}&type=output"
            )

            # 下载图片并算 MD5
            img_resp = await client.get(image_url)
            if img_resp.status_code != 200:
                raise RuntimeError(f"下载图片失败，状态码 {img_resp.status_code}")
            md5 = hashlib.md5(img_resp.content).hexdigest()
            results.append((image_url, md5))

    return results


@app.get("/records")
def list_records(limit: int = 20):
    """
    查看最近的生成记录，默认返回最新 20 条。
    """
    with Session(engine) as session:
        statement = select(GenerationRecord).order_by(GenerationRecord.id.desc()).limit(
            limit
        )
        records = session.exec(statement).all()

    items = []
    for r in records:
        try:
            image_urls = json.loads(r.image_urls)
        except Exception:
            image_urls = []
        try:
            md5_list = json.loads(r.md5_list)
        except Exception:
            md5_list = []

        items.append(
            {
                "id": r.id,
                "text": r.text,
                "style": r.style,
                "prompt_text": r.prompt_text,
                "image_urls": image_urls,
                "md5_list": md5_list,
                "created_at": r.created_at,
                "duration_ms": r.duration_ms,
            }
        )

    return items


# ---------- 异步任务接口（v3 队列版） ----------


class CreateTaskRequest(BaseModel):
    text: str = Field(..., description="文案内容")
    style: str = Field(..., description="风格描述")
    count: int = Field(1, ge=1, le=8, description="生成图片数量")
    use_llm: bool = Field(
        default=False,
        description="是否在任务中使用单模型 LLM 提示词优化（等价于 /generate 的 use_llm）",
    )
    llm_name: Optional[str] = Field(
        default=None,
        description="单模型模式下使用的 llm 名称（llm_config.json 中的 key）",
    )
    use_collab: bool = Field(
        default=False,
        description="是否在任务中使用多模型协同提示词（等价于 /generate_collab 的逻辑）",
    )
    planner_llm_name: Optional[str] = Field(
        default=None,
        description="协同模式下第一阶段模型名称",
    )
    reviewer_llm_name: Optional[str] = Field(
        default=None,
        description="协同模式下第二阶段模型名称，默认与 planner 相同",
    )


@app.post("/tasks")
def create_task(req: CreateTaskRequest):
    """创建异步任务，立即返回 task_id。后台 worker 会排队执行。

    支持三种模式：
    - 纯 ComfyUI：use_llm/use_collab 均为 false（默认）；
    - 单模型 LLM 提示词优化：use_llm=true；
    - 多模型协同：use_collab=true（优先级高于 use_llm）。
    """
    with Session(engine) as session:
        task = GenerationTask(
            text=req.text,
            style=req.style,
            count=req.count,
            use_llm=req.use_llm,
            llm_name=req.llm_name,
            use_collab=req.use_collab,
            planner_llm_name=req.planner_llm_name,
            reviewer_llm_name=req.reviewer_llm_name,
        )
        session.add(task)
        session.commit()
        session.refresh(task)
    return {"task_id": task.id}


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    """查询任务状态。若 status=SUCCESS，会附带 record 中的图片链接和 MD5。"""
    with Session(engine) as session:
        task = session.get(GenerationTask, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")

        out = {
            "task_id": task.id,
            "text": task.text,
            "style": task.style,
            "count": task.count,
            "status": task.status.value,
            "error_message": task.error_message,
            "record_id": task.record_id,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
        }

        if task.status == TaskStatus.SUCCESS and task.record_id:
            record = session.get(GenerationRecord, task.record_id)
            if record:
                try:
                    image_urls = json.loads(record.image_urls)
                except Exception:
                    image_urls = []
                try:
                    md5_list = json.loads(record.md5_list)
                except Exception:
                    md5_list = []
                out["images"] = [
                    {"image_url": url, "md5": md5}
                    for url, md5 in zip(image_urls, md5_list)
                ]
                out["duration_ms"] = record.duration_ms

    return out


@app.get("/tasks")
def list_tasks(limit: int = 20, status: Optional[str] = None):
    """列出最近的任务。可选按 status 筛选（PENDING/RUNNING/SUCCESS/FAILED）。"""
    with Session(engine) as session:
        stmt = select(GenerationTask)
        if status:
            try:
                stmt = stmt.where(GenerationTask.status == TaskStatus(status))
            except ValueError:
                pass
        stmt = stmt.order_by(GenerationTask.id.desc()).limit(limit)
        tasks = session.exec(stmt).all()

    items = []
    for t in tasks:
        items.append({
            "task_id": t.id,
            "text": t.text,
            "style": t.style,
            "count": t.count,
            "status": t.status.value,
            "record_id": t.record_id,
            "created_at": t.created_at,
        })
    return items


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    接收文案 text 和风格 style，组合成 prompt，调用 ComfyUI 生成图片。
    当 use_llm=true 时，会先通过 LangChain + 大模型对提示词进行优化，再注入工作流。
    """
    # 默认直接拼接风格 + 文案
    prompt_text = f"{req.style}, {req.text}".strip(", ")

    if req.use_llm:
        try:
            # 利用 LangChain + LLM 生成更细致的英文提示词
            optimized = await generate_image_prompt(
                text=req.text, style=req.style, llm_name=req.llm_name
            )
            if optimized:
                prompt_text = optimized
        except Exception as e:
            # 为了健壮性，LLM 失败时退回到简单拼接模式，而不是直接报错
            raise HTTPException(
                status_code=500,
                detail=f"使用 LLM 优化提示词失败: {e}",
            )
    start_ts = time.perf_counter()

    try:
        workflow = _load_workflow_template()
        workflow = _inject_prompt_into_workflow(workflow, prompt_text, req.count)
        results = await _call_comfyui(workflow, req.count)
        duration_ms = int((time.perf_counter() - start_ts) * 1000)

        # 入库
        with Session(engine) as session:
            record = GenerationRecord(
                text=req.text,
                style=req.style,
                prompt_text=prompt_text,
                image_urls=json.dumps([url for url, _ in results], ensure_ascii=False),
                md5_list=json.dumps([md5 for _, md5 in results]),
                duration_ms=duration_ms,
            )
            session.add(record)
            session.commit()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {e}")

    images = [
        GeneratedImage(image_url=url, md5=md5) for url, md5 in results
    ]
    return JSONResponse(content={"images": [img.model_dump() for img in images]})


@app.post("/llm/prompt_preview")
async def llm_prompt_preview(req: PromptPreviewRequest):
    """
    仅调用 LangChain + LLM，返回优化后的英文提示词，不触发 ComfyUI。
    可用于运营/算法同学在线调试 Prompt、对比不同版本的效果。
    """
    try:
        optimized = await generate_image_prompt(
            text=req.text, style=req.style, llm_name=req.llm_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成优化提示词失败: {e}")

    return {
        "text": req.text,
        "style": req.style,
        "optimized_prompt": optimized,
    }


@app.post("/llm/prompt_collab")
async def llm_prompt_collab(req: PromptCollabRequest):
    """
    多模型协同生成提示词：
    - planner_llm 负责从文案+风格生成基础英文提示词；
    - reviewer_llm 负责在基础提示词上做审稿和强化。

    适合在 JD 场景下展示“多大模型协同工作流编排”的能力。
    """
    try:
        result = await generate_image_prompt_collab(
            text=req.text,
            style=req.style,
            planner_llm_name=req.planner_llm_name,
            reviewer_llm_name=req.reviewer_llm_name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"多模型协同生成提示词失败: {e}"
        )

    return {
        "text": req.text,
        "style": req.style,
        "planner_llm_name": req.planner_llm_name,
        "reviewer_llm_name": req.reviewer_llm_name or req.planner_llm_name,
        **result,
    }


@app.post("/generate_collab", response_model=GenerateResponse)
async def generate_collab(req: GenerateCollabRequest):
    """
    多模型协同版生成接口：
    1. 先调用 generate_image_prompt_collab，使用两个 LLM 协同产出 base_prompt 和 final_prompt；
    2. 使用 final_prompt 注入 ComfyUI 工作流生成图片；
    3. 将 final_prompt 写入数据库的 prompt_text 字段，便于后续分析与复盘。
    """
    # 第一步：多模型协同生成提示词
    try:
        collab_result = await generate_image_prompt_collab(
            text=req.text,
            style=req.style,
            planner_llm_name=req.planner_llm_name,
            reviewer_llm_name=req.reviewer_llm_name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"多模型协同生成提示词失败: {e}"
        )

    final_prompt = collab_result.get("final_prompt") or collab_result.get(
        "base_prompt"
    )
    if not final_prompt:
        raise HTTPException(status_code=500, detail="协同提示词结果为空")

    start_ts = time.perf_counter()

    # 第二步：调用 ComfyUI 出图
    try:
        workflow = _load_workflow_template()
        workflow = _inject_prompt_into_workflow(workflow, final_prompt, req.count)
        results = await _call_comfyui(workflow, req.count)
        duration_ms = int((time.perf_counter() - start_ts) * 1000)

        # 写入数据库（记录最终使用的提示词）
        with Session(engine) as session:
            record = GenerationRecord(
                text=req.text,
                style=req.style,
                prompt_text=final_prompt,
                image_urls=json.dumps([url for url, _ in results], ensure_ascii=False),
                md5_list=json.dumps([md5 for _, md5 in results]),
                duration_ms=duration_ms,
            )
            session.add(record)
            session.commit()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"协同生成图片失败: {e}")

    images = [
        GeneratedImage(image_url=url, md5=md5) for url, md5 in results
    ]
    return JSONResponse(content={"images": [img.model_dump() for img in images]})


@app.get("/health")
async def health_check():
    return {"status": "ok"}

