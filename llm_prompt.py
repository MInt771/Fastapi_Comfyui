import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


LLM_CONFIG_PATH = os.getenv("LLM_CONFIG_PATH", "llm_config.json")


@lru_cache(maxsize=1)
def _load_llm_config() -> Dict[str, Any]:
    """
    从外部 JSON 加载多模型配置：
    {
      "default": "deepseek",
      "providers": {
        "deepseek": {
          "api_base": "...",
          "api_key": "...",
          "model": "deepseek-chat",
          "temperature": 0.7
        },
        "qwen": {
          "api_base": "...",
          "api_key": "...",
          "model": "qwen-max",
          "temperature": 0.6
        }
      }
    }
    如果文件不存在，则退回到纯环境变量配置（兼容老用法）。
    """
    if not os.path.exists(LLM_CONFIG_PATH):
        return {}

    with open(LLM_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=16)
def _get_llm(llm_name: Optional[str] = None) -> ChatOpenAI:
    """
    按名称获取一个 LLM 实例：
    - 优先从 llm_config.json 中读取指定 llm_name 的配置；
    - 如果未指定，则使用配置里的 default；
    - 如果没有配置文件，则退回到环境变量 LLM_API_KEY / LLM_API_BASE 等。
    """
    cfg = _load_llm_config()
    providers = cfg.get("providers") or {}

    if llm_name is None:
        default_name = cfg.get("default")
        if default_name and default_name in providers:
            llm_name = default_name

    if llm_name and llm_name in providers:
        p = providers[llm_name] or {}
        model_name = p.get("model") or os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini")
        temperature = float(p.get("temperature", os.getenv("LLM_TEMPERATURE", "0.7")))
        api_key = p.get("api_key") or os.getenv("LLM_API_KEY") or os.getenv(
            "OPENAI_API_KEY"
        )
        base_url = p.get("api_base") or os.getenv("LLM_API_BASE") or os.getenv(
            "OPENAI_BASE_URL"
        )
    else:
        # 配置文件缺失或未命中指定名称时，使用环境变量兜底
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )


@lru_cache(maxsize=1)
def _get_prompt() -> ChatPromptTemplate:
    """
    针对“文案 + 风格 -> 稳定扩散提示词”的专用 Prompt。
    输出为一段适合 Stable Diffusion / ComfyUI 的英文提示词，包含画面主体、风格、构图等信息。
    """
    template = """
你是一名资深的 AIGC 提示词工程师，擅长把中文广告文案和风格描述转化为高质量的英文 Stable Diffusion 提示词。

要求：
1. 用英文输出最终的 positive prompt，一行内完成，适合图像生成（如 Stable Diffusion / SD3）。
2. 保留用户文案中的关键信息（人物/商品/场景/情绪）。
3. 根据风格描述补充画面细节，比如光照、镜头、构图、画风等。
4. 不要输出解释说明，只输出最终的英文提示词。

用户文案："{text}"
目标风格："{style}"
"""
    return ChatPromptTemplate.from_template(template)


@lru_cache(maxsize=1)
def _get_review_prompt() -> ChatPromptTemplate:
    """
    第二个模型作为“审稿人/协同模型”，在已有英文提示词的基础上做润色和强化。
    """
    template = """
You are an expert AIGC prompt reviewer.

You will receive:
- Original Chinese copy and style
- A first version of the English Stable Diffusion prompt

Your task:
1. Keep all key information (subject, scene, style, emotion).
2. Improve clarity, composition hints, lighting, camera angle, and details.
3. Keep the output as a single English positive prompt, suitable for SD/SD3/ComfyUI.
4. Do NOT add explanations, only output the final prompt.

Chinese text: "{text}"
Style: "{style}"
First English prompt: "{base_prompt}"
"""
    return ChatPromptTemplate.from_template(template)


async def generate_image_prompt(
    text: str, style: str, llm_name: Optional[str] = None
) -> str:
    """
    使用 LangChain + LLM，把简单的「文案 + 风格」优化为更适合 ComfyUI 的英文提示词。

    由于 FastAPI 端点已经是 async，这里直接使用异步调用。
    可通过 llm_name 指定使用的模型（如 "deepseek" / "qwen"），对应配置在 llm_config.json。
    """
    llm = _get_llm(llm_name=llm_name)
    prompt = _get_prompt()
    chain = prompt | llm

    resp = await chain.ainvoke({"text": text, "style": style})
    content = getattr(resp, "content", None) or str(resp)
    return content.strip()


async def generate_image_prompt_collab(
    text: str,
    style: str,
    planner_llm_name: Optional[str] = None,
    reviewer_llm_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    使用两个模型协同完成提示词：
    1. 模型 A（planner_llm_name）生成基础英文提示词；
    2. 模型 B（reviewer_llm_name）在基础提示词上做审稿和强化。

    返回一个 dict，包含：
    - base_prompt: 第一阶段输出
    - final_prompt: 第二阶段协同后的最终提示词
    """
    # 第一步：规划模型
    planner_llm = _get_llm(llm_name=planner_llm_name)
    base_chain = _get_prompt() | planner_llm
    base_resp = await base_chain.ainvoke({"text": text, "style": style})
    base_prompt = (getattr(base_resp, "content", None) or str(base_resp)).strip()

    # 第二步：审稿/协同模型（默认与 planner 相同）
    reviewer_llm = _get_llm(llm_name=reviewer_llm_name or planner_llm_name)
    review_chain = _get_review_prompt() | reviewer_llm
    final_resp = await review_chain.ainvoke(
        {"text": text, "style": style, "base_prompt": base_prompt}
    )
    final_prompt = (getattr(final_resp, "content", None) or str(final_resp)).strip()

    return {
        "base_prompt": base_prompt,
        "final_prompt": final_prompt,
    }

