Import os
import json
import asyncio
from typing import List
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

# ==========================================
# 1. 系統初始化與模型配置 (支援 Cloud/Local 雙引擎)
# ==========================================
mcp = FastMCP("SpecLinter")

OLLAMA_URL = os.environ.get("OLLAMA_URL")

if OLLAMA_URL:
    client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="local-key")
    LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:32b")
    PERTURB_MODEL = LOCAL_MODEL
    GENERATE_MODEL = LOCAL_MODEL
    JUDGE_MODEL = LOCAL_MODEL
else:
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    PERTURB_MODEL = "gpt-4o-mini"
    GENERATE_MODEL = "gpt-4o-mini"
    JUDGE_MODEL = "gpt-4o"

# ==========================================
# 2. 定義符號化結構 (Neuro-symbolic 橋樑)
# ==========================================
class PerturbedSpecs(BaseModel):
    versions: List[str] = Field(description="改寫後的規格書版本列表")

class LogicFlow(BaseModel):
    endpoints: List[str] = Field(description="此規格涉及的 API 路徑或核心函式")
    database_operations: List[str] = Field(description="資料庫的讀寫順序與條件")
    error_handling_conditions: List[str] = Field(description="例外處理邏輯與邊界條件")
    assumptions: List[str] = Field(description="規格中未明確說明、但模型自行補充的隱含假設")

# ==========================================
# 3. 通用 structured output 解析器 (雲端/本地通用)
# ==========================================
def _parse_structured<br_or_newline>(response, model_cls: type[BaseModel], fallback_split: str = None):
    """
    優先使用 .parsed (OpenAI native)；
    若為 None (部分 Ollama 模型) 則 fallback 到 model_validate_json；
    若仍失敗且有 fallback_split，嘗試切割純文字。
    """
    msg = response.choices[0].message

    # 1st try: native parsed
    if getattr(msg, "parsed", None) is not None:
        return msg.parsed

    # 2nd try: manual JSON parse
    try:
        return model_cls.model_validate_json(msg.content)
    except Exception:
        pass

    # 3rd try: 純文字 fallback (僅 PerturbedSpecs 使用)
    if fallback_split and model_cls is PerturbedSpecs:
        parts = [v.strip() for v in msg.content.split(fallback_split) if v.strip()]
        if parts:
            return PerturbedSpecs(versions=parts)

    raise ValueError(f"無法解析模型輸出為 {model_cls.__name__}，原始內容：{msg.content[:200]}")


# ==========================================
# 4. 核心運算邏輯
# ==========================================
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def perturb_spec(original_spec: str, num_versions: int, variation_style: str) -> list[str]:
    """生成 N 個語義等價但措辭不同的規格版本"""

    style_instructions = {
        "light":      "措辭輕微調整，主要改變用詞選擇。",
        "moderate":   "句式、主被動語態大幅翻新，但保留原始結構感。",
        "aggressive": "完全重組段落順序與句式，盡可能讓表面看起來截然不同。"
    }
    style_hint = style_instructions.get(variation_style, style_instructions["moderate"])

    prompt = f"""
請將以下軟體規格書(SDD)改寫成 {num_versions} 個版本。
要求：
1. 語義必須 100% 等價。
2. {style_hint}
3. 嚴禁修改技術專有名詞（如 JWT、PostgreSQL、REST）。

以 JSON 格式回傳，結構為 {{"versions": ["版本1內容", "版本2內容", ...]}}。

原始規格：
{original_spec}
"""

    response = await client.beta.chat.completions.parse(
        model=PERTURB_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=PerturbedSpecs,
        temperature=0.7
    )

    parsed = _parse_structured(response, PerturbedSpecs, fallback_split="---")
    return parsed.versions[:num_versions]


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def generate_logic(spec: str, version_id: int, temperature: float) -> dict:
    """根據規格抽取結構化邏輯，並揭露隱含假設"""

    prompt = f"""
你是一個嚴謹的系統架構師。請根據以下規格，萃取出系統的核心運作邏輯。
請忽略自然語言的修飾，專注於系統行為、資料流與例外處理。

特別注意：請在 assumptions 欄位中列出規格中「沒有明確說明、但你為了解讀規格而自行補充的假設」。
這些假設是最有價值的歧義訊號。

規格：
{spec}

以 JSON 格式回傳，包含 endpoints、database_operations、error_handling_conditions、assumptions 四個欄位。
"""

    response = await client.beta.chat.completions.parse(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=LogicFlow,
        temperature=temperature,
        seed=42
    )

    parsed = _parse_structured(response, LogicFlow)
    return {
        "id": version_id,
        "spec": spec,
        "logic_flow": parsed.model_dump()
    }


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def judge_consensus(original_spec: str, results: list[dict]) -> str:
    """AI 裁判比對多份 JSON 邏輯的符號一致性，重點關注隱含假設分歧"""

    logic_texts = "\n\n".join([
        f"### 版本 {r['id']} 抽取邏輯:\n{json.dumps(r['logic_flow'], ensure_ascii=False, indent=2)}"
        for r in results
    ])

    prompt = f"""
你是一個嚴格的邏輯審查員。以下是多個 AI 根據同一份規格書的不同措辭版本，所萃取出的「結構化系統邏輯 (JSON)」。

原始規格：
{original_spec}

{logic_texts}

請比對這些 JSON 結構的一致性，專注於：
- endpoints 是否有分歧？
- database_operations 的操作順序或鎖定條件是否有差異？
- error_handling_conditions 是否有遺漏或衝突？
- **assumptions（隱含假設）是否不同？** 這是最重要的歧義訊號，不同版本對同一規格做出不同假設，代表原文存在語義漏洞。

請輸出 Markdown 報告，包含：
1. **綜合穩定度分數** (0% - 100%)
2. **歧義點分析**：對每個不一致，精準指出原始規格中的哪一句話造成了語義分叉
3. **修補提案**：針對每個歧義點，提供 1~2 個消除歧義的改寫建議

格式範例：
| 歧義點 | 原始問題句 | 建議改寫 |
|--------|-----------|----------|
| ... | ... | ... |
"""

    response = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    return response.choices[0].message.content


# ==========================================
# 5. MCP 工具端點定義
# ==========================================
@mcp.tool()
async def run_stress_test(
    spec_text: str,
    num_versions: int = 3,
    generation_temperature: float = 0.0,
    variation_style: str = "moderate"
) -> str:
    """
    對傳入的軟體規格書 (SDD) 執行語義變質壓測，返回歧義分析報告。
    使用時機：在 AI Agent 開始撰寫程式碼之前，驗證人類需求的邏輯穩定度。

    參數：
    - spec_text: 規格書內容
    - num_versions: 擾動版本數量（建議 3-5，預設 3）
    - generation_temperature: 邏輯抽取溫度（0.0 = 確定性最高，預設 0.0）
    - variation_style: 擾動強度，可選 'light' / 'moderate' / 'aggressive'
    """
    if not 2 <= num_versions <= 7:
        raise ValueError("參數錯誤：num_versions 請設定在 2 到 7 之間。")
    if not 0.0 <= generation_temperature <= 1.0:
        raise ValueError("參數錯誤：generation_temperature 請設定在 0.0 到 1.0 之間。")
    if variation_style not in ("light", "moderate", "aggressive"):
        raise ValueError("參數錯誤：variation_style 請選擇 'light'、'moderate' 或 'aggressive'。")

    try:
        # 1. 擾動階段
        versions = await perturb_spec(spec_text, num_versions, variation_style)

        # 2. 平行邏輯抽取
        tasks = [
            generate_logic(ver, i + 1, generation_temperature)
            for i, ver in enumerate(versions)
        ]
        generation_results = await asyncio.gather(*tasks)

        # 3. 裁判比對
        report = await judge_consensus(spec_text, list(generation_results))
        return report

    except Exception as e:
        raise RuntimeError(f"SpecLinter 壓測執行失敗: {str(e)}")


if __name__ == "__main__":
    mcp.run()

這個呢
搜尋並思考
