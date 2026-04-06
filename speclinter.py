import os
import re
import ast
import json
import hashlib
import asyncio
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt

# ==========================================
# 1. 系統初始化與模型配置
# ==========================================
mcp = FastMCP("SpecLinter")

OLLAMA_URL   = os.environ.get("OLLAMA_URL")
IS_LOCAL     = bool(OLLAMA_URL)
HYBRID_MODE  = bool(os.environ.get("HYBRID_MODE"))  # 擾動用本地，Generate/Judge 用雲端

def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"必要環境變數未設定：{key}")
    return val


def _parse_json_arg(value: str, param_name: str) -> object:
    """
    相容 JSON（雙引號）與 Python literal（單引號）兩種格式。
    Agent 常吐出 Python 風格的單引號陣列，json.loads 會直接報錯。
    解析失敗時拋出明確的 ValueError，不靜默吞掉。
    """
    # 1st try: 標準 JSON
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    # 2nd try: Python literal（處理單引號格式）
    try:
        result = ast.literal_eval(value)
        # 確保結果型別合理（list 或 dict）
        if not isinstance(result, (list, dict)):
            raise ValueError
        return result
    except Exception:
        pass

    raise ValueError(
        f"參數 {param_name} 格式錯誤，無法解析為 JSON。\n"
        f"請使用雙引號 JSON 格式，例如：\n"
        f"  synonym_groups: [[\"用戶\", \"使用者\"], [\"訂單\", \"購買單\"]]\n"
        f"  distinct_terms: [\"會員\", \"訪客\"]\n"
        f"收到的值：{value[:100]}"
    )


if HYBRID_MODE:
    # 擾動走本地（免費），Generate/Judge 走雲端（高品質）
    _perturb_client = AsyncOpenAI(base_url=_require_env("OLLAMA_URL"), api_key="local-key")
    _cloud_client   = AsyncOpenAI(api_key=_require_env("OPENAI_API_KEY"))
    _local          = _require_env("LOCAL_MODEL")
    PERTURB_MODEL   = os.environ.get("PERTURB_MODEL",  _local)
    GENERATE_MODEL  = _require_env("GENERATE_MODEL")
    JUDGE_MODEL     = _require_env("JUDGE_MODEL")
    CTX_CHAR_LIMIT  = int(os.environ.get("CTX_CHAR_LIMIT", "80000"))
    _perturb_is_local = True
elif IS_LOCAL:
    _perturb_client = _cloud_client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="local-key")
    _local          = _require_env("LOCAL_MODEL")
    PERTURB_MODEL   = os.environ.get("PERTURB_MODEL",  _local)
    GENERATE_MODEL  = os.environ.get("GENERATE_MODEL", _local)
    JUDGE_MODEL     = os.environ.get("JUDGE_MODEL",    _local)
    CTX_CHAR_LIMIT  = int(os.environ.get("CTX_CHAR_LIMIT", "6000"))
    _perturb_is_local = True
else:
    _perturb_client = _cloud_client = AsyncOpenAI(api_key=_require_env("OPENAI_API_KEY"))
    PERTURB_MODEL   = _require_env("PERTURB_MODEL")
    GENERATE_MODEL  = _require_env("GENERATE_MODEL")
    JUDGE_MODEL     = _require_env("JUDGE_MODEL")
    CTX_CHAR_LIMIT  = int(os.environ.get("CTX_CHAR_LIMIT", "80000"))
    _perturb_is_local = False

# IS_LOCAL 且非 HYBRID：generate/judge 也走 Ollama（_cloud_client 此時即 Ollama client）
# HYBRID 或純雲端：generate/judge 走 OpenAI
_cloud_is_local = IS_LOCAL and not HYBRID_MODE

# 溫度
PERTURB_TEMPERATURE = float(os.environ.get("PERTURB_TEMPERATURE", "0.7"))
JUDGE_TEMPERATURE   = float(os.environ.get("JUDGE_TEMPERATURE",   "0.0"))

# 規格長度上限
MAX_SPEC_CHARS = int(os.environ.get("MAX_SPEC_CHARS", "12000"))

# Retry 設定
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "3"))
RETRY_MIN_WAIT = int(os.environ.get("RETRY_MIN_WAIT", "2"))
RETRY_MAX_WAIT = int(os.environ.get("RETRY_MAX_WAIT", "10"))

# run_stress_test 預設值
DEFAULT_NUM_VERSIONS  = int(os.environ.get("DEFAULT_NUM_VERSIONS",  "3"))
DEFAULT_VARIATION     = os.environ.get("DEFAULT_VARIATION",          "moderate")
DEFAULT_PASS_THRESHOLD = float(os.environ.get("DEFAULT_PASS_THRESHOLD", "0.7"))
DEFAULT_TIMEOUT       = int(os.environ.get("DEFAULT_TIMEOUT",        "120"))

# 穩定度評分
SEMANTIC_WEIGHT  = float(os.environ.get("SEMANTIC_WEIGHT",  "0.3"))
EMBED_MODEL      = _require_env("EMBED_MODEL")   # 向量相似度模型（雲端用 text-embedding-*，本地用 nomic-embed-text 等）

# ==========================================
# 2. 定義符號化結構 (Neuro-symbolic 橋樑)
# ==========================================
class PerturbedSpecs(BaseModel):
    versions: List[str] = Field(description="改寫後的規格書版本列表")

class LogicFlow(BaseModel):
    endpoints: List[str] = Field(description="此規格涉及的 API 路徑或核心函式")
    database_operations: List[str] = Field(description="資料庫的讀寫順序與條件，需標註 table 名稱")
    error_handling_conditions: List[str] = Field(description="例外處理邏輯與邊界條件")
    assumptions: List[str] = Field(description="規格中未明確說明、但模型自行補充的隱含假設")
    schema_violations: List[str] = Field(description="操作中引用了但不存在於提供的 DB schema 的 table 或 column；若無提供 schema 則留空")
    api_violations: List[str] = Field(description="規格中引用了但不存在於提供的 API context 的端點或方法；若無提供 API context 則留空")

class SynonymCandidate(BaseModel):
    terms: List[str] = Field(description="可能被混用的術語列表")
    reason: str = Field(description="判斷為候選同義詞的理由（例如：在規格中交替出現、語義相近）")
    confidence: float = Field(description="判斷為真正同義詞的信心值，0.0=幾乎確定不同義，1.0=幾乎確定同義")
    recommendation: str = Field(description="建議：'synonym'（可互換）/ 'distinct'（語義不同）/ 'ask_user'（需人工確認）")

class SynonymReport(BaseModel):
    candidates: List[SynonymCandidate] = Field(description="候選同義詞組")
    domain_terms: List[str] = Field(description="規格中所有重要領域術語（供使用者參考完整詞彙表）")
    auto_synonym_groups: List[List[str]] = Field(description="信心值 >= 0.85 的同義詞組，Agent 可直接傳給 run_stress_test 的 synonym_groups")
    auto_distinct_terms: List[str] = Field(description="信心值 <= 0.15 的術語（確定不同義），Agent 可直接傳給 run_stress_test 的 distinct_terms")
    needs_human_review: List[str] = Field(description="信心值介於中間、建議人工確認的術語組描述")

class GWTScenario(BaseModel):
    given: str = Field(description="前置條件：系統或使用者的初始狀態")
    when: str  = Field(description="觸發事件：使用者的操作或系統事件")
    then: str  = Field(description="預期結果：系統應有的明確行為，需包含具體欄位或 API 名稱")

class Ambiguity(BaseModel):
    point: str = Field(description="歧義點的簡短描述")
    original_quote: str = Field(description="【必須一字不漏】從原始規格中直接擷取造成歧義的完整句子，不得改寫或摘要")
    flaw_type: str = Field(description="缺陷類型：'Missing Logic'（缺少必要條件）/ 'Contradiction'（自相矛盾）/ 'Vague Terminology'（術語模糊）")
    suggestions: List[GWTScenario] = Field(description="1-2 個消除歧義的 Given/When/Then 場景，Then 必須明確到可直接寫成測試")

class JudgeOutput(BaseModel):
    reasoning_process: str = Field(
        description=(
            "【必須先填寫此欄位，再填寫其他欄位】"
            "逐步推導各版本邏輯的差異：先比對 endpoints，再比對 database_operations，"
            "再比對 error_handling_conditions，最後深挖 assumptions 的矛盾。"
            "找出每個分歧點對應的原始規格句子。完成推導後再輸出 ambiguities。"
        )
    )
    semantic_stability_score: float = Field(
        description=(
            "語義一致性分數，0.0-1.0。"
            "基於上方 reasoning_process 的推導結果評分，忽略措辭差異，"
            "只評估業務意圖、邊界條件、操作順序的一致程度。"
        )
    )
    ambiguities: List[Ambiguity] = Field(description="所有發現的歧義點，按嚴重程度排序")
    schema_violation_summary: List[str] = Field(description="各版本中幻覺或錯誤引用的 schema 元素彙整；格式：'table.column — 責任歸屬'")
    api_violation_summary: List[str] = Field(description="各版本中幻覺或錯誤引用的 API 端點彙整；格式：'METHOD /path — 責任歸屬'")
    assumption_conflicts: List[str] = Field(description="各版本假設差異最大的前 3 點")

class LintResult(BaseModel):
    run_id: str                  # 格式：{spec_hash}-{timestamp}，用於追蹤與快取
    spec_hash: str               # spec_text 的 SHA256 前 8 碼
    timestamp: str               # ISO 8601 UTC
    analyzed_versions: int       # 實際成功分析的版本數
    failed_versions: int         # 因錯誤略過的版本數
    vector_score: float          # 詞彙向量餘弦相似度（embedding 跨版本一致性）
    semantic_score: float        # AI 語義一致性評估（理解自然語言差異）
    stability_score: float       # 加權合併：(1-SEMANTIC_WEIGHT)*vector + SEMANTIC_WEIGHT*semantic
    pass_threshold: float
    passed: bool
    ambiguities: List[Ambiguity]
    schema_violations: List[str]
    api_violations: List[str]
    assumption_conflicts: List[str]
    suggestions: List[str]       # 從所有 ambiguities 攤平的改寫建議，方便 Agent 直接引用
    raw_report: str              # Markdown 格式摘要，供人類閱讀

# ==========================================
# 3. 通用 structured output 解析器 (雲端/本地通用)
# ==========================================
def _extract_json_str(content: str) -> str:
    """從 Markdown 代碼塊或純文字中提取 JSON 字串（Ollama 常把 JSON 包在 ```json ... ```）"""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
    if match:
        return match.group(1)
    match = re.search(r"(\{[\s\S]+\})", content)
    if match:
        return match.group(1)
    return content


def _parse_structured(response, model_cls: type[BaseModel], fallback_split: str = None):
    """
    優先使用 .parsed (OpenAI native)；
    若為 None (Ollama 或 plain create) 則嘗試手動解析，
    包含 Markdown 代碼塊剝離與純文字 fallback。
    """
    msg = response.choices[0].message

    # 1st try: native parsed (OpenAI beta.parse)
    if getattr(msg, "parsed", None) is not None:
        return msg.parsed

    content = msg.content or ""

    # 2nd try: 直接 JSON parse（含 markdown 剝離）
    for candidate in [content, _extract_json_str(content)]:
        try:
            return model_cls.model_validate_json(candidate)
        except Exception:
            pass

    # 3rd try: 純文字 fallback (僅 PerturbedSpecs 使用)
    if fallback_split and model_cls is PerturbedSpecs:
        parts = [v.strip() for v in content.split(fallback_split) if v.strip()]
        if parts:
            return PerturbedSpecs(versions=parts)

    raise ValueError(f"無法解析模型輸出為 {model_cls.__name__}，原始內容：{content[:200]}")


# ==========================================
# 4. 向量穩定度計算（不依賴 LLM 自評）
# ==========================================
async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """批次取得文字的 embedding 向量"""
    resp = await _cloud_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def _compute_stability(results: list[dict]) -> float:
    """
    對每個欄位，將各版本的項目串接成文件後取 embedding，
    計算成對餘弦相似度的加權平均作為跨版本向量一致性分數。
    assumptions 權重最高，因為假設分歧是最關鍵的歧義訊號。
    """
    weights = {
        "endpoints":                0.20,
        "database_operations":      0.25,
        "error_handling_conditions": 0.20,
        "assumptions":              0.35,
    }
    score = 0.0
    for field, weight in weights.items():
        docs = [
            " ".join(r["logic_flow"].get(field, [])) or "_empty_"
            for r in results
        ]
        vectors = await _embed_texts(docs)
        pairs, total = 0, 0.0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total += _cosine(vectors[i], vectors[j])
                pairs += 1
        field_score = total / pairs if pairs else 1.0
        score += field_score * weight
    return round(score, 4)


# ==========================================
# 5. Ollama 相容 API wrapper
# ==========================================
async def _api_parse(
    model: str,
    messages: list,
    response_format: type[BaseModel],
    fallback_split: str = None,
    use_local: bool = False,
    **kwargs,
) -> BaseModel:
    """
    Ollama-safe structured output wrapper。
    use_local=True  → 走 _perturb_client（Ollama 相容路徑）
    use_local=False → 走 _cloud_client（OpenAI native structured output）
    """
    if not use_local:
        response = await _cloud_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )
        return _parse_structured(response, response_format, fallback_split)

    # Ollama 路徑：不傳 seed（部分模型不支援）
    kwargs.pop("seed", None)

    try:
        response = await _perturb_client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return _parse_structured(response, response_format, fallback_split)
    except Exception:
        pass

    response = await _perturb_client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    return _parse_structured(response, response_format, fallback_split)


def _compress_logic(result: dict) -> dict:
    """
    Judge context 超限時，只保留高鑑別度欄位。
    database_operations 和 error_handling_conditions 在超限時犧牲，
    保留 assumptions（最關鍵）、violations、endpoints。
    """
    lf = result["logic_flow"]
    return {
        "id": result["id"],
        "logic_flow": {
            "endpoints":        lf.get("endpoints", []),
            "assumptions":      lf.get("assumptions", []),
            "schema_violations": lf.get("schema_violations", []),
            "api_violations":   lf.get("api_violations", []),
        },
    }


# ==========================================
# 6. 核心運算邏輯
# ==========================================
@retry(wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT), stop=stop_after_attempt(RETRY_ATTEMPTS))
async def perturb_spec(original_spec: str, num_versions: int, variation_style: str, db_schema: Optional[str] = None, api_context: Optional[str] = None, synonym_groups: Optional[str] = None, distinct_terms: Optional[str] = None) -> list[str]:
    """生成 N 個語義等價但措辭不同的規格版本"""

    style_instructions = {
        "light":      "措辭輕微調整，主要改變用詞選擇。",
        "moderate":   "句式、主被動語態大幅翻新，但保留原始結構感。",
        "aggressive": "完全重組段落順序與句式，盡可能讓表面看起來截然不同。"
    }
    style_hint = style_instructions.get(variation_style, style_instructions["moderate"])

    extra_constraints = ""
    if synonym_groups:
        groups = _parse_json_arg(synonym_groups, "synonym_groups")
        formatted = "、".join(f"[{' = '.join(g)}]" for g in groups)
        extra_constraints += f"\n- 以下術語組已由使用者確認為同義詞，改寫時可在組內自由替換：{formatted}"

    if distinct_terms:
        terms = _parse_json_arg(distinct_terms, "distinct_terms")
        formatted = "、".join(f"「{t}」" for t in terms)
        extra_constraints += f"\n- 以下術語雖然相似，但使用者確認語義不同，改寫時嚴禁互換或混用：{formatted}"

    if db_schema:
        extra_constraints += f"""
4. 以下是系統的 DB Schema，改寫時所有 table 名稱、欄位名稱必須保持原樣，嚴禁自行替換或省略：
{db_schema}
"""
    if api_context:
        idx = 5 if db_schema else 4
        extra_constraints += f"""
{idx}. 以下是系統現有的 API 定義，改寫時所有端點路徑、HTTP 方法、參數名稱必須保持原樣：
{api_context}
"""

    term_constraints = f"\n術語規範：{extra_constraints}" if extra_constraints else ""

    prompt = f"""
請將以下軟體規格書(SDD)改寫成 {num_versions} 個版本。
要求：
1. 語義必須 100% 等價。
2. {style_hint}
3. 嚴禁修改技術專有名詞（如 JWT、PostgreSQL、REST）。{term_constraints}

以 JSON 格式回傳，結構為 {{"versions": ["版本1內容", "版本2內容", ...]}}。

原始規格：
{original_spec}
"""

    parsed = await _api_parse(
        model=PERTURB_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=PerturbedSpecs,
        fallback_split="---",
        use_local=_perturb_is_local,
        temperature=PERTURB_TEMPERATURE,
    )
    return parsed.versions[:num_versions]


@retry(wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT), stop=stop_after_attempt(RETRY_ATTEMPTS))
async def generate_logic(spec: str, version_id: int, temperature: float, db_schema: Optional[str] = None, api_context: Optional[str] = None) -> dict:
    """根據規格抽取結構化邏輯，並揭露隱含假設"""

    # system message 放穩定內容（schema、API、角色指令），讓 N+1 次呼叫共用同一個 cached prefix
    system_parts = [
        "你是一個嚴謹的系統架構師。請根據使用者提供的規格，萃取出系統的核心運作邏輯。"
        "請忽略自然語言的修飾，專注於系統行為、資料流與例外處理。"
    ]
    extra_instructions = []

    if db_schema:
        system_parts.append(f"以下是系統的 DB Schema，請以此為唯一真實來源：\n{db_schema}")
        extra_instructions.append(
            "請在 database_operations 中標註每個操作實際對應的 table 名稱。"
            "請在 schema_violations 中列出規格描述中引用了但 schema 中不存在的 table 或 column。"
        )
    else:
        extra_instructions.append("schema_violations 欄位請留空陣列。")

    if api_context:
        system_parts.append(f"以下是系統現有的 API 定義，請以此為唯一真實來源：\n{api_context}")
        extra_instructions.append(
            "請在 endpoints 中只列出規格真正涉及的 API 路徑，並確認它們存在於上方 API 定義中。"
            "請在 api_violations 中列出規格描述中引用了但 API 定義中不存在的端點或方法。"
        )
    else:
        extra_instructions.append("api_violations 欄位請留空陣列。")

    system_parts.append(
        "特別注意：\n"
        "- 請在 assumptions 欄位中列出規格中「沒有明確說明、但你為了解讀規格而自行補充的假設」。這些假設是最有價值的歧義訊號。\n"
        "- " + "\n- ".join(extra_instructions) + "\n"
        "以 JSON 格式回傳，包含 endpoints、database_operations、error_handling_conditions、assumptions、schema_violations、api_violations 六個欄位。"
    )

    parsed = await _api_parse(
        model=GENERATE_MODEL,
        messages=[
            {"role": "system", "content": "\n\n".join(system_parts)},  # 穩定前綴，跨 N+1 次呼叫被 cache
            {"role": "user", "content": f"規格：\n{spec}"},
        ],
        response_format=LogicFlow,
        use_local=_cloud_is_local,
        temperature=temperature,
        seed=version_id,
    )
    return {
        "id": version_id,
        "spec": spec,
        "logic_flow": parsed.model_dump(),
        "has_schema": db_schema is not None,
        "has_api_context": api_context is not None
    }


@retry(wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT), stop=stop_after_attempt(RETRY_ATTEMPTS))
async def judge_consensus(original_spec: str, results: list[dict], db_schema: Optional[str] = None, api_context: Optional[str] = None) -> JudgeOutput:
    """AI 裁判比對多份 JSON 邏輯的符號一致性，回傳結構化審查結果"""

    # 先估算是否需要壓縮，再依結果決定 prompt 內容，避免 prompt 與資料矛盾
    logic_size = sum(len(json.dumps(r["logic_flow"])) for r in results)
    is_compressed = (len(original_spec) + logic_size) > CTX_CHAR_LIMIT

    effective_results = [_compress_logic(r) for r in results] if is_compressed else results
    label = "抽取邏輯 (壓縮，僅含高鑑別度欄位)" if is_compressed else "抽取邏輯"
    logic_texts = "\n\n".join([
        f"### 版本 {r['id']} {label}:\n{json.dumps(r['logic_flow'], ensure_ascii=False, indent=2)}"
        for r in effective_results
    ])

    # system message 放穩定內容（角色、schema、API），user message 放每次不同的 logic texts
    system_parts = [
        "你是一個嚴格的邏輯審查員。"
        "你會收到多個 AI 根據同一份規格書的不同措辭版本所萃取出的「結構化系統邏輯 (JSON)」，"
        "請比對它們的一致性，找出歧義點與假設衝突。"
    ]

    # extra_checks 與實際提供的欄位保持一致，避免模型對不存在的欄位產生幻覺
    if is_compressed:
        extra_checks = (
            "分析重點（注意：本次因 context 限制僅提供 endpoints、assumptions 及 violations 欄位，"
            "請勿要求或推測 database_operations、error_handling_conditions）：\n"
            "- endpoints 是否有分歧？\n"
            "- assumptions（隱含假設）是否不同？不同版本對同一規格做出不同假設，代表原文存在語義漏洞。"
        )
    else:
        extra_checks = (
            "分析重點：\n"
            "- endpoints 是否有分歧？\n"
            "- database_operations 的操作順序或鎖定條件是否有差異？\n"
            "- error_handling_conditions 是否有遺漏或衝突？\n"
            "- assumptions（隱含假設）是否不同？不同版本對同一規格做出不同假設，代表原文存在語義漏洞。"
        )

    if db_schema:
        system_parts.append(f"DB Schema（唯一真實來源）：\n{db_schema}")
        extra_checks += "\n- schema_violation_summary：彙整各版本幻覺或錯誤引用的 table/column，格式為 'table.column — 責任歸屬（規格錯誤/模型幻覺）'"

    if api_context:
        system_parts.append(f"API 定義（唯一真實來源）：\n{api_context}")
        extra_checks += "\n- api_violation_summary：彙整各版本幻覺或錯誤引用的 API 端點，格式為 'METHOD /path — 責任歸屬（規格錯誤/模型幻覺）'"

    system_parts.append(
        extra_checks + "\n\n"
        "回傳 JSON，包含：\n"
        "- semantic_stability_score（0.0-1.0）：忽略措辭差異，純粹評估各版本在業務意圖、"
        "邊界條件、操作順序上的語義一致程度。"
        "注意：措辭不同但意圖相同應給高分；假設或行為有實質分歧才扣分。\n"
        "- ambiguities、schema_violation_summary、api_violation_summary、assumption_conflicts"
    )

    system_text = "\n\n".join(system_parts)
    user_content = f"原始規格：\n{original_spec}\n\n{logic_texts}"

    return await _api_parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_text},  # 穩定前綴，跨多次執行被 cache
            {"role": "user", "content": user_content},
        ],
        response_format=JudgeOutput,
        use_local=_cloud_is_local,
        temperature=JUDGE_TEMPERATURE,
    )


def _build_lint_result(
    judge: JudgeOutput,
    vector_score: float,
    pass_threshold: float,
    run_id: str,
    spec_hash: str,
    timestamp: str,
    analyzed_versions: int,
    failed_versions: int,
) -> LintResult:
    """將 JudgeOutput 轉換為 LintResult，並生成 Markdown raw_report"""
    semantic_score = max(0.0, min(1.0, judge.semantic_stability_score))
    stability_score = round(
        (1 - SEMANTIC_WEIGHT) * vector_score + SEMANTIC_WEIGHT * semantic_score, 4
    )

    # suggestions 攤平成可讀字串（Given/When/Then）供 Agent 快速引用
    suggestions = [
        f"Given {s.given} / When {s.when} / Then {s.then}"
        for a in judge.ambiguities
        for s in a.suggestions
    ]

    lines = [
        f"## SpecLinter 報告",
        f"",
        f"**Run ID**：`{run_id}`　**分析版本**：{analyzed_versions}"
        + (f"　**略過（錯誤）**：{failed_versions}" if failed_versions else ""),
        f"",
        f"**穩定度分數**：{stability_score * 100:.1f}% "
        f"({'✓ 通過' if stability_score >= pass_threshold else '✗ 未通過'}，門檻 {pass_threshold * 100:.0f}%)"
        f"　向量 {vector_score * 100:.1f}% / 語義 {semantic_score * 100:.1f}%",
        f"",
    ]

    if judge.ambiguities:
        lines += ["### 歧義點分析", ""]
        for a in judge.ambiguities:
            lines += [
                f"**[{a.flaw_type}] {a.point}**",
                f"> 原文：「{a.original_quote}」",
                "",
            ]
            for s in a.suggestions:
                lines += [
                    f"- **Given** {s.given}",
                    f"  **When** {s.when}",
                    f"  **Then** {s.then}",
                    "",
                ]

    if judge.assumption_conflicts:
        lines += ["### 假設衝突（前 3 點）", ""]
        lines += [f"- {c}" for c in judge.assumption_conflicts]
        lines.append("")

    if judge.schema_violation_summary:
        lines += ["### Schema 問題", ""]
        lines += [f"- {v}" for v in judge.schema_violation_summary]
        lines.append("")

    if judge.api_violation_summary:
        lines += ["### API 問題", ""]
        lines += [f"- {v}" for v in judge.api_violation_summary]
        lines.append("")

    return LintResult(
        run_id=run_id,
        spec_hash=spec_hash,
        timestamp=timestamp,
        analyzed_versions=analyzed_versions,
        failed_versions=failed_versions,
        vector_score=vector_score,
        semantic_score=semantic_score,
        stability_score=stability_score,
        pass_threshold=pass_threshold,
        passed=stability_score >= pass_threshold,
        ambiguities=judge.ambiguities,
        schema_violations=judge.schema_violation_summary,
        api_violations=judge.api_violation_summary,
        assumption_conflicts=judge.assumption_conflicts,
        suggestions=suggestions,
        raw_report="\n".join(lines),
    )


# ==========================================
# 7. MCP 工具端點定義
# ==========================================
@mcp.tool()
async def extract_synonyms(spec_text: str) -> str:
    """
    從規格書中找出可能被混用的術語，供使用者確認後傳給 run_stress_test。
    使用時機：在呼叫 run_stress_test 之前，先確認哪些術語是真正的同義詞、哪些語義不同。

    回傳 JSON，包含：
    - candidates：候選同義詞組（terms + reason），請使用者逐一確認
    - domain_terms：規格中所有重要領域術語

    確認後的使用方式：
    - synonym_groups：確認為同義的術語組，傳給 run_stress_test（擾動時可互換）
    - distinct_terms：確認為不同義的術語，傳給 run_stress_test（擾動時嚴禁互換）
    """
    prompt = f"""
你是一個術語分析專家。請分析以下規格書，找出可能被混用或誤認為同義的術語。

重點關注：
- 在規格中交替出現、指涉對象可能相同的詞（例如「用戶」與「使用者」）
- 語義相近但在特定領域可能有明確區別的詞（例如「訂單」與「購買紀錄」）
- 縮寫與全名（例如「DB」與「資料庫」）

對每組候選術語，請評估：
- confidence：0.0-1.0，判斷它們為真正同義詞的信心值
- recommendation：'synonym'（confidence >= 0.85，可直接視為同義）/ 'distinct'（confidence <= 0.15，確定不同義）/ 'ask_user'（其餘，需人工確認）

並根據上述判斷，分別填入：
- auto_synonym_groups：recommendation='synonym' 的術語組（Agent 可直接使用）
- auto_distinct_terms：recommendation='distinct' 的術語（Agent 可直接使用）
- needs_human_review：recommendation='ask_user' 的術語組描述（需人工確認）

規格：
{spec_text}
"""
    parsed = await _api_parse(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=SynonymReport,
        temperature=0.0,
        use_local=_cloud_is_local,
    )
    return parsed.model_dump_json(indent=2)


@mcp.tool()
async def run_stress_test(
    spec_text: str,
    num_versions: int = DEFAULT_NUM_VERSIONS,
    generation_temperature: float = 0.0,
    variation_style: str = DEFAULT_VARIATION,
    db_schema: Optional[str] = None,
    api_context: Optional[str] = None,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    timeout_seconds: int = DEFAULT_TIMEOUT,
    synonym_groups: Optional[str] = None,
    distinct_terms: Optional[str] = None,
) -> str:
    """
    對傳入的軟體規格書 (SDD) 執行語義變質壓測，返回 JSON 格式的 LintResult。
    使用時機：在 AI Agent 開始撰寫程式碼之前，驗證人類需求的邏輯穩定度。
    當 passed=false 時，Agent 應停止開發，將 ambiguities 與 suggestions 回報給人類。

    建議先呼叫 extract_synonyms 確認術語後再呼叫此工具。

    參數：
    - spec_text: 規格書內容
    - num_versions: 擾動版本數量（建議 3-5，預設 3）
    - generation_temperature: 邏輯抽取溫度（0.0 = 確定性最高，預設 0.0）
    - variation_style: 擾動強度，可選 'light' / 'moderate' / 'aggressive'
    - db_schema: （選填）DB Schema，可貼入 CREATE TABLE DDL 或 JSON Schema
    - api_context: （選填）現有 API 定義，可貼入 OpenAPI/Swagger YAML、JSON 或純文字端點列表
    - pass_threshold: 通過門檻，0.0-1.0（預設 0.7）；stability_score 低於此值時 passed=false
    - timeout_seconds: 整個流程的超時秒數（預設 120）；Agent 框架超時前會先收到明確錯誤
    - synonym_groups: （選填）JSON 字串，確認為同義的術語組，擾動時可互換。
                      格式：[["用戶", "使用者"], ["訂單", "購買單"]]
    - distinct_terms: （選填）JSON 字串，確認語義不同、嚴禁互換的術語列表。
                      格式：["會員", "訪客", "管理員"]
    """
    if not 2 <= num_versions <= 7:
        raise ValueError("參數錯誤：num_versions 請設定在 2 到 7 之間。")
    if not 0.0 <= generation_temperature <= 1.0:
        raise ValueError("參數錯誤：generation_temperature 請設定在 0.0 到 1.0 之間。")
    if variation_style not in ("light", "moderate", "aggressive"):
        raise ValueError("參數錯誤：variation_style 請選擇 'light'、'moderate' 或 'aggressive'。")
    if not 0.0 <= pass_threshold <= 1.0:
        raise ValueError("參數錯誤：pass_threshold 請設定在 0.0 到 1.0 之間。")
    if len(spec_text) > MAX_SPEC_CHARS:
        raise ValueError(
            f"規格書過長（{len(spec_text):,} 字元），請控制在 {MAX_SPEC_CHARS:,} 字元以內。"
        )

    # 版本追蹤資訊
    spec_hash = hashlib.sha256(spec_text.encode()).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{spec_hash}-{timestamp}"

    async def _run() -> str:
        # 1. 擾動階段
        versions = await perturb_spec(
            spec_text, num_versions, variation_style, db_schema, api_context,
            synonym_groups=synonym_groups, distinct_terms=distinct_terms,
        )

        # 2. 平行邏輯抽取（版本 0 = 原始規格，作為 baseline）
        all_specs = [spec_text] + versions
        tasks = [
            generate_logic(spec, i, generation_temperature, db_schema, api_context)
            for i, spec in enumerate(all_specs)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 部分失敗容錯：只要 >= 2 個版本成功即可繼續
        generation_results = [r for r in raw_results if not isinstance(r, Exception)]
        failed_count = len(raw_results) - len(generation_results)

        if len(generation_results) < 2:
            errors = [str(r) for r in raw_results if isinstance(r, Exception)]
            raise RuntimeError(
                f"成功抽取的版本數不足（{len(generation_results)}/{len(all_specs)}），"
                f"無法進行比對。錯誤：{errors}"
            )

        # 3. 向量相似度分數（embedding 跨版本一致性）
        vector_score = await _compute_stability(generation_results)

        # 4. 裁判比對（含語義分數）→ 結構化結果
        judge_output = await judge_consensus(spec_text, generation_results, db_schema, api_context)
        result = _build_lint_result(
            judge=judge_output,
            vector_score=vector_score,
            pass_threshold=pass_threshold,
            run_id=run_id,
            spec_hash=spec_hash,
            timestamp=timestamp,
            analyzed_versions=len(generation_results),
            failed_versions=failed_count,
        )
        return result.model_dump_json(indent=2)

    try:
        return await asyncio.wait_for(_run(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise RuntimeError(f"SpecLinter 執行逾時（{timeout_seconds}s），請增加 timeout_seconds 或縮短規格長度。")
    except Exception as e:
        raise RuntimeError(f"SpecLinter 壓測執行失敗 [{type(e).__name__}]: {str(e)}")


def _cli_entry():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="speclinter",
        description="SpecLinter：對規格書進行語義壓測，找出歧義與隱性假設。",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- lint 子命令 ---
    lint_p = subparsers.add_parser("lint", help="對規格書執行壓測")
    lint_p.add_argument("spec", nargs="?", help="規格文字（省略則從 stdin 讀取）")
    lint_p.add_argument("--db-schema", metavar="FILE", help="DB schema 檔案路徑")
    lint_p.add_argument("--api-context", metavar="FILE", help="API context 檔案路徑")
    lint_p.add_argument("--versions", type=int, default=DEFAULT_NUM_VERSIONS, metavar="N",
                        help=f"擾動版本數（預設 {DEFAULT_NUM_VERSIONS}）")
    lint_p.add_argument("--variation", default=DEFAULT_VARIATION,
                        choices=["light", "moderate", "aggressive"],
                        help=f"擾動強度（預設 {DEFAULT_VARIATION}）")
    lint_p.add_argument("--threshold", type=float, default=DEFAULT_PASS_THRESHOLD, metavar="T",
                        help=f"通過門檻 0~1（預設 {DEFAULT_PASS_THRESHOLD}）")
    lint_p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, metavar="S",
                        help=f"逾時秒數（預設 {DEFAULT_TIMEOUT}）")
    lint_p.add_argument("--json", action="store_true", help="輸出原始 JSON（不格式化）")

    # --- synonyms 子命令 ---
    syn_p = subparsers.add_parser("synonyms", help="預檢規格書中的同義詞")
    syn_p.add_argument("spec", nargs="?", help="規格文字（省略則從 stdin 讀取）")
    syn_p.add_argument("--json", action="store_true", help="輸出原始 JSON")

    # --- 無子命令：啟動 MCP server ---
    args = parser.parse_args()

    if args.command is None:
        mcp.run()
        sys.exit(0)

    # 讀取規格文字
    def _read_spec(args_spec):
        if args_spec:
            return args_spec
        if not sys.stdin.isatty():
            return sys.stdin.read()
        parser.error("請提供規格文字（引數或 stdin）")

    def _read_file(path):
        if not path:
            return None
        with open(path, encoding="utf-8") as f:
            return f.read()

    async def _cli_main():
        if args.command == "synonyms":
            spec = _read_spec(args.spec)
            output = await extract_synonyms(spec_text=spec)
            if args.json:
                print(output)
            else:
                data = json.loads(output)
                print("=== 同義詞預檢 ===")
                for group in data.get("auto_synonym_groups", []):
                    print(f"  同義詞組：{' / '.join(group)}")
                for term in data.get("auto_distinct_terms", []):
                    print(f"  明確不同：{term}")
                review = data.get("needs_human_review", [])
                if review:
                    print(f"  需人工確認：{', '.join(review)}")

        elif args.command == "lint":
            spec      = _read_spec(args.spec)
            db_schema = _read_file(args.db_schema)
            api_ctx   = _read_file(args.api_context)

            output = await run_stress_test(
                spec_text=spec,
                db_schema=db_schema or "",
                api_context=api_ctx or "",
                num_versions=args.versions,
                variation_style=args.variation,
                pass_threshold=args.threshold,
                timeout_seconds=args.timeout,
            )

            if args.json:
                print(output)
                return

            data = json.loads(output)
            status = "PASS ✓" if data["passed"] else "FAIL ✗"
            print(f"\n=== SpecLinter 結果：{status} ===")
            print(f"穩定度：{data['stability_score']:.0%}  "
                  f"（向量 {data['vector_score']:.0%} / "
                  f"語義 {data['semantic_score']:.0%}）  "
                  f"門檻：{data['pass_threshold']:.0%}")
            print(f"分析版本：{data['analyzed_versions']}  失敗：{data['failed_versions']}")

            if data["ambiguities"]:
                print("\n歧義點：")
                for a in data["ambiguities"]:
                    print(f"  [{a['flaw_type']}] {a['point']}")
                    print(f"    原文：「{a['original_quote']}」")
                    for s in a.get("suggestions", []):
                        print(f"    Given {s['given']}")
                        print(f"    When  {s['when']}")
                        print(f"    Then  {s['then']}")

            if data["assumption_conflicts"]:
                print("\n假設衝突：")
                for c in data["assumption_conflicts"]:
                    print(f"  • {c}")

            if data["schema_violations"]:
                print("\nSchema 違規：")
                for v in data["schema_violations"]:
                    print(f"  • {v}")

            if data["api_violations"]:
                print("\nAPI 違規：")
                for v in data["api_violations"]:
                    print(f"  • {v}")

            if not data["passed"]:
                sys.exit(1)

    asyncio.run(_cli_main())


if __name__ == "__main__":
    _cli_entry()
