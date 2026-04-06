# SpecLinter MCP

> **在 AI Agent 開始寫程式之前，先驗證你的規格書夠不夠嚴謹。**

`SpecLinter` 是一個 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 伺服器，作為軟體設計文件（SDD）的**語義 Linter**。它不是幫你生成程式碼，而是在開發開始前，主動找出規格書裡隱藏的歧義——那些最終會讓 AI 寫出「技術正確但邏輯錯誤」的程式的根源。

---

## 問題：AI Agent 的猜謎遊戲

當你把一份模糊的規格丟給 AI Agent（Claude、GPT、Gemini 等）：

1. **Agent 自己腦補**：邏輯缺口由模型的隱含假設填補。
2. **靜默失敗**：它寫出 500 行技術上正確、但業務邏輯錯誤的程式碼。
3. **無盡 Debug**：你花幾小時追的 bug，根源只是規格書裡一句話的歧義。

問題不在 AI，在規格本身。

---

## 解法：語義變質壓測

SpecLinter 的核心思想來自**對抗性語義擾動（Adversarial Semantic Perturbation）**：

> 如果一份規格書的語義是清晰且唯一的，那麼無論用什麼措辭描述它，任何讀者都應該推導出完全相同的系統行為。

利用這個性質，SpecLinter 強迫 AI 自己證明規格夠嚴謹，再允許開發開始。

---

## 運作原理

```
原始規格
    │
    ▼
┌─────────────────────────────────────────────┐
│  Stage 1：擾動（Perturb）                    │
│  用不同措辭、句式、主被動語態改寫成 N 個版本   │
│  語義必須 100% 等價                           │
└───────────────────┬─────────────────────────┘
                    │  N 份語義等價但表面不同的規格
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 2：邏輯抽取（Generate）               │
│  對每個版本（含原版作為 baseline）平行萃取：  │
│  - endpoints / database_operations           │
│  - error_handling_conditions                 │
│  - assumptions（最關鍵：隱含假設）           │
│  - schema_violations / api_violations        │
└───────────────────┬─────────────────────────┘
                    │  N+1 份結構化 JSON
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 3a：向量穩定度評分                    │
│  對各欄位取 embedding，計算跨版本餘弦相似度   │
│  assumptions 權重 35%（歧義訊號最強）        │
│  → vector_score（客觀、可重現）              │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  Stage 3b：裁判分析（Judge）                 │
│  高推理模型比對 JSON 差異，找出：            │
│  - 造成語義分叉的原始句子                    │
│  - 幻覺引用的 table / API endpoint           │
│  - 消除歧義的 Given/When/Then 場景           │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
              LintResult (JSON)
         passed / ambiguities / suggestions
```

### 為什麼 assumptions 是關鍵？

當規格有歧義，模型不會報錯，而是**靜默地做假設**。同一句話在不同措辭版本下觸發不同假設，就代表原文存在語義漏洞。SpecLinter 把這些假設顯式化，讓人類在開發前決策。

---

## 功能

- **Neuro-symbolic 橋樑**：擾動是神經網路的；評分是 embedding 向量餘弦相似度，不依賴 LLM 自評
- **支援 Cloud / Local 雙引擎**：OpenAI 或 Ollama 本地模型
- **DB Schema & API Context**：注入真實 schema 和 API 定義，偵測幻覺欄位與不存在的端點
- **Prompt Caching**：schema/API 等穩定內容置於 system message，跨 N+1 次呼叫共用 cached prefix
- **容錯機制**：單一版本抽取失敗不影響整體，≥ 2 個版本成功即繼續
- **Context 自動壓縮**：本地模型 context 超限時自動保留高鑑別度欄位，並同步調整 prompt
- **GWT 改寫建議**：歧義點的修正建議直接輸出為 Given/When/Then 格式，可貼進測試框架
- **Agent 可直接決策**：`passed` 欄位讓 Agent 決定是否繼續開發，`suggestions` 直接回報給人類

---

## 安裝


```bash
pip install fastmcp openai pydantic tenacity
```

---

## 設定

**雲端（OpenAI）**
```bash
export OPENAI_API_KEY=sk-...

# 模型（必填，無預設值）
export PERTURB_MODEL=<model>   # 擾動：生成語義等價的改寫版本
export GENERATE_MODEL=<model>  # 邏輯抽取：萃取結構化系統邏輯
export JUDGE_MODEL=<model>     # 裁判：比對差異、找出歧義點
export EMBED_MODEL=<model>     # 向量評分：計算跨版本 embedding 相似度

# 溫度（選填）
export PERTURB_TEMPERATURE=0.7     # 擾動溫度，越高版本差異越大
export JUDGE_TEMPERATURE=0.0       # 裁判溫度，建議保持 0（確定性）

# Hybrid Routing（選填）
export HYBRID_MODE=1               # 擾動走本地 Ollama，Generate/Judge 走雲端
                                   # 啟用時須同時設定 OLLAMA_URL、LOCAL_MODEL、OPENAI_API_KEY

# 限制（選填）
export CTX_CHAR_LIMIT=80000        # judge prompt 字元上限，超過則自動壓縮
export MAX_SPEC_CHARS=12000        # 規格書字元上限

# Retry（選填）
export RETRY_ATTEMPTS=3
export RETRY_MIN_WAIT=2
export RETRY_MAX_WAIT=10

# run_stress_test 預設值（選填）
export DEFAULT_NUM_VERSIONS=3
export DEFAULT_VARIATION=moderate  # light / moderate / aggressive
export DEFAULT_PASS_THRESHOLD=0.7
export DEFAULT_TIMEOUT=120
```

**本地（Ollama，全本地）**
```bash
export OLLAMA_URL=http://localhost:11434/v1
export LOCAL_MODEL=<your-model>        # 必填，作為三個角色的基礎模型
export EMBED_MODEL=<embed-model>       # 必填，建議使用 nomic-embed-text 等 embedding 專用模型

# 各角色可獨立指定不同模型（選填）
export PERTURB_MODEL=<small-model>     # 擾動任務較簡單，可用小模型節省資源
export GENERATE_MODEL=<your-model>
export JUDGE_MODEL=<large-model>       # 裁判需要較強推理能力，建議用大模型

# 限制（選填，本地預設值較保守）
export CTX_CHAR_LIMIT=6000
export MAX_SPEC_CHARS=12000

# Retry（選填）
export RETRY_ATTEMPTS=3
export RETRY_MIN_WAIT=2
export RETRY_MAX_WAIT=10

# run_stress_test 預設值（選填）
export DEFAULT_NUM_VERSIONS=3
export DEFAULT_VARIATION=moderate
export DEFAULT_PASS_THRESHOLD=0.7
export DEFAULT_TIMEOUT=120
```

**Hybrid Routing（擾動本地、分析雲端）**
```bash
export HYBRID_MODE=1

# 本地端（擾動用）
export OLLAMA_URL=http://localhost:11434/v1
export LOCAL_MODEL=<small-local-model>   # 擾動只需語意重組，小模型即可

# 雲端（Generate/Judge/Embed 用，必填，無預設值）
export OPENAI_API_KEY=sk-...
export GENERATE_MODEL=<cloud-model>      # 邏輯抽取
export JUDGE_MODEL=<cloud-model>         # 裁判，建議用最強推理模型
export EMBED_MODEL=<cloud-embed-model>   # 向量評分
```

**MCP 設定（以 Claude Desktop 為例）**

```json
{
  "mcpServers": {
    "speclinter": {
      "command": "python",
      "args": ["/path/to/speclinter.py"]
    }
  }
}
```

---

## 使用方式

### CLI

```bash
# 壓測規格書
python speclinter.py lint "使用者登入後，系統應更新最後登入時間並記錄 IP。"

# 從檔案讀取 + 注入 schema
python speclinter.py lint --db-schema schema.sql --api-context api.txt < spec.txt

# 輸出原始 JSON（可接 jq）
python speclinter.py lint "..." --json | jq .passed

# 同義詞預檢
python speclinter.py synonyms "使用者登入後..."

# 調整參數
python speclinter.py lint "..." --versions 5 --variation aggressive --threshold 0.8
```

### MCP Tool

```python
await run_stress_test(
    spec_text="使用者登入後，系統應更新最後登入時間並記錄 IP。",

    # 選填：提供 context 可偵測幻覺引用
    db_schema="""
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            email TEXT NOT NULL,
            last_login_at TIMESTAMP,
            last_login_ip INET
        );
    """,
    api_context="""
        POST /auth/login
        POST /auth/logout
        GET  /users/{id}
    """,

    # 調整參數
    num_versions=3,              # 擾動版本數（2-7，預設 3）
    variation_style="moderate",  # light / moderate / aggressive
    pass_threshold=0.7,          # 穩定度門檻（預設 0.7）
    timeout_seconds=120,         # 整體逾時（預設 120s）
)
```

### 回傳結果（LintResult）

```json
{
  "run_id": "a1b2c3d4-20260406T120000Z",
  "spec_hash": "a1b2c3d4",
  "timestamp": "20260406T120000Z",
  "analyzed_versions": 4,
  "failed_versions": 0,
  "vector_score": 0.58,
  "semantic_score": 0.71,
  "stability_score": 0.62,
  "pass_threshold": 0.7,
  "passed": false,
  "ambiguities": [
    {
      "point": "更新時機不明",
      "original_quote": "系統應更新最後登入時間",
      "flaw_type": "Missing Logic",
      "suggestions": [
        {
          "given": "使用者提交正確的帳號密碼",
          "when": "JWT 簽發成功",
          "then": "系統應立即更新 users.last_login_at 並寫入 last_login_ip"
        }
      ]
    }
  ],
  "schema_violations": [],
  "api_violations": [],
  "assumption_conflicts": [
    "版本1假設登入失敗也應記錄 IP，版本2假設僅成功登入才記錄"
  ],
  "suggestions": [
    "Given 使用者提交正確的帳號密碼 / When JWT 簽發成功 / Then 系統應立即更新 users.last_login_at 並寫入 last_login_ip"
  ],
  "raw_report": "## SpecLinter 報告\n..."
}
```

### Agent 標準用法

```python
import json

result = json.loads(await run_stress_test(spec_text=spec))

if not result["passed"]:
    # 停止開發，把問題回報給人類
    lines = []
    for a in result["ambiguities"]:
        lines.append(f"- [{a['flaw_type']}] {a['point']}")
        lines.append(f"  原文：「{a['original_quote']}」")
        for s in a["suggestions"]:
            lines.append(f"  Given {s['given']}")
            lines.append(f"  When  {s['when']}")
            lines.append(f"  Then  {s['then']}")
    return (
        f"規格不穩定（穩定度 {result['stability_score']:.0%}），"
        f"請先釐清以下歧義：\n" + "\n".join(lines)
    )

# 通過 → 繼續開發
```

---

## 授權

MIT
