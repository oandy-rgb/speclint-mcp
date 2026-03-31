🛡️ SpecLinter MCP
Stop letting AI Agents "hallucinate" your business logic.

SpecLinter is a Model Context Protocol (MCP) server that acts as a Semantic Linter for your Software Design Documents (SDD). It stress-tests your requirements by generating semantic perturbations and detecting logic divergence before a single line of code is written.

❌ The Problem
You give a vague prompt to an AI Agent (like Claude or GPT-4). It "guesses" the edge cases, writes 500 lines of code, and you spend 2 hours debugging its "assumptions."

✅ The Solution (SpecLinter)
Perturb: Rewrites your spec into 3 different semantic styles.

Extract: Flattens them into structured logic rules (JSON).

Judge: Detects where the logic conflicts.

Report: Tells the Agent exactly where the spec is ambiguous.

🚀 Quick Start (For Claude Desktop / Cursor / Aider)
Install via uv:

Bash
uv pip install speclinter-mcp
Add to your MCP Config:

JSON
"mcpServers": {
  "speclinter": {
    "command": "python",
    "args": ["-m", "speclinter.server"],
    "env": {
      "OPENAI_API_KEY": "your-key-here"
    }
  }
}
📊 Example Output
Stability Score: 65/100
⚠️ Conflict found in "Refund Logic":

Version A assumed "Instant Refund".

Version B assumed "Pending Approval".

Source of Ambiguity: The sentence "Refunds are processed accordingly" in your spec.
