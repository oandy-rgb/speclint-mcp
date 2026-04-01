# 🛡️ SpecLint MCP

> **Stop letting AI Agents "hallucinate" your business logic. Validate your specs before you write a single line of code.**

`SpecLint` is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server designed to act as a **Semantic Linter** for Software Design Documents (SDD) and requirement specs. 

Instead of just generating code, it **stress-tests** your requirements by using adversarial semantic perturbations to find hidden ambiguities that usually lead to AI-generated bugs.

---

## ❌ The Problem: "The Guessing Game"
When you give a vague or ambiguous spec to an AI Agent (like Claude, GPT, or Gemini):
1.  **The Agent Guesses**: It fills the logical gaps with its own assumptions.
2.  **The "Silent Fail"**: It writes 500 lines of technically correct but logically wrong code.
3.  **The Debug Loop**: You spend hours fixing bugs that originated from a single ambiguous sentence in your spec.

## ✅ The Solution: SpecLint
SpecLint breaks the cycle by forcing the AI to prove the spec is robust **before** development starts.



### How it works:
1.  **Perturb**: Rewrites your spec into 3 distinct semantic styles (e.g., Concise, Legalistic, User-Centric) while maintaining 100% semantic equivalence.
2.  **Implement**: Simultaneously generates 3 independent "shadow implementations" (pseudocode) based on these variations.
3.  **Judge**: A high-reasoning "Judge" model compares the implementations. If they diverge, it means your original spec is ambiguous.
4.  **Report**: Pinpoints the exact sentence in your spec that caused the logic to fork.

---
