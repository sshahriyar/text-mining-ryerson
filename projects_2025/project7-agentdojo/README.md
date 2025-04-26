# AgentDojo Prompt Injection Benchmark Analysis

## Overview
This project investigates the security and utility performance of large language models (LLMs) when subjected to prompt injection attacks. We use the [AgentDojo](https://github.com/ethz-spylab/agentdojo) framework to evaluate how well the `tool_filter` defense mitigates the `tool_knowledge` attack across various task environments.

## Methodology
We replicate the benchmarking procedure from the NeurIPS 2024 AgentDojo paper using the `agentdojo` package. The evaluation involves:
- Model: `gpt-3.5-turbo-0125`
- Attack Strategy: `tool_knowledge` (injects malicious instructions)
- Defense Strategy: `tool_filter` (removes suspicious tool instructions)
- Suites: Workspace, Travel, Banking, Slack

Each suite includes:
- User tasks (benign)
- Injection tasks (malicious instructions)
- Evaluation metrics:
  -  **Targeted ASR (Attack Success Rate)** – how often the model is tricked into executing a malicious instruction.
  -  **Benign Utility** – how often the model succeeds in legitimate tasks without interference.

## Set Up
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. **Install dependencies**
   ```bash
   pip install agentdojo
3. **Set your OpenAI API key**
    ```python
    import os
    os.environ["OPENAI_API_KEY"] = "your key"
4. **Run the Benchmark**
    ```bash
    python -m agentdojo.scripts.benchmark \
    --model gpt-3.5-turbo-0125 \
    --defense tool_filter \
    --attack tool_knowledge
