"""
llm.py
======
Language model interface using OpenAI GPT-3.5-turbo API.

Replaces the local flan-t5-base model with GPT-3.5-turbo which:
  - Properly follows instruction prompts
  - Correctly handles TA-ARE yes/no decision prompts
  - Matches the model family used in the original paper (Zhang et al. 2024)
  - Costs ~$0.15 for the full 250-question experiment

"""

import openai
import time

# ── API Configuration ─────────────────────────────────────────────────────────

# Paste your OpenAI API key here
OPENAI_API_KEY = "*************"

# Model used throughout the project
# gpt-3.5-turbo matches the paper's model family and costs ~$0.15 for 500 calls
OPENAI_MODEL = "gpt-3.5-turbo"

# Global client — initialised when load_model() is called
_client = None


def load_model(model_name=OPENAI_MODEL):
    """
    Initialise the OpenAI client.

    Replaces the flan-t5-base loader. No weights are downloaded —
    the client just authenticates with the OpenAI API.

    Args:
        model_name (str): OpenAI model to use (default: gpt-3.5-turbo)

    Returns:
        tuple: (None, model_name) — keeps same return signature
               so existing notebook cells don't need to change
    """
    global _client

    print(f"Initialising OpenAI client...")
    print(f"  Model : {model_name}")

    _client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Quick connectivity test
    try:
        test = _client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
            temperature=0.0
        )
        print(f"  API test response : {test.choices[0].message.content.strip()}")
        print(f"  OpenAI client ready!")
    except Exception as e:
        print(f"  ERROR connecting to OpenAI API: {e}")
        raise

    return None, model_name


def generate_answer(prompt, max_tokens=100):
    """
    Generate an answer using GPT-3.5-turbo via the OpenAI API.

    Used for ALL generation tasks in this project:
      1. No Retrieval strategy — LLM answers from memory
      2. Always Retrieval strategy — LLM answers with context
      3. Oracle Adaptive strategy — same as above with oracle routing
      4. TA-ARE decision prompt — yes/no retrieval decision
      5. TA-ARE final answer — answer after decision

    temperature=0.0 ensures deterministic outputs, matching the
    paper's experimental setup (Zhang et al. 2024).

    Args:
        prompt     (str): The full prompt string to send to the model
        max_tokens (int): Maximum tokens in the response (default 100)

    Returns:
        str: The model's response text, stripped of whitespace
    """
    global _client

    if _client is None:
        raise RuntimeError("Call load_model() before generate_answer().")

    try:
        response = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    except openai.RateLimitError:
        print("  Rate limit hit — waiting 15 seconds then retrying...")
        time.sleep(15)
        return generate_answer(prompt, max_tokens)

    except openai.APIError as e:
        print(f"  OpenAI API error: {e}")
        return ""

    except Exception as e:
        print(f"  Unexpected error: {e}")
        return ""
