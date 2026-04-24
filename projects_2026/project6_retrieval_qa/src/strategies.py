"""
strategies.py
=============
Prompt-building functions for the four retrieval strategies:

  1. No Retrieval       — LLM answers from memory only
  2. Always Retrieval   — LLM always receives retrieved context
  3. Adaptive (Oracle)  — Uses gold param_knowledge_answerable label
  4. TA-ARE             — Time-Aware Adaptive REtrieval (Zhang et al. 2024)
                          The LLM itself decides whether to retrieve,
                          guided by today's date + 2 yes / 2 no examples.
                          This is the paper's main proposed method.

"""


def build_prompt_no_retrieval(question):
    """
    Strategy 1 — No Retrieval.
    The LLM answers purely from its parametric (training-time) memory.
    No external context is provided.
    """
    return (
        f"Answer the following question in a few words.\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_prompt_always_retrieval(question, contexts):
    """
    Strategy 2 — Always Retrieval (Core RAG Method).
    Retrieved passages are always prepended to the prompt before
    the LLM generates an answer.

    Context passages are capped at 200 characters each to stay
    within the flan-t5-base 512-token input limit.
    """
    context_str = "\n".join([
        f"- {ctx[:200]}"
        for ctx in contexts
        if ctx.strip()
    ])

    return (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_prompt_adaptive(question, contexts, needs_retrieval):
    """
    Strategy 3 — Adaptive Retrieval (Oracle / Simplified Version).
    Uses the gold param_knowledge_answerable label to decide whether
    to retrieve. This is NOT the paper's method — it is an upper-bound
    oracle that shows what perfect retrieval decisions would achieve.

    needs_retrieval == 0  ->  question requires external knowledge
    needs_retrieval == 1  ->  question is answerable from LLM memory
    """
    if needs_retrieval == 0:
        return build_prompt_always_retrieval(question, contexts)
    else:
        return build_prompt_no_retrieval(question)


# ── TA-ARE: Time-Aware Adaptive REtrieval ────────────────────────────────────

# Two YES and two NO in-context examples matching the paper's TA-ARE prompt
# (Table 4, Zhang et al. 2024). These teach the model to distinguish
# questions needing retrieval from those it can answer from memory.
_TAARE_EXAMPLES = """Examples:
Q: Who is the current Prime Minister of the United Kingdom?
Date hint: This changes over time — you may not know the latest.
Answer: Yes

Q: What is the name of the river that runs through Paris?
Date hint: This is a stable geographic fact known before your training.
Answer: No

Q: Who won the most recent FIFA World Cup?
Date hint: This changes every four years — check if you are confident.
Answer: Yes

Q: What is the chemical formula for water?
Date hint: This is a permanent scientific fact.
Answer: No"""


def build_prompt_taare_decision(question, anchor_date="January 2024"):
    """
    TA-ARE Step 1 — Retrieval Decision Prompt.

    Asks the LLM whether external retrieval is needed for this question,
    using three signals from the paper (Zhang et al. 2024, Section 4):
      (a) Today's date  — helps the model reason about temporal relevance
      (b) In-context examples — 2 YES and 2 NO demonstrations
      (c) Direct yes/no question — forces a binary retrieval decision

    The model should output 'yes' or 'no'. Any other output is treated
    as 'yes' (retrieve by default — conservative strategy).

    Args:
        question    (str): The question to evaluate
        anchor_date (str): Date string anchoring temporal reasoning
                           (paper uses the dataset collection date)

    Returns:
        str: The decision prompt to pass to generate_answer()
    """
    return (
        f"Today's date is {anchor_date}.\n\n"
        f"{_TAARE_EXAMPLES}\n\n"
        f"Now answer for this question:\n"
        f"Q: {question}\n"
        f"Date hint: Consider whether this question asks about recent events, "
        f"fast-changing facts, or long-tail knowledge you may not have.\n"
        f"Answer (yes or no):"
    )


def parse_taare_decision(raw_output):
    """
    Parse GPT-3.5's yes/no decision output.
    GPT-3.5 may respond with full sentences like:
      "Yes, I need retrieval..." or "No, I don't need..."
    So we find positions of yes/no and take whichever appears first.
    Defaults to True (retrieve) when ambiguous.
    """
    text = raw_output.lower().strip()

    yes_pos = text.find('yes')
    no_pos  = text.find('no')

    if yes_pos == -1 and no_pos == -1:
        return True    # ambiguous → retrieve by default

    if yes_pos == -1:
        return False   # only 'no' found

    if no_pos == -1:
        return True    # only 'yes' found

    # Both found — whichever comes FIRST wins
    return yes_pos < no_pos        # ambiguous → retrieve by default (conservative)


def build_prompt_taare(question, contexts, anchor_date="January 2024",
                       generate_fn=None):
    """
    TA-ARE Step 2 — Full Strategy Entry Point.

    Calls generate_fn() with the decision prompt to get a yes/no retrieval
    decision from the LLM itself (no oracle labels used). Then builds the
    appropriate QA prompt based on that decision.

    If generate_fn is None (e.g. during testing), defaults to retrieval.

    Args:
        question    (str):       The question to answer
        contexts    (list[str]): Pre-retrieved passages (used if retrieval chosen)
        anchor_date (str):       Date string for temporal reasoning
        generate_fn (callable):  The generate_answer() function from llm.py

    Returns:
        tuple:
            prompt       (str):  The final QA prompt to pass to generate_answer()
            did_retrieve (bool): Whether retrieval was triggered (for logging)
    """
    if generate_fn is None:
        return build_prompt_always_retrieval(question, contexts), True

    # Step 1 — ask the LLM whether retrieval is needed
    decision_prompt = build_prompt_taare_decision(question, anchor_date)
    raw_decision    = generate_fn(decision_prompt)
    did_retrieve    = parse_taare_decision(raw_decision)

    # Step 2 — build the QA prompt based on the LLM's own decision
    if did_retrieve:
        qa_prompt = build_prompt_always_retrieval(question, contexts)
    else:
        qa_prompt = build_prompt_no_retrieval(question)

    return qa_prompt, did_retrieve