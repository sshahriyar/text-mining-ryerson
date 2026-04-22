"""
evaluation.py
=============
All evaluation logic for the RAG project:

  - normalize(), exact_match(), token_f1()       — metric functions
  - run_experiments()                             — runs 3 baseline strategies
  - run_taare_experiment()                        — runs TA-ARE (paper's method)
  - compute_scores()                              — EM + F1 for baselines
  - compute_taare_scores()                        — 4-strategy comparison table
  - compute_scores_by_source()                    — per data-source breakdown
  - compute_retrieval_accuracy()                  — TA-ARE retrieval decisions
  - error_analysis()                              — RAG helped vs failed
  - retrieval_decision_error_analysis()           — TA-ARE decision errors

"""

import re
import json
import string
import time
import pandas as pd


# ── Metric Functions ──────────────────────────────────────────────────────────

def normalize(text):
    """Lowercase and strip punctuation for fair string comparison."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()


def exact_match(prediction, gold_list):
    """1 if normalized prediction matches any normalized gold answer."""
    pred_norm = normalize(prediction)
    return int(any(normalize(g) == pred_norm for g in gold_list))


def token_f1(prediction, gold_list):
    """Token-level F1 maximised over all gold answers."""
    pred_tokens = normalize(prediction).split()
    best_f1     = 0.0
    for gold in gold_list:
        gold_tokens = normalize(gold).split()
        common      = set(pred_tokens) & set(gold_tokens)
        if not common:
            continue
        precision = len(common) / len(pred_tokens)
        recall    = len(common) / len(gold_tokens)
        f1        = 2 * precision * recall / (precision + recall)
        best_f1   = max(best_f1, f1)
    return best_f1


# ── Baseline Experiment Runner (Strategies 1–3) ───────────────────────────────

def run_experiments(sample_df, generate_answer_fn,
                    build_no_ret_fn, build_always_fn, build_adaptive_fn):
    """
    Run the three baseline strategies on every question in sample_df.

    Strategies:
      1. No Retrieval   — LLM answers from memory
      2. Always Retrieval — LLM gets pre-retrieved context
      3. Adaptive (Oracle) — uses gold param_knowledge_answerable label

    With GPT-3.5-turbo this takes ~5-10 minutes for 250 questions.

    Returns:
        list[dict]: One result dict per question
    """
    results = []
    print(f"Running 3 baseline strategies on {len(sample_df)} questions...")
    print(f"  Using GPT-3.5-turbo via OpenAI API")

    for i, (_, row) in enumerate(sample_df.iterrows()):

        question        = row['question']
        ground_truth    = list(row['ground_truth'])
        needs_retrieval = row['param_knowledge_answerable']
        data_source     = row['data_source']

        # Extract top-3 pre-retrieved context passages
        context_list = []
        for ctx_str in list(row['context'])[:3]:
            try:
                ctx  = json.loads(ctx_str)
                text = ctx.get('text', '').strip()
                if text:
                    context_list.append(text)
            except Exception:
                continue

        pred_no_ret   = generate_answer_fn(build_no_ret_fn(question))
        pred_always   = generate_answer_fn(build_always_fn(question, context_list))
        pred_adaptive = generate_answer_fn(
            build_adaptive_fn(question, context_list, needs_retrieval)
        )

        results.append({
            'question'        : question,
            'ground_truth'    : ground_truth,
            'needs_retrieval' : needs_retrieval,
            'data_source'     : data_source,
            'pred_no_ret'     : pred_no_ret,
            'pred_always'     : pred_always,
            'pred_adaptive'   : pred_adaptive,
        })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(sample_df)} done...")

    print(f"\nAll {len(results)} questions completed!")
    return results


# ── TA-ARE Experiment Runner (Strategy 4) ────────────────────────────────────

def run_taare_experiment(sample_df, generate_answer_fn,
                         build_prompt_taare_fn, anchor_date="January 2024"):
    """
    Run TA-ARE strategy (Zhang et al. 2024, Section 4) on sample_df.

    For each question GPT-3.5:
      1. Reads the date + 2 yes/2 no examples
      2. Decides yes/no whether to retrieve
      3. Answers with or without context based on its own decision

    No oracle labels used — genuine adaptive retrieval.
    With GPT-3.5-turbo this takes ~10-15 minutes for 250 questions
    (2 API calls per question).

    Returns:
        list[dict]: One result dict per question with retrieval decision logged
    """
    taare_results = []
    print(f"Running TA-ARE on {len(sample_df)} questions...")
    print(f"  Model       : GPT-3.5-turbo")
    print(f"  Anchor date : {anchor_date}")
    print(f"  Note        : 2 API calls per question (decision + answer)")

    for i, (_, row) in enumerate(sample_df.iterrows()):

        question        = row['question']
        ground_truth    = list(row['ground_truth'])
        needs_retrieval = row['param_knowledge_answerable']
        data_source     = row['data_source']

        context_list = []
        for ctx_str in list(row['context'])[:3]:
            try:
                ctx  = json.loads(ctx_str)
                text = ctx.get('text', '').strip()
                if text:
                    context_list.append(text)
            except Exception:
                continue

        # TA-ARE: GPT-3.5 makes its own retrieval decision
        qa_prompt, did_retrieve = build_prompt_taare_fn(
            question,
            context_list,
            anchor_date = anchor_date,
            generate_fn = generate_answer_fn
        )

        pred_taare = generate_answer_fn(qa_prompt)

        taare_results.append({
            'question'           : question,
            'ground_truth'       : ground_truth,
            'needs_retrieval'    : needs_retrieval,
            'data_source'        : data_source,
            'taare_did_retrieve' : did_retrieve,
            'pred_taare'         : pred_taare,
        })

        if (i + 1) % 10 == 0:
            retrieve_count = sum(1 for r in taare_results if r['taare_did_retrieve'])
            print(f"  Progress: {i+1}/{len(sample_df)} done "
                  f"(retrieved so far: {retrieve_count}/{i+1})")

    print(f"\nTA-ARE complete! {len(taare_results)} questions evaluated.")
    retrieve_total = sum(1 for r in taare_results if r['taare_did_retrieve'])
    print(f"  GPT-3.5 chose to retrieve : {retrieve_total}/{len(taare_results)} "
          f"({retrieve_total/len(taare_results)*100:.1f}%)")
    return taare_results


# ── Scoring Functions ─────────────────────────────────────────────────────────

def compute_scores(results):
    """
    Compute EM and Token F1 for strategies 1–3.
    Prints Overall / Retrieval-Needed / Parametric splits.
    Returns scored DataFrame.
    """
    for r in results:
        r['em_no_ret']   = exact_match(r['pred_no_ret'],   r['ground_truth'])
        r['em_always']   = exact_match(r['pred_always'],   r['ground_truth'])
        r['em_adaptive'] = exact_match(r['pred_adaptive'], r['ground_truth'])
        r['f1_no_ret']   = token_f1(r['pred_no_ret'],      r['ground_truth'])
        r['f1_always']   = token_f1(r['pred_always'],      r['ground_truth'])
        r['f1_adaptive'] = token_f1(r['pred_adaptive'],    r['ground_truth'])

    results_df = pd.DataFrame(results)

    print("=" * 55)
    print(f"OVERALL RESULTS ({len(results_df)} questions)")
    print("=" * 55)
    print(f"{'Strategy':<22} {'Exact Match':>12} {'Token F1':>10}")
    print("-" * 46)
    for col, label in [('no_ret',   'No Retrieval'),
                       ('always',   'Always Retrieval'),
                       ('adaptive', 'Adaptive (Oracle)')]:
        print(f"{label:<22} "
              f"{results_df[f'em_{col}'].mean():>12.3f} "
              f"{results_df[f'f1_{col}'].mean():>10.3f}")

    ret_df = results_df[results_df['needs_retrieval'] == 0]
    par_df = results_df[results_df['needs_retrieval'] == 1]

    for subset, label in [
        (ret_df, f"RETRIEVAL-NEEDED (n={len(ret_df)})"),
        (par_df, f"PARAMETRIC ONLY  (n={len(par_df)})")
    ]:
        print(f"\n{'='*55}")
        print(label)
        print("=" * 55)
        print(f"{'Strategy':<22} {'Exact Match':>12} {'Token F1':>10}")
        print("-" * 46)
        for col, name in [('no_ret',   'No Retrieval'),
                          ('always',   'Always Retrieval'),
                          ('adaptive', 'Adaptive (Oracle)')]:
            print(f"{name:<22} "
                  f"{subset[f'em_{col}'].mean():>12.3f} "
                  f"{subset[f'f1_{col}'].mean():>10.3f}")

    return results_df


def compute_taare_scores(results_df, taare_results):
    """
    Score TA-ARE and print a 4-strategy comparison table.
    Mirrors Table 1 from the original paper.

    Returns:
        pd.DataFrame: taare_df with em_taare and f1_taare columns
    """
    for r in taare_results:
        r['em_taare'] = exact_match(r['pred_taare'], r['ground_truth'])
        r['f1_taare'] = token_f1(r['pred_taare'],    r['ground_truth'])

    taare_df = pd.DataFrame(taare_results)

    print("\n" + "=" * 60)
    print("4-STRATEGY COMPARISON (mirrors Table 1, Zhang et al. 2024)")
    print("=" * 60)
    print(f"{'Strategy':<26} {'Exact Match':>12} {'Token F1':>10}")
    print("-" * 50)

    for col, label in [('no_ret',   'No Retrieval'),
                       ('always',   'Always Retrieval'),
                       ('adaptive', 'Adaptive (Oracle)')]:
        print(f"{label:<26} "
              f"{results_df[f'em_{col}'].mean():>12.3f} "
              f"{results_df[f'f1_{col}'].mean():>10.3f}")

    print(f"{'TA-ARE (paper method)':<26} "
          f"{taare_df['em_taare'].mean():>12.3f} "
          f"{taare_df['f1_taare'].mean():>10.3f}")

    ret_base  = results_df[results_df['needs_retrieval'] == 0]
    par_base  = results_df[results_df['needs_retrieval'] == 1]
    ret_taare = taare_df[taare_df['needs_retrieval'] == 0]
    par_taare = taare_df[taare_df['needs_retrieval'] == 1]

    for base_sub, ta_sub, label in [
        (ret_base, ret_taare, f"RETRIEVAL-NEEDED (n={len(ret_taare)})"),
        (par_base, par_taare, f"PARAMETRIC ONLY  (n={len(par_taare)})")
    ]:
        print(f"\n{'='*60}")
        print(label)
        print("=" * 60)
        print(f"{'Strategy':<26} {'Exact Match':>12} {'Token F1':>10}")
        print("-" * 50)
        for col, name in [('no_ret',   'No Retrieval'),
                          ('always',   'Always Retrieval'),
                          ('adaptive', 'Adaptive (Oracle)')]:
            print(f"{name:<26} "
                  f"{base_sub[f'em_{col}'].mean():>12.3f} "
                  f"{base_sub[f'f1_{col}'].mean():>10.3f}")
        print(f"{'TA-ARE (paper method)':<26} "
              f"{ta_sub['em_taare'].mean():>12.3f} "
              f"{ta_sub['f1_taare'].mean():>10.3f}")

    return taare_df


def compute_scores_by_source(results_df, taare_df=None):
    """
    Per-source EM breakdown for all strategies.
    Mirrors the per-source analysis in Table 1 of the paper.
    """
    print("\n" + "=" * 72)
    print("PER-SOURCE BREAKDOWN — Exact Match (mirrors Table 1, Zhang et al. 2024)")
    print("=" * 72)

    has_taare = taare_df is not None
    header = f"{'Source':<14} {'n':>4}  {'No Ret':>8}  {'Always':>8}  {'Oracle':>8}"
    if has_taare:
        header += f"  {'TA-ARE':>8}"
    print(header)
    print("-" * (72 if has_taare else 56))

    for source in sorted(results_df['data_source'].unique()):
        r_sub = results_df[results_df['data_source'] == source]
        row   = (f"{source:<14} {len(r_sub):>4}  "
                 f"{r_sub['em_no_ret'].mean():>8.3f}  "
                 f"{r_sub['em_always'].mean():>8.3f}  "
                 f"{r_sub['em_adaptive'].mean():>8.3f}")
        if has_taare:
            t_sub = taare_df[taare_df['data_source'] == source]
            row  += f"  {t_sub['em_taare'].mean():>8.3f}"
        print(row)

    print("-" * (72 if has_taare else 56))
    total_row = (f"{'TOTAL':<14} {len(results_df):>4}  "
                 f"{results_df['em_no_ret'].mean():>8.3f}  "
                 f"{results_df['em_always'].mean():>8.3f}  "
                 f"{results_df['em_adaptive'].mean():>8.3f}")
    if has_taare:
        total_row += f"  {taare_df['em_taare'].mean():>8.3f}"
    print(total_row)


def compute_retrieval_accuracy(taare_df):
    """
    Compute TA-ARE retrieval decision accuracy.

    This is the PRIMARY metric in the paper — how often did GPT-3.5
    correctly decide when to retrieve vs skip retrieval?

    Paper benchmarks:
      GPT-3.5 Vanilla prompting : ~49.3%
      GPT-3.5 TA-ARE            : ~86.3%

    Returns:
        float: Retrieval accuracy (0.0 – 1.0)
    """
    gold  = taare_df['needs_retrieval'] == 0
    model = taare_df['taare_did_retrieve'] == True

    correct = (gold == model)
    acc     = correct.mean()

    tp = int(( gold  &  model).sum())
    tn = int((~gold  & ~model).sum())
    fp = int((~gold  &  model).sum())
    fn = int(( gold  & ~model).sum())

    print("\n" + "=" * 55)
    print("TA-ARE RETRIEVAL ACCURACY (GPT-3.5-turbo)")
    print("=" * 55)
    print(f"  Overall retrieval accuracy   : {acc:.3f} ({acc*100:.1f}%)")
    print(f"\n  Confusion breakdown:")
    print(f"  ✓ Correctly retrieved   (TP) : {tp:>3}  (needed & retrieved)")
    print(f"  ✓ Correctly skipped     (TN) : {tn:>3}  (not needed & skipped)")
    print(f"  ✗ Unnecessary retrieval (FP) : {fp:>3}  (not needed but retrieved)")
    print(f"  ✗ Missed retrieval      (FN) : {fn:>3}  (needed but skipped)")
    print(f"\n  Paper benchmarks (GPT-3.5):")
    print(f"    Vanilla prompting : ~49.3%")
    print(f"    TA-ARE            : ~86.3%")
    print(f"  Our GPT-3.5 TA-ARE  : {acc*100:.1f}%")

    return acc


def error_analysis(results_df):
    """
    Qualitative error analysis — where RAG helped vs failed.
    """
    print("ERROR ANALYSIS — Selected Examples")
    print("=" * 60)

    rag_helps = results_df[
        (results_df['em_no_ret'] == 0) &
        (results_df['em_always'] == 1)
    ].head(3)

    print("\n[CASE 1] RAG HELPED — No Retrieval wrong, Always RAG correct")
    print("-" * 60)
    for _, r in rag_helps.iterrows():
        print(f"Question     : {r['question']}")
        print(f"Gold answer  : {r['ground_truth']}")
        print(f"No Retrieval : '{r['pred_no_ret']}'  ✗ WRONG")
        print(f"Always RAG   : '{r['pred_always']}'  ✓ CORRECT")
        print(f"Data source  : {r['data_source']}")
        print()

    rag_fails = results_df[
        (results_df['em_no_ret']   == 0) &
        (results_df['em_always']   == 0) &
        (results_df['needs_retrieval'] == 0)
    ].head(2)

    print("\n[CASE 2] RAG FAILED — Both strategies wrong")
    print("-" * 60)
    for _, r in rag_fails.iterrows():
        print(f"Question     : {r['question']}")
        print(f"Gold answer  : {r['ground_truth']}")
        print(f"No Retrieval : '{r['pred_no_ret']}'  ✗ WRONG")
        print(f"Always RAG   : '{r['pred_always']}'  ✗ WRONG")
        print(f"Data source  : {r['data_source']}")
        print()

    print("=" * 60)
    print("Summary (No Retrieval vs Always Retrieval):")
    print(f"  RAG helped (wrong -> correct) : "
          f"{len(results_df[(results_df['em_no_ret']==0)&(results_df['em_always']==1)])}")
    print(f"  RAG hurt   (correct -> wrong) : "
          f"{len(results_df[(results_df['em_no_ret']==1)&(results_df['em_always']==0)])}")
    print(f"  Both correct                  : "
          f"{len(results_df[(results_df['em_no_ret']==1)&(results_df['em_always']==1)])}")
    print(f"  Both wrong                    : "
          f"{len(results_df[(results_df['em_no_ret']==0)&(results_df['em_always']==0)])}")


def retrieval_decision_error_analysis(taare_df):
    """
    Analyse TA-ARE retrieval decision errors.
    Mirrors Figure 1 (bottom) and Figure 3 from Zhang et al. 2024.
    """
    print("\n" + "=" * 60)
    print("TA-ARE RETRIEVAL DECISION ERROR ANALYSIS (GPT-3.5-turbo)")
    print("(mirrors Figure 3, Zhang et al. 2024)")
    print("=" * 60)

    needs  = taare_df['needs_retrieval'] == 0
    did    = taare_df['taare_did_retrieve'] == True
    correct = taare_df['em_taare'] == 1

    categories = {
        "Retrieved + Correct answer"  : int(( did  &  correct).sum()),
        "Retrieved + Wrong answer"    : int(( did  & ~correct).sum()),
        "Skipped + Correct answer"    : int((~did  &  correct).sum()),
        "Missed retrieval (FN)"       : int(( needs & ~did).sum()),
        "Unnecessary retrieval (FP)"  : int((~needs &  did).sum()),
    }

    for label, count in categories.items():
        bar = "█" * min(count, 40)
        print(f"  {label:<38}: {count:>3}  {bar}")

    fn_examples = taare_df[needs & ~did].head(2)
    if len(fn_examples) > 0:
        print("\nExamples where GPT-3.5 SKIPPED retrieval but should have retrieved:")
        print("-" * 60)
        for _, r in fn_examples.iterrows():
            print(f"  Q    : {r['question']}")
            print(f"  Gold : {r['ground_truth']}")
            print(f"  Pred : '{r['pred_taare']}'")
            print()
