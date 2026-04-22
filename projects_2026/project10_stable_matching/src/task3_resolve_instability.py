import time
import pandas as pd
from langchain_groq import ChatGroq
from src.task1_stable_matching import process_model_response
from src.config import get_basic_model, get_reasoning_model
from src.data_utils import (
    load_matching_csv,
    extract_matching_from_response,
    prepare_instance,
    check_validity,
    check_stability,
    exact_match_with_ground_truth,
)
from src.task2_detect_instability import create_unstable_matching_from_ground_truth




# =========================================
# 3.1 prepare unstable matching for task 3
# =========================================

def prepare_task3_instance(row):
    men_prefs, women_prefs, men_text, women_text, ground_truth = prepare_instance(row)

    expected_size = len(men_prefs)

    unstable_dict, unstable_ok, unstable_msg = create_unstable_matching_from_ground_truth(
        ground_truth,
        expected_size
    )

    return {
        "men_prefs": men_prefs,
        "women_prefs": women_prefs,
        "men_text": men_text,
        "women_text": women_text,
        "ground_truth": ground_truth,
        "expected_size": expected_size,
        "unstable_matching_dict": unstable_dict,
        "unstable_ok": unstable_ok,
        "unstable_msg": unstable_msg
    }



# =========================================
# 3.2 build task 3 prompt
# =========================================

def build_task3_prompt(men_text, women_text, unstable_matching_dict, expected_size):
    unstable_json_lines = []
    for i in range(1, expected_size + 1):
        man = f"M{i}"
        woman = unstable_matching_dict[man]
        unstable_json_lines.append(f'"{man}": "{woman}"')

    unstable_json_text = "{\n" + ",\n".join(unstable_json_lines) + "\n}"

    prompt = f"""
You are an intelligent assistant who is an expert in algorithms. Consider the following instance of the two-sided matching problem, where men are to be matched with women.

Here are the preference lists for all individuals:

<preferences>
{{
M: {{
{men_text}
}},
W: {{
{women_text}
}}
}}
</preferences>

Here is an unstable matching:

<answer>
{unstable_json_text}
</answer>

Your task is to modify the given unstable matching to make it equivalent to the proposer-optimal stable matching.

Once you have found a stable matching, return your final matching in the JSON format below inside <answer></answer> tags:

<answer>
{{
"M1": "<woman matched with M1>",
"M2": "<woman matched with M2>",
...
}}
</answer>

Make sure each man is matched with exactly one woman and each woman is matched with exactly one man.
"""
    return prompt



# =========================================
# 3.3 basic model wrapper
# =========================================

def task3_basic_model(csv_file, num_instances=10, start_index=0, temperature=0, num_examples_to_show=2):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_retries=30
    )

    df = load_matching_csv(csv_file)
    end_index = min(start_index + num_instances, len(df))

    detailed_results = []
    summary_stats = {
        "total_instances": 0,
        "parsed_count": 0,
        "valid_count": 0,
        "stable_count": 0,
        "exact_match_count": 0,
        "total_blocking_pairs": 0
    }

    for idx in range(start_index, end_index):
        row = df.iloc[idx]
        instance = prepare_task3_instance(row)

        if not instance["unstable_ok"]:
            continue

        prompt = build_task3_prompt(
            instance["men_text"],
            instance["women_text"],
            instance["unstable_matching_dict"],
            instance["expected_size"]
        )

        response = llm.invoke(prompt)
        time.sleep(9.0)
        raw_output = response.content

        result = process_model_response(
            raw_output,
            instance["men_prefs"],
            instance["women_prefs"],
            instance["ground_truth"],
            expected_size=instance["expected_size"],
            show_output=((idx - start_index) < num_examples_to_show)
        )

        eval_result = result["evaluation"]

        summary_stats["total_instances"] += 1

        if result["parsed_ok"]:
            summary_stats["parsed_count"] += 1

        if eval_result["is_valid"]:
            summary_stats["valid_count"] += 1

        if eval_result["is_stable"]:
            summary_stats["stable_count"] += 1

        if eval_result["exact_match"]:
            summary_stats["exact_match_count"] += 1

        summary_stats["total_blocking_pairs"] += len(eval_result["blocking_pairs"])

        detailed_results.append({
            "instance_idx": idx,
            "raw_response": raw_output,
            "parsed_matching": result["parsed_matching"],
            "evaluation": eval_result
        })

    avg_blocking_pairs = (
        summary_stats["total_blocking_pairs"] / summary_stats["total_instances"]
        if summary_stats["total_instances"] > 0 else 0.0
    )

    summary = {
        "total_instances": summary_stats["total_instances"],
        "parsed_count": summary_stats["parsed_count"],
        "valid_count": summary_stats["valid_count"],
        "stable_count": summary_stats["stable_count"],
        "exact_match_count": summary_stats["exact_match_count"],
        "avg_blocking_pairs": round(avg_blocking_pairs, 2)
    }

    return detailed_results, summary





# =========================================
# 3.4 reasoning model wrapper
# =========================================

def task3_reasoning_model(csv_file, num_instances=10, start_index=0, temperature=0, num_examples_to_show=2):
    llm_reasoning = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_retries=30
    )

    df = load_matching_csv(csv_file)
    end_index = min(start_index + num_instances, len(df))

    detailed_results = []
    summary_stats = {
        "total_instances": 0,
        "parsed_count": 0,
        "valid_count": 0,
        "stable_count": 0,
        "exact_match_count": 0,
        "total_blocking_pairs": 0
    }

    for idx in range(start_index, end_index):
        row = df.iloc[idx]
        instance = prepare_task3_instance(row)

        if not instance["unstable_ok"]:
            continue

        prompt = build_task3_prompt(
            instance["men_text"],
            instance["women_text"],
            instance["unstable_matching_dict"],
            instance["expected_size"]
        )

        response = llm_reasoning.invoke(prompt)
        time.sleep(9.0)
        raw_output = response.content

        result = process_model_response(
            raw_output,
            instance["men_prefs"],
            instance["women_prefs"],
            instance["ground_truth"],
            expected_size=instance["expected_size"],
            show_output=((idx - start_index) < num_examples_to_show)
        )

        eval_result = result["evaluation"]

        summary_stats["total_instances"] += 1

        if result["parsed_ok"]:
            summary_stats["parsed_count"] += 1

        if eval_result["is_valid"]:
            summary_stats["valid_count"] += 1

        if eval_result["is_stable"]:
            summary_stats["stable_count"] += 1

        if eval_result["exact_match"]:
            summary_stats["exact_match_count"] += 1

        summary_stats["total_blocking_pairs"] += len(eval_result["blocking_pairs"])

        detailed_results.append({
            "instance_idx": idx,
            "raw_response": raw_output,
            "parsed_matching": result["parsed_matching"],
            "evaluation": eval_result
        })

    avg_blocking_pairs = (
        summary_stats["total_blocking_pairs"] / summary_stats["total_instances"]
        if summary_stats["total_instances"] > 0 else 0.0
    )

    summary = {
        "total_instances": summary_stats["total_instances"],
        "parsed_count": summary_stats["parsed_count"],
        "valid_count": summary_stats["valid_count"],
        "stable_count": summary_stats["stable_count"],
        "exact_match_count": summary_stats["exact_match_count"],
        "avg_blocking_pairs": round(avg_blocking_pairs, 2)
    }

    return detailed_results, summary



