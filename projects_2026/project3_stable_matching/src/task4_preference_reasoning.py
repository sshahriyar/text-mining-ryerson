import time
import re
import pandas as pd
from langchain_groq import ChatGroq

from src.config import get_basic_model, get_reasoning_model
from src.data_utils import (
    load_matching_csv,
    prepare_instance,
    convert_prefs_to_dict,
    prefers,
)

# =========================================
# 4.1 prepare task 4 instance
# =========================================

def prepare_task4_instance(row):
    men_prefs, women_prefs, men_text, women_text, ground_truth = prepare_instance(row)

    return {
        "men_prefs": men_prefs,
        "women_prefs": women_prefs,
        "men_text": men_text,
        "women_text": women_text,
        "ground_truth": ground_truth,
        "level1_q": row["level1_q"],
        "level1_a": str(row["level1_a"]).strip(),
        "level2_q": row["level2_q"],
        "level2_a": str(row["level2_a"]).strip().upper(),
        "level2n_q": row["level2n_q"],
        "level2n_a": str(row["level2n_a"]).strip().upper()
    }


# =========================================
# 4.2 build task 4 prompt
# =========================================

def build_task4_prompt(men_text, women_text, question_text, question_type="level1"):
    if question_type == "level1":
        answer_instruction = """
Once you determine the answer, return only the single agent name inside <answer></answer> tags.

Example:
<answer>M2</answer>
"""
    else:
        answer_instruction = """
Once you determine the answer, return only YES or NO inside <answer></answer> tags.

Example:
<answer>YES</answer>
"""

    prompt = f"""
You are an intelligent assistant tasked with analyzing preference lists in a two-sided matching problem.

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

Now answer the following question:

<question>
{question_text}
</question>

{answer_instruction}
Do not include any explanation outside the final answer.
"""
    return prompt




# =========================================
# 4.3 parse task 4 answer
# =========================================

def parse_task4_answer(raw_output, question_type="level1"):
    if raw_output is None:
        return None, False, "no output"

    text = str(raw_output).strip()

    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip()
    else:
        answer_text = text.strip()

    if question_type == "level1":
        agent_match = re.search(r'\b([MW]\d+)\b', answer_text, re.IGNORECASE)
        if agent_match:
            return agent_match.group(1).upper(), True, "parsed agent answer"
        return None, False, "could not parse agent answer"

    answer_upper = answer_text.upper()
    if "YES" in answer_upper:
        return "YES", True, "parsed yes"
    if "NO" in answer_upper:
        return "NO", True, "parsed no"

    return None, False, "could not parse yes/no answer"



# =========================================
# 4.4 evaluate task 4 answer
# =========================================

def evaluate_task4_answer(parsed_answer, true_answer):
    true_answer_clean = str(true_answer).strip().upper() if true_answer is not None else None

    if parsed_answer is None:
        return {
            "parsed_successfully": False,
            "is_correct": False,
            "predicted_answer": None,
            "true_answer": true_answer_clean
        }

    is_correct = (str(parsed_answer).strip().upper() == true_answer_clean)

    return {
        "parsed_successfully": True,
        "is_correct": is_correct,
        "predicted_answer": str(parsed_answer).strip().upper(),
        "true_answer": true_answer_clean
    }


def process_task4_model_response(raw_output, true_answer, question_type="level1", show_output=True):
    parsed_answer, parsed_ok, parse_msg = parse_task4_answer(raw_output, question_type=question_type)

    if not parsed_ok:
        result = {
            "parsed_ok": False,
            "parse_msg": parse_msg,
            "parsed_answer": None,
            "evaluation": {
                "parsed_successfully": False,
                "is_correct": False,
                "predicted_answer": None,
                "true_answer": str(true_answer).strip().upper()
            }
        }
    else:
        evaluation = evaluate_task4_answer(parsed_answer, true_answer)

        result = {
            "parsed_ok": True,
            "parse_msg": parse_msg,
            "parsed_answer": parsed_answer,
            "evaluation": evaluation
        }

    if show_output:
        eval_result = result["evaluation"]
        print("parsed:", result["parsed_ok"])
        print("predicted:", eval_result["predicted_answer"])
        print("true answer:", eval_result["true_answer"])
        print("correct:", eval_result["is_correct"])

    return result



# =========================================
# 4.6 reasoning model wrapper
# =========================================

def task4_reasoning_model(csv_file, num_instances=10, start_index=0, temperature=0, num_examples_to_show=2):
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
        "correct_count": 0
    }

    for idx in range(start_index, end_index):
        row = df.iloc[idx]
        instance = prepare_task4_instance(row)

        q_type_idx = (idx - start_index) % 3

        if q_type_idx == 0:
            question_text = instance["level1_q"]
            true_answer = instance["level1_a"]
            q_type = "level1"
        elif q_type_idx == 1:
            question_text = instance["level2_q"]
            true_answer = instance["level2_a"]
            q_type = "level2"
        else:
            question_text = instance["level2n_q"]
            true_answer = instance["level2n_a"]
            q_type = "level2"

        prompt = build_task4_prompt(
            instance["men_text"],
            instance["women_text"],
            question_text,
            question_type=q_type
        )

        response = llm_reasoning.invoke(prompt)
        time.sleep(9.0)
        raw_output = response.content

        result = process_task4_model_response(
            raw_output,
            true_answer,
            question_type=q_type,
            show_output=((idx - start_index) < num_examples_to_show)
        )

        summary_stats["total_instances"] += 1

        if result["parsed_ok"]:
            summary_stats["parsed_count"] += 1

        if result["evaluation"]["is_correct"]:
            summary_stats["correct_count"] += 1

        detailed_results.append({
            "instance_idx": idx,
            "question_type": q_type,
            "raw_response": raw_output,
            "parsed_answer": result["parsed_answer"],
            "evaluation": result["evaluation"]
        })

    accuracy = (
        summary_stats["correct_count"] / summary_stats["total_instances"]
        if summary_stats["total_instances"] > 0 else 0.0
    )

    summary = {
        "total_instances": summary_stats["total_instances"],
        "parsed_count": summary_stats["parsed_count"],
        "correct_count": summary_stats["correct_count"],
        "accuracy": round(accuracy, 2)
    }

    return detailed_results, summary


# =========================================
# 4.5 basic model wrapper
# =========================================

def task4_basic_model(csv_file, num_instances=10, start_index=0, temperature=0, num_examples_to_show=2):
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
        "correct_count": 0
    }

    for idx in range(start_index, end_index):
        row = df.iloc[idx]
        instance = prepare_task4_instance(row)

        # cycle through question types
        q_type_idx = (idx - start_index) % 3

        if q_type_idx == 0:
            question_text = instance["level1_q"]
            true_answer = instance["level1_a"]
            q_type = "level1"
        elif q_type_idx == 1:
            question_text = instance["level2_q"]
            true_answer = instance["level2_a"]
            q_type = "level2"
        else:
            question_text = instance["level2n_q"]
            true_answer = instance["level2n_a"]
            q_type = "level2"

        prompt = build_task4_prompt(
            instance["men_text"],
            instance["women_text"],
            question_text,
            question_type=q_type
        )

        response = llm.invoke(prompt)
        time.sleep(9.0)
        raw_output = response.content

        result = process_task4_model_response(
            raw_output,
            true_answer,
            question_type=q_type,
            show_output=((idx - start_index) < num_examples_to_show)
        )

        summary_stats["total_instances"] += 1

        if result["parsed_ok"]:
            summary_stats["parsed_count"] += 1

        if result["evaluation"]["is_correct"]:
            summary_stats["correct_count"] += 1

        detailed_results.append({
            "instance_idx": idx,
            "question_type": q_type,
            "raw_response": raw_output,
            "parsed_answer": result["parsed_answer"],
            "evaluation": result["evaluation"]
        })

    accuracy = (
        summary_stats["correct_count"] / summary_stats["total_instances"]
        if summary_stats["total_instances"] > 0 else 0.0
    )

    summary = {
        "total_instances": summary_stats["total_instances"],
        "parsed_count": summary_stats["parsed_count"],
        "correct_count": summary_stats["correct_count"],
        "accuracy": round(accuracy, 2)
    }

    return detailed_results, summary