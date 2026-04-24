import time
import re
import pandas as pd
from langchain_groq import ChatGroq
from src.data_utils import (
    load_matching_csv,
    prepare_instance,
    parse_ground_truth_pairs,
    check_stability,
)

from src.data_utils import normalize_matching_dict



# =========================================
# 2.1 parse ground truth string
# =========================================

def parse_ground_truth_string(ground_truth_string):
    if ground_truth_string is None:
        return None

    text = str(ground_truth_string).strip()

    pairs = re.findall(r'\[\s*(M\d+)\s*,\s*(W\d+)\s*\]', text)

    if not pairs:
        return None

    parsed_dict = {}
    for man, woman in pairs:
        parsed_dict[man] = woman

    return parsed_dict


# =========================================
# 2.2 create stable and unstable examples
# =========================================

def matching_dict_to_pair_list(matching_dict, expected_size):
    pairs = []
    for i in range(1, expected_size + 1):
        man = f"M{i}"
        woman = matching_dict[man]
        pairs.append([man, woman])
    return pairs



def create_unstable_matching_from_ground_truth(ground_truth_string, expected_size):
    ground_truth_dict = parse_ground_truth_string(ground_truth_string)

    if ground_truth_dict is None:
        return None, False, "could not parse ground truth"

    unstable_dict = ground_truth_dict.copy()

    if expected_size < 2:
        return None, False, "need at least 2 pairs to create unstable matching"

    m1 = "M1"
    m2 = "M2"

    w1 = unstable_dict[m1]
    w2 = unstable_dict[m2]

    unstable_dict[m1] = w2
    unstable_dict[m2] = w1

    return unstable_dict, True, "unstable matching created"

def prepare_task2_instance(row):
    men_prefs, women_prefs, men_text, women_text, ground_truth = prepare_instance(row)

    expected_size = len(men_prefs)

    stable_dict = parse_ground_truth_string(ground_truth)
    unstable_dict, unstable_ok, unstable_msg = create_unstable_matching_from_ground_truth(
        ground_truth,
        expected_size
    )

    stable_pairs = matching_dict_to_pair_list(stable_dict, expected_size) if stable_dict else None
    unstable_pairs = matching_dict_to_pair_list(unstable_dict, expected_size) if unstable_ok else None

    return {
        "men_prefs": men_prefs,
        "women_prefs": women_prefs,
        "men_text": men_text,
        "women_text": women_text,
        "ground_truth": ground_truth,
        "expected_size": expected_size,
        "stable_matching_dict": stable_dict,
        "unstable_matching_dict": unstable_dict,
        "stable_matching_pairs": stable_pairs,
        "unstable_matching_pairs": unstable_pairs,
        "unstable_ok": unstable_ok,
        "unstable_msg": unstable_msg
    }


# =========================================
# 2.2 build task 2 prompt
# =========================================

def build_task2_prompt(men_text, women_text, matching_pairs):
    matching_text = "["
    matching_text += ",".join([f"[{pair[0]}, {pair[1]}]" for pair in matching_pairs])
    matching_text += "]"

    prompt = f"""
Consider the following instance of the two-sided matching problem, where men are to be matched with women.

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

Your task is to determine whether the following matching is stable or not.

<matching>
{matching_text}
</matching>

Please return 'Yes' if the provided matching is stable and 'No' if it is unstable.

Return your final answer inside <answer></answer> tags only.
"""
    return prompt




# =========================================
# 2.3 parse yes no answer
# =========================================

def parse_yes_no_answer(raw_output):
    if raw_output is None:
        return None, False, "no output"

    text = raw_output.lower()

    # try to extract from <answer> tags first
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        answer_text = match.group(1).strip()
    else:
        answer_text = text.strip()

    # normalize
    if "yes" in answer_text:
        return "YES", True, "parsed yes"
    elif "no" in answer_text:
        return "NO", True, "parsed no"
    else:
        return None, False, "could not parse yes/no"
    
# =========================================
# 2.4 evaluate task 2 answer
# =========================================

def evaluate_task2_answer(parsed_answer, true_label):
    if parsed_answer is None:
        return {
            "parsed_successfully": False,
            "is_correct": False,
            "predicted_label": None,
            "true_label": true_label
        }

    is_correct = (parsed_answer == true_label)

    return {
        "parsed_successfully": True,
        "is_correct": is_correct,
        "predicted_label": parsed_answer,
        "true_label": true_label
    }


def process_task2_model_response(raw_output, true_label, show_output=True):
    parsed_answer, parsed_ok, parse_msg = parse_yes_no_answer(raw_output)

    if not parsed_ok:
        result = {
            "parsed_ok": False,
            "parse_msg": parse_msg,
            "parsed_answer": None,
            "evaluation": {
                "parsed_successfully": False,
                "is_correct": False,
                "predicted_label": None,
                "true_label": true_label
            }
        }
    else:
        evaluation = evaluate_task2_answer(parsed_answer, true_label)

        result = {
            "parsed_ok": True,
            "parse_msg": parse_msg,
            "parsed_answer": parsed_answer,
            "evaluation": evaluation
        }

    if show_output:
        eval_result = result["evaluation"]
        print("parsed:", result["parsed_ok"])
        print("predicted:", eval_result["predicted_label"])
        print("true label:", eval_result["true_label"])
        print("correct:", eval_result["is_correct"])

    return result



# =========================================
# 2.5 basic model wrapper
# =========================================

def task2_basic_model(csv_file, num_instances=10, start_index=0, temperature=0, num_examples_to_show=2):
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
        instance = prepare_task2_instance(row)

        if not instance["unstable_ok"]:
            continue

        if (idx - start_index) % 2 == 0:
            matching_pairs = instance["stable_matching_pairs"]
            true_label = "YES"
        else:
            matching_pairs = instance["unstable_matching_pairs"]
            true_label = "NO"

        prompt = build_task2_prompt(
            instance["men_text"],
            instance["women_text"],
            matching_pairs
        )

        response = llm.invoke(prompt)
        time.sleep(9.0)
        raw_output = response.content

        result = process_task2_model_response(
            raw_output,
            true_label,
            show_output=((idx - start_index) < num_examples_to_show)
        )

        summary_stats["total_instances"] += 1

        if result["parsed_ok"]:
            summary_stats["parsed_count"] += 1

        if result["evaluation"]["is_correct"]:
            summary_stats["correct_count"] += 1

        detailed_results.append({
            "instance_idx": idx,
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
# 2.6 reasoning model wrapper
# =========================================

def task2_reasoning_model(csv_file, num_instances=10, start_index=0, temperature=0, num_examples_to_show=2):
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
        instance = prepare_task2_instance(row)

        if not instance["unstable_ok"]:
            continue

        if (idx - start_index) % 2 == 0:
            matching_pairs = instance["stable_matching_pairs"]
            true_label = "YES"
        else:
            matching_pairs = instance["unstable_matching_pairs"]
            true_label = "NO"

        prompt = build_task2_prompt(
            instance["men_text"],
            instance["women_text"],
            matching_pairs
        )

        response = llm_reasoning.invoke(prompt)
        time.sleep(9.0)
        raw_output = response.content

        result = process_task2_model_response(
            raw_output,
            true_label,
            show_output=((idx - start_index) < num_examples_to_show)
        )

        summary_stats["total_instances"] += 1

        if result["parsed_ok"]:
            summary_stats["parsed_count"] += 1

        if result["evaluation"]["is_correct"]:
            summary_stats["correct_count"] += 1

        detailed_results.append({
            "instance_idx": idx,
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
