import time
from langchain_groq import ChatGroq

from src.data_utils import (
    load_matching_csv,
    infer_expected_size,
    prepare_instance,
    build_answer_template,
    extract_matching_from_response,
    check_validity,
    check_stability,
    exact_match_with_ground_truth,
    summarize_results,
)

from src.data_utils import (
    convert_prefs_to_dict,
    prefers,
    parse_ground_truth_pairs,
)

# =========================================
# 1.1 build prompt
# =========================================

def build_prompt(men_text, women_text, expected_size):
    answer_template = build_answer_template(expected_size)

    prompt = f"""
You are an intelligent assistant who is an expert in algorithms. Consider the following instance of the two-sided matching problem, where {expected_size} men are to be matched with {expected_size} women.

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

Your task is to find the proposer-optimal stable matching.

Return ONLY the final matching.
Do NOT provide Python code.
Do NOT explain your reasoning.
Do NOT describe the algorithm.
Your final response must contain only one JSON object enclosed in <answer></answer> tags.

<answer>
{{
{answer_template}
}}
</answer>

Make sure that each man is matched with exactly ONE woman and each woman is matched with exactly ONE man.
""".strip()

    return prompt






def evaluate_matching(llm_matching, men_prefs, women_prefs, ground_truth, expected_size):
    result = {
        "parsed_successfully": llm_matching is not None,
        "is_valid": False,
        "is_stable": False,
        "exact_match": False,
        "validity_message": "",
        "blocking_pairs": [],
        "ground_truth_dict": None
    }

    if llm_matching is None:
        result["validity_message"] = "no matching parsed"
        return result

    is_valid, validity_message = check_validity(llm_matching, expected_size)
    result["is_valid"] = is_valid
    result["validity_message"] = validity_message

    if not is_valid:
        return result

    is_stable, blocking_pairs = check_stability(llm_matching, men_prefs, women_prefs)
    result["is_stable"] = is_stable
    result["blocking_pairs"] = blocking_pairs

    exact_match, ground_truth_dict = exact_match_with_ground_truth(llm_matching, ground_truth)
    result["exact_match"] = exact_match
    result["ground_truth_dict"] = ground_truth_dict

    return result


# =========================================
# 1.4 process one raw response
# =========================================

def process_model_response(
    raw_output,
    men_prefs,
    women_prefs,
    ground_truth_string,
    expected_size,
    show_output=True
):
    parsed_matching, parsed_ok, parse_msg = extract_matching_from_response(raw_output)

    if not parsed_ok:
        result = {
            "parsed_ok": False,
            "parse_msg": parse_msg,
            "parsed_matching": None,
            "evaluation": {
                "parsed_successfully": False,
                "is_valid": False,
                "is_stable": False,
                "exact_match": False,
                "validity_message": parse_msg,
                "blocking_pairs": [],
                "ground_truth_dict": None
            }
        }
    else:
        evaluation = evaluate_matching(
            parsed_matching,
            men_prefs,
            women_prefs,
            ground_truth_string,
            expected_size
        )

        result = {
            "parsed_ok": True,
            "parse_msg": parse_msg,
            "parsed_matching": parsed_matching,
            "evaluation": evaluation
        }

    if show_output:
        eval_result = result["evaluation"]

        print("parsed:", result["parsed_ok"])
        print("valid:", eval_result["is_valid"])
        print("stable:", eval_result["is_stable"])
        print("exact match:", eval_result["exact_match"])
        print("validity message:", eval_result["validity_message"])
        print("blocking pairs count:", len(eval_result["blocking_pairs"]))

        if len(eval_result["blocking_pairs"]) > 0:
            print("blocking pairs:", eval_result["blocking_pairs"])

        print()

    return result



# =========================================
# 1.6 shared batch runner
# =========================================

def run_model_on_instances(model, csv_file, num_instances=5, start_index=0, num_examples_to_show=2):
    df = load_matching_csv(csv_file)
    expected_size = infer_expected_size(csv_file=csv_file)

    end_index = min(start_index + num_instances, len(df))

    detailed_results = []
    compact_results = []

    for idx in range(start_index, end_index):
        row = df.iloc[idx]

        men_prefs, women_prefs, men_text, women_text, ground_truth = prepare_instance(row)
        prompt = build_prompt(men_text, women_text, expected_size)

        response = model.invoke(prompt)

        # only print first num_examples_to_show instances
        should_print = (idx - start_index) < num_examples_to_show

        result = process_model_response(
            raw_output=response.content,
            men_prefs=men_prefs,
            women_prefs=women_prefs,
            ground_truth_string=ground_truth,
            expected_size=expected_size,
            show_output=should_print
        )

        evaluation = result["evaluation"]

        detailed_results.append({
            "instance_idx": idx,
            "raw_response": response.content,
            "parsed_matching": result["parsed_matching"],
            "evaluation": evaluation
        })

        compact_results.append({
            "instance_idx": idx,
            "parsed_ok": result["parsed_ok"],
            "is_valid": evaluation["is_valid"],
            "is_stable": evaluation["is_stable"],
            "exact_match": evaluation["exact_match"],
            "blocking_pairs_count": len(evaluation["blocking_pairs"])
        })

    summary = summarize_results(compact_results)
    print(summary)

    return detailed_results, compact_results, summary

 
# =========================================
# 1.7 basic model wrapper
# =========================================


def basic_model(csv_file, num_instances=5, start_index=0, temperature=0, num_examples_to_show=2):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_retries=30
    )

    return run_model_on_instances(
        model=llm,
        csv_file=csv_file,
        num_instances=num_instances,
        start_index=start_index,
        num_examples_to_show=num_examples_to_show
    )


# =========================================
# 1.8 reasoning model wrapper
# =========================================

def reasoning_model(csv_file, num_instances=5, start_index=0, temperature=0, num_examples_to_show=2):
    llm_reasoning = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_retries=30
    )

    return run_model_on_instances(
        model=llm_reasoning,
        csv_file=csv_file,
        num_instances=num_instances,
        start_index=start_index,
        num_examples_to_show=num_examples_to_show
    )