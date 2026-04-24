import re
import json
import pandas as pd
import os

from src.config import DATA_DIR

# =========================
# COMMON SETUP AND HELPERS
# =========================
# Goal: Define shared functions and utilities used across all tasks.

# Includes:
# - dataset loading
# - preference parsing
# - instance preparation
# - answer template generation
# - shared utility functions


# =========================================
# phase 3 load csv file
# =========================================

def load_matching_csv(csv_file, base_path=DATA_DIR):
    full_path = os.path.join(base_path, csv_file)
    df = pd.read_csv(full_path)
    return df


# =========================================
# phase 4 infer expected size
# =========================================

def infer_expected_size(csv_file=None, row=None):
    if csv_file is not None:
        match = re.match(r"(\d+)_", csv_file)
        if match:
            return int(match.group(1))

    if row is not None:
        return len(row["man_pref_string"].split("\n"))

    raise ValueError("could not infer expected size")


# =========================================
# phase 5 prepare one instance
# =========================================

def prepare_instance(row):
    men_prefs_raw = row["man_pref_string"].split("\n")
    women_prefs_raw = row["woman_pref_string"].split("\n")

    men_prefs = [list(map(int, line.split(","))) for line in men_prefs_raw]
    women_prefs = [list(map(int, line.split(","))) for line in women_prefs_raw]

    men_lines = []
    for i, prefs in enumerate(men_prefs):
        men_lines.append(f"M{i+1}: [" + ",".join([f"W{w}" for w in prefs]) + "]")

    women_lines = []
    for i, prefs in enumerate(women_prefs):
        women_lines.append(f"W{i+1}: [" + ",".join([f"M{m}" for m in prefs]) + "]")

    men_text = ",\n".join(men_lines)
    women_text = ",\n".join(women_lines)

    ground_truth = row["men_opt"]

    return men_prefs, women_prefs, men_text, women_text, ground_truth


# =========================================
# phase 6 build answer template
# =========================================

def build_answer_template(expected_size):
    answer_lines = []

    for i in range(1, expected_size + 1):
        comma = "," if i < expected_size else ""
        answer_lines.append(f'  "M{i}": "<woman matched with M{i}>"{comma}')

    return "\n".join(answer_lines)




# =========================================
# 1.2 parse model response
# =========================================

def normalize_agent_label(label, prefix):
    label = str(label).strip().upper()
    match = re.match(rf"{prefix}\s*(\d+)$", label)
    if match:
        return f"{prefix}{match.group(1)}"
    return label

def normalize_matching_dict(matching):
    normalized = {}

    for k, v in matching.items():
        man = normalize_agent_label(k, "M")
        woman = normalize_agent_label(v, "W")
        normalized[man] = woman

    return normalized


def extract_matching_from_response(raw_output):
    answer_match = re.search(
        r"<answer>\s*(\{[\s\S]*?\})\s*</answer>",
        raw_output,
        re.IGNORECASE
    )

    if answer_match:
        candidate_json = answer_match.group(1)
        try:
            parsed = json.loads(candidate_json)
            return normalize_matching_dict(parsed), True, "parsed from answer tags"
        except Exception as e:
            return None, False, f"json parse error in answer tags: {str(e)}"

    matches = re.findall(r"\{[\s\S]*?\}", raw_output)

    for block in reversed(matches):
        cleaned = block.replace("'", '"')
        cleaned = re.sub(r",\s*}", "}", cleaned)

        try:
            parsed = json.loads(cleaned)
            return normalize_matching_dict(parsed), True, "parsed from fallback json"
        except:
            continue

    return None, False, "no valid json object found"

def convert_prefs_to_dict(prefs, side_prefix, other_prefix):
    converted = {}

    for i, pref_list in enumerate(prefs, start=1):
        converted[f"{side_prefix}{i}"] = [f"{other_prefix}{x}" for x in pref_list]

    return converted

def prefers(preference_list, a, b):
    return preference_list.index(a) < preference_list.index(b)


# =========================================
# 1.3 evaluate matching
# =========================================

def check_validity(llm_matching, expected_size):
    if llm_matching is None:
        return False, "no matching parsed"

    expected_men = {f"M{i}" for i in range(1, expected_size + 1)}
    expected_women = {f"W{i}" for i in range(1, expected_size + 1)}

    if set(llm_matching.keys()) != expected_men:
        return False, "wrong or incomplete men labels"

    assigned_women = list(llm_matching.values())

    if len(assigned_women) != expected_size:
        return False, "wrong number of assignments"

    if not set(assigned_women).issubset(expected_women):
        return False, "invalid women labels"

    if len(set(assigned_women)) != expected_size:
        return False, "duplicate women assigned"

    return True, "valid one-to-one matching"

def check_stability(matching, men_prefs, women_prefs):
    men_prefs_dict = convert_prefs_to_dict(men_prefs, "M", "W")
    women_prefs_dict = convert_prefs_to_dict(women_prefs, "W", "M")

    reverse_matching = {w: m for m, w in matching.items()}
    blocking_pairs = []

    for m in men_prefs_dict:
        current_w = matching[m]

        for w in men_prefs_dict[m]:
            if w == current_w:
                break

            current_m_for_w = reverse_matching[w]

            if prefers(women_prefs_dict[w], m, current_m_for_w):
                blocking_pairs.append((m, w))

    is_stable = len(blocking_pairs) == 0
    return is_stable, blocking_pairs

def parse_ground_truth_pairs(ground_truth_string):
    pairs = re.findall(r"\[M(\d+),\s*W(\d+)\]", ground_truth_string)
    return {f"M{m}": f"W{w}" for m, w in pairs}

def exact_match_with_ground_truth(llm_matching, ground_truth_string):
    ground_truth_dict = parse_ground_truth_pairs(ground_truth_string)
    return llm_matching == ground_truth_dict, ground_truth_dict



# =========================================
# 1.5 summarize results
# =========================================

def summarize_results(compact_results):
    total_instances = len(compact_results)

    if total_instances == 0:
        return {
            "total_instances": 0,
            "parsed_count": 0,
            "valid_count": 0,
            "stable_count": 0,
            "exact_match_count": 0,
            "avg_blocking_pairs": 0.0
        }

    parsed_count = sum(x["parsed_ok"] for x in compact_results)
    valid_count = sum(x["is_valid"] for x in compact_results)
    stable_count = sum(x["is_stable"] for x in compact_results)
    exact_match_count = sum(x["exact_match"] for x in compact_results)
    avg_blocking_pairs = sum(x["blocking_pairs_count"] for x in compact_results) / total_instances

    return {
        "total_instances": total_instances,
        "parsed_count": parsed_count,
        "valid_count": valid_count,
        "stable_count": stable_count,
        "exact_match_count": exact_match_count,
        "avg_blocking_pairs": round(avg_blocking_pairs, 2)
    }