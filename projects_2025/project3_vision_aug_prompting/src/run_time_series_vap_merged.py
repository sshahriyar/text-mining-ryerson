#!/usr/bin/env python3
"""

Implements a 3-stage pipeline for time series forecasting:
  1) Planning      -> The model outputs a short plan in JSON.
  2) Iterative     -> The model simulates step-by-step refinement.
  3) Conclusive    -> The model provides a final forecast as JSON: {"final_forecast":[...]}

Logs:
 - A "figures/" folder for saved plots
 - A "result.csv" file with forecasts and errors
 - A "summary.log" with average error metrics
"""

import os
import re
import json
import time
import argparse
import datetime
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from call_gpt import call_gpt  # Your GPT API call function


# Helper: Clean GPT response

def clean_response(response_text: str) -> str:
    """
    Removes any markdown fences (```), leading/trailing whitespace, etc.
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines.pop(0)
        if lines and lines[-1].startswith("```"):
            lines.pop()
        cleaned = "\n".join(lines).strip()
    return cleaned

# Helper: Plot the time series (for local record)

def draw_time_series(sequence, out_filename="time_series.png"):
    """
    Creates and saves a PNG plot of the time series data, but we do not pass it to GPT.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(sequence, marker='o', linestyle='-')
    plt.title("Input Time Series")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.savefig(out_filename, dpi=300)
    plt.close()

# Stage 1: Planning

def planning_stage(sequence_str: str, model: str, max_tokens: int = 500) -> dict:

    prompt = f"""You are a time series forecasting expert.
You have this time series (as text):
{sequence_str}

Please output a short plan in JSON with these keys:
  - "analysis": short analysis
  - "forecasting_method": method you'll use
  - "detailed_steps": array of steps
  - "termination_condition": condition to stop reasoning

Output ONLY valid JSON, no extra text.
"""
    response = call_gpt(
        question=prompt,
        model=model,
        temperature=0,
        max_tokens=max_tokens
    )
    cleaned = clean_response(response)
    if not cleaned:
        print("Warning: Cleaned planning response is empty!")
    try:
        plan_json = json.loads(cleaned)
    except Exception as e:
        print("Error parsing plan JSON:", e)
        plan_json = {}
    return plan_json

# Stage 2: Iterative Reasoning

def iterative_reasoning_stage(sequence_str: str, plan: dict, model: str, max_tokens: int = 500) -> list:

    plan_text = json.dumps(plan)
    prompt = f"""You are a time series forecasting expert.
You have a high-level plan: {plan_text}
Here is the time series again: {sequence_str}

Simulate iterative reasoning to refine your forecast. 
Output a JSON array, each element:
{{
  "iteration": <number>,
  "thought": "...",
  "forecast_update": "..."
}}

No extra commentary, just that JSON.
"""
    response = call_gpt(
        question=prompt,
        model=model,
        temperature=0,
        max_tokens=max_tokens
    )
    cleaned = clean_response(response)
    if not cleaned:
        print("Warning: Cleaned iterative reasoning response is empty!")
    try:
        iterations = json.loads(cleaned)
    except Exception as e:
        print("Error parsing iterative reasoning JSON:", e)
        iterations = []
    return iterations


# Stage 3: Conclusive Reasoning

def conclusive_reasoning_stage(sequence_str: str, plan: dict, iterations: list, model: str, max_tokens: int = 500) -> list:

    plan_text = json.dumps(plan)
    iter_text = json.dumps(iterations)
    prompt = f"""You are a time series forecasting expert.
Given:
1) The time series: {sequence_str}
2) The high-level plan: {plan_text}
3) The iterative steps: {iter_text}

Provide the final forecast in JSON as:
{{
  "final_forecast": [number1, number2, ...]
}}

No extra text.
"""
    response = call_gpt(
        question=prompt,
        model=model,
        temperature=0,
        max_tokens=max_tokens
    )
    cleaned = clean_response(response)
    if not cleaned:
        print("Warning: Cleaned conclusive reasoning response is empty!")
    try:
        final_json = json.loads(cleaned)
        final_forecast = final_json.get("final_forecast", [])
        # Ensure they are floats
        final_forecast = [float(x) for x in final_forecast]
    except Exception as e:
        print("Error parsing final forecast JSON:", e)
        final_forecast = []
    return final_forecast


# Error Metrics

def calculate_mae(actual: list, predicted: list) -> float:
    if len(actual) != len(predicted) or not actual:
        return float('inf')
    return sum(abs(a - p) for a,p in zip(actual,predicted))/len(actual)

def calculate_mape(actual: list, predicted: list) -> float:
    if len(actual) != len(predicted) or not actual:
        return float('inf')
    totpct = 0.0
    count = 0
    for a, p in zip(actual, predicted):
        if a != 0:
            totpct += abs((a - p)/a)
        count+=1
    return (totpct/count)*100 if count>0 else float('inf')


# Main Pipeline

def main():
    parser = argparse.ArgumentParser(description="3-Stage pipeline for time series forecasting")
    parser.add_argument('--dataset_dir', type=str, default='../dataset', help='Dataset directory')
    parser.add_argument('--task', type=str, default='time_series_prediction', help='Folder name for the data')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Which GPT model to use')
    parser.add_argument('--max_tokens', type=int, default=1500, help='Max tokens for GPT calls')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps to forecast in the future')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    task = args.task
    model = args.model
    max_tokens = args.max_tokens
    forecast_steps = args.steps

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir_base = os.path.join("logs", task + "_vap_noimage")
    os.makedirs(log_dir_base, exist_ok=True)

    run_dir = os.path.join(log_dir_base, f"{model}_{current_time_str}")
    os.makedirs(run_dir, exist_ok=True)

    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    result_csv_path = os.path.join(run_dir, "result.csv")
    summary_path = os.path.join(run_dir, "summary.log")

    result_file = open(result_csv_path, "w", encoding="utf-8")
    result_file.write("sample_id,final_forecast,mae,mape\n")

    # Load dataset
    data_path = os.path.join(dataset_dir, task, "task.json")
    with open(data_path,"r",encoding="utf-8") as f:
        metadata = json.load(f)

    total_mae = 0.0
    total_mape = 0.0
    count_samples = 0

    for i, item in enumerate(tqdm(metadata)):
        input_series = item["input"]
        ground_truth = item["output"][:forecast_steps]

        # Convert input to string
        seq_str = ",".join(str(x) for x in input_series)+","

        figure_name = os.path.join(figures_dir, f"time_series_{i}.png")
        draw_time_series(input_series, figure_name)

        # Stage 1: Planning
        plan = planning_stage(seq_str, model, max_tokens=max_tokens)

        # Stage 2: Iterative
        iterations = iterative_reasoning_stage(seq_str, plan, model, max_tokens=max_tokens)

        # Stage 3: Conclusive
        final_forecast = conclusive_reasoning_stage(seq_str, plan, iterations, model, max_tokens=max_tokens)

        # If forecast is empty or too short
        needed = len(ground_truth) - len(final_forecast)
        if needed>0:
            if not final_forecast:
                fallback_val = 0.0
                final_forecast = [fallback_val]*len(ground_truth)
            else:
                final_forecast += [final_forecast[-1]]*needed
        elif needed<0:
            final_forecast = final_forecast[:len(ground_truth)]

        # Evaluate
        mae = calculate_mae(ground_truth, final_forecast)
        mape= calculate_mape(ground_truth, final_forecast)
        total_mae+=mae
        total_mape+=mape
        count_samples+=1

        # Log to CSV
        fcast_str = " ".join(map(str, final_forecast))
        result_file.write(f"{i},{fcast_str},{mae},{mape}\n")
        result_file.flush()

        print(f"Sample={i}")
        print(f"Ground Truth={ground_truth}")
        print(f"Forecast={final_forecast}")
        print(f"MAE={mae}, MAPE={mape}%\n{'-'*30}")

        time.sleep(5)  # minor delay to reduce rate-limits

    result_file.close()

    avg_mae  = total_mae / count_samples if count_samples>0 else float("inf")
    avg_mape = total_mape / count_samples if count_samples>0 else float("inf")

    with open(summary_path,"w",encoding="utf-8") as sf:
        sf.write(f"Total Samples: {count_samples}\n")
        sf.write(f"Average MAE: {avg_mae}\n")
        sf.write(f"Average MAPE: {avg_mape}%\n")

    print("\nCompleted all samples.")
    print(f"Avg MAE={avg_mae}, Avg MAPE={avg_mape}%")
    print(f"Logs and figures in: {run_dir}")

if __name__=="__main__":
    main()
