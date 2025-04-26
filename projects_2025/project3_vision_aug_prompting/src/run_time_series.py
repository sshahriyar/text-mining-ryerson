#!/usr/bin/env python3
"""

Runs a time series prediction task using either standard or chain-of-thought (CoT) prompting.
- Loads time series problems from dataset/time_series_prediction/task.json.
- Sends text prompts to the LLM (e.g., GPT-3.5-Turbo, GPT-4o-mini, etc.).
- Extracts the predicted continuation sequence from GPT's output.
- Computes Mean Absolute Error (MAE) against the ground truth.
- Logs results and performance summary in a log folder.
This file uses integrated prompt definitions for both the "standard" and "chain-of-thought" methods.
"""

import os
import re
import json
import time
from datetime import datetime
from tqdm import tqdm
import argparse
from call_gpt import call_gpt  # This function handles the GPT API call



# Standard prompt: The model is instructed to simply continue the sequence.
standard_prompt = '''You are a time predictor. The user will provide a sequence of numbers (represented as decimal strings separated by commas) and you will predict the remaining sequence.

Please continue the following sequence without producing any additional text. Do not say anything like "the next terms in the sequence are", just return the numbers separated by commas.
Sequence:
{sequence}'''

# Chain-of-Thought (CoT) prompt: The model must briefly explain its reasoning and then output only the predicted numbers.
cot_prompt = '''You are a time predictor with chain-of-thought reasoning abilities. First, briefly describe your reasoning about the trend of the sequence. Then, on the next line, output only the final predicted numbers (decimal values) separated by commas, without any extra text.
Sequence:
{sequence}
Output:
Reasoning:
Final Answer:'''


# Helper function to calculate Mean Absolute Error (MAE)

def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error (MAE) between two lists of floats.
    """
    if len(actual) != len(predicted):
        raise ValueError("The length of actual and predicted lists must be the same.")
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
    return mae


# Main execution logic for time series prediction

def main():
    parser = argparse.ArgumentParser(description='Run time series prediction task')
    parser.add_argument('--dataset_dir', type=str, default='../dataset', help='Directory of the dataset')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='LLM model to use')
    parser.add_argument('--method', type=str, default='standard', choices=['standard', 'cot'], help='Method to use')
    parser.add_argument('--log_dir', type=str, default='log', help='Directory for logs')
    parser.add_argument('--k_samples', type=int, default=5, help='Number of samples for CoT method (if applicable)')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    task = 'time_series_prediction'
    model = args.model
    method = args.method.lower()
    log_dir = args.log_dir
    k_samples = args.k_samples

    # Create a base log folder for the task
    log_dir_base = os.path.join(log_dir, task)
    if not os.path.exists(log_dir_base):
        os.makedirs(log_dir_base)
    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(log_dir_base, f'{model}_{method}_{current_time_str}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Prepare the result CSV file for logging
    result_path = os.path.join(log_dir, 'result.csv')
    result_file = open(result_path, 'w', encoding='utf8')
    result_file.write('sample_id,pred_sequence,mae\n')
    result_file.flush()

    # Load the dataset (expects the dataset JSON to contain a list of samples, each with "input" and "output")
    metadata_path = os.path.join(dataset_dir, task, 'task.json')
    with open(metadata_path, 'r', encoding='utf8') as f:
        metadata = json.load(f)

    total_mae = 0.0
    sample_count = 0

    # Process each sample in the dataset
    for sample_id, item in enumerate(tqdm(metadata)):
        # For each sample, assume:
        #   item["input"] is a list of numbers (the given sequence)
        #   item["output"] is a list of numbers (the ground truth continuation)
        input_sequence = item["input"]
        ground_truth = item["output"]

        # Build a comma-separated string from the input sequence (append a comma at the end)
        sequence_str = ",".join(str(x) for x in input_sequence) + ","

        # Choose the appropriate prompt and maximum tokens based on the method
        if method == 'standard':
            prompt = standard_prompt.format(sequence=sequence_str)
            max_tokens = 150  # Increase as needed to allow full continuation response
        elif method == 'cot':
            prompt = cot_prompt.format(sequence=sequence_str)
            max_tokens = 2048  # Increase token limit for longer chain-of-thought responses
        else:
            prompt = standard_prompt.format(sequence=sequence_str)
            max_tokens = 150

        # Call the GPT API with the prompt
        result = call_gpt(prompt, model=model, max_tokens=max_tokens)
        print("Prompt:")
        print(prompt)
        print("Raw GPT Result:")
        print(result)
        
        # Parse the result into a predicted sequence of floats
        if method == 'standard':
            pred_sequence = []
            # Expecting the response to be only the numbers separated by commas
            for token in result.split(","):
                token = token.strip()
                try:
                    if token:  # Skip empty strings
                        pred_sequence.append(float(token))
                except Exception:
                    continue
        elif method == 'cot':
            # In the CoT prompt we expect a reasoning section followed by "Final Answer:" and the numbers.
            final_answer_match = re.search(r'final answer\s*:\s*(.*)', result.lower(), re.DOTALL)
            if final_answer_match:
                pred_text = final_answer_match.group(1)
            else:
                pred_text = result
            pred_sequence = []
            for token in pred_text.split(","):
                token = token.strip()
                try:
                    if token:
                        pred_sequence.append(float(token))
                except Exception:
                    continue
        else:
            pred_sequence = []

        # If no prediction was parsed, default to repeating the last input value
        if not pred_sequence:
            pred_sequence = [input_sequence[-1]] * len(ground_truth)
        # Adjust the predicted sequence length to match the ground truth length
        if len(pred_sequence) < len(ground_truth):
            pred_sequence.extend([pred_sequence[-1]] * (len(ground_truth) - len(pred_sequence)))
        elif len(pred_sequence) > len(ground_truth):
            pred_sequence = pred_sequence[:len(ground_truth)]

        # Calculate the MAE for this sample
        try:
            mae = calculate_mae(ground_truth, pred_sequence)
        except Exception as e:
            mae = float('inf')
        total_mae += mae
        sample_count += 1

        # Log the result for this sample
        result_file.write(f'{sample_id},"{" ".join(map(str, pred_sequence))}",{mae}\n')
        result_file.flush()

        print(f"Sample {sample_id}:")
        print(f"Input sequence: {input_sequence}")
        print(f"Ground truth: {ground_truth}")
        print(f"Predicted sequence: {pred_sequence}")
        print(f"MAE: {mae}")
        print("-" * 40)

        # Sleep between API calls to help avoid rate limits
        time.sleep(5)

    # Calculate and print the average MAE
    avg_mae = total_mae / sample_count if sample_count > 0 else float('inf')
    print(f"Performance Report - Model: {model}, Method: {method}")
    print(f"Total Samples: {sample_count}")
    print(f"Average MAE: {avg_mae}")

    with open(os.path.join(log_dir, 'summary.log'), 'w', encoding='utf8') as f:
        f.write(f"Total Samples: {sample_count}\n")
        f.write(f"Average MAE: {avg_mae}\n")

    result_file.close()

if __name__ == "__main__":
    main()
