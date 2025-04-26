import os
import re
import json
import time
from datetime import datetime
from tqdm import tqdm
from call_gpt import call_gpt
from prompts.intersection_count import standard_prompt, cot_prompt
import argparse

parser = argparse.ArgumentParser(description='Run geometry intersection task')

parser.add_argument('--dataset_dir', type=str, default='../dataset', help='Directory of the dataset')
parser.add_argument('--task', type=str, default='intersection_geometry', help='Name of the task')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to use')
parser.add_argument('--method', type=str, default='standard', choices=['standard', 'cot', 'cot-sc'], help='Method to use')
parser.add_argument('--log_dir', type=str, default='log', help='Directory for logs')
parser.add_argument('--k_samples', type=int, default=5, help='Number of samples for cot-sc method')

args = parser.parse_args()
dataset_dir = args.dataset_dir
task = 'intersection_geometry'
model = args.model
method = args.method
log_dir = args.log_dir
k_samples = args.k_samples

method = method.lower()

log_dir_base = os.path.join(log_dir, task)

current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')

if not os.path.exists(log_dir_base):
    os.makedirs(log_dir_base)

log_dir = os.path.join(log_dir_base, f'{model}_{method}_{current_time_str}')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

result_path = os.path.join(log_dir, f'result.csv')
result_file = open(result_path, 'w', encoding='utf8')
result_file.write('problem_id,pred_answer,is_correct,num_shape\n')
result_file.flush()

def check_identity(ground_truth, pred):
    # Convert type of pred to the type of ground_truth
    try:
        converted_pred = type(ground_truth)(pred)
    except BaseException as e:
        return False

    # Return True if they are identical, otherwise return False
    return converted_pred == ground_truth


metadata_path = os.path.join(dataset_dir, task, 'task.json')
with open(metadata_path, 'r', encoding='utf8') as f:
    metadata = json.load(f)

right_count = 0
current_count = 0
for question_id, item in enumerate(tqdm(metadata[:120])):

    log_path = os.path.join(log_dir, f'{question_id}.log')
    log_file = open(log_path, 'w', encoding='utf8')

    # temperature = 0
    # Unify temperature to 0.7
    temperature = 0.7
    if method == 'standard':
        prompt = standard_prompt
        max_tokens = 150
    elif method == 'cot':
        prompt = cot_prompt
        max_tokens = 2048
    elif method == 'cot-sc':
        prompt = cot_prompt
        max_tokens = 512
        # temperature = 0.7

    result = call_gpt(prompt.format(problem=item['input']), model, max_tokens=max_tokens)

    print(result)

    if method == 'standard':
        try:
            pred_answer = int(result)
        except:
            pred_answer = 0
    elif method == 'cot':
        try:
            m = re.findall('answer:\s*(\d+)', result.lower())[0]
            pred_answer = int(m)
        except:
            pred_answer = 0
    elif method == 'cot-sc':
        count_map = {}
        for _ in range(k_samples):
            try:
                m = re.findall('answer:\s*(\d+)', result.lower())[0]
                pred_answer = int(m)
            except:
                pred_answer = 0
            count_map[pred_answer] = count_map.get(pred_answer, 0) + 1
        pred_answer = sorted(list(count_map.items()), key=lambda x: x[1], reverse=True)[0][0]

    log_file.write('=' * 40 + '\n')
    log_file.write(f'problem: {item["input"]}\n')
    log_file.write(f'result: {result}\n')
    log_file.write(f'pred_answer: {pred_answer}\n')
    log_file.write(f'ground truth: {item["answer"]}\n')
    log_file.flush()
    print('=' * 40 + '\n')
    print(f'problem: {item["input"]}\n')
    print(f'result: {result}\n')
    print(f'pred_answer: {pred_answer}\n')
    print(f'ground truth: {item["answer"]}\n')

    is_correct = check_identity(item['answer'], pred_answer)
    if is_correct:
        right_count += 1
    current_count += 1

    result_file.write(f'{current_count-1},{pred_answer},{is_correct},{item["num_shape"]}\n')
    result_file.flush()

    print(f'Accuracy: {right_count / current_count * 100:.2f}%')

    log_file.close()

    time.sleep(5)

print(f'Accuracy: {right_count / len(metadata) * 100:.2f}%')

with open(os.path.join(log_dir, 'summary.log'), 'w', encoding='utf8') as f:
    f.write(f'\n\nAccuracy: {right_count / len(metadata) * 100:.2f}%\n')
