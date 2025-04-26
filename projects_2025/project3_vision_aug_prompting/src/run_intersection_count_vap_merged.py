#!/usr/bin/env python3
"""

This script implements a three-stage Vision-Augmented Prompting (VAP) pipeline in three separate API calls:
1. Planning: Generate a high-level plan for visualizing the problem.
2. Iterative Reasoning: Simulate iterative drawing updates with thought process.
3. Conclusive Reasoning: Use the accumulated information to produce the final intersection count.

"""

import os
import json
import re
import argparse
import datetime
import matplotlib.pyplot as plt

from call_gpt import call_gpt  # Assumes call_gpt is defined elsewhere to interface with the LLM API

# ---------------------------
# Helper function to clean the API response
# ---------------------------

def clean_response(response_text):
    """
    Cleans the raw GPT response by removing markdown code block markers and any leading "json" text.
    Returns the cleaned text.
    """
    cleaned = response_text.strip()
    # If the response is wrapped in triple backticks, remove them.
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Remove the first line if it starts with ``` (e.g., ```json)
        if lines and lines[0].startswith("```"):
            lines.pop(0)
        # Remove the last line if it is the closing ```
        if lines and lines[-1].startswith("```"):
            lines.pop()
        # If the first line is "json" (case-insensitive), remove it.
        if lines and lines[0].strip().lower() == "json":
            lines.pop(0)
        cleaned = "\n".join(lines).strip()
    return cleaned


# Helper Functions for Geometry Parsing & Drawingdef parse_geometry_description(problem_text):
    """
    Parses a geometry problem and returns a list of shapes.
    Each shape is represented as:
      ("circle", (cx, cy), radius)
      ("line", (x1, y1), (x2, y2))
      ("polygon", [(x1, y1), (x2, y2), ...])
    """
    shapes = []
    text_lower = problem_text.lower()

    # Find circles
    circle_matches = re.findall(
        r'circle centered at \(([-\d\.]+),\s*([-\d\.]+)\)\s*with radius\s*([-\d\.]+)',
        text_lower
    )
    for (cx, cy, r_str) in circle_matches:
        r_clean = r_str.rstrip('.')  # Remove trailing dot if any
        shapes.append(("circle", (float(cx), float(cy)), float(r_clean)))

    # Find line segments
    line_matches = re.findall(
        r'line segment from \(([-\d\.]+),\s*([-\d\.]+)\)\s*to\s*\(([-\d\.]+),\s*([-\d\.]+)\)',
        text_lower
    )
    for (x1, y1, x2, y2) in line_matches:
        shapes.append(("line", (float(x1), float(y1)), (float(x2), float(y2))))

    # Find polygons
    poly_matches = re.findall(
        r'polygon with coordinates\s*\[([^\]]+)\]', text_lower
    )
    for coords_str in poly_matches:
        pair_matches = re.findall(r'\(([-\d\.]+),\s*([-\d\.]+)\)', coords_str)
        if pair_matches:
            pts = [(float(px), float(py)) for (px, py) in pair_matches]
            shapes.append(("polygon", pts))

    return shapes

def draw_geometry(shapes, out_filename="figure.png"):
    """
    Uses matplotlib to draw the given shapes and saves the figure.
    - Circles are drawn with blue outlines.
    - Lines are drawn as red lines.
    - Polygons are drawn as green lines (closed by connecting the first point again).
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    for shape in shapes:
        if shape[0] == "circle":
            (cx, cy), r = shape[1], shape[2]
            circle = plt.Circle((cx, cy), r, fill=False, color='b')
            ax.add_patch(circle)
        elif shape[0] == "line":
            (x1, y1), (x2, y2) = shape[1], shape[2]
            ax.plot([x1, x2], [y1, y2], color='r')
        elif shape[0] == "polygon":
            pts = shape[1]
            if pts:
                pts_closed = pts + [pts[0]]
                xs = [p[0] for p in pts_closed]
                ys = [p[1] for p in pts_closed]
                ax.plot(xs, ys, color='g')

    plt.title("Visualized Geometry Problem")
    plt.savefig(out_filename)
    plt.close(fig)

# Stage 1: Planning

def planning_stage(problem_text, model, max_tokens=500):

    prompt = f"""Your task is to visualize a geometry problem by creating an image.
Before drawing, provide a high-level plan that includes the following three aspects:
1. Tool Selection: Which drawing tool you will use (choose from Python Matplotlib, Python Turtle, or DALLE3).
2. Initialization: How you will set up the image (for example, the coordinate limits and initial shapes).
3. Iterative Drawing Approach: A detailed description of how you will add details step by step.

Output your plan in JSON format following this exact structure:
{{
    "tool": "<name of the tool>",
    "initialization": "<description of initialization>",
    "iterative_approach": ["<step 1 description>", "<step 2 description>", ...],
    "termination_condition": "<criteria to stop drawing>"
}}

Problem Description: {problem_text}
"""
    response = call_gpt(question=prompt, model=model, temperature=0, max_tokens=max_tokens)
    
    # Debug: Print raw and cleaned planning response
    print("Raw Planning Response:")
    print(response)
    cleaned = clean_response(response)
    print("Cleaned Planning Response:")
    print(cleaned)
    if not cleaned:
        print("Warning: The cleaned planning response is empty!")
    
    try:
        plan_json = json.loads(cleaned)
    except Exception as e:
        print("Error parsing planning response:", e)
        plan_json = {}
    return plan_json

# Stage 2: Iterative Reasoning

def iterative_reasoning_stage(problem_text, figure_filename, plan, model, max_tokens=500):

    prompt = f"""You are a problem visualizer. Based on the problem description and the following plan, simulate an iterative drawing process.
The current partial drawing is saved in the image referenced by {figure_filename}.
Use the plan below to guide your iterative reasoning.
Plan: {json.dumps(plan)}

Problem Description: {problem_text}

Output your iterative steps as a JSON array, where each element is of the form:
{{
    "iteration": <number>,
    "thought": "<your reasoning for this step>",
    "iterative_draw_step": "<description of the drawing update>"
}}

Do not output any additional commentary.
"""
    response = call_gpt(question=prompt, model=model, image_path=figure_filename, temperature=0, max_tokens=max_tokens)
    
    # Debug: Print raw and cleaned iterative reasoning response
    print("Raw Iterative Reasoning Response:")
    print(response)
    cleaned = clean_response(response)
    print("Cleaned Iterative Reasoning Response:")
    print(cleaned)
    if not cleaned:
        print("Warning: The cleaned iterative reasoning response is empty!")
    
    try:
        iterations = json.loads(cleaned)
    except Exception as e:
        print("Error parsing iterative reasoning response:", e)
        iterations = []
    return iterations

# Stage 3: Conclusive Reasoning


def conclusive_reasoning_stage(problem_text, figure_filename, plan, iterations, model, max_tokens=500):

    prompt = f"""You are a problem visualizer. You are provided with:
1. The original problem: {problem_text}
2. The high-level plan: {json.dumps(plan)}
3. The iterative reasoning steps: {json.dumps(iterations)}
4. A reference image of the final drawing saved at {figure_filename}

Based on the above information, determine the final number of intersection points in the drawing.
Output ONLY a JSON string in the following format:
{{
    "final_answer": <an integer>
}}
Do not include any extra commentary.
"""
    response = call_gpt(question=prompt, model=model, image_path=figure_filename, temperature=0, max_tokens=max_tokens)
    
    # Debug: Print raw and cleaned conclusive reasoning response
    print("Raw Conclusive Reasoning Response:")
    print(response)
    cleaned = clean_response(response)
    print("Cleaned Conclusive Reasoning Response:")
    print(cleaned)
    if not cleaned:
        print("Warning: The cleaned conclusive reasoning response is empty!")
    
    try:
        final_json = json.loads(cleaned)
        final_answer = int(final_json.get("final_answer", 0))
    except Exception as e:
        print("Error parsing conclusive reasoning response:", e)
        final_answer = 0
    return final_answer


# Main Pipeline Logic


def main():
    parser = argparse.ArgumentParser(description='Run 3-stage Vision-Augmented Prompting for geometry intersection')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Directory of the dataset')
    parser.add_argument('--task', type=str, default='intersection_geometry', help='Task folder name')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='LLM to use')
    parser.add_argument('--max_tokens', type=int, default=1500, help='Maximum tokens for each API call')
    parser.add_argument('--max_iterations', type=int, default=1, help='(Unused in separate stages)')
    args = parser.parse_args()

    # Load the dataset of geometry problems
    metadata_path = os.path.join(args.dataset_dir, args.task, 'task.json')
    with open(metadata_path, 'r', encoding='utf8') as f:
        metadata = json.load(f)

    # Set up directories for logs and figures
    current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir_base = os.path.join('logs', args.task + '_vap_separate')
    os.makedirs(log_dir_base, exist_ok=True)
    log_dir = os.path.join(log_dir_base, f'{args.model}_separate_{current_time_str}')
    os.makedirs(log_dir, exist_ok=True)
    figures_dir = os.path.join(log_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    result_csv = os.path.join(log_dir, 'result.csv')
    summary_log = os.path.join(log_dir, 'summary.log')
    result_file = open(result_csv, 'w', encoding='utf8')
    result_file.write('problem_id,plan,iterations,final_answer,is_correct,num_shape\n')
    result_file.flush()

    right_count = 0
    total_count = 0

    # Process each problem
    for idx, item in enumerate(metadata):
        problem_text = item['input']
        gt_answer_str = str(item['answer'])
        num_shape = item.get('num_shape', 0)
        figure_filename = os.path.join(figures_dir, f"figure_{idx}.png")

        # Generate and save the initial drawing
        shapes = parse_geometry_description(problem_text)
        draw_geometry(shapes, out_filename=figure_filename)

        # Stage 1: Planning
        plan = planning_stage(problem_text, model=args.model, max_tokens=500)
        print(f"[{idx}] Plan: {plan}")

        # Stage 2: Iterative Reasoning
        iterations = iterative_reasoning_stage(problem_text, figure_filename, plan, model=args.model, max_tokens=500)
        print(f"[{idx}] Iterations: {iterations}")

        # Stage 3: Conclusive Reasoning
        final_answer = conclusive_reasoning_stage(problem_text, figure_filename, plan, iterations, model=args.model, max_tokens=500)
        print(f"[{idx}] Final predicted answer: {final_answer}")

        is_correct = (str(final_answer) == gt_answer_str)
        if is_correct:
            right_count += 1
        total_count += 1

        result_file.write(f"{idx},{json.dumps(plan)},{json.dumps(iterations)},{final_answer},{is_correct},{num_shape}\n")
        result_file.flush()
        print(f"[{idx}] Predicted: {final_answer} | Ground Truth: {gt_answer_str} | Correct: {is_correct}")

    result_file.close()
    accuracy = 100.0 * right_count / total_count if total_count else 0
    with open(summary_log, 'w', encoding='utf8') as sf:
        sf.write(f"Total: {total_count}, Correct: {right_count}\n")
        sf.write(f"Accuracy: {accuracy:.2f}%\n")

    print(f"\nCompleted: {right_count}/{total_count} correct | Accuracy: {accuracy:.2f}%")
    print(f"Logs and figures saved to: {log_dir}")

if __name__ == "__main__":
    main()
