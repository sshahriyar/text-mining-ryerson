from pathlib import Path
import os
from tqdm import tqdm
from ollama import chat

def generate_all_prompts(datasets_path, embeddings=False):
  data_sets = os.listdir(datasets_path)


  SYSTEM_PROMPT = f"""
    You are a domain expert with years of experience in building the best-performing decision trees.
    You have an astounding ability to identify the best features for the task at hand, and you know how to combine them to get the best predictions.
    Impressively, your profound world knowledge allows you to do that without looking at any real-world data.

    # Do the following Steps
    ### STEP 1
    I want you to induce a decision tree classifier based on features and a prediction target.
    I first give an example of the decision tree. Given Features and a new prediction target, I then want you to build a decision tree using the most important features.

    ### STEP 2
    Format the decision tree as a Python function that returns a single prediction as well as a list representing the truth values of the inner nodes.
    The entries of this list should be 1 if the condition of the corresponding inner node is satisfied, and 0 otherwise.
    Use only the feature names that I provide, generate the decision tree without training it on actual data, and return the Python function.
    Ensure that the input the funciton is just a dictionary or dataframe row, for example def func(feature_row):
    At the start of the function make sure to always precompute the binary vector embeddings 
    Example: 
    node1 = int(a > 20)
    node2 = int(b < 15)
    ...
    emb = [node_1, node_2, ..., node_n]
    Then apply the decision tree logic after that returns prediction, emb
    """
  dataset_prompts = {}
  for dataset in tqdm(data_sets, desc="Generating DTs for each dataset"):
    # Read dataset description
    with open(os.path.join(datasets_path, dataset, 'description.txt')) as f:
      description_part = f.read()

    # Read specific prompt for dataset
    
    with open(os.path.join(datasets_path, dataset, 'prompt_emb.txt' if embeddings else 'prompt.txt')) as f:
      final_part = f.read()

    with open(os.path.join(datasets_path, dataset, 'feature_description.txt')) as f:
      feat_desc = f.read()


    user_prompt = f"""
    ### DESCRIPTION OF DATASET
    <description>
    {description_part}
    </description>
    
    ### Feature description
    <feature_description>
    {feat_desc}
    </feature_description>
    
    ### USER
    <user>
    {final_part}
    </user>

    ### OUTPUT INSTRUCTION
    Only return the decision tree function. Do not add any reasoning or other materials.
    """
    
    dataset_prompts[dataset] = {'system': SYSTEM_PROMPT, 'user': user_prompt }
    
  return dataset_prompts



def get_dt_functions_out(prompts, output_path, temperature=1.0, dt_trees=1):
    models = {'gpt-oss:20b': 'gpt_oss', 'qwen3:14B': 'qwen3', 'gemma3:12b': 'gemma3', 'mistral-small3.2:24b': 'mistral_small3'}

    for model in tqdm(models, desc="Generateing dt functions from each model"):
        print(f"Runing {model}")
        model_path = Path(output_path) / models[model]
        for dataset in prompts:
            dataset_path = model_path / dataset
            dataset_path.mkdir(exist_ok=True, parents=True)
            prompt = prompts[dataset]
            for i in range(dt_trees):
                dt_path = dataset_path / f"dt_func_{i}.txt"
                if dt_path.exists(): # skip datasets already done
                  continue
                response = chat(
                    model=model,
                    messages=[{'role': 'system', 'content': prompt['system']},
                            {'role': 'user', 'content': prompt['user']}],
                    options={
                        "temperature": temperature
                    })
                    
                output = response.message.content
                with open(dt_path, 'w') as f:
                    f.write(output)
    