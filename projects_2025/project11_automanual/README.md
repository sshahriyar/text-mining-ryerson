# AutoManual: Autonomous Manual Construction for LLM Agents
AutoManual is a framework that enables Large Language Model (LLM) agents to autonomously build their understanding of new environments through interaction, without requiring extensive human-crafted instructions or expert prompts.
Traditional approaches to LLM agent deployment face a key limitation: they rely heavily on human intervention to customize agents for specific environments, which is time-consuming, requires expertise, and doesn't scale well. AutoManual addresses this challenge by enabling agents to build their own instruction manuals through environmental interaction.

The AutoManual Framework operates in three stages:

Building Stage: Two agents work collaboratively in an alternating process:

A Planner Agent interacts with the environment using code-based plans, analyzing situations, identifying relevant rules, and executing Python code with assertions
A Builder Agent extracts and updates rules from these interactions, organizing them into a structured rule system with attributes for type, content, examples, and validation logs

Formulating Stage: A Formulator Agent categorizes the collected rules based on their application scenarios, drafts introductions for each category, and compiles everything into a comprehensive manual in Markdown format.
Testing Stage: A test-time Planner agent utilizes the generated manual to complete new tasks, demonstrating the effectiveness of the learned rules.

# Environment Setup
We experiment with Python 3.9 for ALFWorld and MiniWoB++

#### ALFWorld

We made some minor changes to the logic of [alfworld](./alfworld) to make it more suitable for code-style plan: remove the article before the object and connect the object and its ID with an underscore.

```bash
# create conda env and install packages
conda create -y --name py39 python=3.9.16
conda activate py39

# install alfworld packages
pip install https://github.com/MarcCote/TextWorld/archive/handcoded_expert_integration.zip
pip install https://github.com/MarcCote/downward/archive/faster_replan.zip
cd automanual/alfworld/
pip install -r requirements.txt
export ALFWORLD_DATA=<root>/automanual/alfworld/downloaded/ # replace <root> with your dir
pip install .
```

#### MiniWoB++

The setup for [MiniWoB++](./automanual_miniwob/computergym) is simple:

```bash
# install requirements
pip install selenium Pillow regex
```

## Experiments

Configurate your OpenAI API Key and OpenAI Base URL

```bash
export OPENAI_API_KEY="<your-api-key>" # a valid OpenAI API key starts with sk-
export OPENAI_BASE_URL="<your-base-url>" # e.g., https://api.openai.com/v1
```

The manuals we built are already included in each environment directory for testing. You can also build it as follows.

**ALFWorld:**

```bash
cd automanual/automanual_alfworld

# Building stage
python main_build.py --agent_type "autobuild_case" --run_name "autobuildcase_logs" \
			--model_name "gpt-4-1106-preview" --assistant_api --num_env_per_task 6

# Formulating stage
python main_test.py --agent_type "autobuild_case" --run_name "autobuildcase_logs" \
			--model_name "gpt-4-1106-preview" --assistant_api --mode "formulating" --is_resume

# Testing stage, 'model_name' can be replace with gpt-3.5-turbo
python main_test.py --agent_type "autobuild_case" --run_name "autobuildcase_logs" \
			--model_name "gpt-4-1106-preview" --assistant_api --mode "testing" \
			--num_env_per_task 25 --is_resume
```

`--agent_type` can be selected from "replan", "autobuild_case", "autobuild_offline".

If the building or testing process stops in the middle, you can add `--is_resume --start_env_id <stop_env_id>` to continue.

**MiniWoB++:**

```bash
cd automanual/automanual_miniwob

# Building stage
python main_build.py --agent_type "autobuild_case" --run_name "autobuildcase_logs" \
			--model_name "gpt-4-1106-preview" --assistant_api --num_env_per_task 6

# Formulating stage
python main_test.py --agent_type "autobuild_case" --run_name "autobuildcase_logs" \
			--model_name "gpt-4-1106-preview" --assistant_api --mode "formulating" --is_resume

# Testing stage
python main_test.py --agent_type "autobuild_case" --run_name "autobuildcase_logs" \
			--model_name "gpt-4-1106-preview" --assistant_api --mode "testing" \
			--num_env_per_task 6 --is_resume
```

optionally add `--headless` 


