# MediQ: Question-Asking LLMs for Adaptive and Reliable Clinical Reasoning

## [[paper](https://arxiv.org/abs/2406.00922)] [[website](https://stellalisy.com/projects/mediQ/)] [[data](https://github.com/stellali7/mediQ/tree/main/data)]

## Overview

This benchmark system simulates an interactive conversation between a patient and an expert. The system evaluates how well participants' expert modules can handle realistic patient queries by either asking relevant questions or making final decisions based on the conversation history.

## Installation

Clone this repository to your local machine using the following command:

```
git clone https://github.com/aadarshachapagain/Meqiq_bench.git
```

Navigate into the project directory:

```
cd MediQ
```

Create a new conda environment with necessary packages (note: you need to be on a GPU node to install PyTorch with CUDA):

```
conda env create -f environment.yml
```

## Project Structure

-   `benchmark.py`: Main script to run the benchmark.
-   `patient.py`: Defines the `Patient` class that simulates patient behavior.
-   `expert.py`: Contains the `Expert` class which participants will extend to implement their response strategies.
-   `args.py`: Handles command-line arguments for the benchmark system.

## Configuration

Before running the benchmark, configure the necessary parameters in `args.py`:

-   `--expert_module`: The file name (without `.py`) where the Expert class is implemented (e.g. expert if your Expert class definition is in `expert.py`)
-   `--expert_class`: The name of the Expert class to be evaluated, this should be defined in the file `[expert_module].py` (e.g. RandomExpert)
-   `--patient_module`: The file name (without `.py`) where the Patient class is implemented (e.g. patient if your Patient class definition is in `patient.py`)
-   `--patient_class`: The name of the Patient class to use for the benchmark, this should be defined in the file `[patient_module].py` (e.g. RandomPatient)
-   `--data_dir`: Directory containing the development data files.
-   `--dev_filename`: Filename for development data.
-   `--log_filename`: Filename for logging general benchmark information.
-   `--history_log_filename`: Filename for logging detailed interaction history.
-   `--message_log_filename`: Filename for logging messages.
-   `--output_filepath`: Path where the output JSONL files will be saved.

## Running the Benchmark

NOTE: if you choose to use an OpenAI model to power the benchmark, you need to put the API key in `src/keys.py`.

```
python mediQ_benchmark.py  --expert_module expert --expert_class FixedExpert \
                        --patient_module patient --patient_class RandomPatient \
                        --data_dir ../data --dev_filename all_dev_good.jsonl \
                        --output_filename out.jsonl --max_questions 10
```

## Expected Benchmark Outcomes

By using the following LLM and parameters, the expected outcome should align these results:

-   Meta-LLAMA 3.1 as the engine for both patient system and expert system
-   Ran benchmark model for 1272 patients
-   Max 30 questions asked to gather information required to make decisions
-   Accuracy 731 patients were successfully diagnosed out of 1272 patients (0.5747)
-   Timeout Rate of 1.0 and Avg Turn of 31 with a confidence level of 4.0 (scale threshold, or 0.8 for the abstrain threshold)

## Project Adaptation: Integration of Meta-Llama-3.1 into MediQ Framework

This project is a fork and adaptation of the original implementation from the paper “MediQ: Question-Asking LLMs for Adaptive and Reliable Clinical Reasoning”. The original codebase has been extended and modified to integrate the Meta-Llama-3.1 model, enabling more advanced and adaptive clinical reasoning capabilities while preserving the core logic and evaluation framework of the original MediQ system.
Ensure to replace the placeholder values with actual parameters relevant to your setup.
