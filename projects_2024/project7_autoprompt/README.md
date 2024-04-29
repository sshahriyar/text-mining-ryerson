<p align="center">
    <!-- community badges -->
    <a href="https://discord.gg/G2rSbAf8uP"><img src="https://img.shields.io/badge/Join-Discord-blue.svg"/></a>
    <!-- license badge -->
    <a href="https://github.com/Eladlev/AutoPrompt/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
</p>

# 📝 Introduction- AutoPrompt


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

**Auto Prompt is a prompt optimization framework designed to enhance and perfect your prompts for real-world use cases.**

The framework automatically generates high-quality, detailed prompts tailored to user intentions. It employs a refinement (calibration) process, where it iteratively builds a dataset of challenging edge cases and optimizes the prompt accordingly. This approach not only reduces manual effort in prompt engineering but also effectively addresses common issues such as prompt [sensitivity](https://arxiv.org/abs/2307.09009) and inherent prompt [ambiguity](https://arxiv.org/abs/2311.04205) issues.


**Our mission:** Empower users to produce high-quality robust prompts using the power of large language models (LLMs).

# Why Auto Prompt?
- **Prompt Engineering Challenges.** The quality of LLMs greatly depends on the prompts used. Even [minor changes](#prompt-sensitivity-example) can significantly affect their performance. 
- **Benchmarking Challenges.**  Creating a benchmark for production-grade prompts is often labour-intensive and time-consuming.
- **Reliable Prompts.** Auto Prompt generates robust high-quality prompts, offering measured accuracy and performance enhancement using minimal data and annotation steps.
- **Modularity and Adaptability.** With modularity at its core, Auto Prompt integrates seamlessly with popular open-source tools such as LangChain, Wandb, and Argilla, and can be adapted for a variety of tasks, including data synthesis and prompt migration.
## Problem Description:

Recent advancements in Large Language Models (LLMs) have showcased significant improvements in generative tasks, yet these models remain highly sensitive to the specific prompts provided by users. This sensitivity manifests as variations in performance not only when prompts are slightly altered, but also when model versions change, leading to unpredictable shifts in behavior across a spectrum of tasks. Current strategies to address this include the use of soft prompts and meta-prompts that require iterative optimization and deep access to the LLMs themselves. However, these methods depend heavily on large, high-quality benchmarks which are often unavailable in real-world scenarios and are costly to iterate over. Moreover, without adequate calibration to user intentions, LLMs might produce inaccurate outputs due to misunderstood prompts.

This project introduces the Intent-based Prompt Calibration (IPC) system based on the paper "Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases" [4], designed to refine and optimize prompts based on synthetic boundary examples specifically generated to encapsulate the user’s task requirements. This system not only aims to mitigate the challenge of prompt sensitivity but also adapts the LLM's output to more accurately reflect the user's intended use case, particularly in environments like content moderation where data distributions are typically imbalanced. The IPC system also incorporates a novel ranking mechanism to further enhance prompt optimization for generative tasks, reducing reliance on extensive manual annotation and large-scale data requirements.

## Context of the Problem:

The importance of addressing prompt sensitivity in Large Language Models (LLMs) is multifaceted. Firstly, it enhances reliability and predictability, essential for consistent AI performance across various applications. Secondly, it promotes scalability and efficiency, enabling LLMs to be fine-tuned for specific tasks without extensive resources, thereby reducing computational and economic costs. Thirdly, optimizing prompts to align with user intentions allows for customized, effective solutions, crucial for broadening AI applicability and user satisfaction. Lastly, tackling this issue aids in democratizing AI technology, ensuring more organizations can leverage these advancements with less specialized knowledge, and improving model explainability and transparency for ethical AI practices. Addressing these challenges is vital for the practical and ethical deployment of LLMs in real-world scenarios.
## System Overview

![System Overview](./docs/AutoPrompt_Diagram.png)

The system is designed for real-world scenarios, such as moderation tasks, which are often  challenged by imbalanced data distributions. The system implements the [Intent-based Prompt Calibration](https://arxiv.org/abs/2402.03099) method. The process begins with a user-provided initial prompt and task description, optionally including user examples. The refinement process iteratively generates diverse samples, annotates them via user/LLM, and evaluates prompt performance, after which an LLM suggests an improved prompt.  

The optimization process can be extended to content generation tasks by first devising a ranker prompt and then performing the prompt optimization with this learned ranker. The optimization concludes upon reaching the budget or iteration limit.  


This joint synthetic data generation and prompt optimization approach outperform traditional methods while requiring minimal data and iterations. Learn more in our paper
[Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases](https://arxiv.org/abs/2402.03099) by E. Levi et al. (2024).

## Solution

The method discussed above addresses the limitations of previous approaches by introducing a system that iteratively generates challenging boundary cases and refines prompts based on user feedback and synthetic examples. This system does not require extensive datasets or direct access to the model's internals, making it adaptable and efficient for real-world applications. By focusing on synthetic data generation and iterative calibration, it also minimizes the need for large, expensive benchmarks, thereby enhancing accessibility and reducing optimization costs.

Background
| Reference |Explanation |  Dataset/Input |Weakness
| --- | --- | --- | --- |
| Pi Liang et al. [1] | Task-specific Embedding (Continuous and Discrete)	| E2E , WebNLG, DART| Requires direct access to the LLM, making it less feasible for proprietary models with restricted access.
| M.Deng et al. [2] | Reinforcement Learning	| Yelp sentiment transfer dataset | 	Depends on access to generated tokens' probabilities or a large dataset, which can be resource-intensive and impractical in real-world scenarios.
| R.Pryzant et al. [3] | LLMs for Prompt Optimization		| Yelp Polarity| 	Still requires robust benchmarks to evaluate and compare prompts effectively, which may not always be available or feasible to construct..
| Elad Levi et al. [4] | Optimizing prompts through iterative calibration based on user intent and synthetic data		| IMDB review dataset | 	Covered in the Future work section


## Methodology
Initial Prompt and Task Description: The process begins with the user providing an initial prompt and a task description. Optionally, the user can also supply a few examples in a few-shot setting, which helps the system understand the context and requirements better.

Sample Generator: Using the initial data, the sample generator creates challenging and diverse boundary cases for the task. This is done to test the robustness of the current prompt and to identify any ambiguities or inaccuracies in prompt interpretation. The generator leverages a meta-prompt, which evolves over iterations based on the history of previous prompts and their performance.

Evaluator: Once the synthetic samples are generated, the evaluator assesses the current prompt's effectiveness on this generated dataset. This evaluation typically involves analyzing the prompt's accuracy using a confusion matrix and identifying misclassifications. This step is crucial for understanding the strengths and weaknesses of the current prompt.

Analyzer: This component receives the results from the evaluator, including prompt scores and detailed error analysis. It synthesizes this information into a comprehensive analysis summary, highlighting major failure cases and suggesting areas for prompt refinement.

Prompt Generator: With the analysis and historical performance data in hand, the prompt generator then suggests a new prompt. This prompt is designed to address the deficiencies identified in the previous iteration, with the aim of achieving a higher score in subsequent evaluations.

Iteration and Optimization: The calibration optimization process is iterative. It repeats the cycle of generating samples, evaluating the prompt, analyzing performance, and generating a new prompt. This loop continues until there is no significant improvement in prompt performance or a maximum number of iterations is reached.


## Implementation
Implementation code of this paper requries access to openAI API key as well as a docker instance of Argilla created either in huggingface space or locally. For that reason I cannot run the code in the google colab instance without having a colab Pro subscription which would give me access to the VM command line.

The github repo(git clone https://github.com/uzairwarraich/NLPProject_AutoPrompt.git) contains all the code and instructions of how to run the code. NOTE: Easier to run the code locally or in a VM, since for annotation Argilla is used, and for Argilla service web browser is required.

<img width="1037" alt="Screenshot 2024-04-19 at 3 00 46 PM" src="https://github.com/uzairwarraich/NLPProject_AutoPrompt/assets/77300361/6e84b60f-dfec-439d-bacb-9b736ea5483a">

## Demo

![pipeline_recording](./docs/autoprompt_recording.gif)

## Evalulation
They used a high-capacity model, GPT-4 Turbo, to create ground truth data. This step is crucial as it provides a benchmark for comparing the effectiveness of different prompts. In addition they used adversarial synthetic data generated from 10 initial samples from the IMDB dataset. It involves creating examples that are specifically designed to challenge or 'fool' the model. These are known as boundary cases.

After synthetic adversarial examples are generated by the system, human annotators evaluate these examples to determine how well they align with the intended task requirements and whether they effectively challenge the model. This helps in ensuring that the synthetic data is of high quality and truly representative of potential real-world complexities.

Human annotators also review the responses generated by the LLM based on the current prompts. They assess whether the responses are accurate, relevant, and appropriately detailed, providing a human judgment that can capture nuances possibly missed by automated metrics

Finally the metaprompts are adjusted based on the human annotations.

The researchers evaluate the effectiveness of their Intent-based Prompt Calibration (IPC) system using three binary classification tasks: spoiler detection, sentiment analysis, and parental guidance detection.

The proposed method outperformed other methods such as the OPRO[5] and PE[6].The other methods exhibit a higher variance in results, which suggests they may be less reliable or consistent, especially when the number of training samples is small. This variance can be detrimental in real-world applications where stable and predictable outputs are necessary.The other methods exhibit a higher variance in results, which suggests they may be less reliable or consistent, especially when the number of training samples is small. This variance can be detrimental in real-world applications where stable and predictable outputs are necessary.

## Conclusion and Future Direction

Future work could be done to refine the meta-prompts themselves. This involves improving how the meta-prompts are generated and used within the system to make them even more effective at guiding the prompt optimization process. All multimodality could be introduced to help the model generate better prompts with help images and audio.

In Conclusion, the paper offers valuable learnings about improving the efficiency and effectiveness of Large Language Models (LLMs) through refined prompt engineering. The most important part according to me was the use of synthetic data to generate boundary cases, which aids in iteratively refining prompts to better align with user intentions, thereby addressing the high sensitivity of LLMs to prompt variations. The IPC method showcases a modular and flexible system architecture, allowing for easy adaptation across different tasks and settings, including multi-modality and in-context learning. Additionally, the paper highlights the benefits of minimizing human annotation efforts by relying on synthetic data for prompt calibration, demonstrating significant improvements in model performance with fewer data requirements. These insights underscore the potential of IPC to enhance the practical utility of LLMs in diverse real-world applications, particularly in scenarios with limited data availability or specific performance criteria.

## References
[1]: X. L. Li and P. Liang. Prefix-tuning: Optimizing continuous prompts for generation. In C. Zong, F. Xia, W. Li, and R. Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582–4597, Online, Aug. 2021. Association for Computational Linguistics.

[2]: M. Deng, J. Wang, C. Hsieh, Y. Wang, H. Guo, T. Shu, M. Song, E. P. Xing, and Z. Hu. Rlprompt: Optimizing discrete text prompts with reinforcement learning. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 3369–3391. Association for Computational Linguistics, 2022.

[3]: R. Pryzant, D. Iter, J. Li, Y. Lee, C. Zhu, and M. Zeng. Automatic prompt optimization with “gradient descent” and beam search. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7957–7968, Singapore, Dec. 2023. Association for Computational Linguistics.

[4]: Elad Levi and Eli Brosh and Matan Friedmann . Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases,arXiv:2402.03099, 5 Feb 2024

[5]: C. Yang, X. Wang, Y. Lu, H. Liu, Q. V. Le, D. Zhou, and X. Chen. Large language models as optimizers. CoRR, abs/2309.03409, 2023.

[6]: Q. Ye, M. Axmed, R. Pryzant, and F. Khani. Prompt engineering a prompt engineer, 2023.

[7]:D. Vila-Suero and F. Aranda. Argilla - open-source framework for data-centric nlp, 2023. cff-version: 1.2.0, message: "If you use this software, please cite it as below.", date-released: 2023-01-12.


## QuickStart
AutoPrompt requires `python <= 3.10`
<br />

> **Step 1** - Download the project

```bash
git clone git@github.com:Eladlev/AutoPrompt.git
cd AutoPrompt
```

<br />

> **Step 2** - Install dependencies

Use either Conda or pip, depending on your preference. Using Conda:
```bash
conda env create -f environment_dev.yml
conda activate AutoPrompt
```

Using pip: 
```bash
pip install -r requirements.txt
```

Using pipenv:
```bash
pip install pipenv
pipenv sync
```

<br />

> **Step 3** - Configure your LLM. 

Set your OpenAI API key  by updating the configuration file `config/llm_env.yml`
- If you need help locating your API key, visit this [link](https://help.openai.com/en/articles/4936850-where-do-i-find-my-api-key).

- We recommend using [OpenAI's GPT-4](https://platform.openai.com/docs/guides/gpt) for the LLM. Our framework also supports other providers and open-source models, as discussed [here](docs/installation.md#configure-your-llm).

<br />

> **Step 4** - Configure your Annotator
- Select an annotation approach for your project. We recommend beginning with a human-in-the-loop method, utilizing [Argilla](https://docs.argilla.io/en/latest/index.html). Follow the [Argilla setup instructions](docs/installation.md#configure-human-in-the-loop-annotator-) to configure your server. Alternatively, you can set up an LLM as your annotator by following these [configuration steps](docs/installation.md#configure-llm-annotator-).

- The default predictor LLM, GPT-3.5, for estimating prompt performance, is configured in the `predictor` section of `config/config_default.yml`.

- Define your budget in the input config yaml file using the `max_usage parameter`. For OpenAI models, `max_usage` sets the maximum spend in USD. For other LLMs, it limits the maximum token count.

<br />


> **Step 5** - Run the pipeline

First, configure your labels by editing `config/config_default.yml`
```
dataset:
    label_schema: ["Yes", "No"]
```


For a **classification pipeline**, use the following command from your terminal within the appropriate working directory: 
```bash
python run_pipeline.py
```
If the initial prompt and task description are not provided directly as input, you will be guided to provide these details.  Alternatively, specify them as command-line arguments:
```bash
python run_pipeline.py \
    --prompt "Does this movie review contain a spoiler? answer Yes or No" \
    --task_description "Assistant is an expert classifier that will classify a movie review, and let the user know if it contains a spoiler for the reviewed movie or not." \
    --num_steps 30
```






