# Vision_Augmented_Prompting
## Overview
This project implements a simplified version of the Vision-Augmented Prompting (VAP) framework proposed in the NeurIPS 2024 paper “Enhancing LLM Reasoning via Vision-Augmented Prompting”.

We demonstrate how combining text + visual reasoning improves accuracy in tasks that are traditionally difficult for text-only models — specifically:
	•	 Geometry Intersection Problems (e.g., “How many times do shapes intersect?”)
	•	 Time Series Prediction (e.g., “What are the next 8 values in this sequence?”)

Unlike standard prompting or chain-of-thought (CoT) prompting, VAP:
	•	Uses visual diagrams (generated using Matplotlib) as additional input
	•	Guides the model through a 3-stage process: planning → iterative reasoning → conclusive reasoning
	•	Enhances reasoning by mimicking how humans sketch and think step-by-step

---
## Folder Structure
```
project-root/
│
├── dataset/                              # Contains task-specific input files (e.g., task.json)
│   └── intersection_geometry/
│       └── task.json
│
├── log/                                  # Generated after running the code, contains logs/results
│   └── intersection_geometry/
│       └── [model_name]_[method]_[timestamp]/
│           ├── result.csv               # Main results
│           ├── summary.log              # Accuracy summary
│           └── *.log                    # Logs for each problem
│
├── prompts/
│   └── intersection_count.py            # Prompt templates for standard and CoT methods
│
├── call_gpt.py                          # Utility function to call the GPT model (OpenAI API wrapper)
├── run_intersection_count.py           # Standard/CoT prompt execution
├── run_intersection_count_vap.py       # VAP-based visual prompting execution
└── README.md                            # Project documentation
```

---
## How to Run the Code
  ``` 
1. Clone the Repository and arrange the files according to the folder structure mentioned above
  ``` 
```
Prerequisites
	•	Python 3.8+
	•	OpenAI API Key (saved inside key_list.json)
	•	Required Libraries installed
  ```
  ``` 
2. Run the Notebook
  Run Vision_Augmented_Prompting.ipynb notebook
  ```

---
## Outputs

  ``` 
Each run creates a folder under log/ with the following contents:
	•	result.csv: All predictions with fields:
    	•	problem_id, pred_answer, is_correct, num_shape (for standard runs)
	    •	plan, iterations, final_answer, is_correct (for VAP runs)
	•	.log files: One log file per problem containing raw GPT response, prediction, ground truth, and debug info
	•	figures/: (VAP only) visualizations of the geometry problems created using Matplotlib
	•	summary.log: Final accuracy percentage over the dataset
  ``` 

---
## Contact us for any suggestions or queries:

-  Ashish Sunuwar- ashish.sunuwar@torontomu.ca
- Yahya Shaikh- yahya.shaikh@torontomu.ca

Links / references to relevant papers:
- [Original Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/328c922d068dd4ccb23cec5c64e6c7fc-Paper-Conference.pdf)
