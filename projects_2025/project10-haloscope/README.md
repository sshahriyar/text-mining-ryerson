# Hallucination Detection in Language Model Generations

This project is based on the works of [HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection](https://github.com/deeplearning-wisc/haloscope)

This project proposes a method (Haloscope) for training a hallucination detector using an unlabeled dataset of prompts and LLM-generated answers. By analyzing the latent activation space from which LLMs generate responses, the researchers aim to identify a subspace that captures the patterns associated with hallucinated outputs. This approach enables the automatic inference of labels—distinguishing hallucinated from truthful answers—based on structural properties of the latent space, allowing for scalable hallucination detection without manual annotation.

In this work, we extend this idea by proposing a novel approach for analyzing internal representations through sequence modeling. Specifically, we treat the depth-wise progression of hidden layers in the transformer as a pseudo-temporal sequence and apply an LSTM to learn patterns across layers. This framing allows the model to implicitly learn which layers encode hallucination-relevant signals, rather than requiring explicit manual selection or dimensionality reduction. Our method explores the temporal dynamics of representation formation within LLMs and leverages them for improved downstream hallucination detection.

---

## Installation Instructions

To install the main libraries required for running the project:

```bash
pip install git+https://github.com/davidbau/baukit
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
pip install huggingface_hub
pip install evaluate
pip install t5
```


You will also need a HuggingFace read token to download the Llama 2 and BLUERT models.
The main project code can be run from ProjectCode.ipynb in src/ folder. The cells have
been broken down into respective tasks.

---

## Project Structure

```
.
├── src/                            # Stores full project with custom libraries
│   ├── llama_iti/                  # library for loading llama model for inference
│   ├── truthfulqa/                 # library for TruthfulQA dataset and functions
│   ├── ylib/                       # library of tools for linear probe classifier  
│   ├── ProjectCode.ipynb           # full project code
│   ├── linear_probe.py             # linear probe classifier model
│   ├── metric_utils.py             # Haloscope custom functions (see source repo)
│   ├── utils.py                    # Haloscope custom functions (see source repo)
├── FinalProjectSubmission.ipynb    # Our paper + important code, explained
├── haloscope.drawio.png            # Diagram of Haloscope architecture
├── roccurve.png                    # ROC curve comparing linear probe vs LSTM classifier

```

---

## Utility files from other projects

llama_iti/, truthfulqa/, ylib/, linear_probe.py, metric_utils.py, utils.py

Those folders contain source code (libraries) from other projects used in this project. 
The main body of work in ProjectCode.ipynb is based off of the Haloscope project. 
The novel addition in this work is utilzing an LSTM based classifer for the final
hallucination detection model. Please go through FinalProjectSubmission.ipynb
for a more detailed explaination. 


---

## Code Outputs

When the full code in ProjectCode.ipynb is run it will output:
- LLM generated answers from Llama-2-7b to combine with TruthfulQA dataset
- automated BLUERT annotations to give ground truth of hallucinated answers
- Trained linear probe classifier
- Trained LSTM classifer

---

## Citation ##
Here is the citation for the original Haloscope body of work:

```
 @inproceedings{du2024haloscope,
      title={ HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection}, 
      author={Xuefeng Du and Chaowei Xiao and Yixuan Li},
      booktitle={Advances in Neural Information Processing Systems},
      year = {2024}
}
```
