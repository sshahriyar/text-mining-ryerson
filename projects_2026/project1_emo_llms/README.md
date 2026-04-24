# EmoLLMs: Experimental Analysis and Implementation

**Project Members**: Jenita John, Nujaimah Ahmed, Farah Taher  
**Emails**: jjohnpau@torontomu.ca, nujaimah.ahmed@torontomu.ca

## Project Overview

This project contains our experimental work with the **EmoLLMs** paper and codebase as part of our Master's degree in NLP. EmoLLMs are a series of Emotional Large Language Models designed for comprehensive affective analysis, including emotion detection, sentiment classification, and emotion intensity scoring.

## Original Research

**Paper**: [EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis](https://arxiv.org/abs/2401.08508)

**Authors**: Liu et al., KDD 2024

The original research introduces multiple emotion-aware language models fine-tuned on the AAID (Affective Analysis Instruction Dataset) for various affective computing tasks.

## Project Structure

```
EmoLLMs-jjohnpau/
|
|-- src/
|   |-- experimental-results.ipynb  # Main experimental notebook
|   |-- inference.py                # Batch inference script
|   |-- run_inference.sh           # Shell script for running inference
|   |-- run_sft.sh                 # Shell script for supervised fine-tuning
|   |-- sft_train.py               # Training implementation
|   |-- models/                    # Model configurations
|   |-- sample_generator.py        # Data sampling utilities
|   |-- utils.py                   # Helper functions
|
|-- data/
|   |-- train.json                 # Training dataset
|   |-- dev.json                   # Development dataset
|   |-- test.json                  # Test dataset
|
|-- EmoLLMs.ipynb                 # Professor's report template and academic writeup
```

## Experimental Setup

### Models Tested
- **Emot5-large**: Fine-tuned T5-large model for emotion tasks
- **Emollama-chat-7b**: Fine-tuned LLaMA2-chat-7B model

### Key Dependencies
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

## Experimental Results

### Task 1: Emotion Detection
**Objective**: Identify presence of emotions in text from a predefined set (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).

**Example 1**:
- **Text**: "Whatever you decide to do make sure it makes you happy."
- **EmoLLMs Output**: `joy, optimism`
- **Analysis**: Correctly identified positive emotions with contextual understanding.

**Example 2**:
- **Text**: "There was so much to do today and not enough time."
- **EmoLLMs Output**: `anticipation, fear, sadness`
- **Analysis**: Multi-emotion detection showing nuanced understanding of stress-related emotions.

### Task 2: Sentiment Classification
**Objective**: Classify text sentiment on a 7-point scale (-3 to +3).

**Example 1**:
- **Text**: "Beyoncé resentment gets me in my feelings every time. ð©"
- **EmoLLMs Output**: `-2: moderately negative emotional state can be inferred`
- **Analysis**: Appropriate classification for emotionally charged content.

**Example 2**:
- **Text**: "That was one of the best concert's I've ever been to."
- **EmoLLMs Output**: `3: very positive emotional state can be inferred`
- **Analysis**: Correct strong positive sentiment detection.

### Task 3: Emotion Intensity Scoring
**Objective**: Assign numerical intensity scores (0.0 to 1.0) for specific emotions.

**Example 1**:
- **Text**: "I can't stop smiling today, everything is perfect!"
- **Emotion**: joy
- **EmoLLMs Output**: `0.86`
- **Analysis**: High intensity score appropriate for strongly positive text.

**Example 2**:
- **Text**: "Things have been chaotic recently."
- **Emotion**: joy
- **EmoLLMs Output**: `0.167`
- **Analysis**: Low joy score correctly reflects negative sentiment.

### Task 4: Batch Testing Results
| Text | Target Emotion | EmoLLMs Intensity Score |
|------|----------------|------------------------|
| "I am so angry I could scream." | anger | 0.812 |
| "This is the best news I've heard all year!" | joy | 0.833 |
| "I feel completely empty inside." | sadness | 0.854 |

**Analysis**: Consistent high-intensity scores for emotionally explicit texts.

## Architecture Overview

The EmoLLMs system follows a comprehensive pipeline for affective analysis:

### 1. Data Source
- **SemEval-2018 Task 1: Affect in Tweets (raw data)** - Initial raw data source for training and evaluation

### 2. Instruction Dataset
- **AAID - Affective Analysis Instruction Dataset (234K samples)** - Derived from raw data for instruction tuning
  - **EI-reg** (Emotion intensity regression)
  - **EI-oc** (Emotion ordinal classification)  
  - **V-reg** (Sentiment strength regression)
  - **V-oc** (Sentiment ordinal classification)
  - **E-c** (Multi-label emotion classification)

### 3. Multi-task Instruction Tuning
- **Training Details**: 3 epochs, AdamW optimizer, DeepSpeed, A100 GPUs
- **Models Tuned**:
  - **EmoLLaMA**: 7B / chat-7B / chat-13B
  - **EmoOPT**: OPT-13B base
  - **EmoBLOOM**: BLOOM-7B base
  - **EmoBART**: BART base
  - **EmoT5**: T5 base

### 4. Affective Evaluation Benchmark (AEB)
- **Details**: 8 regression tasks, 6 classification tasks, 14 datasets
- **Components**:
  - **AEB-1** (Training effectiveness test): Same domain as AAID
  - **AEB-2** (Generalization test): Cross-domain datasets

### 5. Comparison Baselines
- **PLMs** (Pre-trained Language Models): BERT, RoBERTa, SentiBERT
- **Zero/few-shot LLMs** (Large Language Models): ChatGPT, GPT-4, LLaMA2, Vicuna, Falcon
- **SA tools** (Sentiment Analysis tools): VADER, TextBlob, rule-based tools

## Architecture Diagram

![EmoLLMs Architecture](images/EmoLLMs_architecture.jpg)

*Figure: Comprehensive pipeline showing data flow from SemEval-2018 raw data through AAID instruction dataset, multi-task instruction tuning, Affective Evaluation Benchmark (AEB), and comparison with baseline models.*

## Key Learnings

### 1. Prompt Engineering Criticality
- Precise task descriptions are essential for accurate outputs
- Consistent formatting improves model reliability
- Multi-emotion detection requires careful prompt design

### 2. Strengths of EmoLLMs
- **Specialized training** on affective data provides domain expertise
- **Multi-task capability** handles classification, regression, and detection
- **Fine-grained emotion understanding** beyond basic sentiment
- **Numerical intensity scoring** provides quantitative emotion analysis

### 3. Limitations Observed
- **Context dependency** - performance varies with text complexity
- **Emotion overlap** - sometimes struggles with ambiguous emotional content
- **Cultural nuances** - may miss culturally specific emotional expressions
- **Computational requirements** - larger models need significant resources

## Comparison with ChatGPT (Latest Version)

### Experimental Comparison Framework

#### ChatGPT GPT-5 Experimental Results

**Sentiment Strength Task**:
- **Text**: "Happy Birthday shorty. Stay fine stay breezy stay wavy @daviistuart ð"
- **Task**: Evaluate valence intensity (0 = most negative, 1 = most positive)
- **ChatGPT Output**: `0.91`
- **EmoLLMs Comparison**: `0.879` (from original experiments)

**Sentiment Classification Task**:
- **Text**: "Beyoncé resentment gets me in my feelings every time. ð©"
- **Task**: Ordinal classification (-3 to +3 scale)
- **ChatGPT Output**: `-1` (slightly negative)
- **EmoLLMs Comparison**: `-2` (moderately negative)

### Comparative Analysis Framework
**Metrics to Compare:**
- Accuracy in emotion detection
- Consistency in sentiment classification
- Precision in intensity scoring
- Response coherence and relevance
- Handling of ambiguous or mixed emotions

**Initial Observations:**
1. **Sentiment Strength**: ChatGPT (0.91) vs EmoLLMs (0.879) - ChatGPT shows slightly higher positive intensity
2. **Sentiment Classification**: ChatGPT (-1) vs EmoLLMs (-2) - ChatGPT less severe in negative classification
3. **Response Format**: ChatGPT provides direct numerical outputs, EmoLLMs includes explanatory text

**Expected Areas of Comparison:**
1. **Task Performance**: Accuracy across different emotion tasks
2. **Response Quality**: Coherence and relevance of outputs
3. **Specialization**: Domain-specific knowledge vs general capability
4. **Consistency**: Reliability across similar inputs
5. **Speed**: Inference time and computational efficiency

## Technical Implementation Details

### Inference Pipeline
```python
def run_task(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Batch Processing
- Supports batch inference for large datasets
- Configurable batch sizes for memory optimization
- Progress tracking with tqdm

### Model Loading
```python
tokenizer = AutoTokenizer.from_pretrained('lzw1008/Emot5-large')
model = AutoModelForCausalLM.from_pretrained('lzw1008/Emot5-large', device_map='auto')
```

## Challenges and Solutions

### 1. Resource Management
- **Challenge**: Large model memory requirements
- **Solution**: Used `device_map='auto'` for automatic GPU/CPU distribution

### 2. Prompt Consistency
- **Challenge**: Maintaining consistent prompt formats
- **Solution**: Created standardized prompt templates for each task type

### 3. Output Parsing
- **Challenge**: Extracting clean numerical scores from model outputs
- **Solution**: Implemented post-processing for result standardization

## Future Work

### 1. Extended Evaluation
- Larger dataset testing
- Cross-domain performance analysis
- Multilingual emotion detection

### 2. Model Improvements
- Fine-tuning on domain-specific data
- Ensemble methods for improved accuracy
- Real-time emotion detection systems

### 3. Applications
- Sentiment analysis for social media monitoring
- Customer feedback emotion analysis
- Mental health support systems
- Content moderation tools

## Usage Instructions

### Running Experiments
```bash
# Clone the repository
git clone [repository-url]
cd EmoLLMs

# Install dependencies
pip install transformers torch tqdm pandas

# Run inference
bash src/run_inference.sh

# View experimental results
jupyter notebook src/experimental-results.ipynb
```

### Custom Experiments
1. Load the experimental notebook
2. Modify prompts and test cases
3. Compare results with baseline models
4. Document findings in the comparison section

## Academic Contributions

This experimental work contributes to:
- **Practical understanding** of emotion-aware LLMs
- **Benchmarking framework** for emotion detection models
- **Comparative analysis** between specialized and general-purpose models
- **Implementation insights** for affective computing applications

## References

1. Liu, Z., Yang, K., Xie, Q., Zhang, T., & Ananiadou, S. (2024). EmoLLMs: A Series of Emotional Large Language Models and Annotation Tools for Comprehensive Affective Analysis. KDD 2024.

2. Original EmoLLMs repository: https://github.com/lzw1008/EmoLLMs

3. Hugging Face model hub: https://huggingface.co/lzw1008/

---

## Academic Report Template (Professor's Format)

The following structured report format is provided in `EmoLLMs.ipynb` following the professor's template:

### Report Structure:
1. **Introduction** - Problem description, context, limitations of existing approaches, proposed solution
2. **Background** - Related work analysis table comparing different approaches
3. **Methodology** - Dataset details, model architecture, evaluation approach
4. **Implementation** - Code examples and explanations
5. **Conclusion and Future Direction** - Learnings, limitations, future extensions
6. **References** - Academic citations

**Note**: This README documents our experimental work and learnings as part of our Master's degree research in Natural Language Processing. The comparison with ChatGPT results will be updated as experiments are completed.
