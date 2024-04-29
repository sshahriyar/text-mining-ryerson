# Unveiling the Power of Large Language Models in Text-to-Image Synthesis Evaluation
---

```
Group Members: 
1.   Muhammad Umer (m1umer@torontomu.ca)
2.   Kami Wang (kami.wang@torontomu.ca)
```

# General Overview 

##### Problem Description:

This paper aims to explore a new framework that offers evaluation scores with multi-granularitybcompositionality by using language models(LLMs) called LLMScore. By merging various features, LLMScore generates comprehensive image descriptions. It uses LLMs to assess aligenment between provide images and text that user provides. This approach provides a powerful method for assessing the accuracy of provide text and images. It provides a more accurate assessment than existing methods.

#### Context of the Problem:

Existing methods use automatic evaluation on text-to-image synthesis provide only a single score that shows how well the text matches the whole image, without considering the details of the objects in the image. This limitation results in a poor correlation with human judgements, since human considers the differences of every single item on the image.

#### Limitation About other Approaches:

The paper also discusses the limitations and potential biases associated with using LLMs. The challenges such as accurately capturing composite text-to-image alignment, for example couting, color, location, etc. They will affect producing the score and evaluating with various objects.

#### Solution:

LLMScore utilizes the incredible text reasoning ad generation ability of Large Language Models(LLMs) to imitate human evaluation processes, offering insights into model performance by assessing compositionality at multi-levels. Then by combining visual and language models with text prompts, LLMScore transforms images into detailed descriptions, enabling the evaluation of alignment with text prompts using LLMs such as GPT-4.

----
# Technical Details

## Package Requirements
1. opencv-python
2. mss 
3. timm 
4. ftfy 
5. regex
6. fasttext 
7. scikit-learn 
8. lvis 
9. nltk 
10. tqdm
11. matplotlib 
12. requests 
13. anytree boto3 
14. scikit-image 
15. pyyaml 
16. inflect 
17. icecream 
18. openai 
19. protobuf 
20. einops 
21. deepspeed
22. dataclasses

The project can be setup and run using three different ways. The details for each is given further below
1. Google Colab / Notebook
2. Docker Image
3. Anaconda Venv

### Google Colab
```
import os
# Set the OPENAI_KEY environment variable (you need to replace with your actual key)
os.environ['OPENAI_KEY'] = # Replace YOUR_OPENAI_KEY with your actual OpenAI API key
os.environ['OPENAI_API_KEY'] = # Replace YOUR_OPENAI_KEY with your actual OpenAI API key

# Clone the LLM repository
!git clone https://github.com/YujieLu10/LLMScore.git
%cd LLMScore

# Initiate and update the Detectron2 submodule
!git submodule update --init submodule/detectron2

# Install the Hugging Face Transformers library
%cd ../../
!pip install git+https://github.com/huggingface/transformers
%cd /content/LLMScore

# Download the GRiT Model
!mkdir -p models
%cd models
!wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
%cd ..

# Install relevant dependencies
%cd /content/LLMScore
!pip install opencv-python mss timm ftfy regex fasttext scikit-learn lvis nltk tqdm matplotlib requests anytree boto3 scikit-image pyyaml inflect icecream openai protobuf einops deepspeed

# Install depedencies for detectron2
%cd /content/LLMScore/submodule/detectron2
!pip install -e .

%cd /content/LLMScore
```
### Docker Image
```
# Use a base image with the necessary dependencies (e.g., Python, CUDA, etc.)
FROM continuumio/miniconda3:latest

# Install necessary system dependencies including libGL
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    wget \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variablesb
ENV OPENAI_KEY=YOUR_KEY
ENV OPENAI_KEY=YOUR_KEY

# Clone the LLMScore repository
#RUN git clone https://github.com/YujieLu10/LLMScore.git /LLMScore

# Set the working directory in the container
WORKDIR /LLMScore

# Copy the local LLMScore repository to the container
COPY . /LLMScore

# Create and activate conda environment
RUN conda create -n llmscore python=3.8 -y && \
    echo "source activate llmscore" > ~/.bashrc
SHELL ["/bin/bash", "-c"]
RUN /bin/bash -c "source activate llmscore"

# Install PyTorch
RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y

# Install requirements
RUN pip install -r requirements.txt && \
    git submodule update --init && \
    cd submodule/detectron2 && \
    pip install -e .

RUN pip install Pillow --upgrade

# Download GRiT model
RUN mkdir models && \
    cd models && \
    wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth && \
    cd ..

# Set default command to start bash shell
CMD ["/bin/bash"]

```
### Conda Venv
```
conda create -n llmscore python==3.8 -y
conda activate llmscore
git clone https://github.com/YujieLu10/LLMScore.git
cd LLMScore
pip install -r requirements.txt

git submodule update --init
cd submodule/detectron2
pip install -e .

pip install git+https://github.com/huggingface/transformers

export OPENAI_KEY=YOUR_OPENAI_KEY
## GRiT Model Downloading
mkdir models && cd models
wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth && cd ..
```

### Post Setup
The entry point for the script is in the file `llm_score.py`. This file can be used to evaluate a picture and a prompt
##### Sample usage
```
python llm_score.py --image sample/sample.png --text_prompt "a red car and a white sheep"
```
---

## License

MIT

