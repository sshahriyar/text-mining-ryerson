Installation
We recommend to install Align in a conda environment.

First clone this repo:

git clone https://github.com/yuh-zha/Align.git
cd Align
Create a virtual conda environment:

conda create -n Align python=3.9
conda activate Align
pip install -e .
Install the required spaCy model

python -m spacy download en_core_web_sm
Checkpoints
We provide two versions of Align checkpoints: Align-base and Align-large. The -base model is based on RoBERTa-base and has 125M parameters. The -large model is based on RoBERTa-large and has 355M parameters.

Align-base: https://huggingface.co/yzha/Align/resolve/main/Align-base.ckpt

Align-large: https://huggingface.co/yzha/Align/resolve/main/Align-large.ckpt

Usage
To get the alignment score of the text pairs (text_a and text_b), use the scorer function of Align:

from align import Align

text_a = ["Your text here"]
text_b = ["Your text here"]

scorer = Align(model="roberta-large", batch_size=32, device="cuda", ckpt_path="path/to/ckpt")
score = scorer(contexts=text_a, claims=text_b)
model: The backbone model of Align. It can be roberta-baseor roberta-large
batch_size: The batch size of inference 
ckpt_path: The path to the checkpoint
