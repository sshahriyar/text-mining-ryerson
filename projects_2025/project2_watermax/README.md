### Setup Instructions:

- Project needs to run on Google Colab as meta-llama/Llama-2-7b-chat-hf model needs to run and it requires good GPU support (we ran the project using A100 GPU on Google Colab)
- In order to clone meta-llama/Llama-2-7b-chat-hf gated model you need to request access on hugging face website from here: https://huggingface.co/meta-llama/Llama-2-7b-hf
- Once the access is approved, you need to setup SSH key on Google Colab using the steps mentioned below (reference: https://huggingface.co/docs/hub/en/security-git-ssh):

1. Generate SSH key
```python
!ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

2. Add SSH key to your Hugging Face account
- Go to Hugging Face > Settings > SSH and GPG keys > Add SSH key
- Copy the public key from the following command
```python
!cat ~/root/.ssh/id_rsa.pub
```

3. Add the following ssh key entries to your ~/root/.ssh/known_hosts file on Google Colab to avoid manually verifying HuggingFace hosts:
```console
hf.co ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDtPB+snz63eZvTrbMY2Qt39a6HYile89JOum55z3lhIqAqUHxLtXFd+q+ED8izQvyORFPSmFIaPw05rtXo37bm+ixL6wDmvWrHN74oUUWmtrv2MNCLHE5VDb3+Q6MJjjDVIoK5QZIuTStlq0cUbGGxQk7vFZZ2VXdTPqgPjw4hMV7MGp3RFY/+Wy8rIMRv+kRCIwSAOeuaLPT7FzL0zUMDwj/VRjlzC08+srTQHqfoh0RguZiXZQneZKmM75AFhoMbP5x4AW2bVoZam864DSGiEwL8R2jMiyXxL3OuicZteZqll0qfRlNopKnzoxS29eBbXTr++ILqYz1QFqaruUgqSi3MIC9sDYEqh2Q8UxP5+Hh97AnlgWDZC0IhojVmEPNAc7Y2d+ctQl4Bt91Ik4hVf9bU+tqMXgaTrTMXeTURSXRxJEm2zfKQVkqn3vS/zGVnkDS+2b2qlVtrgbGdU/we8Fux5uOAn/dq5GygW/DUlHFw412GtKYDFdWjt3nJCY8=
hf.co ssh-dss AAAAB3NzaC1kc3MAAACBAORXmoE8fn/UTweWy7tCYXZxigmODg71CIvs/haZQN6GYqg0scv8OFgeIQvBmIYMnKNJ7eoo5ZK+fk1yPv8aa9+8jfKXNJmMnObQVyObxFVzB51x8yvtHSSrL4J3z9EAGX9l9b+Fr2+VmVFZ7a90j2kYC+8WzQ9HaCYOlrALzz2VAAAAFQC0RGD5dE5Du2vKoyGsTaG/mO2E5QAAAIAHXRCMYdZij+BYGC9cYn5Oa6ZGW9rmGk98p1Xc4oW+O9E/kvu4pCimS9zZordLAwHHWwOUH6BBtPfdxZamYsBgO8KsXOWugqyXeFcFkEm3c1HK/ysllZ5kM36wI9CUWLedc2vj5JC+xb5CUzhVlGp+Xjn59rGSFiYzIGQC6pVkHgAAAIBve2DugKh3x8qq56sdOH4pVlEDe997ovEg3TUxPPIDMSCROSxSR85fa0aMpxqTndFMNPM81U/+ye4qQC/mr0dpFLBzGuum4u2dEpjQ7B2UyJL9qhs1Ubby5hJ8Z3bmHfOK9/hV8nhyN8gf5uGdrJw6yL0IXCOPr/VDWSUbFrsdeQ==
hf.co ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBL0wtM52yIjm8gRecBy2wRyEMqr8ulG0uewT/IQOGz5K0ZPTIy6GIGHsTi8UXBiEzEIznV3asIz2sS7SiQ311tU=
hf.co ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINJjhgtT9FOQrsVSarIoPVI1jFMh3VSHdKfdqp/O776s
```

4. Verify the SSH connection
```python
!ssh -T git@hf.co
```

5. Clone the meta-llama/Llama-7b-hf model within the src/ directory
```python
!git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

6. Generate API token from HuggingFace
- Go to https://huggingface.co/settings/tokens
- Click on "Create new token" and generate a new token
- Copy the token to clipboard

7. Login to HuggingFace from your python notebook using the following code and copy the token from the clipboard
```python
from huggingface_hub import login
login()
```

8. To run the code execute any of the following commands
```python
!python watermax.py --model_name meta-llama/Llama-2-7b-chat-hf --generate --detect --seed=815 --ngram=4 --n=2 --N=2 --prompts data/test_prompts.txt

!python watermax.py --model_name meta-llama/Llama-2-7b-chat-hf --generate --detect --seed=926 --ngram=6 --n=2 --N=2 --fp16 --prompts "What was Spinoza's relationship with Leibniz?" "Which philospher spoke about the multicolored cow?"
```