import openai
from keys import mykey
import gradio as gr

# Set your OpenAI API key
openai.api_key = mykey["mediQ"]

# Chat function
def gpt_response(message, history=[]):
    # Reformat history into proper OpenAI chat message format
    messages = [
        {"role": "system", "content": "You are a careful and experienced clinical reasoning expert."}
    ]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    # Call OpenAI GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )

    reply = response["choices"][0]["message"]["content"]
    return reply

# Gradio chat interface
gr.ChatInterface(
    fn=gpt_response,
    title="🧠 MediQ GPT-4 Clinical Reasoning",
    description="Ask a clinical question. GPT-4 will simulate adaptive expert reasoning.",
).launch()
