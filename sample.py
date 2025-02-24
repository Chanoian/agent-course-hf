import os
from huggingface_hub import InferenceClient

with open(".hf_token", "r") as file:
    hf_token = file.read().strip()

os.environ["HF_TOKEN"] = hf_token
client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct")

# Using Text Generation
output = client.text_generation(
    "The Capital of Armenia is",
    max_new_tokens=100,
)
print(output)

# Using Prompt
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
The capital of Armenia is<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt_output = client.text_generation(
    prompt,
    max_new_tokens=100,
)
print("\n\n")
print(prompt_output)


# Using Chat Method
chat_output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of Armenia is"},
    ],
    stream=False,
    max_tokens=1024,
)
print("\n\n")
print(chat_output.choices[0].message.content)

