import gradio as gr
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline
)
import torch

model = T5ForConditionalGeneration.from_pretrained("./T5-Fine-Tuned")
tokenizer = T5Tokenizer.from_pretrained("./T5-Fine-Tuned")

pipe = pipeline(
    "text2text-generation",
    model = model,
    tokenizer = tokenizer,
    device = 0 if torch.cuda.is_available() else -1
)

def preprocess(task, input_text, context = None):
    if task == "Summary":
       prompt =  f"summarize: {input_text}"
    elif task == "Sentiment":
       prompt =  f"sentiment: {input_text}"
    elif task == "Questions & Answers":
       prompt =  f"question: {input_text} context: {context}"
    prediction = pipe(prompt, max_length = 128)
    return prediction[0]['generated_text']

with gr.Blocks() as demo:
    task = gr.Dropdown(["Summary", "Sentiment", "Questions & Answers"], label="Task")
    text = gr.Textbox(label="Input Text")
    context = gr.Textbox(label="Context (QA only)", visible=False)
    output = gr.Textbox(label="Result")
    
    task.change(
        lambda x: gr.Textbox(visible=x == "Questions & Answers"),
        inputs=task,
        outputs=context
    )
    
    btn = gr.Button("Run")
    btn.click(preprocess, inputs=[task, text, context], outputs=output)

demo.launch()