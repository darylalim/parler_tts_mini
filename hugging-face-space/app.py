import spaces
import gradio as gr
import torch

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_name = "parler-tts/parler_tts_mini_v0.1"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

sr = feature_extractor.sampling_rate

examples = [
    [
        "Hey, how are you doing today?",
        "A female speaker with a slightly high-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."
    ],
    [
        "The life of the land is perpetuated in righteousness.",
        "A male speaker with a low-pitched voice delivers his words at a slightly slow pace and a dramatic tone, in a very spacious environment, accompanied by noticeable background noise."
    ]
]

@spaces.GPU
def generate_speech(text, description):
    """
    Generate speech with a text prompt.
    """
    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    generation = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids,
        do_sample=True,
        temperature=1.0
    )
    audio_arr = generation.cpu().numpy().squeeze()
    
    return sr, audio_arr

with gr.Blocks() as demo:
    gr.Markdown("# Parler-TTS Mini")
    gr.Markdown(
        """
        Tips:
        - Include term "very clear audio" and/or "very noisy audio"
        - Use punctuation for prosody
        - Control gender, speaking rate, pitch, reverberation in prompt
        """
    )
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text",
                lines=2,
                elem_id="input_text"
            )
            description = gr.Textbox(
                label="Description",
                lines=2,
                elem_id="input_description"
            )
            run_button = gr.Button("Generate Audio", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(
                label="Parler-TTS generation",
                type="numpy",
                elem_id="audio_out"
            )
    
    inputs = [input_text, description]
    outputs = [audio_out]
    gr.Examples(
        examples=examples,
        fn=generate_speech,
        inputs=inputs,
        outputs=outputs,
        cache_examples=True
    )
    run_button.click(
        fn=generate_speech,
        inputs=inputs,
        outputs=outputs,
        queue=True
    )

demo.queue()

demo.launch()