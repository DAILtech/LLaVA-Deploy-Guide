import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

def load_model(model_name="llava-1.5-7b", use_4bit=True, device_map="auto"):
    """
    Load the LLaVA model and processor.
    - model_name: can be a shorthand (e.g., 'llava-1.5-7b', 'llava-1.6-13b') or a path to a local model.
    - use_4bit: whether to load the model in 4-bit quantized mode (requires bitsandbytes).
    - device_map: device placement for model. 'auto' will spread layers across available GPUs/CPU.
    Returns: (model, processor)
    """
    # Map shorthand names to huggingface repo IDs
    hf_repo = model_name
    if model_name == "llava-1.5-7b":
        hf_repo = "liuhaotian/llava-v1.5-7b"
    elif model_name == "llava-1.5-13b":
        hf_repo = "liuhaotian/llava-v1.5-13b"
    elif model_name == "llava-1.6-7b":
        hf_repo = "liuhaotian/llava-v1.6-vicuna-7b"
    elif model_name == "llava-1.6-13b":
        hf_repo = "liuhaotian/llava-v1.6-vicuna-13b"
    # Load the model
    load_kwargs = {}
    if use_4bit:
        load_kwargs.update({
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "device_map": device_map
        })
    else:
        load_kwargs.update({
            "torch_dtype": torch.float16,
            "device_map": device_map,
            "low_cpu_mem_usage": True
        })
    model = LlavaForConditionalGeneration.from_pretrained(hf_repo, **load_kwargs)
    processor = AutoProcessor.from_pretrained(hf_repo)
    return model, processor

def load_image(image_path):
    """Load an image from the given path and return a PIL Image in RGB format."""
    image = Image.open(image_path)
    return image.convert("RGB")

def format_prompt(question):
    """
    Format the user question into the LLaVA conversation prompt format.
    LLaVA expects a prompt with a special <image> token in the USER query.
    """
    question = question.strip()
    # Ensure the prompt ends with the ASSISTANT: prefix for model to generate answer
    prompt = f"USER: <image>\\n{question} ASSISTANT:"
    return prompt

def answer_question(model, processor, image, question):
    """Generate answer from the model for a given image and question."""
    if isinstance(image, str):
        image = load_image(image)
    prompt = format_prompt(question)
    # Prepare inputs for the model
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    # Generate answer
    outputs = model.generate(**inputs, max_new_tokens=256)
    # Decode the generated tokens to text
    output_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    # The output may contain the prompt (USER and ASSISTANT). Extract answer after 'ASSISTANT:' if present.
    if "ASSISTANT:" in output_text:
        answer = output_text.split("ASSISTANT:")[-1].strip()
    else:
        answer = output_text.strip()
    return answer
