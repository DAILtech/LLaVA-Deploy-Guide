#!/usr/bin/env python3
import argparse
import gradio as gr
from llava_deploy import utils

def main():
    parser = argparse.ArgumentParser(description="Launch Gradio web UI for LLaVA model.")
    parser.add_argument("--model", type=str, default="llava-1.5-7b",
                        help="Model version or path to load (default: llava-1.5-7b).")
    parser.add_argument("--share", action="store_true",
                        help="If set, share the Gradio app publicly (creates a shareable link).")
    args = parser.parse_args()

    # Load model and processor
    print(f"[Info] Loading model {args.model} for web UI... (this may take a while)")
    model, processor = utils.load_model(args.model, use_4bit=True, device_map="auto")
    print(f"[Info] Model {args.model} loaded. Starting Gradio interface...")

    def answer_question(image, question):
        if image is None or question is None or question.strip() == "":
            return "Please provide both an image and a question."
        # Use the global model and processor
        return utils.answer_question(model, processor, image, question)

    # Prepare example inputs (using demo images and prompts)
    examples = [
        [ "examples/images/demo1.jpg", "What is in this image?" ],
        [ "examples/images/demo1.jpg", "这张图片中有什么？" ],
        [ "examples/images/demo2.png", "Describe the scene." ],
        [ "examples/images/demo2.png", "描述一下这个场景。" ]
    ]

    # Create Gradio interface
    description_text = "Upload an image and ask a question. The model will describe or answer questions about the image."
    interface = gr.Interface(
        fn=answer_question,
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Textbox(lines=1, placeholder="Ask a question about the image...", label="Question")
        ],
        outputs=gr.Textbox(label="Answer"),
        examples=examples,
        title="LLaVA-Deploy-Tutorial Web UI",
        description=description_text,
    )
    # Launch the app
    interface.launch(server_name="0.0.0.0", share=args.share)

if __name__ == "__main__":
    main()
