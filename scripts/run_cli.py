#!/usr/bin/env python3
import argparse
import sys
from llava_deploy import utils

def main():
    parser = argparse.ArgumentParser(description="Run LLaVA model in command-line interactive mode.")
    parser.add_argument("--model", type=str, default="llava-1.5-7b",
                        help="Model version or path (e.g., llava-1.5-7b, llava-1.6-13b, or local path to model directory).")
    parser.add_argument("--image", type=str, required=True, 
                        help="Path to the image file for the question.")
    parser.add_argument("--question", type=str, default=None,
                        help="Question to ask about the image. If not provided, enters interactive chat mode.")
    args = parser.parse_args()

    # Load image
    try:
        image = utils.load_image(args.image)
    except Exception as e:
        print(f"[Error] Failed to load image: {e}")
        sys.exit(1)
    # Load model and processor (with 4-bit quantization by default)
    print(f"[Info] Loading model {args.model} (this may take a while)...")
    model, processor = utils.load_model(args.model, use_4bit=True, device_map="auto")
    print(f"[Info] Model {args.model} loaded successfully.")

    # Interactive mode
    if args.question is None:
        print(f"Image '{args.image}' loaded. You can now ask questions about the image.")
        print("Type 'exit' to quit.")
        while True:
            try:
                user_input = input("Question: ")
            except (EOFError, KeyboardInterrupt):
                print("\n[Info] Exiting.")
                break
            if user_input.strip().lower() in ["exit", "quit"]:
                print("[Info] Exiting interactive mode.")
                break
            if user_input.strip() == "":
                continue  # skip empty question
            # Get answer
            answer = utils.answer_question(model, processor, image, user_input)
            print(f"Answer: {answer}")
    else:
        # Single question mode
        answer = utils.answer_question(model, processor, image, args.question)
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
