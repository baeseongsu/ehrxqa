import os
import json
import time
import pickle
import argparse
from datetime import datetime
from pathlib import Path

# custom package
from parser.openai_parser import OpenAIParser


def create_timestamped_subdirectory(base_dir):
    """
    Create a subdirectory in the specified base directory with a name based on the current date and time.
    """
    current_datetime = datetime.now()
    subdir_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S_%f")
    directory_path = Path(base_dir) / subdir_name

    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        if directory_path.is_dir():
            print(f"Directory '{directory_path}' created successfully.")
            return str(directory_path)
        else:
            print(f"Directory '{directory_path}' already exists.")
            return str(directory_path)
    except OSError as error:
        print(f"Error creating directory '{directory_path}': {error}")
        return None


def main(args):
    # Build paths
    args.prompt_file_path = os.path.join(args.prompt_file_path) if args.prompt_file_path else ""
    args.save_dir = create_timestamped_subdirectory(os.path.join(args.save_dir))
    os.makedirs(args.save_dir, exist_ok=True)

    start_time = time.time()

    # Load dataset
    with open(os.path.join(args.dataset_dir, args.database, args.examples_dataset_name), "r") as f:
        examples_dataset = json.load(f)
    with open(os.path.join(args.dataset_dir, args.database, args.target_dataset_name), "r") as f:
        target_dataset = json.load(f)

    if args.debug:
        print("We are in debug mode. Only using the first 5 examples.")
        target_dataset = target_dataset[:5]

    # Load OpenAI API key
    with open(args.api_keys_file, "r") as f:
        openai_api_keys = f.readlines()
    openai_api_key = openai_api_keys[0]

    # Initialize OpenAIParser
    parser = OpenAIParser(
        api_key=openai_api_key,
        model_name=args.model_name,
        max_context_length=args.max_context_length,
        max_tokens=args.max_tokens,
    )

    # Parse target dataset
    responses = parser.parse(
        target_dataset=target_dataset,
        examples_dataset=examples_dataset,
        num_examples=args.num_examples,
        prompt_file_path=args.prompt_file_path,
        return_content_only=args.return_content_only,
        verbose=args.verbose,
    )

    assert len(responses) == len(target_dataset)

    # Extract predictions
    predictions = {}
    for i, response in enumerate(responses):
        if args.return_content_only:
            predictions[target_dataset[i]["id"]] = response
        else:
            predictions[target_dataset[i]["id"]] = response.choices[0].message.content if response is not None else ""

    # Save results
    with open(os.path.join(args.save_dir, "arguments.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    with open(os.path.join(args.save_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=4)

    if not args.return_content_only:
        with open(os.path.join(args.save_dir, "parsed_results.pickle"), "wb") as f:
            pickle.dump(responses, f)

    print(f"Elapsed time: {time.time() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # File path or name
    parser.add_argument("--dataset", type=str, default="ehrxqa", choices=["ehrxqa"])
    parser.add_argument("--database", type=str, default="mimic_iv_cxr", choices=["mimic_iv_cxr"])
    parser.add_argument("--dataset_dir", type=str, default="../../dataset")
    parser.add_argument("--examples_dataset_name", type=str, default="train.json", help="Dataset to use as few-shot examples")
    parser.add_argument("--target_dataset_name", type=str, default="valid.json", choices=["valid.json", "test.json"], help="Dataset to parse")
    parser.add_argument("--api_keys_file", type=str, default="./parser/api_key/OPENAI_API_KEY.txt")
    parser.add_argument("--save_dir", type=str, default="./results/parser")

    # Multiprocess options
    parser.add_argument("--n_processes", type=int, default=1)

    # General options
    parser.add_argument("--seed", type=int, default=42)

    # Prompt options
    parser.add_argument("--prompt_file_path", type=str, default=None)
    parser.add_argument("--prompt_style", type=str, default="retrieval-bm25", choices=["", "retrieval-bm25"])
    parser.add_argument("--num_examples", type=int, default=10)

    # GPT options
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo-0613",
        choices=[
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
        ],
    )
    parser.add_argument("--max_context_length", type=int, default=4097)
    parser.add_argument("--max_tokens", type=int, default=768)
    # parser.add_argument("--temperature", type=float, default=0.0)
    # parser.add_argument("--sampling_n", type=int, default=1)
    # parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--stop_tokens", type=str, default="\n\n", help="Split stop tokens with ||")
    parser.add_argument("--return_content_only", action="store_true")

    # Debug options
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args.stop_tokens = args.stop_tokens.split("||")
    print("Args info:")
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    main(args)
