import os
import json
import logging
import argparse
import pandas as pd

from tqdm import tqdm


def configure_logging():
    logging.basicConfig(level=logging.INFO)


def load_template_idx_to_template_info():
    _dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_dir, "./template_idx_to_template_info.json"), "r") as f:
        template_idx_to_template_info = json.load(f)
    return template_idx_to_template_info


def process_sample(sample, template_idx_to_template_info, split, sample_idx):
    try:
        question = str(sample["question"]).lower()
        template = str(sample["para_info"]["ori_template"]).lower()
        query = str(sample["query"]).lower()
        value = sample["value"]
        q_tag = template_idx_to_template_info[str(sample["template_idx"])]["question_template"]
        t_tag = sample["t_tag"]
        o_tag = sample["o_tag"]
        v_tag = sample["v_tag"]
        tag = sample["tag"]

        # NOTE: will be removed later
        _gold_program = str(sample["gold_sql"]).lower()
        _answer = sample["answer"]

        new_sample = {
            "db_id": "mimic_iv_cxr",
            "split": split,
            "id": sample_idx,
            "question": question,
            "template": template,
            "query": query,
            "value": value,
            "q_tag": q_tag,
            "t_tag": t_tag,
            "o_tag": o_tag,
            "v_tag": v_tag,
            "tag": tag,
            "para_type": "machine",
            "is_impossible": False,
            "_gold_program": _gold_program,
        }

        if args.debug:
            new_sample["_answer"] = _answer

    except KeyError as e:
        logging.error("KeyError: Missing expected field in sample: %s", e)
        return None

    except Exception as e:
        logging.error("Unexpected error while processing sample: %s", e)
        return None

    return new_sample


def formatting_qa_dataset(qa_file_path, save_file_path, template_idx_to_template_info, split):
    try:
        dataset = pd.read_json(qa_file_path).to_dict("records")
    except FileNotFoundError:
        logging.error("QA data not found: %s", qa_file_path)
        return
    except pd.errors.EmptyDataError:
        logging.error("QA data is empty: %s", qa_file_path)
        return

    logging.info("len(dataset): %d", len(dataset))
    logging.debug("dataset[0]: %s", dataset[0])

    new_dataset = []
    for sample_idx, sample in enumerate(tqdm(dataset)):
        processed_sample = process_sample(sample, template_idx_to_template_info, split, sample_idx)
        if processed_sample is not None:
            new_dataset.append(processed_sample)

    if not new_dataset:
        logging.warning("No valid samples processed. Skipping file write.")
        return

    # save
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))

    with open(save_file_path, "w") as f:
        json.dump(new_dataset, f, indent=2)

    # check
    with open(save_file_path, "r") as f:
        new_dataset = json.load(f)
    logging.info("len(new_dataset): %d", len(new_dataset))
    logging.debug("new_dataset[0]: %s", new_dataset[0])


def main(args):
    configure_logging()
    template_idx_to_template_info = load_template_idx_to_template_info()
    for split in ["train", "valid", "test"]:
        csv_file_path = os.path.join(args.data_dir, f"{split}_para.json")
        json_file_name = f"_{split}_debug.json" if args.debug else f"_{split}.json"
        json_file_path = os.path.join(args.save_dir, json_file_name)
        formatting_qa_dataset(csv_file_path, json_file_path, template_idx_to_template_info, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert QA dataset.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the original QA json files.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the JSON files.")
    args = parser.parse_args()
    main(args)

    # python ./dataset_builder/internal_utils/_format_qa_dataset.py --debug  --data_dir="/nfs_data_storage/ehrxqa/dataset/mimic_iv_cxr/" --save_dir="./dataset/mimic_iv_cxr/"
    # python ./dataset_builder/internal_utils/_format_qa_dataset.py  --data_dir="/nfs_data_storage/ehrxqa/dataset/mimic_iv_cxr/" --save_dir="./dataset/mimic_iv_cxr/"
