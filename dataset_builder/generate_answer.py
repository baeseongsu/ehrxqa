import os
import re
import json
import pandas as pd
import logging
from tqdm import tqdm


def load_database(db_file_path):
    import sqlite3

    conn = sqlite3.connect(db_file_path)
    cur = conn.cursor()
    return cur


def post_process_sql(
    query,
    current_time="2105-12-31 23:59:00",
    precomputed_dict={
        "temperature": (35.5, 38.1),
        "sao2": (95.0, 100.0),
        "heart rate": (60.0, 100.0),
        "respiration": (12.0, 18.0),
        "systolic bp": (90.0, 120.0),
        "diastolic bp": (60.0, 90.0),
        "mean bp": (60.0, 110.0),
    },
):
    # handle current_time
    if "current_time" in query:
        query = query.replace("current_time", f"'{current_time}'")

    # handle vital signs
    if re.search("[ \n]+([a-zA-Z0-9_]+_lower)", query) and re.search("[ \n]+([a-zA-Z0-9_]+_upper)", query):
        vital_lower_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_lower)", query)[0]
        vital_upper_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_upper)", query)[0]
        vital_name_list = list(set(re.findall("([a-zA-Z0-9_]+)_lower", vital_lower_expr) + re.findall("([a-zA-Z0-9_]+)_upper", vital_upper_expr)))
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace("_", " ")
            if processed_vital_name in precomputed_dict:
                vital_range = precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")

    # handle etc.
    query = query.replace("''", "'").replace("< =", "<=")
    query = query.replace("%y", "%Y").replace("%j", "%J")
    query = query.replace("‘", "'").replace("’", "'")
    query = query.replace("\u201c", '"').replace("\u201d", '"')

    return query


def post_process_answer(answer, round_digit=6, sorted_answer=False):
    assert isinstance(answer, list)

    if len(answer) == 0:  # empty answer
        assert answer == []
    else:
        # tuple data preprocessing
        if isinstance(answer[0], tuple):
            assert len(answer[0]) == 1
            answer = [ans[0] for ans in answer]

        if isinstance(answer[0], float):  # float-type answer
            answer = [round(ans, round_digit) for ans in answer]
        elif isinstance(answer[0], str):  # string-type answer
            if sorted_answer:
                answer = sorted(answer)
        else:
            pass

    return answer


def main(args):
    # Initialize error count
    answer_error_cnt = 0

    # Read json file
    dataset = json.load(open(args.json_file_path))

    # Load database
    cur = load_database(db_file_path=args.db_file_path)

    new_dataset = []
    for data in tqdm(dataset):
        try:
            db_id = data["db_id"]
            assert db_id == "mimic_iv_cxr", "db_id should be mimic_iv_cxr."

            gold_program = data["_gold_program"]
            gold_program = post_process_sql(gold_program)

            res = cur.execute(gold_program)
            answer = res.fetchall()
            answer = post_process_answer(answer)

            data.pop("_gold_program")
            new_data = {
                **data,
                "answer": answer,
            }

            # Debugging block
            if args.debug:
                try:
                    _answer = data["_answer"]
                    _answer = post_process_answer(_answer)
                    assert answer == _answer, f"answer: {answer}, _answer: {_answer}"
                except:
                    answer_error_cnt += 1
                    print(f"answer_error_cnt: {answer_error_cnt}")
                    print(f"Answer mismatch at {db_id}: retrieved answer {answer}, gold answer {_answer}")

                new_data.pop("_answer")

            assert not any([k.startswith("_") for k in new_data.keys()])
            new_dataset.append(new_data)

        except Exception as e:
            print(f"Error processing data {db_id}: {e}")
            breakpoint()
            # continue

    print(f"Total answer errors: {answer_error_cnt}")

    # Store new dataset
    with open(args.output_path, "w") as f:
        json.dump(new_dataset, f, indent=4, default=str)

    print(f"Saved new dataset to {args.output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mimic_iv_dir", type=str, required=True)
    parser.add_argument("--mimic_cxr_jpg_dir", type=str, required=True)
    parser.add_argument("--chest_imagenome_dir", type=str, required=True)
    parser.add_argument("--json_file_path", type=str, default="../dataset/mimic_iv_cxr/_test.json")
    parser.add_argument("--db_file_path", type=str, default="../database/mimic_iv_cxr/test/mimic_iv_cxr.db")
    parser.add_argument("--output_path", type=str, default="../dataset/mimic_iv_cxr/test.json")
    args = parser.parse_args()

    # split
    args.split = os.path.basename(args.output_path).split(".")[0]

    # postprocess json_file_path
    if args.debug and "_debug" not in args.json_file_path:
        args.json_file_path = args.json_file_path.replace(".json", "_debug.json")

    # postprocess db_file_path
    if args.split == "valid":
        args.db_file_path = args.db_file_path.replace("/valid/", "/train/")

    print(args)
    main(args)
