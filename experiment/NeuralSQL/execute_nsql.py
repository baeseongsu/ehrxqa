import os
import re
import json
from tqdm import tqdm

# custom package
from executor.nsql_executor import NeuralSQLExecutor
from executor.sqlglot.executor import Table


def post_process_sql(query):
    current_time = "2105-12-31 23:59:00"
    precomputed_dict = {
        "temperature": (35.5, 38.1),
        "sao2": (95.0, 100.0),
        "heart rate": (60.0, 100.0),
        "respiration": (12.0, 18.0),
        "systolic bp": (90.0, 120.0),
        "diastolic bp": (60.0, 90.0),
        "mean bp": (60.0, 110.0),
    }

    # Handle current_time
    query = query.replace("current_time", f"'{current_time}'")

    # Handle vital signs
    vital_lower_match = re.search("[ \n]+([a-zA-Z0-9_]+_lower)", query)
    vital_upper_match = re.search("[ \n]+([a-zA-Z0-9_]+_upper)", query)

    if vital_lower_match and vital_upper_match:
        vital_lower_expr = vital_lower_match.group(1)
        vital_upper_expr = vital_upper_match.group(1)
        vital_name_list = list(set(re.findall("([a-zA-Z0-9_]+)_lower", vital_lower_expr) + re.findall("([a-zA-Z0-9_]+)_upper", vital_upper_expr)))

        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace("_", " ")
            if processed_vital_name in precomputed_dict:
                vital_range = precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, str(vital_range[0])).replace(vital_upper_expr, str(vital_range[1]))

    # Handle etc.
    query = query.replace("''", "'").replace("< =", "<=")
    query = query.replace("%y", "%Y").replace("%j", "%J")
    query = query.replace("'", "'").replace("'", "'")
    query = query.replace("\u201c", '"').replace("\u201d", '"')

    return query


def post_process_answer(answer, round_digit=6, sorted_answer=False):
    assert isinstance(answer, list) or answer == "null"

    if answer == "null":
        return answer

    if not answer:
        assert answer == []
        return answer

    # Tuple data preprocessing
    if isinstance(answer[0], tuple):
        assert len(answer[0]) == 1  # NOTE: currently, only support single column output
        answer = [ans[0] for ans in answer]  # unpack tuple

    if isinstance(answer[0], float):
        # Float-type answer
        answer = [round(ans, round_digit) for ans in answer]  # round to specified digit
    elif isinstance(answer[0], str):
        # String-type answer
        if sorted_answer:
            answer = sorted(answer)

    return answer


def run_execution_for_pred_query(args, executor, parsed_result):
    executed_result = {}
    for idx, query in enumerate(tqdm(parsed_result.values())):
        query = post_process_sql(query)
        try:
            result = executor.execute_nsql(query)
            if isinstance(result, Table):
                result = result.rows
            result = post_process_answer(result)
        except Exception as e:
            print(f"Error executing query {idx}: {e}")
            result = "null"  # NOTE: For NeuralSQL, we will abstain as "null" if the query execution fails
        executed_result[str(idx)] = result

    if not args.debug:
        with open(os.path.join(args.save_dir, "predictions.json"), "w") as f:
            json.dump(executed_result, f, indent=4)
        with open(os.path.join(args.save_dir, "arguments.json"), "w") as f:
            json.dump(vars(args), f, indent=4)


def run_execution_for_gt_query(args, executor, parsed_result):
    executed_result = {}
    for idx, query in enumerate(tqdm(parsed_result.values())):
        query = post_process_sql(query)
        try:
            result = executor.execute_nsql(query)
            if isinstance(result, Table):
                result = result.rows
            result = post_process_answer(result)
        except Exception as e:
            print(f"Error executing query {idx}: {e}")
            result = "null"  # NOTE: For NeuralSQL, we will abstain as "null" if the query execution fails
        executed_result[str(idx)] = result
    if not args.debug:
        with open(os.path.join(args.save_dir, "predictions_gt.json"), "w") as f:
            json.dump(executed_result, f, indent=4)


def main(args):
    if args.save_dir is None:
        assert os.path.exists(args.parsed_file_path)
        assert "results/parser/" in args.parsed_file_path
        assert args.parsed_file_path.endswith(".json")
        args.save_dir = os.path.dirname(args.parsed_file_path).replace("/parser/", "/executor/")
    os.makedirs(args.save_dir, exist_ok=True)

    executor = NeuralSQLExecutor(
        args.mimic_iv_cxr_db_dir,
        args.mimic_cxr_image_dir,
        vqa_module_type=args.vqa_module_type,
    )

    if args.execute_gt:
        """
        Ground-truth dataset
        """
        gt_file_path = os.path.join(os.path.dirname(__file__), f"../../dataset/mimic_iv_cxr/{args.dataset_split}.json")
        with open(gt_file_path, "r") as f:
            gt_dataset = json.load(f)
        assert isinstance(gt_dataset, list)
        parsed_result_gt = {str(item["id"]): item["query"] for item in gt_dataset}
        run_execution_for_gt_query(args, executor, parsed_result_gt)

    if args.execute_pred:
        """
        Predicted dataset
        """
        with open(args.parsed_file_path, "r") as f:
            parsed_result_pred = json.load(f)
        assert isinstance(parsed_result_pred, dict)
        run_execution_for_pred_query(args, executor, parsed_result_pred)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--mimic_iv_cxr_db_dir", type=str, default="../../database/mimic_iv_cxr/train", help="MIMIC-IV CXR database directory")
    parser.add_argument("--mimic_cxr_image_dir", type=str, required=True, help="MIMIC-CXR image directory")
    parser.add_argument("--parsed_file_path", type=str, default="results/parser/predictions.json", help="Parsed file path")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")

    parser.add_argument("--dataset_split", type=str, default="valid", help="Dataset split", choices=["train", "valid", "test"])
    parser.add_argument("--vqa_module_type", type=str, default="m3ae", help="VQA module type", choices=["m3ae", "yes"])

    parser.add_argument("--execute_gt", action="store_true", help="Execute ground-truth queries")
    parser.add_argument("--execute_pred", action="store_true", help="Execute predicted queries")
    args = parser.parse_args()

    # assertion
    if args.dataset_split in ["train", "valid"]:
        assert args.mimic_iv_cxr_db_dir.endswith("train")
    else:
        assert args.mimic_iv_cxr_db_dir.endswith("test")

    main(args)
