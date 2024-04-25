import json

# custom package
from reliability_score import ReliabilityScore
from post_processing import post_process_answer


def evaluate(gt_path, pred_path):
    gt = json.load(open(gt_path))
    pred = json.load(open(pred_path))

    assert len(gt) == len(pred), f"Length mismatch: GT {len(gt)} / Pred {len(pred)}"

    # post-process
    if isinstance(gt, list) and isinstance(gt[0], dict):
        gt = sorted(gt, key=lambda x: x["id"])
        gt = {str(item["id"]): item["answer"] for item in gt}
    elif isinstance(gt, dict):
        gt = {str(key): gt[key] for key in sorted(gt)}
    else:
        raise ValueError("Invalid GT format")

    gt = {key: post_process_answer(gt[key]) for key in gt}
    pred = {key: post_process_answer(pred[key]) for key in pred}

    # accuracy
    correct = 0
    for key in gt:
        if gt[key] == pred[key]:
            correct += 1

    accuracy = correct / len(gt) * 100
    print(f"Execution Accuracy: {accuracy:.2f}%")

    # reliability
    reliability = ReliabilityScore(real_result=gt, pred_result=pred, abstain_key="null")
    reliability_scores = reliability.compute(penalties=["0", "5", "10", "N"])
    print(reliability_scores)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    args = parser.parse_args()

    evaluate(gt_path=args.gt_path, pred_path=args.pred_path)
