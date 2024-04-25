import re


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


if __name__ == "__main__":

    import json

    # read dataset
    with open("../dataset/mimic_iv_cxr/test.json") as f:
        data = json.load(f)

    for item in data:
        query = item["query"]
        answer = item["answer"]
        print(post_process_sql(query))
        print(post_process_answer(answer))
