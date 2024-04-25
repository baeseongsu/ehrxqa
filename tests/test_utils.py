import os
import re
import json

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# def map_index(idx):
#     """
#     Map the index to modality and patient scope.

#     modality scope: T (Table), I (Image), M (Multi-modal)
#     patient scope: N (None), S (Single), G (Group)
#     """
#     if isinstance(idx, str):
#         idx = int(idx)

#     if 0 <= idx < 1000:
#         modality = "T"
#         if 0 <= idx <= 5:
#             patient_scope = "N"
#         elif 6 <= idx <= 121:
#             patient_scope = "S"
#         elif 122 <= idx <= 182:
#             patient_scope = "G"
#     elif 1000 <= idx < 5000:
#         if 1000 <= idx < 4000:
#             modality = "I"
#             patient_scope = "S"
#         elif 4000 <= idx < 5000:
#             modality = "I"
#             patient_scope = "G"
#     elif 5000 <= idx < 7000:
#         modality = "M"
#         if 5000 <= idx < 6000:
#             patient_scope = "S"
#         elif 6000 <= idx < 7000:
#             patient_scope = "G"
#     else:
#         return "Invalid index"

#     return f"{modality}-{patient_scope}"


def map_index(idx, detailed=True):

    if isinstance(idx, str):
        idx = int(idx)

    modality_map = {
        range(0, 6): ("TABLE", "NONE"),  # 0-5
        range(6, 122): ("TABLE", "SINGLE"),  # 6-121
        range(122, 183): ("TABLE", "GROUP"),  # 122-182
        range(1000, 2000): ("IMAGE", "SINGLE-1"),
        range(2000, 3000): ("IMAGE", "SINGLE-2"),
        range(3000, 4000): ("IMAGE", "SINGLE-N"),
        range(4000, 5000): ("IMAGE", "GROUP-N"),
        range(5000, 6000): ("MULTIMODAL", "SINGLE"),
        range(6000, 7000): ("MULTIMODAL", "GROUP"),
    }

    for index_range, (modality, patient_scope) in modality_map.items():
        if idx in index_range:
            if not detailed:
                modality = modality.split("-")[0]
            return f"{modality}-{patient_scope}"
    return "Invalid index"


def post_process_sql(query):

    __current_time = "2105-12-31 23:59:00"
    __precomputed_dict = {
        "temperature": (35.5, 38.1),
        "sao2": (95.0, 100.0),
        "heart rate": (60.0, 100.0),
        "respiration": (12.0, 18.0),
        "systolic bp": (90.0, 120.0),
        "diastolic bp": (60.0, 90.0),
        "mean bp": (60.0, 110.0),
    }

    # handle current_time
    if "current_time" in query:
        query = query.replace("current_time", f"'{__current_time}'")

    # handle vital signs
    if re.search("[ \n]+([a-zA-Z0-9_]+_lower)", query) and re.search("[ \n]+([a-zA-Z0-9_]+_upper)", query):
        vital_lower_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_lower)", query)[0]
        vital_upper_expr = re.findall("[ \n]+([a-zA-Z0-9_]+_upper)", query)[0]
        vital_name_list = list(set(re.findall("([a-zA-Z0-9_]+)_lower", vital_lower_expr) + re.findall("([a-zA-Z0-9_]+)_upper", vital_upper_expr)))
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace("_", " ")
            if processed_vital_name in __precomputed_dict:
                vital_range = __precomputed_dict[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")

    # handle etc.
    query = query.replace("''", "'").replace("< =", "<=")
    query = query.replace("%y", "%Y").replace("%j", "%J")
    query = query.replace("‘", "'").replace("’", "'")
    query = query.replace("\u201c", '"').replace("\u201d", '"')

    return query


def post_process_answer(answer, round_digit=6, sorted_answer=False):
    assert isinstance(answer, list) or answer == "null"

    if answer == "null":
        return answer

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


def filter_dataset_by_scope(dataset, scope="TABLE-NONE"):
    with open(os.path.join(ROOT_PATH, "dataset_builder/internal_utils/template_idx_to_template_info.json"), "r") as f:
        template_info_map = json.load(f)
    template_info_map_inv = {v["question_template"]: k for k, v in template_info_map.items()}
    new_dataset = []
    for item in dataset:
        template = item["q_tag"]
        template_idx = template_info_map_inv[template]
        modality = map_index(template_idx)
        # if modality == scope:
        #     new_dataset.append(item)
        if modality.startswith(scope):
            new_dataset.append(item)

    print(f"Original dataset size ({scope}):", len(dataset))
    print(f"Filtered dataset size ({scope}):", len(new_dataset))
    return new_dataset
