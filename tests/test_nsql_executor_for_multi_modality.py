import os
import sys
import json
from tqdm import tqdm
import unittest

from test_utils import post_process_sql, post_process_answer, filter_dataset_by_scope

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "experiment/NeuralSQL/executor"))
from sqlglot.executor import Table

sys.path.append(os.path.join(ROOT_PATH, "experiment/NeuralSQL/executor/neuralsql"))
from nsql_executor import NeuralSQLExecutor


class TestExecutorForMultiModality(unittest.TestCase):
    def setUp(self):
        split = "test"
        mimic_iv_cxr_db_dir = f"/nfs_edlab/ssbae/2023-ehrxqa/nips-2023/ehrxqa/database/mimic_iv_cxr/{split}"
        mimic_cxr_image_dir = "/nfs_data_storage/mmehrqg/mimic-cxr-jpg/20230110/re512_3ch_contour_cropped"

        self.executor = NeuralSQLExecutor(
            mimic_iv_cxr_db_dir,
            mimic_cxr_image_dir,
            vqa_module_type="m3ae",
        )

        self.dataset_dir = os.path.join(ROOT_PATH, "dataset/mimic_iv_cxr")
        with open(os.path.join(self.dataset_dir, f"{split}.json"), "r") as f:
            self.dataset = json.load(f)

    def run_pipeline(self, dataset, executor):
        correct = 0
        error = 0
        for item in tqdm(dataset):
            query = item["query"]
            answer = item["answer"]

            query = post_process_sql(query)
            try:
                result = executor.execute_nsql(query)
            except:
                error += 1
                continue

            if isinstance(result, Table):
                result = result.rows
                result = post_process_answer(result)

            if result == answer:
                correct += 1

        print(f"total: {len(dataset)}")
        print(f"correct: {correct}, error: {error}, incorrect: {len(dataset) - correct - error}")
        print(f"accuracy: {correct / len(dataset)}")

    def test_query_case_multimodal_single(self):
        """
        dataset scope: MULTIMODAL-SINGLE
        executor: execute_nsql
        """
        dataset = self.dataset
        dataset = filter_dataset_by_scope(dataset, scope="MULTIMODAL-SINGLE")
        self.run_pipeline(dataset, self.executor)

    # def test_query_case_multimodal_group(self):
    #     """
    #     dataset scope: MULTIMODAL-GROUP
    #     executor: execute_nsql
    #     """
    #     dataset = self.dataset
    #     dataset = filter_dataset_by_scope(dataset, scope="MULTIMODAL-GROUP")
    #     self.run_pipeline(dataset, self.executor)

    # def test_query_case_multimodal(self):
    #     """
    #     dataset scope: MULTIMODAL
    #     executor: execute_nsql
    #     """
    #     dataset = self.dataset
    #     dataset = filter_dataset_by_scope(dataset, scope="MULTIMODAL")
    #     self.run_pipeline(dataset, self.executor)


if __name__ == "__main__":
    unittest.main()
