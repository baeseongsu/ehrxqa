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


class TestExecutorForTableModality(unittest.TestCase):
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
        error_cnt = 0
        for item in tqdm(dataset):
            query = item["query"]
            answer = item["answer"]

            query = post_process_sql(query)
            result = executor.execute_nsql(query)

            if isinstance(result, Table):
                result = result.rows
                result = post_process_answer(result)

            if result == answer:
                pass
            else:
                error_cnt += 1
        print(error_cnt)

    # def test_query_case_table_none(self):
    #     """
    #     dataset scope: TABLE-NONE
    #     executor: execute_nsql
    #     """
    #     dataset = self.dataset
    #     dataset = filter_dataset_by_scope(dataset, scope="TABLE-NONE")
    #     self.run_pipeline(dataset, self.executor)

    # def test_query_case_table_single(self):
    #     """
    #     dataset scope: TABLE-SINGLE
    #     executor: execute_nsql
    #     """
    #     dataset = self.dataset
    #     dataset = filter_dataset_by_scope(dataset, scope="TABLE-SINGLE")
    #     self.run_pipeline(dataset, self.executor)

    # def test_query_case_table_group(self):
    #     """
    #     dataset scope: TABLE-GROUP
    #     executor: execute_nsql
    #     """
    #     dataset = self.dataset
    #     dataset = filter_dataset_by_scope(dataset, scope="TABLE-GROUP")
    #     self.run_pipeline(dataset, self.executor)

    # def test_query_case_table(self):
    #     """
    #     dataset scope: TABLE
    #     executor: execute_nsql
    #     """
    #     dataset = self.dataset
    #     dataset = filter_dataset_by_scope(dataset, scope="TABLE")
    #     self.run_pipeline(dataset, self.executor)


if __name__ == "__main__":
    unittest.main()
