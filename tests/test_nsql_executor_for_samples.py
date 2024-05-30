import os
import sys
import json
from tqdm import tqdm
import unittest

from test_utils import post_process_sql, post_process_answer, filter_dataset_by_scope

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_PATH, "experiment/NeuralSQL"))
from executor.visual_module import get_vqa_module

sys.path.append(os.path.join(ROOT_PATH, "experiment/NeuralSQL/executor"))
from sqlglot.executor import Table

sys.path.append(os.path.join(ROOT_PATH, "experiment/NeuralSQL/executor/neuralsql"))
from nsql_executor import NeuralSQLExecutor


class TestExecutorForMultiModality(unittest.TestCase):
    def setUp(self):
        split = "train"
        mimic_iv_cxr_db_dir = f"/nfs_edlab/ssbae/2023-ehrxqa/nips-2023/ehrxqa/database/mimic_iv_cxr/{split}"
        mimic_cxr_image_dir = "/nfs_data_storage/mmehrqg/mimic-cxr-jpg/20230110/re512_3ch_contour_cropped"

        self.executor = NeuralSQLExecutor(
            mimic_iv_cxr_db_dir,
            mimic_cxr_image_dir,
            # vqa_module_type="m3ae",
            vqa_module_type="debug",
        )

        self.dataset_dir = os.path.join(ROOT_PATH, "dataset/mimic_iv_cxr")
        with open(os.path.join(self.dataset_dir, f"{split}.json"), "r") as f:
            self.dataset = json.load(f)

    def run_pipeline(self, dataset, executor):
        correct = 0
        for item in tqdm(dataset):
            query = item["query"]
            answer = item["answer"]

            query = post_process_sql(query)
            result = executor.execute_nsql(query)

            if isinstance(result, Table):
                result = result.rows
                result = post_process_answer(result)

            if result == answer:
                correct += 1

    def test_query_case_1(self):
        try:
            # NOTE: Valid-Q19
            query = """
            select ( 
                select (func_vqa(\"is there evidence of any tubes/lines?\", t1.study_id) = false) 
                from ( select tb_cxr.study_id from tb_cxr where tb_cxr.study_id in ( select distinct tb_cxr.study_id from tb_cxr where tb_cxr.hadm_id in ( select admissions.hadm_id from admissions where admissions.subject_id = 12183689 and admissions.dischtime is not null order by admissions.admittime desc limit 1 ) order by tb_cxr.studydatetime asc limit 1 ) ) as t1 
            ) and ( 
                select (func_vqa(\"is there evidence of any tubes/lines?\", t2.study_id) = true) 
                from ( select tb_cxr.study_id from tb_cxr where tb_cxr.study_id in ( select distinct tb_cxr.study_id from tb_cxr where tb_cxr.hadm_id in ( select admissions.hadm_id from admissions where admissions.subject_id = 12183689 and admissions.dischtime is not null order by admissions.admittime asc limit 1 ) order by tb_cxr.studydatetime desc limit 1 ) ) as t2 
            )
            """
            query = post_process_sql(query)
            result = self.executor.execute_nsql(query)
            print(result)

            # NOTE: Valid-Q19:subQ1
            query = """
            select func_vqa(\"is there evidence of any tubes/lines?\", t1.study_id)
            from ( 
                select tb_cxr.study_id 
                from tb_cxr 
                where tb_cxr.study_id in ( 
                    select distinct tb_cxr.study_id 
                    from tb_cxr 
                    where tb_cxr.hadm_id in ( 
                        select admissions.hadm_id 
                        from admissions 
                        where admissions.subject_id = 12183689 
                        and admissions.dischtime is not null 
                        order by admissions.admittime desc limit 1 
                    )
                    order by tb_cxr.studydatetime asc limit 1 
                ) 
            ) as t1
            """
            query = post_process_sql(query)
            result = self.executor.execute_nsql(query)
            print(result)
            assert len(result) == 1

            # NOTE: Valid-Q19:subQ2
            query = """
            select func_vqa(\"is there evidence of any tubes/lines?\", t2.study_id)
            from ( 
                select tb_cxr.study_id 
                from tb_cxr 
                where tb_cxr.study_id in ( 
                    select distinct tb_cxr.study_id 
                    from tb_cxr 
                    where tb_cxr.hadm_id in ( 
                        select admissions.hadm_id 
                        from admissions 
                        where admissions.subject_id = 12183689 
                        and admissions.dischtime is not null 
                        order by admissions.admittime asc limit 1 
                    ) 
                    order by tb_cxr.studydatetime desc limit 1 
                ) 
            ) as t2 
            """
            query = post_process_sql(query)
            result = self.executor.execute_nsql(query)
            print(result)
            assert len(result) == 1
        except:
            # Go to debug
            query = """
            select t2.study_id
            from ( 
                select tb_cxr.study_id 
                from tb_cxr 
                where tb_cxr.study_id in ( 
                    select distinct tb_cxr.study_id 
                    from tb_cxr 
                    where tb_cxr.hadm_id in ( 
                        select admissions.hadm_id 
                        from admissions 
                        where admissions.subject_id = 12183689 
                        and admissions.dischtime is not null 
                        order by admissions.admittime asc limit 1 
                    ) 
                    order by tb_cxr.studydatetime desc limit 1 
                ) 
            ) as t2 
            """
            query = post_process_sql(query)
            result = self.executor.execute_nsql(query)
            print(result)
            assert result.rows == [(50806492,)] and result.columns == ("study_id",)

    # def test_query_case_datetime(self):
    #     query = """

    #     SELECT tb_cxr.studydatetime
    #     FROM tb_cxr
    #     WHERE tb_cxr.study_id IN (
    #         SELECT DISTINCT t2.study_id
    #         FROM (
    #             SELECT DISTINCT tb_cxr.study_id
    #             FROM tb_cxr
    #             WHERE tb_cxr.study_id IN (
    #                 SELECT DISTINCT tb_cxr.study_id
    #                 FROM tb_cxr
    #                 WHERE tb_cxr.subject_id = 10938464
    #                 AND DATETIME(tb_cxr.studydatetime) >= DATETIME('2105-12-31 23:59:00', '-1 year')
    #             )
    #         ) AS t2
    #         WHERE FUNC_VQA("is picc revealed in the cardiac silhouette?", t2.study_id) = TRUE
    #     )
    #     """
    #     query = post_process_sql(query)
    #     result = self.executor.execute_nsql(query)

    #     print(result)


if __name__ == "__main__":
    unittest.main()
