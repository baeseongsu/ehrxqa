import os
import sys
import json
from tqdm import tqdm
from PIL import Image
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT_DIR, "experiment/NeuralSQL")
sys.path.append(EXP_DIR)

from executor.visual_module import M3AEVQAModule


class TestVQAModules(unittest.TestCase):
    def setUp(self):
        self.mimic_cxr_image_dir = "/nfs_data_storage/mmehrqg/mimic-cxr-jpg/20230110/re512_3ch_contour_cropped"
        self.mimiccxrvqa_dataset_dir = "/nfs_edlab/ssbae/2023-ehrxqa/mimic-cxr-vqa/mimic-cxr-vqa-70/mimiccxrvqa/dataset"

    def test_m3ae_vqa_module(self):

        with open(os.path.join(self.mimiccxrvqa_dataset_dir, "test.json"), "r") as f:
            dataset = json.load(f)

        vqa_model = M3AEVQAModule(model_path=os.path.join(EXP_DIR, "./executor/visual_module/checkpoints/m3ae/seed42.ckpt"))

        batch_size = 64
        dataset_updated = []
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            images = [Image.open(os.path.join(self.mimic_cxr_image_dir, item["image_path"])) for item in batch]
            questions = [item["question"] for item in batch]
            answers = [item["answer"] for item in batch]

            # inference
            preds = vqa_model(images, questions)
            answers = vqa_model.postprocess_output(answers)

            assert len(answers) == len(preds)
            dataset_updated.extend([{**item, "correct": (answer == pred)} for item, answer, pred in zip(batch, answers, preds)])

        correct_count = sum(item["correct"] for item in dataset_updated)
        print(f"Overall accuracy: {correct_count / len(dataset_updated)}")

        content_types = ["presence", "abnormality", "attribute", "anatomy", "size", "gender", "plane"]
        for content_type in content_types:
            dataset_updated_c = [item for item in dataset_updated if item["content_type"] == content_type]
            correct_count = sum(item["correct"] for item in dataset_updated_c)
            print(f"Content type: {content_type}, Accuracy: {correct_count / len(dataset_updated_c) if dataset_updated_c else 'N/A'}")


if __name__ == "__main__":
    unittest.main()
