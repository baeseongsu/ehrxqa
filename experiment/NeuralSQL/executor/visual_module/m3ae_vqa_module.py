from .vqa_module import VQAModule

import os
import sys
import json
import json
import torch
from torchvision import transforms
from transformers import RobertaTokenizerFast

sys.path.append(os.path.join(os.path.dirname(__file__), "backbones"))

from m3ae.modules import M3AETransformerSS


CONFIG = {
    "exp_name": "task_infer_vqa_mmehr",
    "seed": 0,
    "datasets": ["vqa_mmehr"],
    "loss_names": {"mlm": 0, "mim": 0, "itm": 0, "vqa": 1, "cls": 0, "irtr": 0},
    "batch_size": 64,
    "max_epoch": 50,
    "max_steps": None,
    "warmup_steps": 0.1,
    "draw_false_image": 0,
    "learning_rate": 5e-06,
    "val_check_interval": 1.0,
    "lr_multiplier_head": 50,
    "lr_multiplier_multi_modal": 5,
    "max_text_len": 34,
    "tokenizer": "roberta-base",
    "vocab_size": 50265,
    "input_text_embed_size": 768,
    "vit": "ViT-B/16",
    "image_size": 512,
    "patch_size": 16,
    "train_transform_keys": ["clip"],
    "val_transform_keys": ["clip"],
    "input_image_embed_size": 768,
    "hidden_size": 768,
    "num_layers": 6,
    "num_heads": 12,
    "mlp_ratio": 4,
    "drop_rate": 0.1,
    "vqa_label_size": 110,
    "num_top_layer": 6,
    "test_only": True,
    "load_path": None,
}


class M3AEVQAModule(VQAModule):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.threshold = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(os.path.dirname(__file__), "ans2idx.json"), "r") as f:
            self.label2ans = {v: k for k, v in json.load(f).items()}

    def load_model(self):
        self.config = CONFIG
        self.config["test_only"] = True
        self.config["load_path"] = self.model_path
        self.model = M3AETransformerSS(self.config).to(self.device).eval()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.config["tokenizer"])

    def preprocess_input(self, images, questions):
        size = self.config["image_size"]
        transform_image = transforms.Compose(
            [
                transforms.Resize([size, size]),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensors = torch.stack([transform_image(image).unsqueeze(0) for image in images])

        text_encodings = self.tokenizer(
            questions,
            padding="max_length",
            truncation=True,
            max_length=self.config["max_text_len"],
            return_tensors="pt",
        )

        batch = {
            "text": questions,
            "text_ids": text_encodings["input_ids"].to(self.device),
            "text_labels": torch.full_like(text_encodings["input_ids"], -100).to(self.device),
            "text_masks": text_encodings["attention_mask"].to(self.device),
            "image": image_tensors.to(self.device),
        }
        return batch

    def postprocess_output(self, raw_output):
        handled_answers = []
        for answer in raw_output:
            if not answer:
                # Handles empty answer case
                handled_answers.append(None)  # Using None to represent no answer
            elif isinstance(answer, list):
                # Handles list of answers
                if all(a.lower() in ["yes", "no"] for a in answer):
                    handled_answers.append([a.lower() == "yes" for a in answer])
                else:
                    handled_answers.append(answer)
            elif answer.lower() in ["yes", "no"]:
                # Handles single answers
                handled_answers.append(answer.lower() == "yes")
            else:
                handled_answers.append(answer)  # Handles non-yes/no answers

        return handled_answers

    def __call__(self, images, questions):
        if self.model is None:
            self.load_model()

        batch = self.preprocess_input(images, questions)

        with torch.no_grad():
            infer = self.model.infer(batch, mask_text=False, mask_image=False)
            logits = torch.sigmoid(self.model.vqa_head(infer["multi_modal_cls_feats"]))
            one_hots = (logits > self.threshold).float()
            raw_output = [[self.label2ans[i.item()] for i in torch.nonzero(one_hot)] for one_hot in one_hots]

        answers = self.postprocess_output(raw_output)

        return answers
