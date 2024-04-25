# NeuralSQL

NeuralSQL is an our proposed baseline approach for solving multi-modal question answering on electronic health records with chest X-ray images. It integrates Large Language Models (LLMs) with an external Visual Question Answering (VQA) module to handle multi-modal questions over a structured database with images.

## Experimental Setup

### Conda Environment

```bash
conda activate ehrxqa
pip install Pillow==8.1.0
pip install transformers==4.10.0
pip install pytorch-lightning==1.3.2
pip install torchmetrics==0.7.2
pip install timm==0.4.12
pip install einops==0.3.0
pip install openai
pip install tiktoken
```

### Data Preparation

For inference only, you need a small part of the MIMIC-CXR-JPG dataset; however, to get the in-domain performance, we recommend downloading the entire MIMIC-CXR-JPG dataset. Also, to ensure smooth inference of VQA modules for the EHRXQA dataset, we recommend using the preprocessed data when best suitable for your VQA module. For example, we use the resized and cropped version of the MIMIC-CXR-JPG dataset because we use those image preprocessing techniques when training the VQA models, especially for the M3AE model.

### VQA Module Preparation

We currently provide skeleton code for two VQA modules: `yes_vqa_module.py` and `m3ae_vqa_module.py`. Since radiological labels related to Chest X-ray images are sourced from the Chest ImaGenome dataset, you can use the `custom_vqa_module.py` to implement your own VQA module.

## Experiment Execution

```bash
python parse_question.py --examples_dataset_name="train.json" --target_dataset_name="valid.json" --return_content_only
```

```bash
python execute_nsql.py --mimic_iv_cxr_db_dir="../../database/mimic_iv_cxr/train" --mimic_cxr_image_dir="/nfs_data_storage/mmehrqg/mimic-cxr-jpg/20230110/re512_3ch_contour_cropped" --parsed_file_path="results/parser/valid.json"
```

## Directory Structure

```
NeuralSQL/
│
├── README.md                    # Project documentation
├── execute_nsql.py              # Script to execute NeuralSQL queries
│
├── executor/
│   ├── nsql_executor.py         # NeuralSQL executor module
│   ├── sqlglot/                 # SQLGlot library for SQL parsing and execution
│   │   └── ...
│   │
│   └── visual_module/           # Visual module for VQA
│       ├── __init__.py
│       ├── ans2idx.json         # Answer-to-index mapping file
│       ├── vqa_module.py        # Base VQA module
│       ├── yes_vqa_module.py    # VQA module for yes/no questions
│       ├── m3ae_vqa_module.py   # VQA module using M3AE model
│       ├── custom_vqa_module.py # Custom VQA module
│       ├── backbones/           # Backbone models for VQA
│       │   └── ...
│       │
│       └── checkpoints/         # Pretrained checkpoints for VQA models
│           └── ...
│
├── parse_question.py            # Script to parse natural language questions into NeuralSQL
│
├── parser/
│   ├── __init__.py
│   ├── openai_parser.py         # OpenAI-based parser module
│   ├── parser_utils.py          # Utility functions for parsing
│   ├── api_key/                 # API keys for LLMs
│   │   └── ...
│   │
│   └── prompt_storage/          # Prompt storage for LLMs
│       └── ...
│
├── results/
│   ├── executor/                # Results from NeuralSQL execution
│   │   └── ...
│   │
│   └── parser/                  # Results from question parsing
│       └── ...
│
└── run_exp.sh                   # Script to run experiments
```

## Workflow

1. **NeuralSQL Parsing:**
   - The parser model translates a natural language question into an executable NeuralSQL query.
   - Image-related and Image+Table-related questions are annotated with NeuralSQL queries.
   - The NeuralSQL query includes a VQA API call function (FUNC_VQA) to handle image-related queries.

2. **NeuralSQL Execution:**
   - The NeuralSQL query is parsed into an abstract syntax tree (AST).
   - The interpreter executes the parsed tree in sequence, including API calls.
   - For VQA API calls, the interpreter loads the corresponding image(s) and feeds them into the VQA model.
   - The VQA model infers information based on the provided question and image(s).
   - The output of the API call is preserved as a column data object, compatible with standard SQL grammar.
   - The NeuralSQL interpreter executes the program and derives the final answer.

This NeuralSQL-based approach effectively handles both structured information and images, enabling multi-modal question answering on EHRs with chest X-ray images.