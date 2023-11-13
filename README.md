# EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images

*A multi-modal question answering dataset that combines structured Electronic Health Records (EHRs) and chest X-ray images, designed to facilitate joint reasoning across imaging and table modalities in EHR Question Answering (QA) systems.*


## Overview
Electronic Health Records (EHRs), which contain patients' medical histories in various multi-modal formats, often overlook the potential for joint reasoning across imaging and table modalities underexplored in current EHR Question Answering (QA) systems. In this paper, we introduce EHRXQA, a novel multi-modal question answering dataset combining structured EHRs and chest X-ray images. To develop our dataset, we first construct two uni-modal resources: 1) The MIMIC-CXR-VQA dataset, our newly created medical visual question answering (VQA) benchmark, specifically designed to augment the imaging modality in EHR QA, and 2) EHRSQL (MIMIC-IV), a refashioned version of a previously established table-based EHR QA dataset. By integrating these two uni-modal resources, we successfully construct a multi-modal EHR QA dataset that necessitates both uni-modal and cross-modal reasoning. To address the unique challenges of multi-modal questions within EHRs, we propose a NeuralSQL-based strategy equipped with an external VQA API. This pioneering endeavor enhances engagement with multi-modal EHR sources and we believe that our dataset can catalyze advances in real-world medical scenarios such as clinical decision-making and research.

- 💡 For a detailed exploration of the MIMIC-CXR-VQA component of our dataset, please visit [here](https://github.com/baeseongsu/mimic-cxr-vqa/).
- 💡 For more details about the multi-modal QA dataset, please refer to our publication [EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images](https://arxiv.org/abs/2310.18652), presented at NeurIPS 2023 (Datasets and Benchmarks Track).


## Features

- [x] Provide a script to download source datasets (MIMIC-CXR-JPG, Chest ImaGenome, and MIMIC-IV) from Physionet.
- [x] Provide a script to preprocess the source datasets.
- [x] Provide a script to construct an integrated database (MIMIC-IV and MIMIC-CXR).
- [x] Provide a script to generate the EHRXQA dataset (with answer information).

## Installation

### For Linux:

Ensure that you have Python 3.8.5 or higher installed on your machine. Set up the environment and install the required packages using the commands below:

```
# Set up the environment
conda create --name ehrxqa python=3.8.5

# Activate the environment
conda activate ehrxqa

# Install required packages
pip install pandas==1.1.3 tqdm==4.65.0 scikit-learn==0.23.2 
pip install dask
```

## Setup

Clone this repository and navigate into it:

```
git clone https://github.com/baeseongsu/ehrxqa.git
cd ehrxqa
```

## Usage

### Privacy

We take data privacy very seriously. All of the data you access through this repository has been carefully prepared to prevent any privacy breaches or data leakage. You can use this data with confidence, knowing that all necessary precautions have been taken.

### Access Requirements

The EHRXQA dataset is constructed from the MIMIC-CXR-JPG (v2.0.0), Chest ImaGenome (v1.0.0), and MIMIC-IV (v2.2). All these source datasets require a credentialed Physionet license. Due to these requirements and in adherence to the Data Use Agreement (DUA), only credentialed users can access the MIMIC-CXR-VQA dataset files (see Access Policy). To access the source datasets, you must fulfill all of the following requirements:

1. Be a [credentialed user](https://physionet.org/settings/credentialing/)
    - If you do not have a PhysioNet account, register for one [here](https://physionet.org/register/).
    - Follow these [instructions](https://physionet.org/credential-application/) for credentialing on PhysioNet.
    - Complete the "CITI Data or Specimens Only Research" [training course](https://physionet.org/about/citi-course/).
2. Sign the data use agreement (DUA) for each project
    - https://physionet.org/sign-dua/mimic-cxr-jpg/2.0.0/
    - https://physionet.org/sign-dua/chest-imagenome/1.0.0/
    - https://physionet.org/sign-dua/mimiciv/2.2/

### Accessing the EHRXQA Dataset


While the complete EHRXQA dataset is being prepared for publication on the Physionet platform, we provide partial access to the dataset via this repository for credentialed users. 

To access the EHRXQA dataset, you can run the provided main script (which requires your unique Physionet credentials) in this repository as follows:

```
bash build_dataset.sh
```

During script execution, enter your PhysioNet credentials when prompted:

- Username: Enter your PhysioNet username and press `Enter`.
- Password: Enter your PhysioNet password and press `Enter`. The password characters won't appear on screen.

This script performs several actions: 1) it downloads the source datasets from Physionet, 2) preprocesses these datasets, and 3) generates the complete EHRXQA dataset by creating ground-truth answer information.

Ensure you keep your credentials secure. If you encounter any issues, please ensure that you have the necessary permissions, a stable internet connection, and all prerequisite tools installed.

### Downloading MIMIC-CXR-JPG Images

<!---
To enhance user convenience, we will provide a script that allows you to download only the CXR images relevant to the MIMIC-CXR-VQA dataset, rather than downloading all the MIMIC-CXR-JPG images.

```
bash download_images.sh
```

During script execution, enter your PhysioNet credentials when prompted:

- Username: Enter your PhysioNet username and press `Enter`.
- Password: Enter your PhysioNet password and press `Enter`. The password characters won't appear on screen.

This script performs several actions: 1) it reads the image paths from the JSON files of the MIMIC-CXR-VQA dataset; 2) uses these paths to download the corresponding images from the MIMIC-CXR-JPG dataset hosted on Physionet; and 3) saves these images locally in the corresponding directories as per their paths.
--->

### Dataset Structure

The dataset is structured as follows:

```
ehrxqa
└── dataset
    ├── _train_.json
    ├── _valid.json
    ├── _test.json
    ├── train.json (available post-script execution)
    ├── valid.json (available post-script execution)
    └── test.json  (available post-script execution)
```

- The `ehrxqa` is the root directory. Within this, the `dataset` directory contains various JSON files that are part of the EHRXQA dataset.
- `_train.json`, `_valid.json`, and `_test.json` are pre-release versions of the dataset files corresponding to the training, validation, and testing sets respectively. These versions are intentionally incomplete to safeguard privacy and prevent the leakage of sensitive information; they do not include certain crucial information, such as the answers.
- Once the main script is executed with valid Physionet credentials, the full versions of these files - `train.json`, `valid.json`, and `test.json` - will be generated. These files contain the complete information, including the corresponding answers for each entry in the respective sets.

### Dataset Description

The QA samples in the EHRXQA dataset are stored in individual .json files. Each file contains a list of Python dictionaries, with each key indicating:

- `db_id`: A string representing the corresponding database ID.
- `split`: The dataset split category (e.g., training, test, validation).
- `id`: A unique identifier for each instance in the dataset.
- `question`: A paraphrased version of the question.
- `template`: The final question template created by injecting real database values into the tag. This template represents the fully specified and contextualized form of the question.
- `query`: The corresponding NeuralSQL/SQL query for the question.
- `value`: Specific key-value pairs relevant to the question, sampled from the database.
- `q_tag`: The initial sampled question template. This serves as the foundational structure for the question.
- `t_tag`: Sampled time templates, used to provide temporal context and specificity to the question.
- `o_tag`: Sampled operational values for the query, often encompassing numerical or computational aspects required for forming the question.
- `v_tag`: Sampled visual values, which include elements like object, category, attribute, and comparison, adding further details to the question.
- `tag`: A comprehensive tag that synthesizes the enhanced q_tag with additional elements (t_tag, o_tag, v_tag). This represents an intermediate, more specified version of the question template before the final template is formed.
- `para_type`: The source of the paraphrase, either from a general machine-generated tool or specifically by GPT-4.
- `is_impossible`: A boolean indicating whether the question is answerable based on the dataset.
- `_gold_program`: A temporary program that is used to generate the answer.

After validating PhysioNet credentials, the create_answer.py script generates the following items:

- `answer`: The answer string based on the query execution.

To be specific, here is the example instance:
```
{
    'db_id': 'mimic_iv_cxr', 
    'split': 'train',
    'id': 0, 
    'question': 'how many days have passed since the last chest x-ray of patient 18679317 depicting any anatomical findings in 2105?', 
    'template': 'how many days have passed since the last time patient 18679317 had a chest x-ray study indicating any anatomicalfinding in 2105?', 
    'query': 'select 1 * ( strftime(\'%J\',current_time) - strftime(\'%J\',t1.studydatetime) ) from ( select tb_cxr.study_id, tb_cxr.studydatetime from tb_cxr where tb_cxr.study_id in ( select distinct tb_cxr.study_id from tb_cxr where tb_cxr.subject_id = 18679317 and strftime(\'%Y\',tb_cxr.studydatetime) = \'2105\' ) ) as t1 where func_vqa("is the chest x-ray depicting any anatomical findings?", t1.study_id) = true', 
    'value': {'patient_id': 18679317}, 
    'q_tag': 'how many [unit_count] have passed since the [time_filter_exact1] time patient {patient_id} had a chest x-ray study indicating any ${category} [time_filter_global1]?', 
    't_tag': ['abs-year-in', '', '', 'exact-last', ''], 
    'o_tag': {'unit_count': {'nlq': 'days', 'sql': '1 * ', 'type': 'days', 'sql_pattern': '[unit_count]'}}, 
    'v_tag': {'object': [], 'category': ['anatomicalfinding'], 'attribute': []}, 
    'tag': 'how many [unit_count:days] have passed since the [time_filter_exact1:exact-last] time patient {patient_id} had a chest x-ray study indicating any anatomicalfinding [time_filter_global1:abs-year-in]?',
    'para_type': 'machine', 
    'is_impossible': False, 
    'answer': 'Will be generated by dataset_builder/generate_answer.py'
}
```

## Versioning

We employ semantic versioning for our dataset, with the current version being v1.0.0. Generally, we will maintain and provide updates only for the latest version of the dataset. However, in cases where significant updates occur or when older versions are required for validating previous research, we may exceptionally retain previous dataset versions for a period of up to one year. For a detailed list of changes made in each version, check out our CHANGELOG.

## Contributing

Contributions to enhance the usability and functionality of this dataset are always welcomed. If you're interested in contributing, feel free to fork this repository, make your changes, and then submit a pull request. For significant changes, please first open an issue to discuss the proposed alterations.

## Contact

For any questions or concerns regarding this dataset, please feel free to reach out to us ([seongsu@kaist.ac.kr](mailto:seongsu@kaist.ac.kr) or [kyungdaeun@kaist.ac.kr](mailto:kyungdaeun@kaist.ac.kr)). We appreciate your interest and are eager to assist.

## Acknowledgements

More details will be provided soon.

## Citation

When you use the EHRXQA dataset, we would appreciate it if you cite the following:
```
@article{bae2023ehrxqa,
  title={EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images},
  author={Bae, Seongsu and Kyung, Daeun and Ryu, Jaehee and Cho, Eunbyeol and Lee, Gyubok and Kweon, Sunjun and Oh, Jungwoo and Ji, Lei and Chang, Eric I and Kim, Tackeun and others},
  journal={arXiv preprint arXiv:2310.18652},
  year={2023}
}
```

## License

The code in this repository is provided under the terms of the MIT License. The final output of the dataset created using this code, the EHRXQA, is subject to the terms and conditions of the original datasets from Physionet: [MIMIC-CXR-JPG License](https://physionet.org/content/mimic-cxr/view-license/2.0.0/), [Chest ImaGenome License](https://physionet.org/content/chest-imagenome/view-license/1.0.0/), and [MIMIC-IV License](https://physionet.org/content/mimiciv/view-license/2.2/).

