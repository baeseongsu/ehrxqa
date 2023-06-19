<div align="center">

# EHRXQA: A Multi-Modal Question Answering Dataset for Electronic Health Records with Chest X-ray Images (🏥, 🩻)

</div>

## Overview
Electronic Health Records (EHRs), despite containing patients' medical histories in various multi-modal formats, often leave the potential for joint reasoning over imaging and table modalities underexplored in current EHR Question Answering (QA) systems. In this paper, we introduce **EHRXQA**, a novel multi-modal question answering dataset for structured EHRs and chest X-ray images. To develop our dataset, we first construct two uni-modal resources: 1) The [MIMIC-CXR-VQA](https://github.com/baeseongsu/mimic-cxr-vqa/) dataset, our newly created medical visual question answering benchmark, specifically designed to augment the imaging modality in EHR QA; 2) EHRSQL (MIMIC-IV), a refashioned version of a previously established table-based EHR QA dataset. By combining these two uni-modal resources, we successfully construct our multi-modal EHR QA dataset, which requires both uni-modal and cross-modal reasoning. To tackle the unique challenge of handling multi-modal questions within EHRs, we propose a NeuralSQL-based strategy equipped with an external VQA API. This pioneering endeavor enhances engagement with multi-modal EHR sources and we believe that our dataset can catalyze advances in real-world medical scenarios such as clinical decision-making and research.

---

**The dataset and code will be available after publication. Please stay tuned!**
