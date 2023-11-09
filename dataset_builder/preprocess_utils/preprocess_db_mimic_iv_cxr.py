import os
import time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
from ast import literal_eval
import pandas as pd
import numpy as np
from collections import Counter
import sqlite3

import warnings

warnings.filterwarnings("ignore")

import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocess_db_utils import Sampler, adjust_time, read_csv, generate_random_date


CHARTEVENT2ITEMID = {
    "Temperature Celsius": "223762",  # body temperature
    "O2 saturation pulseoxymetry": "220277",  # Sao2
    "Heart Rate": "220045",  # heart rate
    "Respiratory Rate": "220210",  # respiration rate
    "Arterial Blood Pressure systolic": "220050",  # systolic blood pressure
    "Arterial Blood Pressure diastolic": "220051",  # diasolic blood pressure
    "Arterial Blood Pressure mean": "220052",  # mean blood pressure
    "Admission Weight (Kg)": "226512",  # weight
    "Height (cm)": "226730",  # height
}


class Build_MIMIC_IV_CXR(Sampler):
    def __init__(
        self,
        mimic_iv_dir,
        mimic_cxr_jpg_dir,
        chest_imagenome_dir,
        out_dir,
        db_name,
        num_patient,
        sample_icu_patient_only,
        split,
        deid=False,
        timeshift=False,
        cur_patient_ratio=0.0,
        start_year=None,
        time_span=None,
        current_time=None,
        verbose=True,
    ):
        super().__init__()
        assert split in ["train", "valid", "test"]
        self.split = split  # train, valid, test

        self.mimic_iv_dir = mimic_iv_dir
        self.mimic_cxr_jpg_dir = mimic_cxr_jpg_dir
        self.chest_imagenome_dir = chest_imagenome_dir
        self.out_dir = os.path.join(out_dir, db_name, split)
        self.preprocessed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "preprocessed_data")

        self.deid = deid
        self.timeshift = timeshift

        self.sample_icu_patient_only = sample_icu_patient_only
        self.num_patient = num_patient
        self.num_cur_patient = int(self.num_patient * cur_patient_ratio)
        self.num_non_cur_patient = self.num_patient - int(self.num_patient * cur_patient_ratio)

        if self.timeshift:
            self.start_year = start_year
            self.start_pivot_datetime = datetime(year=self.start_year, month=1, day=1)
            self.time_span = time_span
            self.current_time = current_time
            if verbose:
                print("timeshift is True")
                print(f"start_year: {self.start_year}")
                print(f"time_span: {self.time_span}")
                print(f"current_time: {self.current_time}")

        self.conn = sqlite3.connect(os.path.join(self.out_dir, db_name + ".db"))
        self.cur = self.conn.cursor()
        with open(os.path.join(self.out_dir, db_name + ".sqlite"), "r") as sql_file:
            sql_script = sql_file.read()
        self.cur.executescript(sql_script)

        self.chartevent2itemid = {k.lower(): v for k, v in CHARTEVENT2ITEMID.items()}  # lower case

    def _load_mimic_cxr_metadata(self, return_raw=True):
        # read
        cxr_meta = pd.read_csv(
            os.path.join(self.mimic_cxr_jpg_dir, "mimic-cxr-2.0.0-metadata.csv"),
            usecols=[
                "dicom_id",
                "subject_id",
                "study_id",
                "ViewPosition",
                "Rows",
                "Columns",
                "StudyDate",
                "StudyTime",
            ],
        )
        cxr_meta = cxr_meta.rename(columns={"dicom_id": "image_id"})
        print(cxr_meta.shape)

        # build a new column: StudyDateTime
        cxr_meta["StudyDateTime"] = pd.to_datetime(cxr_meta.StudyDate.astype(str).apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}") + " " + cxr_meta.StudyTime.apply(lambda x: "%010.3f" % x))

        # build a new column: StudyOrder
        cxr_meta_ = cxr_meta.copy()
        cxr_meta_ = cxr_meta_.sort_values(by=["subject_id", "study_id", "StudyDateTime"])
        cxr_meta_ = cxr_meta_.drop_duplicates(subset=["subject_id", "study_id"], keep="first").copy()
        cxr_meta_["StudyDateTime_study_id"] = cxr_meta_["StudyDateTime"].astype(str) + cxr_meta_["study_id"].astype(str)
        cxr_meta_["StudyDateTime_study_id"] = pd.to_datetime(cxr_meta_["StudyDateTime_study_id"])
        cxr_meta_["StudyOrder"] = cxr_meta_.groupby(["subject_id"])["StudyDateTime_study_id"].rank(method="dense")
        cxr_meta["StudyOrder"] = cxr_meta["study_id"].map(cxr_meta_[["study_id", "StudyOrder"]].set_index("study_id")["StudyOrder"])

        # remove overlapped columns
        del cxr_meta["StudyDate"]
        del cxr_meta["StudyTime"]

        # after base preprocessing, keep all data
        self.mimic_cxr_metadata = cxr_meta.copy()
        if return_raw:
            return cxr_meta

        # Assumption: Use only frontal images (AP/PA)
        cxr_meta = cxr_meta[cxr_meta["ViewPosition"].isin(["AP", "PA"])].reset_index(drop=True)
        print(cxr_meta.shape)

        # Assumption: Given the same study_id, use only one image (studydatetime=first, dicom_id=first)
        cxr_meta = cxr_meta.sort_values(["study_id", "StudyDateTime", "image_id"], ascending=[True, True, True])
        cxr_meta = cxr_meta[cxr_meta["image_id"].isin(cxr_meta.groupby(["study_id"])["image_id"].first().values)]
        assert cxr_meta.groupby(["study_id", "StudyDateTime"])["image_id"].nunique().value_counts().size == 1
        print(cxr_meta.shape)

        return cxr_meta

    def _get_bbox_information(self):
        if self.split == "test":
            # For the 1st/2nd image of each study, we use "gold" dataset
            gold_bbox_info = pd.read_csv(
                os.path.join(
                    self.chest_imagenome_dir,
                    "gold_dataset",
                    "gold_bbox_coordinate_annotations_1000images.csv",
                ),
                usecols=["image_id", "bbox_name", "coord224"],
            )
            gold_bbox_info["image_id"] = gold_bbox_info["image_id"].str.replace(".dcm", "")
            gold_bbox_info = gold_bbox_info.rename(columns={"bbox_name": "object"})

            # For >2nd images of each study, we use "silver" dataset
            silver_bbox_info = pd.read_csv(
                os.path.join(
                    self.chest_imagenome_dir,
                    "silver_dataset",
                    "scene_tabular",
                    "bbox_objects_tabular.txt",
                ),
                sep="\t",
                usecols=["object_id", "bbox_name", "x1", "y1", "x2", "y2"],
            )
            assert sum(silver_bbox_info["object_id"].apply(lambda x: x.split("_")[-1]) != silver_bbox_info["bbox_name"]) == 0
            silver_bbox_info["image_id"] = silver_bbox_info["object_id"].apply(lambda x: x.split("_")[0])
            silver_bbox_info["coord224"] = [
                str([int(x1), int(y1), int(x2), int(y2)])
                for x1, y1, x2, y2 in zip(
                    silver_bbox_info["x1"],
                    silver_bbox_info["y1"],
                    silver_bbox_info["x2"],
                    silver_bbox_info["y2"],
                )
            ]
            silver_bbox_info = silver_bbox_info[["image_id", "bbox_name", "coord224"]]
            silver_bbox_info = silver_bbox_info.rename(columns={"bbox_name": "object"})

            # we replace the bbox_info to gold_bbox_info if the image_id is in gold_bbox_info
            gold_bbox_image_ids = gold_bbox_info["image_id"].unique()
            silver_bbox_info = silver_bbox_info[~silver_bbox_info["image_id"].isin(gold_bbox_image_ids)]
            bbox_info = pd.concat([gold_bbox_info, silver_bbox_info], axis=0).reset_index(drop=True)

        else:
            bbox_info = pd.read_csv(
                os.path.join(
                    self.chest_imagenome_dir,
                    "silver_dataset",
                    "scene_tabular",
                    "bbox_objects_tabular.txt",
                ),
                sep="\t",
                usecols=["object_id", "bbox_name", "x1", "y1", "x2", "y2"],
            )
            assert sum(bbox_info["object_id"].apply(lambda x: x.split("_")[-1]) != bbox_info["bbox_name"]) == 0
            bbox_info["image_id"] = bbox_info["object_id"].apply(lambda x: x.split("_")[0])
            bbox_info["coord224"] = [str([int(x1), int(y1), int(x2), int(y2)]) for x1, y1, x2, y2 in zip(bbox_info["x1"], bbox_info["y1"], bbox_info["x2"], bbox_info["y2"])]
            bbox_info = bbox_info[["image_id", "bbox_name", "coord224"]]
            bbox_info = bbox_info.rename(columns={"bbox_name": "object"})

        return bbox_info

    def build_tb_cxr_table(self, flag_for_plus=False):
        if not flag_for_plus:
            print("\nProcessing tb_cxr")
        else:
            print("\nProcessing tb_cxr_plus")

        start_time = time.time()

        # read
        cxr_meta_raw = self._load_mimic_cxr_metadata(return_raw=True)
        cxr_meta_raw = cxr_meta_raw.rename(columns={col: col.lower() for col in cxr_meta_raw.columns})  # column - lowercase

        # get image id list
        if self.split == "test":
            df_cohort_gold = pd.read_csv(os.path.join(self.preprocessed_dir, "cohort_gold.csv"))
            # study_id_list = df_cohort_gold["study_id"].unique().tolist()  # 2338 studies
            image_id_list = df_cohort_gold["image_id"].unique().tolist()  # 2338 images
            # gold_1st_iids = self._get_gold_1st_image_ids()  # 500 images
        else:
            df_cohort_silver = pd.read_csv(os.path.join(self.preprocessed_dir, "cohort_silver.csv"))
            if self.split in ["valid", "train"]:
                valid_dataset = pd.read_csv(os.path.join(self.preprocessed_dir, "valid_dataset.csv"))
                # study_id_list = valid_dataset.study_id.unique().tolist()  # 8653 studies
                # assert all([study_id in df_cohort_silver['study_id'].unique().tolist() for study_id in study_id_list])
                image_id_list = valid_dataset["image_id"].unique().tolist()  # 8653 images
                assert all([image_id in df_cohort_silver["image_id"].unique().tolist() for image_id in image_id_list])
            else:
                raise NotImplementedError()

        # (deprecated) constrained by cohort (filter by study_id), not using image_id due to the discrepancy in _load_mimic_cxr_metadata()
        # NOTE: we use image_id to filter so that each study gets the exact one image
        cxr_meta = cxr_meta_raw[cxr_meta_raw["image_id"].isin(image_id_list)].reset_index(drop=True)
        print(cxr_meta.shape)
        assert cxr_meta["image_id"].nunique() == len(image_id_list)
        assert cxr_meta["study_id"].nunique() == cxr_meta["image_id"].nunique()

        # append columns: subject_id, study_id, image_id, studydatetime (studydate, studytime), viewpos
        TB_CXR_table = cxr_meta[
            [
                "subject_id",
                "study_id",
                "image_id",
                "viewposition",
                "studydatetime",
                "studyorder",
            ]
        ].copy()
        # preprocess studydatetime (e.g., 2150-03-30 04:54:38.906 -> 2150-03-30 04:54:38)
        TB_CXR_table.loc[:, "studydatetime"] = TB_CXR_table["studydatetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        TB_CXR_table.loc[:, "studydatetime"] = pd.to_datetime(TB_CXR_table["studydatetime"])

        # append column: hadm_id
        # NOTE: why transfers (20230308 meeting)
        # time-related columns in transfers table has a higher resolution than those in admissions table
        transfer_table = pd.read_csv(os.path.join(self.mimic_iv_dir, "hosp/transfers.csv"))
        transfer_table.sort_values(by=["subject_id", "intime"], inplace=True)
        transfer_table.loc[:, "intime"] = pd.to_datetime(transfer_table["intime"])
        transfer_table.loc[:, "outtime"] = pd.to_datetime(transfer_table["outtime"])

        # Init
        study_id_to_hadm_id = {}
        transfer_table_grouped = transfer_table.groupby("subject_id")
        transfer_table_grouped = {subject_id: group for subject_id, group in transfer_table_grouped}

        for subject_id, study_id, study_datetime in TB_CXR_table[["subject_id", "study_id", "studydatetime"]].values:
            if subject_id not in transfer_table_grouped:
                study_id_to_hadm_id[study_id] = None
                continue

            transfer_records = transfer_table_grouped[subject_id]
            transfer_records = transfer_records[(transfer_records["intime"] <= study_datetime) & (transfer_records["outtime"] >= study_datetime)]

            # Determine hadm_id
            if len(transfer_records) == 1:
                study_id_to_hadm_id[study_id] = transfer_records["hadm_id"].values[0]
            elif len(transfer_records) == 0:
                study_id_to_hadm_id[study_id] = None
            else:
                if subject_id in [13891219]:  # exception in train/valid split (preprocessed_data/valid_dataset.csv)
                    continue
                raise ValueError()

        TB_CXR_table["hadm_id"] = TB_CXR_table["study_id"].map(study_id_to_hadm_id)
        # NOTE: some studies have no hadm_id, but we still keep them
        print("Number of studies: {}".format(TB_CXR_table["study_id"].nunique()))
        print("Number of studies with no hadm_id: {}".format(TB_CXR_table[TB_CXR_table["hadm_id"].isnull()]["study_id"].nunique()))
        print("Number of studies with hadm_id: {}".format(TB_CXR_table[TB_CXR_table["hadm_id"].notnull()]["study_id"].nunique()))

        print("Number of patients: {}".format(TB_CXR_table["subject_id"].nunique()))
        print("Number of patients with at least one hadm_id: {}".format(TB_CXR_table.dropna(subset=["hadm_id"])["subject_id"].nunique()))

        print("This statistics will be changed after pre-processing admission/transfer tables")

        # keep patients
        self.cxr_patient_list = TB_CXR_table["subject_id"].unique()

        if not flag_for_plus:
            """
            build tb_cxr.csv
            """
            TB_CXR_table = TB_CXR_table[
                [
                    "subject_id",
                    "hadm_id",
                    "study_id",
                    "image_id",
                    "viewposition",
                    "studydatetime",
                ]
            ]
            TB_CXR_table = TB_CXR_table.reset_index().rename(columns={"index": "row_id"})
            TB_CXR_table["row_id"] = range(len(TB_CXR_table))
            TB_CXR_table.to_csv(os.path.join(self.out_dir, "tb_cxr.csv"), index=False)
            print(f"tb_cxr processed (took {round(time.time() - start_time, 4)} secs)")

        else:
            # append columns: object, relation, attribute, category
            if self.split == "test":
                # NOTE: load gold+_dataset.csv (due to the some labels are not from gold)
                label_dataset = pd.read_csv(os.path.join(self.preprocessed_dir, "gold+_dataset.csv"))
                label_dataset = label_dataset[
                    [
                        "study_id",
                        "image_id",
                        "bbox",
                        "relation",
                        "label_name",
                        "categoryID",
                    ]
                ]
                label_dataset = label_dataset.rename(
                    columns={
                        "bbox": "object",
                        "label_name": "attribute",
                        "categoryID": "category",
                    }
                )
            elif self.split in ["valid", "train"]:
                # NOTE: For train/valid, we use valid dataset (due to the size of train dataset is too large)
                label_dataset = pd.read_csv(os.path.join(self.preprocessed_dir, "valid_dataset.csv"))
                label_dataset = label_dataset[
                    [
                        "study_id",
                        "image_id",
                        "bbox",
                        "relation",
                        "label_name",
                        "categoryID",
                    ]
                ]
                label_dataset = label_dataset.rename(
                    columns={
                        "bbox": "object",
                        "label_name": "attribute",
                        "categoryID": "category",
                    }
                )
            else:
                raise ValueError()

            # append columns: ct_ratio, mt_ratio
            bbox_info = self._get_bbox_information()
            bbox_info = bbox_info[bbox_info["image_id"].isin(label_dataset["image_id"].unique())]

            # we filter out other objects except for 4 objects (e.g., "left lung", "right lung", "cardiac silhouette", "upper mediastinum")
            bbox_info = bbox_info[
                bbox_info["object"].isin(
                    [
                        "left lung",
                        "right lung",
                        "cardiac silhouette",
                        "upper mediastinum",
                    ]
                )
            ]
            # compute the number of unique bboxes for each image
            bbox_info["num_unique_bbox"] = bbox_info.groupby(["image_id"])["object"].transform("nunique")
            # filter image ids where the number of unique bboxes is less than 4
            bbox_info = bbox_info[bbox_info["num_unique_bbox"] == 4].reset_index(drop=True)
            bbox_info = bbox_info.drop(columns=["num_unique_bbox"])
            assert (bbox_info.groupby(["image_id"])["object"].nunique() == 4).all()

            def _calculate_ct_ratio(features):
                features = features.set_index("object")
                coord_ll = literal_eval(features.loc["left lung", "coord224"])
                coord_rl = literal_eval(features.loc["right lung", "coord224"])
                coord_cd = literal_eval(features.loc["cardiac silhouette", "coord224"])
                thorax_width = max(coord_ll[2], coord_ll[0]) - min(coord_rl[2], coord_rl[0])
                cardiac_width = abs(coord_cd[2] - coord_cd[0])
                ct_ratio = int(cardiac_width) / int(thorax_width)
                return ct_ratio

            def _calculate_mt_ratio(features):
                # NOTE: here, we use "upper mediastinum' as the reference not "mediastinum"
                features = features.set_index("object")
                coord_ll = literal_eval(features.loc["left lung", "coord224"])
                coord_rl = literal_eval(features.loc["right lung", "coord224"])
                coord_um = literal_eval(features.loc["upper mediastinum", "coord224"])
                thorax_width = max(coord_ll[2], coord_ll[0]) - min(coord_rl[2], coord_rl[0])
                mediastinum_width = abs(coord_um[2] - coord_um[0])
                mt_ratio = int(mediastinum_width) / int(thorax_width)
                return mt_ratio

            print("Calculating CT ratio...")
            bbox_info_ct_ratio = bbox_info.groupby(["image_id"]).apply(_calculate_ct_ratio).reset_index().rename(columns={0: "ct_ratio"})
            print("Calculating MT ratio...")
            bbox_info_mt_ratio = bbox_info.groupby(["image_id"]).apply(_calculate_mt_ratio).reset_index().rename(columns={0: "mt_ratio"})

            label_dataset = pd.merge(label_dataset, bbox_info_ct_ratio, on=["image_id"], how="left")
            label_dataset = pd.merge(label_dataset, bbox_info_mt_ratio, on=["image_id"], how="left")
            # NOTE: some images do not have bbox information
            print(f"Number of images that ct_ratio is null: {label_dataset[label_dataset['ct_ratio'].isnull()]['image_id'].nunique()}")
            print(f"Number of images that mt_ratio is null: {label_dataset[label_dataset['mt_ratio'].isnull()]['image_id'].nunique()}")

            """
            build tb_cxr_plus.csv
            """
            TB_CXR_PLUS_table = TB_CXR_table.merge(label_dataset, on=["image_id", "study_id"], how="right")
            TB_CXR_PLUS_table = TB_CXR_PLUS_table[
                [
                    "subject_id",
                    "hadm_id",
                    "study_id",
                    "image_id",
                    "viewposition",
                    "studydatetime",
                    "studyorder",
                    "object",
                    "relation",
                    "attribute",
                    "category",
                    "ct_ratio",
                    "mt_ratio",
                ]
            ]
            TB_CXR_PLUS_table = TB_CXR_PLUS_table.reset_index().rename(columns={"index": "row_id"})
            TB_CXR_PLUS_table["row_id"] = range(len(TB_CXR_PLUS_table))
            TB_CXR_PLUS_table.to_csv(os.path.join(self.out_dir, "tb_cxr_plus.csv"), index=False)
            print(f"tb_cxr_plus processed (took {round(time.time() - start_time, 4)} secs)")

        if self.timeshift:
            print("We will change the studydatetime to the timeshifted version after we have the timeshifted data")
            print("Therefore, we further preprocess TB_CXR/TB_CXR_PLUS in the following step: `build_admission_table()`")

    def build_admission_table(self):
        print("Processing patients, admissions, icustays, transfers")
        start_time = time.time()

        # read patients
        PATIENTS_table = read_csv(
            self.mimic_iv_dir,
            "hosp/patients.csv",
            columns=["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
            lower=True,
        )
        PATIENTS_table = PATIENTS_table.reset_index().rename(columns={"index": "row_id"})

        subjectid2anchor_year = {
            pid: anch_year
            for pid, anch_year in zip(
                PATIENTS_table["subject_id"].values,
                PATIENTS_table["anchor_year"].values,
            )
        }
        subjectid2anchor_age = {pid: anch_age for pid, anch_age in zip(PATIENTS_table["subject_id"].values, PATIENTS_table["anchor_age"].values)}

        # add new column `dob` and remove anchor columns
        PATIENTS_table = PATIENTS_table.assign(dob=lambda x: x["anchor_year"] - x["anchor_age"])
        PATIENTS_table = PATIENTS_table[["row_id", "subject_id", "gender", "dob", "dod"]]
        # NOTE: the month and day of dob are randomly sampled
        PATIENTS_table["dob"] = PATIENTS_table["dob"].apply(lambda x: generate_random_date(x))
        # NOTE: time format (dob/dod)
        for col in ["dob", "dod"]:
            PATIENTS_table[col] = pd.to_datetime(PATIENTS_table[col], format="%Y-%m-%d")
            PATIENTS_table[col] = PATIENTS_table[col].dt.strftime(date_format="%Y-%m-%d 00:00:00")

        # read admissions
        ADMISSIONS_table = read_csv(
            self.mimic_iv_dir,
            "hosp/admissions.csv",
            columns=[
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "admission_type",
                "admission_location",
                "discharge_location",
                "insurance",
                "language",
                "marital_status",
                # "ethnicity",
            ],
            lower=True,
        )
        ADMISSIONS_table = ADMISSIONS_table.reset_index().rename(columns={"index": "row_id"})

        # compute admission age
        ADMISSIONS_table["age"] = [
            int((datetime.strptime(admtime, "%Y-%m-%d %H:%M:%S")).year) - subjectid2anchor_year[pid] + subjectid2anchor_age[pid]
            for pid, admtime in zip(
                ADMISSIONS_table["subject_id"].values,
                ADMISSIONS_table["admittime"].values,
            )
        ]

        # remove age outliers
        ADMISSIONS_table = ADMISSIONS_table[(ADMISSIONS_table["age"] > 10) & (ADMISSIONS_table["age"] < 90)]

        # remove hospital stay outlier
        hosp_stay_dict = {
            hosp: (datetime.strptime(dischtime, "%Y-%m-%d %H:%M:%S") - datetime.strptime(admtime, "%Y-%m-%d %H:%M:%S")).days
            for hosp, admtime, dischtime in zip(
                ADMISSIONS_table["hadm_id"].values,
                ADMISSIONS_table["admittime"].values,
                ADMISSIONS_table["dischtime"].values,
            )
        }
        threshold_offset = np.percentile(list(hosp_stay_dict.values()), q=99)  # remove greater than 99% (31 days) or 95% (14 days) (for MIMIC-IV)
        print(f"99% of hospital stays are less than {threshold_offset} days")
        ADMISSIONS_table = ADMISSIONS_table[ADMISSIONS_table["hadm_id"].isin([hosp for hosp in hosp_stay_dict if hosp_stay_dict[hosp] < threshold_offset])]

        # NOTE: save original admittime
        self.HADM_ID2admtime_dict = {hadm: admtime for hadm, admtime in zip(ADMISSIONS_table["hadm_id"].values, ADMISSIONS_table["admittime"].values)}
        self.HADM_ID2dischtime_dict = {hadm: dischtime for hadm, dischtime in zip(ADMISSIONS_table["hadm_id"].values, ADMISSIONS_table["dischtime"].values)}

        # read icustays
        ICUSTAYS_table = read_csv(
            self.mimic_iv_dir,
            "icu/icustays.csv",
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "first_careunit",
                "last_careunit",
                "intime",
                "outtime",
            ],
            lower=True,
        )
        ICUSTAYS_table = ICUSTAYS_table.reset_index().rename(columns={"index": "row_id"})

        # subset only icu patients
        if self.sample_icu_patient_only:
            raise NotImplementedError("We do not support this option for MIMIC-IV yet.")
            # ADMISSIONS_table = ADMISSIONS_table[ADMISSIONS_table["subject_id"].isin(set(ICUSTAYS_table["subject_id"]))]

        # read transfer
        TRANSFERS_table = read_csv(
            self.mimic_iv_dir,
            "hosp/transfers.csv",
            columns=[
                "subject_id",
                "hadm_id",
                "transfer_id",
                "eventtype",
                "careunit",
                "intime",
                "outtime",
            ],
            lower=True,
        )
        TRANSFERS_table = TRANSFERS_table.reset_index().rename(columns={"index": "row_id"})
        TRANSFERS_table = TRANSFERS_table.dropna(subset=["intime"])

        # NOTE: since we focus on patients who have CXR images, we sample patients who have at least one CXR image
        self.sample_cxr_patient_only = True
        if self.sample_cxr_patient_only:
            PATIENTS_table = PATIENTS_table[PATIENTS_table["subject_id"].isin(self.cxr_patient_list)]
            ADMISSIONS_table = ADMISSIONS_table[ADMISSIONS_table["subject_id"].isin(self.cxr_patient_list)]
        else:
            raise NotImplementedError("Not implemented for sampling together for patients who have no CXR images")

        ################################################################################
        """
        Decide the offset (optimized for the MIMIC-IV + MIMIC-CXR)
        """
        if self.timeshift:
            # 1) get the earliest admission time of each patient, compute the offset, and save it
            ADMITTIME_earliest = {subj_id: min(ADMISSIONS_table["admittime"][ADMISSIONS_table["subject_id"] == subj_id].values) for subj_id in ADMISSIONS_table["subject_id"].unique()}
            self.subjectid2admittime_dict = {
                subj_id: self.first_admit_year_sampler(
                    self.start_year,
                    self.time_span,
                    datetime.strptime(ADMITTIME_earliest[subj_id], "%Y-%m-%d %H:%M:%S").year,
                )
                for subj_id in ADMISSIONS_table["subject_id"].unique()
            }

            message = """
            To maximize the number of patients who have at least one CXR image,
            we override the admission time of the patient with the earliest studydatetime which is right after shifting the year.
            Please keep in mind that `self.subjectid2admittime_dict` is used to override the admission time of the patient.
            """
            print(message)
            TB_CXR_table = pd.read_csv(os.path.join(self.out_dir, "tb_cxr.csv"))

            # 2) get the earliest studydatetime of the patient, compute the offset, and override to `self.subjectid2admittime_dict`
            STUDYDATETIME_earliest = {subj_id: min(TB_CXR_table["studydatetime"][TB_CXR_table["subject_id"] == subj_id].values) for subj_id in TB_CXR_table["subject_id"].unique()}
            self.subjectid2admittime_dict = {
                subj_id: self.first_admit_year_sampler(
                    self.start_year,
                    self.time_span,
                    datetime.strptime(STUDYDATETIME_earliest[subj_id], "%Y-%m-%d %H:%M:%S").year,
                )
                for subj_id in TB_CXR_table["subject_id"].unique()
            }

            # 3) get the optimized offset for some patients, and override to `self.subjectid2admittime_dict`
            random.seed(0)  # NOTE: fix the random seed
            _NUM_CUR_PATTIENTS = 40 if self.split == "test" else 80  # 10% of the patients (400/800 patients)
            target_subject_ids = random.sample(ADMISSIONS_table["subject_id"].unique().tolist(), _NUM_CUR_PATTIENTS)
            for idx, subj_id in enumerate(target_subject_ids):
                # pick the last admission of the patient
                table = ADMISSIONS_table[ADMISSIONS_table["subject_id"] == subj_id]
                table = table.sort_values(by="admittime", ascending=False)
                hadm_id = table["hadm_id"].values[0]

                # get the admittime and dischtime (of last admission)
                admittime = ADMISSIONS_table[ADMISSIONS_table["hadm_id"] == hadm_id]["admittime"]
                dischtime = ADMISSIONS_table[ADMISSIONS_table["hadm_id"] == hadm_id]["dischtime"]

                # randomly choose the time between admittime and dischtime
                admittime = datetime.strptime(admittime.values[0], "%Y-%m-%d %H:%M:%S")
                dischtime = datetime.strptime(dischtime.values[0], "%Y-%m-%d %H:%M:%S")
                random_time = random.uniform(admittime.timestamp(), dischtime.timestamp())
                random_time = datetime.fromtimestamp(random_time)

                # compute the offset (minutes)
                current_time = datetime.strptime(self.current_time, "%Y-%m-%d %H:%M:%S")
                offset = int((current_time - random_time).total_seconds() / 60)
                self.subjectid2admittime_dict[subj_id] = offset

        ################################################################################
        """
        Since we start to adjust the time of the patients after this point,
        please do not change the offset (i.e., `self.subjectid2admittime_dict`)
        """
        # process patients
        if self.timeshift:
            PATIENTS_table["dob"] = adjust_time(
                PATIENTS_table,
                "dob",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            PATIENTS_table["dod"] = adjust_time(
                PATIENTS_table,
                "dod",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            PATIENTS_table = PATIENTS_table.dropna(subset=["dob"])

        # process admissions
        if self.timeshift:
            ADMISSIONS_table["admittime"] = adjust_time(
                ADMISSIONS_table,
                "admittime",
                start_year=self.start_year,  # To avoid the admittime less than the start_year
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            ADMISSIONS_table["dischtime"] = adjust_time(
                ADMISSIONS_table,
                "dischtime",
                start_year=self.start_year,  # To avoid the dischtime less than the start_year
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            ADMISSIONS_table = ADMISSIONS_table.dropna(subset=["admittime"])

        # process icustays
        if self.timeshift:
            ICUSTAYS_table["intime"] = adjust_time(
                ICUSTAYS_table,
                "intime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            ICUSTAYS_table["outtime"] = adjust_time(
                ICUSTAYS_table,
                "outtime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            ICUSTAYS_table = ICUSTAYS_table.dropna(subset=["intime"])

        # process transfers
        if self.timeshift:
            TRANSFERS_table["intime"] = adjust_time(
                TRANSFERS_table,
                "intime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            TRANSFERS_table["outtime"] = adjust_time(
                TRANSFERS_table,
                "outtime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            TRANSFERS_table = TRANSFERS_table.dropna(subset=["intime"])

        # process tb_cxr
        if self.timeshift:
            TB_CXR_table = pd.read_csv(os.path.join(self.out_dir, "tb_cxr.csv"))
            TB_CXR_table["studydatetime"] = adjust_time(
                TB_CXR_table,
                "studydatetime",
                start_year=self.start_year,  # To avoid the studydatetime less than the start_year
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            print(
                "The number of invalid time points in TB_CXR table: ",
                len(TB_CXR_table[TB_CXR_table["studydatetime"].isna()]),
            )
            TB_CXR_table = TB_CXR_table.dropna(subset=["studydatetime"])

        # process tb_cxr_plus
        if self.timeshift:
            TB_CXR_PLUS_table = pd.read_csv(os.path.join(self.out_dir, "tb_cxr_plus.csv"))
            TB_CXR_PLUS_table["studydatetime"] = adjust_time(
                TB_CXR_PLUS_table,
                "studydatetime",
                start_year=self.start_year,  # To avoid the studydatetime less than the start_year
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            print(
                "The number of invalid time points in TB_CXR_PLUS table: ",
                len(TB_CXR_PLUS_table[TB_CXR_PLUS_table["studydatetime"].isna()]),
            )
            TB_CXR_PLUS_table = TB_CXR_PLUS_table.dropna(subset=["studydatetime"])

        ################################################################################
        """
        Decide the final cohort of patients: `self.cur_patient_list` and `self.non_cur_patient`
        """
        # sample current patients
        try:
            self.cur_patient_list = self.rng.choice(
                ADMISSIONS_table["subject_id"][ADMISSIONS_table["dischtime"].isnull()].unique(),
                self.num_cur_patient,
                replace=False,
            ).tolist()
        except:
            print("Cannot take a larger sample than population when 'replace=False")
            print("Use all available patients instead.")
            self.cur_patient_list = ADMISSIONS_table["subject_id"][ADMISSIONS_table["dischtime"].isnull()].unique().tolist()

        # sample non-current patients
        try:
            self.non_cur_patient = self.rng.choice(
                ADMISSIONS_table["subject_id"][(ADMISSIONS_table["dischtime"].notnull()) & (~ADMISSIONS_table["subject_id"].isin(self.cur_patient_list))].unique(),
                self.num_non_cur_patient,
                replace=False,
            ).tolist()
        except:
            print("Cannot take a larger sample than population when 'replace=False")
            print("Use all available patients instead.")
            self.non_cur_patient = ADMISSIONS_table["subject_id"][(ADMISSIONS_table["dischtime"].notnull()) & (~ADMISSIONS_table["subject_id"].isin(self.cur_patient_list))].unique().tolist()

        self.patient_list = self.cur_patient_list + self.non_cur_patient
        print(f"num_cur_patient: {len(self.cur_patient_list)}")
        print(f"num_non_cur_patient: {len(self.non_cur_patient)}")
        print(f"num_patient: {len(self.patient_list)}")

        PATIENTS_table = PATIENTS_table[PATIENTS_table["subject_id"].isin(self.patient_list)]
        ADMISSIONS_table = ADMISSIONS_table[ADMISSIONS_table["subject_id"].isin(self.patient_list)]

        TB_CXR_table = TB_CXR_table[TB_CXR_table["subject_id"].isin(self.patient_list)]
        TB_CXR_PLUS_table = TB_CXR_PLUS_table[TB_CXR_PLUS_table["subject_id"].isin(self.patient_list)]

        self.hadm_list = list(set(ADMISSIONS_table["hadm_id"]))
        ICUSTAYS_table = ICUSTAYS_table[ICUSTAYS_table["hadm_id"].isin(self.hadm_list)]
        TRANSFERS_table = TRANSFERS_table[TRANSFERS_table["hadm_id"].isin(self.hadm_list)]

        if self.deid:  # de-identification
            rng_val = np.random.default_rng(0)  # init random generator
            random_indices = rng_val.choice(len(ICUSTAYS_table), len(ICUSTAYS_table), replace=False)

            careunit_mapping = {
                original: shuffled
                for original, shuffled in zip(
                    ICUSTAYS_table["first_careunit"],
                    ICUSTAYS_table["first_careunit"].iloc[random_indices],
                )
            }
            careunit_mapping.update(
                {
                    original: shuffled
                    for original, shuffled in zip(
                        ICUSTAYS_table["last_careunit"],
                        ICUSTAYS_table["last_careunit"].iloc[random_indices],
                    )
                }
            )

            # shuffle ICUStays_table
            ICUSTAYS_table["first_careunit"] = ICUSTAYS_table["first_careunit"].map(careunit_mapping)
            ICUSTAYS_table["last_careunit"] = ICUSTAYS_table["last_careunit"].map(careunit_mapping)

            # shuffle TRANSFERS_table
            TRANSFERS_table["careunit"] = TRANSFERS_table["careunit"].replace(careunit_mapping)

        PATIENTS_table["row_id"] = range(len(PATIENTS_table))
        ADMISSIONS_table["row_id"] = range(len(ADMISSIONS_table))
        ICUSTAYS_table["row_id"] = range(len(ICUSTAYS_table))
        TRANSFERS_table["row_id"] = range(len(TRANSFERS_table))

        PATIENTS_table.to_csv(os.path.join(self.out_dir, "patients.csv"), index=False)
        ADMISSIONS_table.to_csv(os.path.join(self.out_dir, "admissions.csv"), index=False)
        ICUSTAYS_table.to_csv(os.path.join(self.out_dir, "icustays.csv"), index=False)
        TRANSFERS_table.to_csv(os.path.join(self.out_dir, "transfers.csv"), index=False)

        if self.deid:  # de-identification
            # shuffle TB_CXR_table
            TB_CXR_table = TB_CXR_table.sort_values(by=["subject_id", "studydatetime"], ascending=[True, True])
            first_dates = TB_CXR_table.groupby("subject_id")["studydatetime"].first()
            TB_CXR_table["studydatetime"] = TB_CXR_table.groupby("subject_id")["studydatetime"].shift(-1)
            TB_CXR_table["studydatetime"] = TB_CXR_table["studydatetime"].fillna(TB_CXR_table["subject_id"].map(first_dates))
            TB_CXR_table = TB_CXR_table.sort_values(by=["subject_id", "studydatetime"], ascending=[True, True])

            # shuffle TB_CXR_PLUS_table
            sid_to_stime = {
                study_id: studydatetime
                for study_id, studydatetime in zip(
                    TB_CXR_table["study_id"].values,
                    TB_CXR_table["studydatetime"].values,
                )
            }
            TB_CXR_PLUS_table["studydatetime"] = TB_CXR_PLUS_table["study_id"].map(sid_to_stime)
            TB_CXR_PLUS_table = TB_CXR_PLUS_table.sort_values(by=["subject_id", "studydatetime"], ascending=[True, True])

        TB_CXR_table["hadm_id"] = TB_CXR_table["hadm_id"].apply(lambda x: x if x in self.hadm_list else np.nan)
        TB_CXR_table["row_id"] = range(len(TB_CXR_table))
        TB_CXR_table.to_csv(os.path.join(self.out_dir, "tb_cxr.csv"), index=False)
        print("TB_CXR_table (size) : ", TB_CXR_table.shape)

        TB_CXR_PLUS_table["hadm_id"] = TB_CXR_PLUS_table["hadm_id"].apply(lambda x: x if x in self.hadm_list else np.nan)
        TB_CXR_PLUS_table["row_id"] = range(len(TB_CXR_PLUS_table))
        TB_CXR_PLUS_table.to_csv(os.path.join(self.out_dir, "tb_cxr_plus.csv"), index=False)
        print("TB_CXR_PLUS_table (size) : ", TB_CXR_PLUS_table.shape)

        print(f"patients, admissions, icustays, transfers processed (took {round(time.time() - start_time, 4)} secs)")

    def build_dictionary_table(self):
        print("Processing dictionary tables (d_icd_diagnoses, d_icd_procedures, d_labitems, d_items)")
        start_time = time.time()

        """
        d_icd_diagnoses
        """
        # read csv
        D_ICD_DIAGNOSES_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="hosp/d_icd_diagnoses.csv",
            columns=["icd_code", "long_title"],
            lower=True,
        )
        D_ICD_DIAGNOSES_table = D_ICD_DIAGNOSES_table.astype({"icd_code": str})

        # preprocess
        D_ICD_DIAGNOSES_table = D_ICD_DIAGNOSES_table.dropna()
        D_ICD_DIAGNOSES_table = D_ICD_DIAGNOSES_table.drop_duplicates(subset=["icd_code"])  # NOTE: some icd codes have multiple long titles
        D_ICD_DIAGNOSES_table = D_ICD_DIAGNOSES_table.drop_duplicates(subset=["long_title"])  # NOTE: some long titles have multiple icd codes

        # save csv
        D_ICD_DIAGNOSES_table = D_ICD_DIAGNOSES_table.reset_index(drop=False)
        D_ICD_DIAGNOSES_table = D_ICD_DIAGNOSES_table.rename(columns={"index": "row_id"})  # add row_id
        D_ICD_DIAGNOSES_table["row_id"] = range(len(D_ICD_DIAGNOSES_table))
        D_ICD_DIAGNOSES_table.to_csv(os.path.join(self.out_dir, "d_icd_diagnoses.csv"), index=False)
        self.D_ICD_DIAGNOSES_dict = {
            item: val
            for item, val in zip(
                D_ICD_DIAGNOSES_table["icd_code"].values,
                D_ICD_DIAGNOSES_table["long_title"].values,
            )
        }

        """
        d_icd_procedures
        """
        # read csv
        D_ICD_PROCEDURES_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="hosp/d_icd_procedures.csv",
            columns=["icd_code", "long_title"],
            lower=True,
        )
        D_ICD_PROCEDURES_table = D_ICD_PROCEDURES_table.astype({"icd_code": str})

        # preprocess
        D_ICD_PROCEDURES_table = D_ICD_PROCEDURES_table.dropna()
        D_ICD_PROCEDURES_table = D_ICD_PROCEDURES_table.drop_duplicates(subset=["icd_code"])  # NOTE: some icd codes have multiple long titles
        D_ICD_PROCEDURES_table = D_ICD_PROCEDURES_table.drop_duplicates(subset=["long_title"])  # NOTE: some long titles have multiple icd codes

        # save csv
        D_ICD_PROCEDURES_table = D_ICD_PROCEDURES_table.reset_index(drop=False)
        D_ICD_PROCEDURES_table = D_ICD_PROCEDURES_table.rename(columns={"index": "row_id"})  # add row_id
        D_ICD_PROCEDURES_table["row_id"] = range(len(D_ICD_PROCEDURES_table))
        D_ICD_PROCEDURES_table.to_csv(os.path.join(self.out_dir, "d_icd_procedures.csv"), index=False)
        self.D_ICD_PROCEDURES_dict = {
            item: val
            for item, val in zip(
                D_ICD_PROCEDURES_table["icd_code"].values,
                D_ICD_PROCEDURES_table["long_title"].values,
            )
        }

        """
        d_labitems
        """
        # read csv
        D_LABITEMS_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="hosp/d_labitems.csv",
            columns=["itemid", "label"],
            lower=True,
        )

        # preprocess
        D_LABITEMS_table = D_LABITEMS_table.dropna(subset=["label"])

        # save csv
        D_LABITEMS_table = D_LABITEMS_table.reset_index(drop=False)
        D_LABITEMS_table = D_LABITEMS_table.rename(columns={"index": "row_id"})  # add row_id
        D_LABITEMS_table["row_id"] = range(len(D_LABITEMS_table))
        D_LABITEMS_table.to_csv(os.path.join(self.out_dir, "d_labitems.csv"), index=False)
        self.D_LABITEMS_dict = {item: val for item, val in zip(D_LABITEMS_table["itemid"].values, D_LABITEMS_table["label"].values)}

        """
        d_items
        """
        # read csv
        D_ITEMS_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="icu/d_items.csv",
            columns=["itemid", "label", "abbreviation", "linksto"],
            lower=True,
        )

        # preprocess
        D_ITEMS_table = D_ITEMS_table.dropna(subset=["label"])
        D_ITEMS_table = D_ITEMS_table[D_ITEMS_table["linksto"].isin(["inputevents", "outputevents", "chartevents"])]

        # save csv
        D_ITEMS_table = D_ITEMS_table.reset_index(drop=False)
        D_ITEMS_table = D_ITEMS_table.rename(columns={"index": "row_id"})  # add row_id
        D_ITEMS_table["row_id"] = range(len(D_ITEMS_table))
        D_ITEMS_table.to_csv(os.path.join(self.out_dir, "d_items.csv"), index=False)
        self.D_ITEMS_dict = {item: val for item, val in zip(D_ITEMS_table["itemid"].values, D_ITEMS_table["label"].values)}

        print(f"d_icd_diagnoses, d_icd_procedures, d_labitems, d_items processed (took {round(time.time() - start_time, 4)} secs)")

    def build_diagnosis_table(self):
        print("Processing diagnoses_icd table")
        start_time = time.time()

        # read csv
        DIAGNOSES_ICD_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="hosp/diagnoses_icd.csv",
            columns=["subject_id", "hadm_id", "icd_code"],
            lower=True,
        )
        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table.astype({"icd_code": str})

        # preprocess
        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table.dropna(subset=["icd_code"])
        DIAGNOSES_ICD_table["charttime"] = [
            self.HADM_ID2admtime_dict[hadm] if hadm in self.HADM_ID2admtime_dict else None for hadm in DIAGNOSES_ICD_table["hadm_id"].values
        ]  # NOTE: assume charttime is at the hospital admission

        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table[DIAGNOSES_ICD_table["icd_code"].isin(self.D_ICD_DIAGNOSES_dict)]

        # de-identification
        if self.deid:
            DIAGNOSES_ICD_table = self.condition_value_shuffler(DIAGNOSES_ICD_table, target_cols=["icd_code"])

        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table[DIAGNOSES_ICD_table["hadm_id"].isin(self.hadm_list)]

        # timeshift
        if self.timeshift:
            DIAGNOSES_ICD_table["charttime"] = adjust_time(
                DIAGNOSES_ICD_table,
                "charttime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            DIAGNOSES_ICD_table = DIAGNOSES_ICD_table.dropna(subset=["charttime"])
            TIME = np.array([datetime.strptime(tt, "%Y-%m-%d %H:%M:%S") for tt in DIAGNOSES_ICD_table["charttime"].values])
            DIAGNOSES_ICD_table = DIAGNOSES_ICD_table[TIME >= self.start_pivot_datetime]

        # save csv
        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table.reset_index(drop=False)
        DIAGNOSES_ICD_table = DIAGNOSES_ICD_table.rename(columns={"index": "row_id"})
        DIAGNOSES_ICD_table["row_id"] = range(len(DIAGNOSES_ICD_table))
        DIAGNOSES_ICD_table.to_csv(os.path.join(self.out_dir, "diagnoses_icd.csv"), index=False)

        print(f"diagnoses_icd processed (took {round(time.time() - start_time, 4)} secs)")

    def build_procedure_table(self):
        print("Processing procedures_icd table")
        start_time = time.time()

        # read csv
        PROCEDURES_ICD_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="hosp/procedures_icd.csv",
            columns=["subject_id", "hadm_id", "icd_code", "chartdate"],
            lower=True,
        )

        PROCEDURES_ICD_table = PROCEDURES_ICD_table.astype({"icd_code": str})

        # NOTE: In MIMIC-IV, only px table has chartdate column, not charttime / here, we use charttime column
        PROCEDURES_ICD_table["charttime"] = pd.to_datetime(PROCEDURES_ICD_table["chartdate"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        PROCEDURES_ICD_table["admittime_"] = [self.HADM_ID2admtime_dict[hadm] if hadm in self.HADM_ID2admtime_dict else None for hadm in PROCEDURES_ICD_table["hadm_id"].values]
        PROCEDURES_ICD_table["dischtime_"] = [self.HADM_ID2dischtime_dict[hadm] if hadm in self.HADM_ID2dischtime_dict else None for hadm in PROCEDURES_ICD_table["hadm_id"].values]
        # charttime = charttime if charttime in [admission, discharge] else discharge_time
        PROCEDURES_ICD_table.loc[
            (PROCEDURES_ICD_table["charttime"] < PROCEDURES_ICD_table["admittime_"]) | (PROCEDURES_ICD_table["charttime"] > PROCEDURES_ICD_table["dischtime_"]),
            "charttime",
        ] = PROCEDURES_ICD_table["dischtime_"]
        # clean columns
        PROCEDURES_ICD_table = PROCEDURES_ICD_table.drop(columns=["chartdate", "admittime_", "dischtime_"])

        PROCEDURES_ICD_table = PROCEDURES_ICD_table[PROCEDURES_ICD_table["icd_code"].isin(self.D_ICD_PROCEDURES_dict)]

        # de-identification
        if self.deid:
            PROCEDURES_ICD_table = self.condition_value_shuffler(table=PROCEDURES_ICD_table, target_cols=["icd_code"])

        PROCEDURES_ICD_table = PROCEDURES_ICD_table[PROCEDURES_ICD_table["hadm_id"].isin(self.hadm_list)]

        # timeshift
        if self.timeshift:
            PROCEDURES_ICD_table["charttime"] = adjust_time(
                PROCEDURES_ICD_table,
                "charttime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            PROCEDURES_ICD_table = PROCEDURES_ICD_table.dropna(subset=["charttime"])
            TIME = np.array([datetime.strptime(tt, "%Y-%m-%d %H:%M:%S") for tt in PROCEDURES_ICD_table["charttime"].values])
            PROCEDURES_ICD_table = PROCEDURES_ICD_table[TIME >= self.start_pivot_datetime]

        # save csv
        PROCEDURES_ICD_table = PROCEDURES_ICD_table.reset_index(drop=False)
        PROCEDURES_ICD_table = PROCEDURES_ICD_table.rename(columns={"index": "row_id"})
        PROCEDURES_ICD_table["row_id"] = range(len(PROCEDURES_ICD_table))
        PROCEDURES_ICD_table.to_csv(os.path.join(self.out_dir, "procedures_icd.csv"), index=False)

        print(f"procedures_icd processed (took {round(time.time() - start_time, 4)} secs)")

    def build_labevent_table(self):
        print("Processing labevents table")
        start_time = time.time()

        # read csv
        LABEVENTS_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="hosp/labevents.csv",
            columns=[
                "subject_id",
                "hadm_id",
                "itemid",
                "charttime",
                "valuenum",
                "valueuom",
            ],
            lower=True,
        )
        LABEVENTS_table = LABEVENTS_table.dropna(subset=["hadm_id", "valuenum", "valueuom"])

        LABEVENTS_table = LABEVENTS_table[LABEVENTS_table["itemid"].isin(self.D_LABITEMS_dict)]

        # de-identification
        if self.deid:
            LABEVENTS_table = self.condition_value_shuffler(table=LABEVENTS_table, target_cols=["itemid", "valuenum", "valueuom"])

        LABEVENTS_table = LABEVENTS_table[LABEVENTS_table["hadm_id"].isin(self.hadm_list)]

        # timeshift
        if self.timeshift:
            LABEVENTS_table["charttime"] = adjust_time(
                LABEVENTS_table,
                "charttime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            LABEVENTS_table = LABEVENTS_table.dropna(subset=["charttime"])
            TIME = np.array([datetime.strptime(tt, "%Y-%m-%d %H:%M:%S") for tt in LABEVENTS_table["charttime"].values])
            LABEVENTS_table = LABEVENTS_table[TIME >= self.start_pivot_datetime]

        # save csv
        LABEVENTS_table = LABEVENTS_table.reset_index(drop=False)
        LABEVENTS_table = LABEVENTS_table.rename(columns={"index": "row_id"})
        LABEVENTS_table["row_id"] = range(len(LABEVENTS_table))
        LABEVENTS_table.to_csv(os.path.join(self.out_dir, "labevents.csv"), index=False)

        print(f"labevents processed (took {round(time.time() - start_time, 4)} secs)")

    def build_prescriptions_table(self):
        print("Processing prescriptions table")
        start_time = time.time()

        PRESCRIPTIONS_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="hosp/prescriptions.csv",
            columns=[
                "subject_id",
                "hadm_id",
                "starttime",
                "stoptime",
                "drug",
                "dose_val_rx",
                "dose_unit_rx",
                "route",
            ],
            lower=True,
        )

        PRESCRIPTIONS_table = PRESCRIPTIONS_table.dropna(subset=["starttime", "stoptime", "dose_val_rx", "dose_unit_rx", "route"])
        PRESCRIPTIONS_table["dose_val_rx"] = [int(str(v).replace(",", "")) if str(v).replace(",", "").isnumeric() else None for v in PRESCRIPTIONS_table["dose_val_rx"].values]
        PRESCRIPTIONS_table = PRESCRIPTIONS_table.dropna(subset=["dose_val_rx"])  # remove not int elements

        drug2unit_dict = {}
        for item, unit in zip(
            PRESCRIPTIONS_table["drug"].values,
            PRESCRIPTIONS_table["dose_unit_rx"].values,
        ):
            if item in drug2unit_dict:
                drug2unit_dict[item].append(unit)
            else:
                drug2unit_dict[item] = [unit]
        drug_name2unit_dict = {item: Counter(drug2unit_dict[item]).most_common(1)[0][0] for item in drug2unit_dict}  # pick only the most frequent unit of measure

        PRESCRIPTIONS_table = PRESCRIPTIONS_table[PRESCRIPTIONS_table["drug"].isin(drug2unit_dict)]
        PRESCRIPTIONS_table = PRESCRIPTIONS_table[PRESCRIPTIONS_table["dose_unit_rx"] == [drug_name2unit_dict[drug] for drug in PRESCRIPTIONS_table["drug"]]]

        # de-identification
        if self.deid:
            PRESCRIPTIONS_table = self.condition_value_shuffler(
                PRESCRIPTIONS_table,
                target_cols=["drug", "dose_val_rx", "dose_unit_rx", "route"],
            )

        PRESCRIPTIONS_table = PRESCRIPTIONS_table[PRESCRIPTIONS_table["hadm_id"].isin(self.hadm_list)]

        # timeshift
        if self.timeshift:
            PRESCRIPTIONS_table["starttime"] = adjust_time(
                PRESCRIPTIONS_table,
                "starttime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            PRESCRIPTIONS_table["stoptime"] = adjust_time(
                PRESCRIPTIONS_table,
                "stoptime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            PRESCRIPTIONS_table = PRESCRIPTIONS_table.dropna(subset=["starttime"])
            TIME = np.array([datetime.strptime(tt, "%Y-%m-%d %H:%M:%S") for tt in PRESCRIPTIONS_table["starttime"].values])
            PRESCRIPTIONS_table = PRESCRIPTIONS_table[TIME >= self.start_pivot_datetime]

        # save csv
        PRESCRIPTIONS_table = PRESCRIPTIONS_table.reset_index(drop=False)
        PRESCRIPTIONS_table = PRESCRIPTIONS_table.rename(columns={"index": "row_id"})
        PRESCRIPTIONS_table["row_id"] = range(len(PRESCRIPTIONS_table))
        PRESCRIPTIONS_table.to_csv(os.path.join(self.out_dir, "prescriptions.csv"), index=False)

        print(f"prescriptions processed (took {round(time.time() - start_time, 4)} secs)")

    def build_cost_table(self):
        print("Processing COST table")
        start_time = time.time()

        DIAGNOSES_ICD_table = read_csv(self.out_dir, "diagnoses_icd.csv").astype({"icd_code": str})
        LABEVENTS_table = read_csv(self.out_dir, "labevents.csv")
        PROCEDURES_ICD_table = read_csv(self.out_dir, "procedures_icd.csv").astype({"icd_code": str})
        PRESCRIPTIONS_table = read_csv(self.out_dir, "prescriptions.csv")

        cnt = 0
        data_filter = []
        mean_costs = self.rng.poisson(lam=10, size=4)

        cost_id = cnt + np.arange(len(DIAGNOSES_ICD_table))
        person_id = DIAGNOSES_ICD_table["subject_id"].values
        hospitaladmit_id = DIAGNOSES_ICD_table["hadm_id"].values
        cost_event_table_concept_id = DIAGNOSES_ICD_table["row_id"].values
        charge_time = DIAGNOSES_ICD_table["charttime"].values
        diagnosis_cost_dict = {item: round(self.rng.normal(loc=mean_costs[0], scale=1.0), 2) for item in sorted(DIAGNOSES_ICD_table["icd_code"].unique())}
        cost = [diagnosis_cost_dict[item] for item in DIAGNOSES_ICD_table["icd_code"].values]
        temp = pd.DataFrame(
            data={
                "row_id": cost_id,
                "subject_id": person_id,
                "hadm_id": hospitaladmit_id,
                "event_type": "diagnoses_icd",
                "event_id": cost_event_table_concept_id,
                "chargetime": charge_time,
                "cost": cost,
            }
        )
        cnt += len(DIAGNOSES_ICD_table)
        data_filter.append(temp)

        cost_id = cnt + np.arange(len(LABEVENTS_table))
        person_id = LABEVENTS_table["subject_id"].values
        hospitaladmit_id = LABEVENTS_table["hadm_id"].values
        cost_event_table_concept_id = LABEVENTS_table["row_id"].values
        charge_time = LABEVENTS_table["charttime"].values
        lab_cost_dict = {item: round(self.rng.normal(loc=mean_costs[1], scale=1.0), 2) for item in sorted(LABEVENTS_table["itemid"].unique())}
        cost = [lab_cost_dict[item] for item in LABEVENTS_table["itemid"].values]
        temp = pd.DataFrame(
            data={
                "row_id": cost_id,
                "subject_id": person_id,
                "hadm_id": hospitaladmit_id,
                "event_type": "labevents",
                "event_id": cost_event_table_concept_id,
                "chargetime": charge_time,
                "cost": cost,
            }
        )
        cnt += len(LABEVENTS_table)
        data_filter.append(temp)

        cost_id = cnt + np.arange(len(PROCEDURES_ICD_table))
        person_id = PROCEDURES_ICD_table["subject_id"].values
        hospitaladmit_id = PROCEDURES_ICD_table["hadm_id"].values
        cost_event_table_concept_id = PROCEDURES_ICD_table["row_id"].values
        charge_time = PROCEDURES_ICD_table["charttime"].values
        procedure_cost_dict = {item: round(self.rng.normal(loc=mean_costs[2], scale=1.0), 2) for item in sorted(PROCEDURES_ICD_table["icd_code"].unique())}
        cost = [procedure_cost_dict[item] for item in PROCEDURES_ICD_table["icd_code"].values]
        temp = pd.DataFrame(
            data={
                "row_id": cost_id,
                "subject_id": person_id,
                "hadm_id": hospitaladmit_id,
                "event_type": "procedures_icd",
                "event_id": cost_event_table_concept_id,
                "chargetime": charge_time,
                "cost": cost,
            }
        )
        cnt += len(PROCEDURES_ICD_table)
        data_filter.append(temp)

        cost_id = cnt + np.arange(len(PRESCRIPTIONS_table))
        person_id = PRESCRIPTIONS_table["subject_id"].values
        hospitaladmit_id = PRESCRIPTIONS_table["hadm_id"].values
        cost_event_table_concept_id = PRESCRIPTIONS_table["row_id"].values
        charge_time = PRESCRIPTIONS_table["starttime"].values
        prescription_cost_dict = {item: round(self.rng.normal(loc=mean_costs[3], scale=1.0), 2) for item in sorted(PRESCRIPTIONS_table["drug"].unique())}
        cost = [prescription_cost_dict[item] for item in PRESCRIPTIONS_table["drug"].values]
        temp = pd.DataFrame(
            data={
                "row_id": cost_id,
                "subject_id": person_id,
                "hadm_id": hospitaladmit_id,
                "event_type": "prescriptions",
                "event_id": cost_event_table_concept_id,
                "chargetime": charge_time,
                "cost": cost,
            }
        )
        cnt += len(PRESCRIPTIONS_table)
        data_filter.append(temp)

        COST_table = pd.concat(data_filter, ignore_index=True)
        COST_table.to_csv(os.path.join(self.out_dir, "cost.csv"), index=False)

        print(f"cost processed (took {round(time.time() - start_time, 4)} secs)")

    def build_chartevent_table(self):
        print("Processing chartevents table")
        start_time = time.time()

        CHARTEVENTS_table_dtype = {
            "subject_id": int,
            "hadm_id": int,
            "stay_id": int,
            "charttime": str,
            "itemid": int,
            "valuenum": float,
            "valueuom": str,
        }

        CHARTEVENTS_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="icu/chartevents.csv",
            dtype={
                "subject_id": int,
                "hadm_id": int,
                "stay_id": int,
                "charttime": str,
                "itemid": str,  # int,
                "valuenum": "float64",
                "valueuom": str,
            },
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "charttime",
                "itemid",
                "valuenum",
                "valueuom",
            ],
            lower=True,
            filter_dict={
                "itemid": self.chartevent2itemid.values(),
                "subject_id": self.patient_list,
            },
            memory_efficient=True,
        )

        CHARTEVENTS_table = CHARTEVENTS_table.dropna()
        CHARTEVENTS_table = CHARTEVENTS_table.astype(CHARTEVENTS_table_dtype)

        if self.timeshift:  # change the order due to the large number of rows in CHARTEVENTS_table
            CHARTEVENTS_table["charttime"] = adjust_time(
                CHARTEVENTS_table,
                "charttime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            CHARTEVENTS_table = CHARTEVENTS_table.dropna(subset=["charttime"])

        # de-identification
        if self.deid:
            CHARTEVENTS_table = self.condition_value_shuffler(CHARTEVENTS_table, target_cols=["itemid", "valuenum", "valueuom"])

        CHARTEVENTS_table = CHARTEVENTS_table[CHARTEVENTS_table["hadm_id"].isin(self.hadm_list)]

        # timeshift
        if self.timeshift:
            TIME = np.array([datetime.strptime(tt, "%Y-%m-%d %H:%M:%S") for tt in CHARTEVENTS_table["charttime"].values])
            CHARTEVENTS_table = CHARTEVENTS_table[TIME >= self.start_pivot_datetime]

        # save csv
        CHARTEVENTS_table = CHARTEVENTS_table.reset_index(drop=False)
        CHARTEVENTS_table = CHARTEVENTS_table.rename(columns={"index": "row_id"})
        CHARTEVENTS_table["row_id"] = range(len(CHARTEVENTS_table))
        CHARTEVENTS_table.to_csv(os.path.join(self.out_dir, "chartevents.csv"), index=False)

        print(f"chartevents processed (took {round(time.time() - start_time, 4)} secs)")

    def build_inputevent_table(self):
        print("Processing inputevents table")
        start_time = time.time()

        INPUTEVENTS_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="icu/inputevents.csv",
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "starttime",
                "itemid",
                "amount",
                "amountuom",
            ],
            lower=True,
        )

        INPUTEVENTS_table = INPUTEVENTS_table.dropna(subset=["hadm_id", "stay_id", "amount", "amountuom"])
        INPUTEVENTS_table = INPUTEVENTS_table[INPUTEVENTS_table["amountuom"] == "ml"]  # Input volume is mostly (~50%) in ml
        del INPUTEVENTS_table["amountuom"]

        INPUTEVENTS_table = INPUTEVENTS_table[INPUTEVENTS_table["itemid"].isin(self.D_ITEMS_dict)]

        # de-identification
        if self.deid:
            INPUTEVENTS_table = self.condition_value_shuffler(table=INPUTEVENTS_table, target_cols=["itemid", "amount"])

        INPUTEVENTS_table = INPUTEVENTS_table[INPUTEVENTS_table["hadm_id"].isin(self.hadm_list)]

        # timeshift
        if self.timeshift:
            INPUTEVENTS_table["starttime"] = adjust_time(
                INPUTEVENTS_table,
                "starttime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            INPUTEVENTS_table = INPUTEVENTS_table.dropna(subset=["starttime"])
            TIME = np.array([datetime.strptime(tt, "%Y-%m-%d %H:%M:%S") for tt in INPUTEVENTS_table["starttime"].values])
            INPUTEVENTS_table = INPUTEVENTS_table[TIME >= self.start_pivot_datetime]

        # save csv
        INPUTEVENTS_table = INPUTEVENTS_table.reset_index(drop=False)
        INPUTEVENTS_table = INPUTEVENTS_table.rename(columns={"index": "row_id"})
        INPUTEVENTS_table["row_id"] = range(len(INPUTEVENTS_table))
        INPUTEVENTS_table.to_csv(os.path.join(self.out_dir, "inputevents.csv"), index=False)

        print(f"inputevents processed (took {round(time.time() - start_time, 4)} secs)")

    def build_outputevent_table(self):
        print("Processing outputevents table")
        start_time = time.time()

        # read csv
        OUTPUTEVENTS_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="icu/outputevents.csv",
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "charttime",
                "itemid",
                "value",
                "valueuom",
            ],
            lower=True,
        )

        # preprocess
        OUTPUTEVENTS_table = OUTPUTEVENTS_table.dropna(subset=["hadm_id", "stay_id", "value", "valueuom"])
        OUTPUTEVENTS_table = OUTPUTEVENTS_table[OUTPUTEVENTS_table["valueuom"] == "ml"]  # Output volume is always in ml
        del OUTPUTEVENTS_table["valueuom"]

        OUTPUTEVENTS_table = OUTPUTEVENTS_table[OUTPUTEVENTS_table["itemid"].isin(self.D_ITEMS_dict)]

        # de-identification
        if self.deid:
            OUTPUTEVENTS_table = self.condition_value_shuffler(
                table=OUTPUTEVENTS_table,
                target_cols=["itemid", "value"],
            )

        OUTPUTEVENTS_table = OUTPUTEVENTS_table[OUTPUTEVENTS_table["hadm_id"].isin(self.hadm_list)]

        # timeshift
        if self.timeshift:
            OUTPUTEVENTS_table["charttime"] = adjust_time(
                OUTPUTEVENTS_table,
                "charttime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            OUTPUTEVENTS_table = OUTPUTEVENTS_table.dropna(subset=["charttime"])
            TIME = np.array([datetime.strptime(tt, "%Y-%m-%d %H:%M:%S") for tt in OUTPUTEVENTS_table["charttime"].values])
            OUTPUTEVENTS_table = OUTPUTEVENTS_table[TIME >= self.start_pivot_datetime]

        # save csv
        OUTPUTEVENTS_table = OUTPUTEVENTS_table.reset_index(drop=False)
        OUTPUTEVENTS_table = OUTPUTEVENTS_table.rename(columns={"index": "row_id"})
        OUTPUTEVENTS_table["row_id"] = range(len(OUTPUTEVENTS_table))
        OUTPUTEVENTS_table.to_csv(os.path.join(self.out_dir, "outputevents.csv"), index=False)

        print(f"outputevents processed (took {round(time.time() - start_time, 4)} secs)")

    def build_microbiology_table(self):
        print("Processing microbiologyevents table")
        start_time = time.time()

        # read csv
        MICROBIOLOGYEVENTS_table = read_csv(
            data_dir=self.mimic_iv_dir,
            filename="hosp/microbiologyevents.csv",
            columns=[
                "subject_id",
                "hadm_id",
                "chartdate",
                "charttime",
                "spec_type_desc",
                "test_name",
                "org_name",
            ],
            lower=True,
        )

        # If charttime is null, use chartdate as charttime
        MICROBIOLOGYEVENTS_table["charttime"] = MICROBIOLOGYEVENTS_table["charttime"].fillna(MICROBIOLOGYEVENTS_table["chartdate"])
        del MICROBIOLOGYEVENTS_table["chartdate"]

        # de-identification
        if self.deid:
            MICROBIOLOGYEVENTS_table = self.condition_value_shuffler(
                table=MICROBIOLOGYEVENTS_table,
                target_cols=["spec_type_desc", "test_name", "org_name"],
            )

        MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table[MICROBIOLOGYEVENTS_table["hadm_id"].isin(self.hadm_list)]

        # timeshift
        if self.timeshift:
            MICROBIOLOGYEVENTS_table["charttime"] = adjust_time(
                MICROBIOLOGYEVENTS_table,
                "charttime",
                current_time=self.current_time,
                offset_dict=self.subjectid2admittime_dict,
                patient_col="subject_id",
            )
            MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table.dropna(subset=["charttime"])
            TIME = np.array([datetime.strptime(tt, "%Y-%m-%d %H:%M:%S") for tt in MICROBIOLOGYEVENTS_table["charttime"].values])
            MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table[TIME >= self.start_pivot_datetime]

        # save csv
        MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table.reset_index(drop=False)
        MICROBIOLOGYEVENTS_table = MICROBIOLOGYEVENTS_table.rename(columns={"index": "row_id"})
        MICROBIOLOGYEVENTS_table["row_id"] = range(len(MICROBIOLOGYEVENTS_table))
        MICROBIOLOGYEVENTS_table.to_csv(os.path.join(self.out_dir, "microbiologyevents.csv"), index=False)

        print(f"microbiologyevents processed (took {round(time.time() - start_time, 4)} secs)")

    def generate_db(self):
        rows = read_csv(self.out_dir, "tb_cxr_plus.csv")
        rows.to_sql("TB_CXR_PLUS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "tb_cxr.csv")
        rows.to_sql("TB_CXR", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "patients.csv")
        rows.to_sql("PATIENTS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "admissions.csv")
        rows.to_sql("ADMISSIONS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "d_icd_diagnoses.csv").astype({"icd_code": str})
        rows.to_sql("D_ICD_DIAGNOSES", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "d_icd_procedures.csv").astype({"icd_code": str})
        rows.to_sql("D_ICD_PROCEDURES", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "d_items.csv")
        rows.to_sql("D_ITEMS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "d_labitems.csv")
        rows.to_sql("D_LABITEMS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "diagnoses_icd.csv").astype({"icd_code": str})
        rows.to_sql("DIAGNOSES_ICD", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "procedures_icd.csv").astype({"icd_code": str})
        rows.to_sql("PROCEDURES_ICD", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "labevents.csv")
        rows.to_sql("LABEVENTS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "prescriptions.csv")
        rows.to_sql("PRESCRIPTIONS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "cost.csv")
        rows.to_sql("COST", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "chartevents.csv")
        rows.to_sql("CHARTEVENTS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "inputevents.csv")
        rows.to_sql("INPUTEVENTS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "outputevents.csv")
        rows.to_sql("OUTPUTEVENTS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "microbiologyevents.csv")
        rows.to_sql("MICROBIOLOGYEVENTS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "icustays.csv")
        rows.to_sql("ICUSTAYS", self.conn, if_exists="replace", index=False)

        rows = read_csv(self.out_dir, "transfers.csv")
        rows.to_sql("TRANSFERS", self.conn, if_exists="replace", index=False)

        query = "SELECT * FROM sqlite_master WHERE type='table'"
        print(pd.read_sql_query(query, self.conn)["name"])  # 17 tables

        self._check_assertion_db_and_csv()

    def _check_assertion_db_and_csv(self, table_names=None):
        if table_names is None:
            table_names = pd.read_sql_query("SELECT * FROM sqlite_master WHERE type='table'", self.conn)["name"]

        for table_name in table_names:
            table_name = table_name.lower()
            csv = read_csv(self.out_dir, f"{table_name}.csv")
            db = pd.read_sql_query(f"select * from {table_name}", self.conn)

            # np.nan => None object (for comparison)
            csv = csv.where(pd.notnull(csv), None)
            db = db.where(pd.notnull(db), None)
            assert set(csv.columns) == set(db.columns.str.lower())
            csv = csv[db.columns.str.lower()]

            assert csv.shape == db.shape
            try:
                if (csv.values == db.values).all():
                    pass
                elif table_name == "tb_cxr":
                    assert (db.isna().sum().values == csv.isna().sum().values).all()
                elif table_name == "tb_cxr_plus":
                    assert (db.isna().sum().values == csv.isna().sum().values).all()
                elif table_name == "prescriptions":
                    pass  # data types of dose_val_rx is not the same
                else:
                    raise AssertionError(f"csv and db are not the same: {table_name}")
            except:
                breakpoint()
