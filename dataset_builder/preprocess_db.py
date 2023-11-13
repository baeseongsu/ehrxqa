import os
import argparse


def config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", default="test", type=str, choices=["train", "valid", "test"], help="split")

    # parser.add_argument("--data_dir", required=True, type=str, help="csv file dir")
    parser.add_argument("--mimic_iv_dir", required=True, type=str, help="mimic iv directory")
    parser.add_argument("--mimic_cxr_jpg_dir", required=True, type=str, help="mimic cxr jpg directory")
    parser.add_argument("--chest_imagenome_dir", required=True, type=str, help="chest imagenome directory")

    parser.add_argument("--db_name", required=True, type=str, choices=["mimic_iii", "eicu", "mimic_iv", "mimic_iv_cxr"], help="choose between mimic_iii, eicu, mimic_iv")
    parser.add_argument("--out_dir", default="../dataset/ehrsql", type=str, help="output file directory")

    parser.add_argument("--sample_icu_patient_only", action="store_true", help="sample only patients who went to the ICU")
    parser.add_argument("--num_patient", default=1000, type=int, help="number of patients")

    parser.add_argument("--deid", action="store_true", help="do deidentification")

    parser.add_argument("--timeshift", action="store_true", help="do time shift")
    parser.add_argument("--start_year", default=None, type=int, help="start sampling year")
    parser.add_argument(
        "--time_span", default=None, type=int, help="time span starting from start_year"
    )  # mimic_iii: 2001 - 2012 => 2100 - 2105 / eicu: 2014 - 2015 => 2100 - 2105  / mimic_iv: 2008 - 2019 => 2100 - 2105
    parser.add_argument("--cur_patient_ratio", default=0.0, type=float, help="ratio of inpatient")
    parser.add_argument("--current_time", default=None, type=str, help="any record past current_time is removed")

    args = parser.parse_args()

    return args


def main(args):
    if args.timeshift:
        assert args.start_year is not None, 'To do a time shift, "start_year" must be specified'
        assert args.time_span is not None, 'To do a time shift, "time_span" must be specified'
        assert args.current_time is not None, 'To do a time shift, "current_time" must be specified'

    if args.db_name == "mimic_iv_cxr":
        from preprocess_utils.preprocess_db_mimic_iv_cxr import Build_MIMIC_IV_CXR

        print(vars(args))

        mimic_writer = Build_MIMIC_IV_CXR(
            split=args.split,
            mimic_iv_dir=args.mimic_iv_dir,
            mimic_cxr_jpg_dir=args.mimic_cxr_jpg_dir,
            chest_imagenome_dir=args.chest_imagenome_dir,
            out_dir=args.out_dir,
            db_name=args.db_name,
            num_patient=args.num_patient,
            sample_icu_patient_only=args.sample_icu_patient_only,
            deid=args.deid,
            timeshift=args.timeshift,
            start_year=args.start_year,
            time_span=args.time_span,
            cur_patient_ratio=args.cur_patient_ratio,
            current_time=args.current_time,
        )

        mimic_writer.build_tb_cxr_table(flag_for_plus=False)  # tb_cxr
        mimic_writer.build_tb_cxr_table(flag_for_plus=True)  # tb_cxr_plus
        mimic_writer.build_admission_table()  # patients, admissions, icustays, transfers
        mimic_writer.build_dictionary_table()  # d_icu_diagnoses, d_icu_procedures, d_items, d_labitems
        mimic_writer.build_diagnosis_table()  # diagnoses_icd
        mimic_writer.build_procedure_table()  # procedures_icd
        mimic_writer.build_labevent_table()  # labevents
        mimic_writer.build_prescriptions_table()  # prescriptions
        mimic_writer.build_cost_table()  # cost
        mimic_writer.build_chartevent_table()  # chartevents
        mimic_writer.build_inputevent_table()  # inputevents
        mimic_writer.build_outputevent_table()  # outputevents
        mimic_writer.build_microbiology_table()  # microbiologyevents

        mimic_writer.generate_db()

    elif args.db_name == "mimic_iv":
        raise NotImplementedError("In this repository, we only support MIMIC-IV-CXR dataset")

    elif args.db_name == "mimic_iii":
        raise NotImplementedError("In this repository, we only support MIMIC-IV-CXR dataset")

    else:
        raise ValueError("Invalid db_name")


if __name__ == "__main__":
    args = config()
    main(args)
    print("Done!\n")
