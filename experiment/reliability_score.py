from typing import List, Dict, Union
import numpy as np


class ReliabilityScore:
    """
    Calculate reliability scores for semantic parsing models.
    Reference: Lee et al. (2024). TrustSQL: A Reliability Benchmark for Text-to-SQL Models with Diverse Unanswerable Questions.
    """

    def __init__(self, real_result: Dict[str, str], pred_result: Dict[str, str], abstain_key: str = "null"):
        self.real_result = real_result
        self.pred_result = pred_result
        self.abstain_key = abstain_key

    def compute(self, penalties: Union[List[Union[int, str]], None] = None) -> Dict[str, float]:
        """
        Compute reliability scores for different penalty values.
        """
        if penalties is None:
            penalties = [0, 5, 10, "N"]
        else:
            penalties = [int(penalty) if isinstance(penalty, int) else penalty for penalty in penalties]

        reliability_score_dict: Dict[str, int] = {}
        for key in self.real_result:
            ans_real = self.real_result[key]
            ans_pred = self.pred_result[key]
            exec_acc = ans_real == ans_pred

            # x in ANS; g(x)=1; Acc(x)=1
            if ans_real != self.abstain_key and exec_acc == True:
                score = 1
            # x in ANS; g(x)=0; Acc(x)={0,1}
            elif ans_real != self.abstain_key and ans_pred == self.abstain_key:
                score = 0
            # x in ANS; g(x)=1; Acc(x)=0
            elif ans_real != self.abstain_key and exec_acc == False:
                score = -1
            # x in UnANS; g(x)=1
            elif ans_real == self.abstain_key and ans_pred != self.abstain_key:
                score = -1
            # x in UnANS; g(x)=0
            elif ans_real == self.abstain_key and ans_pred == self.abstain_key:
                score = 1
            else:
                raise NotImplementedError()

            reliability_score_dict[key] = score

        reliability_results = {}
        for penalty in penalties:
            penalty_score = len(reliability_score_dict) if penalty == "N" else int(penalty)
            reliability_result = 100 * np.mean([score * penalty_score if score == -1 else score for score in reliability_score_dict.values()])
            reliability_results[str(penalty)] = reliability_result

        return reliability_results


if __name__ == "__main__":
    real_result = {
        "1": "a",
        "2": "b",
        "3": "c",
        "4": "d",
        "5": "null",
    }
    pred_result = {
        "1": "a",
        "2": "b",
        "3": "a",
        "4": "a",
        "5": "null",
    }
    reliability = ReliabilityScore(real_result=real_result, pred_result=pred_result)
    reliability_scores = reliability.compute()
    print(reliability_scores)
