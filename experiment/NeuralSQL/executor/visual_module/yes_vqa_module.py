from .vqa_module import VQAModule


class YesVQAModule(VQAModule):
    def __init__(self):
        pass

    def load_model(self):
        # No need to load any model
        pass

    def preprocess_input(self, images, questions):
        # No preprocessing required
        return images, questions

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
        # Preprocess the input images and questions (no-op)
        preprocessed_images, preprocessed_questions = self.preprocess_input(images, questions)

        # Simulate obtaining a raw_output list that is always "yes" for each question
        raw_output = ["yes"] * len(preprocessed_questions)  # Generate a "yes" for each question

        # Postprocess the output to obtain the handled answers
        handled_answers = self.postprocess_output(raw_output)

        return handled_answers
