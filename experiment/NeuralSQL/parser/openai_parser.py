import os
import logging
from typing import List, Dict, Union
from parser.parser_utils import BM25Okapi
import time
from openai import OpenAI
import tiktoken

# Set up logging at the module level
logging.basicConfig(level=logging.INFO)


DEFAULT_SYSTEM_MESSAGE = (
    "/* Generate NeuralSQL (i.e., extended SQL with the following conditions) given the question to answer the question correctly.\n"
    "If the question can only be answered by examining a chest x-ray image and requires a VQA model, use the new syntax func_vqa() to create a query.\n"
    "When the VQA sentence in func_vqa() syntax contains logical operations such as union, difference, intersection, disjunction, or conjunction, decompose the VQA statement into minimal semantic units and use the SQL syntax to generate NeuralSQL.\n"
    "For example, decompose the original sentence 'are there any technicalassessment or tubesandlines?' into 'are there any technicalassessment?' and 'are there any tubesandlines?' by separating the logical disjunction (or) and creating two separate questions.*/"
)


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class OpenAIParser:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_context_length: int = 4097,
        max_tokens: int = 768,
        temperature: float = 0.0,
        top_p: float = 1.0,
        n: int = 1,
        stop: List[str] = ["\n\n"],
    ):
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.max_context_length = max_context_length
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stop = stop
        self.client = OpenAI(api_key=self.api_key)

    def _build_few_shot_prompt_from_dataset(
        self,
        target_data: Dict,
        examples_dataset: List[Dict],
        num_examples: int,
        lower_case_query: bool = True,
        retrieval_method: str = "retrieval-bm25",
        question_tag: str = "Q",
        query_tag: str = "NeuralSQL",
        verbose: bool = False,
    ) -> str:
        """
        Constructs a few-shot learning prompt from a dataset using the specified retrieval method.
        Supports dynamic tagging for questions and responses.
        """
        prompts = []
        if retrieval_method == "retrieval-bm25":
            questions_corpus = [example["question"] for example in examples_dataset]
            questions_tokenized = [question.split() for question in questions_corpus]
            bm25 = BM25Okapi(corpus=questions_tokenized)

            target_question = target_data["question"]
            target_tokens = target_question.split()
            retrieved_questions = bm25.get_top_n(target_tokens, questions_corpus, n=num_examples)

            for question in retrieved_questions:
                example_index = questions_corpus.index(question)
                corresponding_query = examples_dataset[example_index]["query"]
                corresponding_query = corresponding_query.lower() if lower_case_query else corresponding_query
                prompt = f"{question_tag}: {question}\n{query_tag}: {corresponding_query}"
                prompts.append(prompt)

            while num_tokens_from_string("\n\n".join(prompts)) > self.max_context_length - (self.max_tokens + 500):
                print("Prompt length is too long, reducing the number of examples.")
                prompts = prompts[:-1]
        else:
            raise ValueError(f"Unsupported retrieval method: {retrieval_method}")

        return "\n\n".join(prompts)

    def _build_few_shot_prompt_from_file(
        self,
        prompt_file_path: str,
        num_examples: int,
        verbose: bool = False,
    ) -> str:
        """
        Constructs a few-shot learning prompt from a file, selecting a specified number of examples.
        Each example should be separated by double newlines in the file.
        """
        try:
            with open(prompt_file_path, "r") as file:
                content = file.read()
        except FileNotFoundError as e:
            logging.error("File not found: %s", e)
            raise FileNotFoundError(f"The specified file was not found: {prompt_file_path}")

        prompts = [prompt.strip() for prompt in content.strip().split("\n\n") if prompt.strip()]
        selected_prompts = prompts[:num_examples]

        while num_tokens_from_string("\n\n".join(prompts)) > self.max_context_length - (self.max_tokens + 500):
            print("Prompt length is too long, reducing the number of examples.")
            selected_prompts = selected_prompts[:-1]

        return "\n\n".join(selected_prompts)

    def _build_target_prompt(
        self,
        target_data: Dict,
        question_tag: str = "Q",
        query_tag: str = "NeuralSQL",
        verbose: bool = False,
    ) -> str:
        """
        Constructs a target prompt for generating a query based on a provided question, referencing earlier examples.
        """
        target_question = target_data["question"]
        return f"-- Refer to the examples above and parse the question into NeuralSQL.\n{question_tag}: {target_question}\n{query_tag}: "

    def build_few_shot_prompt(
        self,
        target_data: Dict,
        examples_dataset: List[Dict],
        num_examples: int,
        prompt_file_path: str = None,
        lower_case_query: bool = True,
        retrieval_method: str = "retrieval-bm25",
        question_tag: str = "Q",
        query_tag: str = "NeuralSQL",
        verbose: bool = False,
    ) -> str:
        if prompt_file_path:
            few_shot_prompt = self._build_few_shot_prompt_from_file(
                prompt_file_path=prompt_file_path,
                num_examples=num_examples,
                verbose=verbose,
            )
        else:
            few_shot_prompt = self._build_few_shot_prompt_from_dataset(
                target_data=target_data,
                examples_dataset=examples_dataset,
                num_examples=num_examples,
                lower_case_query=lower_case_query,
                retrieval_method=retrieval_method,
                question_tag=question_tag,
                query_tag=query_tag,
                verbose=verbose,
            )

        target_prompt = self._build_target_prompt(target_data, question_tag, query_tag)
        return f"{few_shot_prompt}\n\n{target_prompt}"

    def _generate(self, prompt: str, default_system_message: str = DEFAULT_SYSTEM_MESSAGE):
        retry_count = 0
        start_time = time.time()
        while retry_count < 10:  # Retry logic inconsistency fixed
            try:
                messages = [
                    {"role": "system", "content": default_system_message},
                    {"role": "user", "content": prompt},
                ]
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n=self.n,
                    stop=self.stop,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                logging.info("OpenAI ChatGPT API inference time: %s seconds", time.time() - start_time)
                return result
            except Exception as e:
                logging.error("Error during generation: %s. Retry %d", e, retry_count)
                time.sleep(10)  # Exponential backoff or more sophisticated error handling can be considered
                retry_count += 1

        logging.error("Failed to generate response after %s retries.", retry_count)
        return None

    def parse(
        self,
        target_dataset: List[Dict],
        examples_dataset: List[Dict],
        num_examples: int,
        prompt_file_path: str = None,
        return_content_only: bool = True,
        verbose: bool = False,
    ):
        responses = []
        for target_data in target_dataset:
            few_shot_prompt = self.build_few_shot_prompt(
                target_data=target_data,
                examples_dataset=examples_dataset,
                num_examples=num_examples,
                prompt_file_path=prompt_file_path,
                verbose=verbose,
            )
            response = self._generate(few_shot_prompt)
            if return_content_only:
                content = response.choices[0].message.content if response is not None else ""
                responses.append(content)
            else:
                responses.append(response)

            if verbose:
                logging.info("Few-shot prompt:\n%s", few_shot_prompt)
                logging.info("Response:\n%s", response)

        return responses
