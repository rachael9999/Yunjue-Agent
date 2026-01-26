import argparse
import textwrap
import dataclasses
import collections
import json
import logging
from pathlib import Path
from typing import Any, Optional, List, Tuple
import concurrent.futures  # Thread pool for parallel grading
from tqdm import tqdm  # CLI-friendly progress bar (no notebook deps)
import math
import numpy
import random
import pandas as pd
import time
from evaluate_predictions import build_eval_model
DEEPSEARCH_QA_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")

GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


# @title Grader Helper Class (ItemRating, ProjectRating) {"display-mode":"form"}
@dataclasses.dataclass
class ItemRatingBase:
  """Abstract item rating class. Should be overridden by each eval type."""
  original_index: int | None = dataclasses.field(
      default=None, kw_only=True, compare=False
  )
  def to_dict(self) -> dict[str, Any]:
    d = dataclasses.asdict(self)
    # del d['original_index'] # Keep original_index for offline analysis
    return d

@dataclasses.dataclass
class ItemRating(ItemRatingBase):
  """Item Rating."""
  example_id: str
  query: str
  response: str
  category_type: str | None = None
  expected_correct_answer: str | None = None

  # This field stores the explanation for the answer correctness.
  answer_correctness_explanation: str | None = None
  # Change this to a list of scores if we want to
  expected_correct_answer_list: list[str] | None = None
  # This field stores excessive answers in the response, which are not in the
  # correct answer list.
  response_wrong_answers_list: list[str] | None = None
  grader_ratings_list: list[bool] | None = None

  empty_model_response: bool = False
  empty_auto_rater_response: bool = False
  invalid_auto_rater_response: bool = False
  # The raw response string from the autoeval model.
  rating_response: str = ''
  # The full prompt sent to the autorater LLM
  rating_prompt: str = ''
  error_message: Optional[str] = None

@dataclasses.dataclass
class ProjectRatingBase:
  """Abstract project rating class. Should be overridden by each eval type."""
  def to_dict(self) -> dict[str, Any]:
    return dataclasses.asdict(self)

@dataclasses.dataclass
class ProjectRating(ProjectRatingBase):
  """Project Rating."""

  num_total_ratings: int = 0
  num_empty_model_response: int = 0
  num_invalid_auto_rater_response: int = 0
  num_empty_auto_rater_response: int = 0
  num_valid_ratings: int = 0
  num_answer_correctness_evaluated: int = 0

  pct_w_ci_all_answers_correct: str = ''
  pct_w_ci_fully_incorrect_items: str = ''
  pct_w_ci_correct_with_excessive_answers: str = ''

  pct_empty_model_response: float = 0.0
  pct_invalid_auto_rater_response: float = 0.0
  pct_empty_auto_rater_response: float = 0.0


  precision: str = ''
  recall: str = ''
  f1_score: str = ''

_DefaultDict = collections.defaultdict

def _parse_json_response(ori_json_response: str) -> Any:
  """Try to parse a json object from the input string.
  Handles cases where JSON is wrapped in '```json' and '```' markers.
  """
  try:
    json_str = ori_json_response.strip()
    # Find the start of the JSON block, looking for '```json'
    start_marker = '```json'
    start_idx = json_str.find(start_marker)

    if start_idx != -1:
      # Extract content after '```json'
      json_str = json_str[start_idx + len(start_marker):].strip()
      # Find the end of the JSON block, looking for '```'
      end_marker = '```'
      end_idx = json_str.rfind(end_marker)
      if end_idx != -1:
        json_str = json_str[:end_idx].strip()

    # Attempt to load the (potentially extracted) JSON string
    return json.loads(json_str)
  except json.JSONDecodeError as e:
    logging.info('json.JSONDecodeError: %s for response: %s', e, ori_json_response)
    return None

def _get_answer_correctness_details(
    json_response: Any,
) -> dict[str, bool] | None:
  """Extract the answer correctness details from the json response."""
  try:
    details = json_response['Answer Correctness']['Correctness Details']
    if isinstance(details, dict):
      all_keys_are_strings = all(isinstance(key, str) for key in details.keys())
      all_values_are_booleans = all(
          isinstance(value, bool) for value in details.values()
      )
      if all_keys_are_strings and all_values_are_booleans:
        return details
    logging.warning(
        'Invalid format for Answer Correctness Details: %s', details
    )
    return None # Return None if format is invalid
  except KeyError as e:
    logging.info(
        'KeyError: %s for path "Answer Correctness.Correctness Details" in'
        ' json_response: %s',
        e,
        json_response,
    )
    return None
  except TypeError: # Handle cases where json_response itself is not a dict or 'Answer Correctness' is not a dict
    logging.warning(
        'TypeError while accessing Correctness Details. JSON response: %s', json_response
    )
    return None


def _get_excessive_answers(
    json_response: Any,
) -> list[str] | None:
  """Extract the excessive answers from the json response.

  Args:
    json_response: The json response from the grader.

  Returns:
    A list of strings if excessive answers are valid.
    Returns an empty list if the value of `Excessive Answers` are also an empty
    list or `Excessive Answers' keys are missing. This will be also valid.
    Returns None if 'Excessive Answers' are present but malformed.
  """
  try:
    excessive_answers = json_response['Answer Correctness']['Excessive Answers']
    if isinstance(excessive_answers, list):
      # Check if all list items are strings
      all_items_are_strings = all(
          isinstance(item, str) for item in excessive_answers
      )
      if all_items_are_strings:
        return excessive_answers
    logging.warning(
        'Invalid format for Excessive Answers: %s', excessive_answers
    )
    return None  # Return None for the format error. Will be handled by caller.
  except KeyError as e:
    logging.info(
        'KeyError: %s for path "Answer Correctness.Excessive Answers" in'
        ' json_response: %s',
        e,
        json_response,
    )
    return []  # Return an empty list.

#@title Grader Utils

def _reduce_llm_response_to_item_rating(
    item_rating: ItemRating,
    row_data: pd.Series,
    grader_llm_response_text: str,
    grader_llm_prompt_text: str
) -> ItemRating:
    """Parses the AutoRater LLM's string response and populates an ItemRating object."""

    item_rating.rating_prompt = grader_llm_prompt_text
    item_rating.rating_response = grader_llm_response_text

    if not item_rating.response:
        item_rating.empty_model_response = True
        item_rating.error_message = "AI response was empty."
        return item_rating

    if not grader_llm_response_text:
        item_rating.empty_auto_rater_response = True
        item_rating.error_message = "Auto-rater response was empty."
        return item_rating

    parsed_json_response = _parse_json_response(grader_llm_response_text)
    if not parsed_json_response:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid JSON response from auto-rater."
        return item_rating

    try:
        answer_correctness_node = parsed_json_response.get('Answer Correctness')
        if not isinstance(answer_correctness_node, dict):
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Missing or malformed 'Answer Correctness' node."
            return item_rating

        # Extract explanation
        grader_explanation = answer_correctness_node.get('Explanation')
        if not isinstance(grader_explanation, str):
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Missing or malformed 'Explanation' in Answer Correctness."
            return item_rating
        item_rating.answer_correctness_explanation = grader_explanation

        # Extract correctness details
        details = _get_answer_correctness_details(parsed_json_response)
        if details is None:
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Invalid 'Correctness Details' in Answer Correctness."
            return item_rating
        item_rating.expected_correct_answer_list = list(details.keys())
        item_rating.grader_ratings_list = list(details.values())

        # Extract excessive answers
        excessive_answers = _get_excessive_answers(parsed_json_response)
        if excessive_answers is None: # None indicates malformed, [] indicates no excessive answers (which is valid)
            item_rating.invalid_auto_rater_response = True
            item_rating.error_message = "Invalid 'Excessive Answers' in Answer Correctness."
            return item_rating
        if excessive_answers: # Only assign if not empty
            item_rating.response_wrong_answers_list = excessive_answers

    except (KeyError, TypeError, ValueError) as e:
        logging.exception('Error processing parsed JSON: %s', e)
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = f"Error during JSON processing: {e}"
        return item_rating

    return item_rating


def _get_grader_model_input_for_row(
    row: pd.Series
) -> str:
    """Constructs the full prompt for the grader LLM for a given data row."""
    # Use direct column access as columns are expected to be present after merge
    prompt = str(row['problem']).strip()
    response = str(row['response']).strip()
    prompt_type = str(row['answer_type']).strip()
    answer = str(row['answer']).strip()

    template = DEEPSEARCH_QA_PROMPT
    rating_output_example = GRADER_RATING_OUTPUT_EXAMPLE
    template += rating_output_example.format(
            prompt=prompt,
            prompt_type=prompt_type,
            answer=answer,
            response=response,
        )
    return template

def process_single_row_for_rating(
    index_row_tuple: Tuple[int, pd.Series]
) -> ItemRating:
    """Takes a tuple of (original_index, row_data), generates prompt, calls LLM, and returns ItemRating."""
    original_idx, row = index_row_tuple

    # Initialize a base ItemRating object with known information
    item_rating = ItemRating(
        original_index=original_idx,
        example_id=str(row.get('example_id', '')).strip(),
        query=str(row.get('problem', '')).strip(),
        response=str(row.get('response', '')).strip(),
        category_type=str(row.get('problem_category', '')).strip(),
        expected_correct_answer=str(row.get('answer', '')).strip(),
    )

    llm, eval_conf = build_eval_model(CONFIG_PATH)
    try:
        rating_prompt_text = _get_grader_model_input_for_row(row)

        grader_llm_response_str = None
        for i in range(3):
            try:
                grader_llm_response_str = llm.invoke(
                    messages=[
                        {"role": "user", "content": rating_prompt_text}
                    ]
                ).content
                break
            except Exception as e:
                logging.error(f"LLM call failed (attempt {i+1}/{3}) for idx {original_idx}: {e}")
                if i == 3 - 1:
                  item_rating.error_message = f"LLM call failed after {3} attempts: {e}"
                  return item_rating
                time.sleep(1 + (2**(i + random.random())))

        if grader_llm_response_str is None:
            item_rating.error_message = "LLM response was None after retries."
            return item_rating

        return _reduce_llm_response_to_item_rating(
            item_rating, # Pass the existing ItemRating object
            row,
            grader_llm_response_str,
            rating_prompt_text
        )

    except Exception as e:
        logging.error(f"Error processing row index {original_idx} ('{item_rating.query[:50]}...'): {e}", exc_info=True)
        item_rating.error_message = str(e)
        return item_rating

def _calculate_ci_str(count: int, total: int, z: float = 1.96) -> str:
    if total == 0:
      return f'N/A ({count}/{total})'
    if count < 0:  # Should not happen
      logging.warning('CI calculation: count %d is less than 0.', count)
      count = 0
    if count > total:  # Should not happen
      logging.warning(
          'CI calculation: count %d is greater than total %d.', count, total
      )
      count = total

    p = count / total
    p_percent = p * 100.0

    try:
      variance = p * (1.0 - p)
      # variance should not be negative if p is between 0 and 1
      margin_of_error = z * math.sqrt(variance / total)
      moe_percent = margin_of_error * 100.0
      result_str = (
          f'{round(p_percent, 2):.2f} ±'
          f' {round(moe_percent, 2):.2f} ({count}/{total})'
      )
      if total <= 5:  # Normal approx. is poor for very small n
        result_str += ' (CI not robust for n<=5)'
      return result_str
    except (
        ValueError,  # math domain error for sqrt if variance became < 0
        # due to float issues
        ZeroDivisionError,
    ):
      return 'N/A'

def _calculate_metric(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
  """Calculates precision, recall, and F1."""
  precision_val = 0.0
  if (true_positives + false_positives) > 0:
    precision_val = true_positives / (true_positives + false_positives)

  recall_val = 0.0
  if (true_positives + false_negatives) > 0:
    recall_val = true_positives / (true_positives + false_negatives)

  f1_score_val = 0.0
  if (precision_val + recall_val) > 0:
    f1_score_val = (
        2 * (precision_val * recall_val) / (precision_val + recall_val)
    )

  return {
      'precision': precision_val,
      'recall': recall_val,
      'f1_score': f1_score_val,
  }


def _aggregate_metrics_and_format_strings(
    per_item_metrics: dict[str, list[float]],
) -> dict[str, str]:
  """Aggregates per-example metrics and formats strings."""
  return {
      'precision': f"{numpy.mean(per_item_metrics['precision']):.2%}",
      'recall': f"{numpy.mean(per_item_metrics['recall']):.2%}",
      'f1_score': f"{numpy.mean(per_item_metrics['f1_score']):.2%}"
  }


def aggregate_ratings(
    item_ratings: list[ItemRating]
) -> ProjectRating:
  """Aggregates item-level ratings into project-level rating."""
  assert item_ratings, 'No item ratings to aggregate.'
  total_items = len(item_ratings)
  project_rating = ProjectRating(num_total_ratings=total_items)

  num_answer_correctness_evaluated = 0
  num_answer_correctness_all_correct = 0
  num_fully_incorrect_items = 0
  num_items_correct_with_excessive_answers = 0

  category_stats = _DefaultDict(lambda: {'evaluated': 0, 'all_correct': 0})
  per_item_metrics = {
      'precision': [],
      'recall': [],
      'f1_score': [],
      'accuracy': [],
  }

  for item_rating in item_ratings:
    if item_rating.invalid_auto_rater_response:
      project_rating.num_invalid_auto_rater_response += 1
      continue
    if item_rating.empty_auto_rater_response:
      project_rating.num_empty_auto_rater_response += 1
      continue
    if item_rating.empty_model_response:
      project_rating.num_empty_model_response += 1
      continue

    project_rating.num_valid_ratings += 1

    current_category = (
        item_rating.category_type if item_rating.category_type else 'Unknown'
    )

    if item_rating.grader_ratings_list is not None:
      num_answer_correctness_evaluated += 1
      category_stats[current_category]['evaluated'] += 1
      ratings = item_rating.grader_ratings_list
      num_correct = sum(1 for r in ratings if r)

      true_positives = num_correct
      false_negatives = len(ratings) - num_correct

      has_expected_answers = bool(ratings)

      all_expected_answers_correct = False
      if has_expected_answers:
        all_expected_answers_correct = num_correct == len(ratings)
        if num_correct == 0:
          num_fully_incorrect_items += 1

      excessive_answers = item_rating.response_wrong_answers_list
      has_excessive_answers = bool(excessive_answers)
      false_positives = 0
      if has_excessive_answers:
        false_positives = len(excessive_answers)
        if (all_expected_answers_correct or not has_expected_answers):
          num_items_correct_with_excessive_answers += 1

      is_all_correct = (
          all_expected_answers_correct or not has_expected_answers
      ) and not has_excessive_answers

      if is_all_correct:
        num_answer_correctness_all_correct += 1
        category_stats[current_category]['all_correct'] += 1

      per_item_metric = _calculate_metric(
          true_positives, false_positives, false_negatives
      )
      for key, value in per_item_metric.items():
        per_item_metrics[key].append(value)

  if total_items > 0:
    project_rating.pct_empty_model_response = round(
        project_rating.num_empty_model_response * 100.0 / total_items, 2
    )
    project_rating.pct_invalid_auto_rater_response = round(
        project_rating.num_invalid_auto_rater_response * 100.0 / total_items,
        2,
    )
    project_rating.pct_empty_auto_rater_response = round(
        project_rating.num_empty_auto_rater_response * 100.0 / total_items, 2
    )

  if num_answer_correctness_evaluated > 0:
    project_rating.num_answer_correctness_evaluated = (
        num_answer_correctness_evaluated
    )
    project_rating.pct_w_ci_all_answers_correct = _calculate_ci_str(
        num_answer_correctness_all_correct, num_answer_correctness_evaluated
    )
    project_rating.pct_w_ci_fully_incorrect_items = _calculate_ci_str(
        num_fully_incorrect_items, num_answer_correctness_evaluated
    )
    project_rating.pct_w_ci_correct_with_excessive_answers = _calculate_ci_str(
        num_items_correct_with_excessive_answers,
        num_answer_correctness_evaluated
    )

    aggregated_metrics = _aggregate_metrics_and_format_strings(per_item_metrics)
    project_rating.precision = aggregated_metrics['precision']
    project_rating.recall = aggregated_metrics['recall']
    project_rating.f1_score = aggregated_metrics['f1_score']

  return project_rating

# Default config path; overridden by CLI.
CONFIG_PATH: Path = Path("conf.yaml")


def _load_dsqa_ground_truth(dataset_path: Path) -> pd.DataFrame:
    """Load DSQA ground truth JSON list into DataFrame."""
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    df = pd.DataFrame(items)
    # Ensure expected columns exist
    required_cols = {"example_id", "problem", "answer", "answer_type", "problem_category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth missing columns: {missing}")
    df["example_id"] = df["example_id"].astype(str)
    return df


def _parse_prediction_payload(payload: dict[str, Any]) -> tuple[str, str]:
    """Extract (id, response) from one prediction JSONL line."""
    pred_raw = payload.get("prediction")
    response = ""
    if isinstance(pred_raw, str):
        try:
            pred_dict = json.loads(pred_raw)
            response = pred_dict.get("final_answer", "")
        except json.JSONDecodeError:
            response = pred_raw or ""
    elif isinstance(pred_raw, dict):
        response = pred_raw.get("final_answer", "") or ""
    else:
        response = str(pred_raw or "")

    pred_id = payload.get("example_id") or payload.get("question_id") or payload.get("question_index")
    if pred_id is None:
        raise ValueError("Prediction entry missing example/question id.")
    return str(pred_id), str(response).strip()


def _load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load predictions JSONL into DataFrame with columns question_index, response."""
    records: list[dict[str, str]] = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                pred_id, response = _parse_prediction_payload(payload)
            except Exception as exc:  # Keep going on bad rows.
                logging.warning("Skipping malformed prediction line %s: %s", line_no, exc)
                continue
            records.append(
                {
                    "question_index": pred_id,
                    "response": response,
                    "problem": payload.get("question") or payload.get("problem", ""),
                }
            )
    if not records:
        raise ValueError("No predictions loaded.")
    df = pd.DataFrame(records)
    df["question_index"] = df["question_index"].astype(str)
    return df


def build_input_df(dataset_path: Path, predictions_path: Path) -> pd.DataFrame:
    """Merge ground truth and predictions into the schema expected by scorer."""
    gt_df = _load_dsqa_ground_truth(dataset_path)
    pred_df = _load_predictions(predictions_path)

    merged = gt_df.merge(
        pred_df,
        left_on="example_id",
        right_on="question_index",
        how="left",
        suffixes=("", "_pred"),
    )
    if merged["response"].isna().any():
        missing = merged[merged["response"].isna()][["example_id"]]
        logging.warning("Missing predictions for %d items; responses set to empty.", len(missing))
        merged["response"] = merged["response"].fillna("")
    return merged


def run_evaluation(input_df: pd.DataFrame, max_workers: int = 30) -> tuple[ProjectRating, list[ItemRating]]:
    """Execute the rating loop and return summary + per-item ratings."""
    all_item_ratings: List[ItemRating] = []
    project_rating_result = ProjectRating()

    if input_df is None or input_df.empty:
        print("Input DataFrame is empty or None. Cannot perform rating.")
        return project_rating_result, all_item_ratings

    tasks = list(input_df.iterrows())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(process_single_row_for_rating, tasks)
        results = list(tqdm(results_iterator, total=len(tasks)))
        all_item_ratings.extend(results)

    print(f"\nFinished processing {len(all_item_ratings)} items.")
    project_rating_result = aggregate_ratings(all_item_ratings)

    print("\n--- Project Summary Rating ---")
    print(json.dumps(project_rating_result.to_dict(), indent=2))
    return project_rating_result, all_item_ratings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSearchQA official evaluator (LLM-graded)")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions JSONL path.")
    parser.add_argument("--dataset", type=Path, required=True, help="DSQA ground truth JSON path.")
    parser.add_argument("--config", type=Path, default=Path("conf.yaml"), help="Config file for eval model.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path for summary+items.")
    parser.add_argument("--max-workers", type=int, default=30, help="Thread pool size.")
    return parser.parse_args()


def main() -> None:
    global CONFIG_PATH
    args = parse_args()
    CONFIG_PATH = args.config  # used inside process_single_row_for_rating

    input_df = build_input_df(args.dataset, args.predictions)
    summary, items = run_evaluation(input_df, max_workers=args.max_workers)

    output_path = args.output
    if output_path is None:
        output_path = args.predictions.parent / "evaluation.json"

    output = {
        "summary": summary.to_dict(),
        "items": [it.to_dict() for it in items],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation to {output_path}")


if __name__ == "__main__":
    main()