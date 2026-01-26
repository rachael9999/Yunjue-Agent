#!/usr/bin/env python3
"""Unified evaluation script for all benchmark datasets (GAIA, HLE, DSQA, DSBENCH)."""

from __future__ import annotations

import argparse
import atexit
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import yaml
from openai import OpenAI
from tqdm import tqdm


class MessageType:
    """Simple message types."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Simple message class."""

    role: str
    content: str


class BenchmarkType(Enum):
    """Supported benchmark types."""

    GAIA = "gaia"
    HLE = "hle"
    DSQA = "dsqa"
    DSBENCH = "dsbench"
    FINSEARCHCOMP = "finsearchcomp"


@dataclass
class Prediction:
    id: str
    question: str
    prediction: Optional[str]
    raw_prediction_json: Optional[Dict[str, Any]] = None
    run_success: Optional[bool] = None


@dataclass
class GroundTruth:
    id: str
    question: str
    answer: str
    answer_type: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OpenAIChatModel:
    """Thin wrapper around the official OpenAI client with a LangChain-like interface."""

    def __init__(self, client: OpenAI, model: str, **invoke_params: Any) -> None:
        self._client = client
        self._model = model
        self._invoke_params = invoke_params

    def invoke(
        self, messages: Iterable[Any], response_format: Optional[Dict[str, str]] = None
    ) -> SimpleNamespace:
        formatted_messages: List[Dict[str, str]] = []
        for message in messages:
            if isinstance(message, Message):
                formatted_messages.append({"role": message.role, "content": message.content})
            elif isinstance(message, dict):
                formatted_messages.append(message)
            else:
                # Fallback for other types
                content = str(message.content if hasattr(message, "content") else message)
                role = "user"
                if hasattr(message, "role"):
                    role = message.role
                formatted_messages.append({"role": role, "content": content})

        call_params = dict(self._invoke_params)
        if response_format:
            call_params["response_format"] = response_format

        response = self._client.chat.completions.create(
            model=self._model,
            messages=formatted_messages,
            **call_params,
        )

        if not response.choices:
            raise RuntimeError("OpenAI response contained no choices")

        choice = response.choices[0]
        message_content = getattr(choice.message, "content", "") or ""
        return SimpleNamespace(content=message_content)


def load_predictions(path: Path, benchmark_type: BenchmarkType) -> List[Prediction]:
    """Load model predictions stored as JSON lines."""

    predictions: List[Prediction] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)

                # Extract prediction based on format
                pred = payload.get("prediction")
                if isinstance(pred, str):
                    try:
                        pred_dict = json.loads(pred)
                        prediction = pred_dict.get("final_answer")
                    except json.JSONDecodeError:
                        # Extract via regex
                        match = re.search(r'"final_answer"\s*:\s*"([^"]*)"', pred)
                        prediction = match.group(1) if match else pred
                else:
                    prediction = pred.get("final_answer") if pred else None

                # Determine ID field based on benchmark type
                if benchmark_type == BenchmarkType.GAIA:
                    pred_id = str(payload.get("question_index", line_no))
                elif benchmark_type == BenchmarkType.HLE:
                    pred_id = payload.get("question_id") or str(payload.get("question_index", line_no))
                elif benchmark_type == BenchmarkType.DSQA:
                    # Use explicit question identifiers to align with ground truth even when
                    # predictions are sparse or out of order.
                    pred_id = (
                        payload.get("example_id")
                        or payload.get("question_id")
                        or payload.get("question_index")
                    )
                    if pred_id is None:
                        # Fallback to sequential index if no id is provided
                        pred_id = line_no - 1
                    pred_id = str(pred_id)
                elif benchmark_type == BenchmarkType.DSBENCH:
                    # Use question_index field which has format like "00000001_question6"
                    pred_id = payload.get("question_index") or payload.get("question_id") or str(line_no - 1)
                elif benchmark_type == BenchmarkType.FINSEARCHCOMP:
                    # Use question_index or question_id for FinSearchComp
                    pred_id = payload.get("question_index") or payload.get("question_id") or str(line_no)
                else:
                    pred_id = str(line_no)

                predictions.append(
                    Prediction(
                        id=pred_id,
                        question=str(payload.get("question") or payload.get("problem", "")),
                        prediction=prediction,
                        raw_prediction_json=payload.get("raw_prediction_json"),
                        run_success=payload.get("run_success"),
                    )
                )
            except Exception as exc:
                print(f"Warning: Failed to parse prediction on line {line_no}: {exc}")

    return predictions


def load_ground_truth(path: Path, benchmark_type: BenchmarkType) -> Tuple[Dict[str, GroundTruth], List[str]]:
    """Load ground truth answers based on benchmark type."""

    with path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    if benchmark_type == BenchmarkType.HLE:
        items = items["data"]
    ground_truth_map: Dict[str, GroundTruth] = {}
    order: List[str] = []

    for idx, item in enumerate(items):
        if benchmark_type == BenchmarkType.GAIA:
            gt_id = item.get("task_id") or str(idx + 1)
            question = item.get("Question")
            answer = item.get("Final answer")
            answer_type = None
            category = item.get("Level")

        elif benchmark_type == BenchmarkType.HLE:
            gt_id = item.get("id") or str(idx)
            question = item.get("question")
            answer = item.get("answer")
            answer_type = item.get("answer_type")
            category = None

        elif benchmark_type == BenchmarkType.DSQA:
            gt_id = str(item.get("example_id", idx))
            question = item.get("problem")
            answer = item.get("answer")
            answer_type = item.get("answer_type", "Single Answer")
            category = item.get("problem_category")

        elif benchmark_type == BenchmarkType.DSBENCH:
            question_id = item.get("question_id")
            dataset_id = item.get("dataset_id")
            # Always combine dataset_id and question_id with underscore
            if dataset_id and question_id:
                gt_id = f"{dataset_id}_{question_id}"
            elif question_id:
                gt_id = question_id
            else:
                gt_id = str(idx)
            question = item.get("question")
            answer = item.get("answer")
            answer_type = None
            category = item.get("name")
        elif benchmark_type == BenchmarkType.FINSEARCHCOMP:
            gt_id = item.get("prompt_id") or str(idx)
            question = item.get("prompt")
            answer = item.get("response_reference")
            answer_type = None
            category = item.get("label")
        else:
            continue

        if not isinstance(question, str) or not isinstance(answer, str):
            continue

        ground_truth_map[gt_id] = GroundTruth(
            id=gt_id,
            question=question,
            answer=answer,
            answer_type=answer_type,
            category=category,
            metadata=item,
        )
        order.append(gt_id)

    return ground_truth_map, order


def normalise_answer(value: Optional[str]) -> str:
    """Normalise answers for direct string comparison."""
    if value is None:
        return ""
    return value.strip().lower()


def normalise_set_answer(value: Optional[str]) -> set:
    """Normalise set answers by splitting on commas and normalising each element."""
    if value is None:
        return set()

    # Split by comma and normalise each element
    elements = [elem.strip().lower() for elem in value.split(",")]
    return set(elem for elem in elements if elem)


def build_eval_model(config_path: Path) -> Tuple[OpenAIChatModel, Dict[str, Any]]:
    """Instantiate the evaluation model from configuration."""

    # Load YAML config directly
    with open(config_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    eval_conf = conf.get("EVAL_MODEL")
    if not isinstance(eval_conf, dict) or not eval_conf:
        raise ValueError("EVAL_MODEL configuration is missing or invalid in conf.yaml")

    llm_conf = dict(eval_conf)

    # Remove configuration keys not recognised by the official OpenAI client
    llm_conf.pop("token_limit", None)
    verify_ssl = llm_conf.pop("verify_ssl", True)

    api_key = llm_conf.pop("api_key", None)
    model_name = llm_conf.pop("model", None)
    base_url = llm_conf.pop("base_url", None)
    max_retries = llm_conf.pop("max_retries", 3)

    if not api_key:
        raise ValueError("EVAL_MODEL.api_key is required for OpenAI client configuration")
    if not model_name:
        raise ValueError("EVAL_MODEL.model is required for OpenAI client configuration")

    http_client = None
    if not verify_ssl:
        http_client = httpx.Client(verify=False)

    client_kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": max_retries}
    if base_url:
        client_kwargs["base_url"] = base_url
    if http_client:
        client_kwargs["http_client"] = http_client

    client = OpenAI(**client_kwargs)

    model = OpenAIChatModel(client, model_name, **llm_conf)

    if http_client:
        atexit.register(http_client.close)

    return model, eval_conf


def judge_answer_finsearchcomp(
    llm: OpenAIChatModel,
    question: str,
    ground_truth: str,
    prediction: str,
    judge_system_prompt: str,
    judge_prompt_template: str,
) -> Dict[str, Any]:
    """Ask the evaluation LLM to judge correctness using FinSearchComp's custom prompts."""

    # Format the user prompt using the provided template
    user_prompt = judge_prompt_template.format(
        prompt=question, response_reference=ground_truth, response=prediction or "<empty>"
    )

    messages = [Message(MessageType.SYSTEM, judge_system_prompt), Message(MessageType.USER, user_prompt)]
    response = llm.invoke(messages, response_format={"type": "json_object"})
    text = response.content
    if isinstance(text, list):
        text = "".join(str(chunk) for chunk in text)

    # Clean up the text
    text = text.strip()

    # Try to extract the answer_score from JSON
    try:
        # Look for {"answer_score": X} pattern
        json_match = re.search(r'\{\s*"answer_score"\s*:\s*(\d+)\s*\}', text)
        if json_match:
            score = int(json_match.group(1))
            # Extract explanation (评分依据)
            explanation_match = re.search(r"【评分依据】[：:](.*?)(?=【JSON】|$)", text, re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else text

            return {"correct": score == 1, "explanation": explanation, "answer_score": score}
    except Exception as e:
        print(f"Warning: Failed to parse FinSearchComp response: {e}")

    # Last resort: return a default structure
    print(f"Warning: Failed to parse FinSearchComp response")
    print(f"Response text: {text[:500]}...")
    return {"correct": False, "explanation": "Failed to parse LLM response", "answer_score": 0}


def judge_answer(
    llm: OpenAIChatModel,
    question: str,
    ground_truth: str,
    prediction: str,
    answer_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask the evaluation LLM to judge correctness."""

    system_prompt = (
        "You are an expert judge for benchmark answers. "
        "You must respond with valid JSON only, using this exact schema:\n"
        "{\n"
        '  "correct": <boolean>,\n'
        '  "explanation": "<string>"\n'
        "}\n"
        "Do not include any text outside the JSON object."
    )

    answer_type_instruction = ""
    if answer_type == "Set Answer":
        answer_type_instruction = (
            "\n\nNote: This is a 'Set Answer' question. The prediction should contain all elements "
            "from the ground truth (order doesn't matter), possibly separated by commas. "
            "The prediction is correct if it contains all required elements, even if formatted differently."
        )

    user_prompt = (
        "Question:\n"
        f"{question}\n\n"
        "Ground truth answer:\n"
        f"{ground_truth}\n\n"
        "Model prediction:\n"
        f"{prediction or '<empty>'}\n\n"
    )

    if answer_type:
        user_prompt += f"Answer type: {answer_type}{answer_type_instruction}\n\n"

    user_prompt += "Decide if the model prediction satisfies the ground truth. Respond with valid JSON matching the schema."

    messages = [Message(MessageType.SYSTEM, system_prompt), Message(MessageType.USER, user_prompt)]
    response = llm.invoke(messages, response_format={"type": "json_object"})
    text = response.content
    if isinstance(text, list):
        text = "".join(str(chunk) for chunk in text)

    # Clean up the text
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError) as e:
        # Try to extract JSON from the text
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        # Last resort: return a default structure
        print(f"Warning: Failed to parse LLM response as JSON: {e}")
        print(f"Response text: {text[:200]}...")
        return {"correct": False, "explanation": "Failed to parse LLM response"}


def evaluate_predictions(
    llm: OpenAIChatModel,
    predictions: Iterable[Prediction],
    ground_truth_map: Dict[str, GroundTruth],
    benchmark_type: Optional[BenchmarkType] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Evaluate predictions and return detailed results plus counters."""

    results: List[Dict[str, Any]] = []
    evaluated = 0
    correct = 0
    to_judge = []

    for pred in predictions:
        truth = ground_truth_map.get(pred.id)
        if not truth:
            results.append(
                {
                    "id": pred.id,
                    "question": pred.question,
                    "prediction": pred.prediction,
                    "status": "missing_ground_truth",
                }
            )
            continue

        predicted_value = pred.prediction or ""

        # Skip auto-matching for FinSearchComp as it uses custom judge prompts
        if benchmark_type != BenchmarkType.FINSEARCHCOMP:
            # Try auto-matching based on answer type
            auto_matched = False

            if truth.answer_type == "Set Answer":
                # For set answers, check if all elements match (order-independent)
                pred_set = normalise_set_answer(predicted_value)
                truth_set = normalise_set_answer(truth.answer)
                if pred_set == truth_set:
                    auto_matched = True
            else:
                # For single answers, do exact match after normalisation
                if normalise_answer(predicted_value) == normalise_answer(truth.answer):
                    auto_matched = True

            if auto_matched:
                results.append(
                    {
                        "id": pred.id,
                        "question": truth.question,
                        "category": truth.category,
                        "answer_type": truth.answer_type,
                        "ground_truth": truth.answer,
                        "prediction": predicted_value,
                        "status": "auto_match",
                        "correct": True,
                        "explanation": "Prediction matches ground truth after normalisation.",
                    }
                )
                evaluated += 1
                correct += 1
                continue

        to_judge.append((pred, truth, predicted_value))

    # Concurrently judge the remaining predictions
    with ThreadPoolExecutor(max_workers=30) as executor:
        # Use different judge function for FinSearchComp
        if benchmark_type == BenchmarkType.FINSEARCHCOMP:
            futures_to_data = {
                executor.submit(
                    judge_answer_finsearchcomp,
                    llm,
                    truth.question,
                    truth.answer,
                    predicted_value,
                    truth.metadata.get("judge_system_prompt", ""),
                    truth.metadata.get("judge_prompt_template", ""),
                ): (pred, truth, predicted_value)
                for pred, truth, predicted_value in to_judge
            }
        else:
            futures_to_data = {
                executor.submit(
                    judge_answer, llm, truth.question, truth.answer, predicted_value, truth.answer_type
                ): (pred, truth, predicted_value)
                for pred, truth, predicted_value in to_judge
            }
        for future in tqdm(as_completed(futures_to_data), total=len(to_judge), desc="Evaluating"):
            pred, truth, predicted_value = futures_to_data[future]
            try:
                judgement = future.result()
                is_correct = bool(judgement.get("correct"))
                explanation = judgement.get("explanation", "")
            except Exception as exc:
                print(f"Warning: Failed to judge prediction for {pred.id}: {exc}")
                is_correct = False
                explanation = f"Evaluation error: {exc}"
                judgement = {}

            results.append(
                {
                    "id": pred.id,
                    "question": truth.question,
                    "category": truth.category,
                    "answer_type": truth.answer_type,
                    "ground_truth": truth.answer,
                    "prediction": predicted_value,
                    "status": "llm_judged",
                    "correct": is_correct,
                    "explanation": explanation,
                    "raw_judgement": judgement,
                }
            )
            evaluated += 1
            if is_correct:
                correct += 1

    return results, evaluated, correct


def sort_results_by_ground_truth(
    results: List[Dict[str, Any]], ground_truth_order: List[str]
) -> List[Dict[str, Any]]:
    """Sort evaluation results to match the order of the ground-truth dataset."""

    order_map = {gt_id: idx for idx, gt_id in enumerate(ground_truth_order)}
    indexed_results = list(enumerate(results))

    def sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        original_idx, result = item
        gt_idx = order_map.get(result.get("id"))
        if gt_idx is None:
            gt_idx = len(order_map) + original_idx
        return gt_idx, original_idx

    indexed_results.sort(key=sort_key)
    return [result for _, result in indexed_results]


def compute_category_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute statistics by category."""

    category_stats: Dict[str, Dict[str, Any]] = {}

    for result in results:
        if "correct" not in result:
            continue

        category = result.get("category") or "Unknown"
        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
            }

        category_stats[category]["total"] += 1
        if result["correct"]:
            category_stats[category]["correct"] += 1

    # Calculate accuracy for each category
    for category in category_stats:
        total = category_stats[category]["total"]
        correct = category_stats[category]["correct"]
        category_stats[category]["accuracy"] = (correct / total) if total > 0 else 0.0

    return category_stats


def detect_benchmark_type(dataset_path: Path) -> BenchmarkType:
    """Detect benchmark type from dataset path or content."""

    path_str = str(dataset_path).lower()

    if "gaia" in path_str:
        return BenchmarkType.GAIA
    elif "hle" in path_str:
        return BenchmarkType.HLE
    elif "dsqa" in path_str or "deepsearchqa" in path_str:
        return BenchmarkType.DSQA
    elif "dsbench" in path_str:
        return BenchmarkType.DSBENCH
    elif "finsearchcomp" in path_str:
        return BenchmarkType.FINSEARCHCOMP

    # Try to detect from content
    try:
        with dataset_path.open("r") as f:
            data = json.load(f)
            if data and isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if "task_id" in first_item or "Question" in first_item:
                    return BenchmarkType.GAIA
                elif "example_id" in first_item and "problem" in first_item:
                    return BenchmarkType.DSQA
                elif "dataset_id" in first_item and "question_id" in first_item:
                    return BenchmarkType.DSBENCH
                elif (
                    "prompt_id" in first_item
                    and "prompt" in first_item
                    and "response_reference" in first_item
                ):
                    return BenchmarkType.FINSEARCHCOMP
                elif "answer_type" in first_item:
                    return BenchmarkType.HLE
    except Exception:
        pass

    raise ValueError(
        f"Could not detect benchmark type from {dataset_path}. Please specify --benchmark explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation for all benchmark datasets")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to the predictions JSONL file",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the ground truth dataset JSON file",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[b.value for b in BenchmarkType],
        help="Benchmark type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf.yaml"),
        help="Path to configuration file containing EVAL_MODEL settings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save detailed evaluation results (JSON). Default: evaluation.json in predictions dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions file not found: {args.predictions}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # Detect or use specified benchmark type
    if args.benchmark:
        benchmark_type = BenchmarkType(args.benchmark)
    else:
        benchmark_type = detect_benchmark_type(args.dataset)

    print(f"Detected benchmark type: {benchmark_type.value.upper()}")

    predictions = load_predictions(args.predictions, benchmark_type)
    ground_truth_map, ground_truth_order = load_ground_truth(args.dataset, benchmark_type)

    llm, eval_conf = build_eval_model(args.config)

    results, evaluated, correct = evaluate_predictions(llm, predictions, ground_truth_map, benchmark_type)
    results = sort_results_by_ground_truth(results, ground_truth_order)

    total_predictions = len(predictions)
    accuracy = (correct / evaluated) if evaluated else 0.0

    # Compute category-wise statistics
    category_stats = compute_category_stats(results)

    summary = {
        "benchmark_type": benchmark_type.value,
        "total_predictions": total_predictions,
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "category_stats": category_stats,
        "model": eval_conf.get("model"),
        "results": results,
    }

    output_path = args.output
    if output_path is None:
        output_path = args.predictions.parent / "evaluation.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nEvaluation complete:")
    print(f"  Benchmark: {benchmark_type.value.upper()}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Evaluated (via auto match or LLM): {evaluated}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.2%}")

    if category_stats:
        print("\n  Category-wise statistics:")
        for category, stats in sorted(category_stats.items()):
            print(f"    {category}:")
            print(f"      Correct: {stats['correct']}/{stats['total']} ({stats['accuracy']:.2%})")

    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
