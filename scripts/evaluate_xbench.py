#!/usr/bin/env python3
"""XBench evaluation script with LLM judge and result persistence."""

import argparse
import json
import re
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluate_predictions import build_eval_model, OpenAIChatModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

###################################################
# Prompt Templates
###################################################

QUESTION_PROMPT = """
你是一个通用人工智能助手。我将向你提出一个学术问题, 请尽可能简洁地给出解题思路, 并用以下模版作为回答的结尾:

最终答案:[你的答案]

不要在最终答案周围添加任何多余的符号, 不要使用换行（在同一行中完成回答）
""".strip()

OBJECTIVE_PROMPT = "对于本题, 你的答案必须是尽可能简洁的数值, 短语, 或者数学表达式; 如果答案有多个, 使用逗号将它们隔开。"
CHOICE_PROMPT = "对于本题, 选出所有符合的选项, 少选、多选或错选都不得分; 如果选项有多个, 连续列出所有选项, 不要使用逗号或空格分隔。"

LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


###################################################
# Data Classes
###################################################

@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    question: str
    question_type: str
    correct_answer: str
    prediction: str
    
    # For each repeat
    scores: List[int]
    extracted_answers: List[str]
    llm_responses: List[str]
    grader_explanations: List[str]
    costs: List[float]
    times: List[float]
    
    # Flags
    length_cutoffs: List[bool]
    safety_cutoffs: List[bool]
    api_errors: List[bool]
    
    # Summary metrics
    average_score: float
    best_score: int
    majority_vote_answer: Optional[str]
    majority_vote_score: int
    average_cost: float
    average_time: float
    
    # Raw judge responses (for debugging)
    judge_responses: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Overall evaluation summary."""
    total_questions: int
    total_repeats: int
    average_score: float
    best_score_rate: float
    majority_vote_score: float
    total_cost: float
    total_time: float
    average_time_per_question: float


###################################################
# Global Variables
###################################################

# Config path for building eval model in worker threads
CONFIG_PATH: Optional[Path] = None


###################################################
# Helper Functions
###################################################

def get_question_prompt(question: str, question_type: str) -> str:
    """Generate full question prompt for LLM."""
    full_prompt = QUESTION_PROMPT

    if question_type == "问答题":
        full_prompt += OBJECTIVE_PROMPT + "\n\n"
    elif question_type == "选择题":
        full_prompt += CHOICE_PROMPT + "\n\n"

    full_prompt += "[问题]: " + question
    return full_prompt


def majority_vote(answers: List[str]) -> Optional[str]:
    """Majority vote function."""
    if not answers:
        return None

    count = Counter(answers)
    max_votes = max(count.values())
    candidates = [answer for answer, votes in count.items() if votes == max_votes]
    return random.choice(candidates)


def parse_match_result(match):
    """解析 Match 结果."""
    if match is None:
        return match

    match = match.group(0)
    try:
        target = match.split(':')[1].strip()
        return target
    except Exception:
        return match


def grade_question(
    llm: OpenAIChatModel,
    question_text: str,
    correct_answer: str,
    llm_response: Optional[str]
) -> Tuple[int, str, str, str]:
    """Grade one question with LLM judge.
    
    Returns:
        (score, extracted_answer, explanation, judge_response)
    """
    if llm_response is None or not llm_response.strip():
        return 0, "", "Response was empty", ""

    # If there's direct match, do not need LLM judge
    simple_match = re.search(r'最终答案:*(.*)', llm_response)
    simple_match_result = parse_match_result(simple_match)
    if simple_match_result and simple_match_result == correct_answer:
        return 1, simple_match_result, "答案完全正确, 无需调用LLM Judge", ""

    # Otherwise, use LLM Judge
    judge_prompt = LLM_JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=correct_answer,
        response=llm_response,
    )

    judge_response = ""
    for i in range(3):
        try:
            judge_response = llm.invoke(
                messages=[{"role": "user", "content": judge_prompt}]
            ).content
            break
        except Exception as e:
            logger.error(f"Judge LLM call failed (attempt {i+1}/3): {e}")
            if i == 2:
                return 0, "", f"Judge Response error: {e}", ""
            time.sleep(1 + (2 ** (i + random.random())))

    if not judge_response:
        return 0, "", "Judge Response error: empty response", ""

    # Extract grader conclusions
    extract_match = re.search(r'最终答案:*(.*)', judge_response)
    extract_match_result = parse_match_result(extract_match)

    correct_match = re.search(r"结论:*.(正确|错误)", judge_response)
    correct_match_result = parse_match_result(correct_match)

    explain_match = re.search(r"解释:*(.*)", judge_response)
    explain_match_result = parse_match_result(explain_match)

    score = 1 if (correct_match_result == "正确") else 0

    return score, extract_match_result or "", explain_match_result or "", judge_response


###################################################
# Data Loading Functions
###################################################

def load_ground_truth(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load ground truth data from JSON file.
    
    Supports two formats:
    1. XBench DeepSearch format: {id, prompt, answer, type}
    2. XBench ScienceQA format: {task_id, task_question, ground_truth, metadata.type}
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ground_truth = {}
    for item in data:
        # Support both formats
        question_id = str(item.get('task_id') or item.get('id', ''))
        question_text = item.get('task_question') or item.get('prompt', '')
        answer = item.get('ground_truth') or item.get('answer', '')
        
        # Type can be in metadata or directly in item
        question_type = item.get('type', '问答题')
        if 'metadata' in item and isinstance(item['metadata'], dict):
            question_type = item['metadata'].get('type', question_type)
        
        ground_truth[question_id] = {
            'question': question_text,
            'answer': answer,
            'type': question_type
        }
    
    return ground_truth


def load_predictions(predictions_path: Path) -> Dict[str, str]:
    """Load predictions from JSONL file.
    
    Expected format: {"question_index": ..., "question": ..., "prediction": ...}
    """
    predictions = {}
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                question_id = str(data.get('question_index', ''))
                
                # Extract prediction - can be string or dict with final_answer
                prediction = data.get('prediction', '')
                if isinstance(prediction, dict):
                    prediction = prediction.get('final_answer', '')
                elif isinstance(prediction, str):
                    # Try to parse as JSON in case it's a JSON string
                    try:
                        pred_dict = json.loads(prediction)
                        prediction = pred_dict.get('final_answer', prediction)
                    except json.JSONDecodeError:
                        pass
                
                predictions[question_id] = str(prediction)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue
    
    return predictions


###################################################
# Evaluation Functions
###################################################

def eval_and_grade_question(
    llm: OpenAIChatModel,
    question_id: str,
    question_text: str,
    question_type: str,
    correct_answer: str,
    prediction: str,
    n_repeats: int = 1
) -> QuestionResult:
    """Evaluate and grade a single question.
    
    For XBench, we typically already have the prediction, so n_repeats=1.
    But we keep the structure flexible for potential re-runs.
    """
    scores = []
    extracted_answers = []
    llm_responses = []
    grader_explanations = []
    costs = []
    times = []
    length_cutoffs = []
    safety_cutoffs = []
    api_errors = []
    judge_responses = []
    
    for i in range(n_repeats):
        start_time = time.time()
        
        # Use the provided prediction directly
        llm_response = prediction
        response_time = time.time() - start_time
        api_cost = 0.0  # No API cost for using existing predictions
        
        # Grade the response
        score, extracted_answer, grader_explanation, judge_response = grade_question(
            llm, question_text, correct_answer, llm_response
        )
        
        llm_responses.append(llm_response)
        extracted_answers.append(extracted_answer)
        scores.append(score)
        grader_explanations.append(grader_explanation)
        costs.append(api_cost)
        times.append(response_time)
        length_cutoffs.append(False)
        safety_cutoffs.append(False)
        api_errors.append(False)
        judge_responses.append(judge_response)
    
    # Calculate summary metrics
    average_score = float(np.mean(scores))
    best_score = int(np.max(scores))
    majority_vote_answer = majority_vote(extracted_answers)
    majority_vote_score = 1 if (majority_vote_answer == correct_answer) else 0
    average_cost = float(np.mean(costs))
    average_time = float(np.mean(times))
    
    return QuestionResult(
        question_id=question_id,
        question=question_text,
        question_type=question_type,
        correct_answer=correct_answer,
        prediction=prediction,
        scores=scores,
        extracted_answers=extracted_answers,
        llm_responses=llm_responses,
        grader_explanations=grader_explanations,
        costs=costs,
        times=times,
        length_cutoffs=length_cutoffs,
        safety_cutoffs=safety_cutoffs,
        api_errors=api_errors,
        average_score=average_score,
        best_score=best_score,
        majority_vote_answer=majority_vote_answer,
        majority_vote_score=majority_vote_score,
        average_cost=average_cost,
        average_time=average_time,
        judge_responses=judge_responses
    )


def eval_single_question_worker(args: Tuple[str, Dict[str, Any], str, int]) -> QuestionResult:
    """Worker function for concurrent evaluation.
    
    Args:
        args: Tuple of (question_id, ground_truth_item, prediction, n_repeats)
    
    Returns:
        QuestionResult for the evaluated question
    """
    question_id, gt_item, prediction, n_repeats = args
    
    # Create a new LLM instance for this worker to avoid thread safety issues
    llm, _ = build_eval_model(CONFIG_PATH)
    
    try:
        result = eval_and_grade_question(
            llm=llm,
            question_id=question_id,
            question_text=gt_item['question'],
            question_type=gt_item['type'],
            correct_answer=gt_item['answer'],
            prediction=prediction,
            n_repeats=n_repeats
        )
        return result
    except Exception as e:
        logger.error(f"Error evaluating question {question_id}: {e}", exc_info=True)
        # Return a result with error information
        return QuestionResult(
            question_id=question_id,
            question=gt_item['question'],
            question_type=gt_item['type'],
            correct_answer=gt_item['answer'],
            prediction=prediction,
            scores=[0],
            extracted_answers=[""],
            llm_responses=[prediction],
            grader_explanations=[f"Error during evaluation: {str(e)}"],
            costs=[0.0],
            times=[0.0],
            length_cutoffs=[False],
            safety_cutoffs=[False],
            api_errors=[True],
            average_score=0.0,
            best_score=0,
            majority_vote_answer=None,
            majority_vote_score=0,
            average_cost=0.0,
            average_time=0.0,
            judge_responses=[]
        )


def run_evaluation(
    ground_truth: Dict[str, Dict[str, Any]],
    predictions: Dict[str, str],
    n_repeats: int = 1,
    output_dir: Optional[Path] = None,
    max_workers: int = 10
) -> Tuple[List[QuestionResult], EvaluationSummary]:
    """Run evaluation on all questions with concurrent processing.
    
    Args:
        ground_truth: Dictionary mapping question IDs to ground truth data
        predictions: Dictionary mapping question IDs to predictions
        n_repeats: Number of times to evaluate each question
        output_dir: Directory to save intermediate results
        max_workers: Maximum number of concurrent worker threads
    
    Returns:
        Tuple of (results list, evaluation summary)
    """
    results = []
    total_cost = 0.0
    total_time = 0.0
    
    # Find common question IDs
    common_ids = set(ground_truth.keys()) & set(predictions.keys())
    logger.info(f"Evaluating {len(common_ids)} questions (Ground truth: {len(ground_truth)}, Predictions: {len(predictions)})")
    
    if len(common_ids) == 0:
        logger.warning("No common question IDs found between ground truth and predictions!")
        return [], EvaluationSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Prepare tasks for concurrent execution
    tasks = [
        (question_id, ground_truth[question_id], predictions[question_id], n_repeats)
        for question_id in sorted(common_ids)
    ]
    
    # Use ThreadPoolExecutor for concurrent evaluation
    logger.info(f"Starting concurrent evaluation with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_qid = {
            executor.submit(eval_single_question_worker, task): task[0]
            for task in tasks
        }
        
        # Process results as they complete with progress bar
        with tqdm(total=len(tasks), desc="Evaluating questions") as pbar:
            for future in as_completed(future_to_qid):
                question_id = future_to_qid[future]
                try:
                    result = future.result()
                    results.append(result)
                    total_cost += result.average_cost
                    total_time += result.average_time
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'avg_score': f"{np.mean([r.average_score for r in results]):.2%}",
                        'current_score': result.average_score
                    })
                    
                    # Save intermediate results periodically
                    if output_dir and len(results) % 10 == 0:
                        intermediate_file = output_dir / f"intermediate_results_{len(results)}.json"
                        save_results(results, intermediate_file)
                        
                except Exception as e:
                    logger.error(f"Error processing result for question {question_id}: {e}")
                    pbar.update(1)
    
    # Sort results by question_id to maintain consistent ordering
    results.sort(key=lambda x: str(x.question_id))
    
    # Calculate summary
    total_questions = len(results)
    average_score = float(np.mean([r.average_score for r in results])) if results else 0.0
    best_score_rate = float(np.mean([r.best_score for r in results])) if results else 0.0
    majority_vote_score = float(np.mean([r.majority_vote_score for r in results])) if results else 0.0
    average_time_per_question = total_time / total_questions if total_questions > 0 else 0.0
    
    summary = EvaluationSummary(
        total_questions=total_questions,
        total_repeats=n_repeats,
        average_score=average_score,
        best_score_rate=best_score_rate,
        majority_vote_score=majority_vote_score,
        total_cost=total_cost,
        total_time=total_time,
        average_time_per_question=average_time_per_question
    )
    
    return results, summary


def save_results(results: List[QuestionResult], output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            [r.to_dict() for r in results],
            f,
            ensure_ascii=False,
            indent=2
        )
    
    logger.info(f"Saved results to {output_path}")


def save_summary(summary: EvaluationSummary, output_path: Path):
    """Save evaluation summary to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            asdict(summary),
            f,
            ensure_ascii=False,
            indent=2
        )
    
    logger.info(f"Saved summary to {output_path}")


def print_summary(summary: EvaluationSummary):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Questions: {summary.total_questions}")
    print(f"Repeats per Question: {summary.total_repeats}")
    print(f"Average Score: {summary.average_score:.2%}")
    print(f"Best Score Rate: {summary.best_score_rate:.2%}")
    print(f"Majority Vote Score: {summary.majority_vote_score:.2%}")
    print(f"Total Cost: ${summary.total_cost:.4f}")
    print(f"Total Time: {summary.total_time:.2f}s")
    print(f"Average Time per Question: {summary.average_time_per_question:.2f}s")
    print("="*60 + "\n")


###################################################
# Main Function
###################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XBench evaluation script with LLM judge and concurrent processing"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSONL file"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf.yaml"),
        help="Path to configuration file (default: conf.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: same as predictions)"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of times to evaluate each question (default: 1)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of concurrent worker threads (default: 10)"
    )
    
    return parser.parse_args()


def main():
    global CONFIG_PATH
    
    args = parse_args()
    
    # Set global config path for worker threads
    CONFIG_PATH = args.config
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.predictions.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate config by loading evaluation model
    logger.info(f"Validating evaluation model config from {args.config}")
    _, eval_conf = build_eval_model(args.config)
    logger.info(f"Using model: {eval_conf.get('model', 'unknown')}")
    logger.info(f"Concurrent workers: {args.max_workers}")
    
    # Load data
    logger.info(f"Loading ground truth from {args.dataset}")
    ground_truth = load_ground_truth(args.dataset)
    logger.info(f"Loaded {len(ground_truth)} ground truth items")
    
    logger.info(f"Loading predictions from {args.predictions}")
    predictions = load_predictions(args.predictions)
    logger.info(f"Loaded {len(predictions)} predictions")
    
    # Run evaluation with concurrent processing
    results, summary = run_evaluation(
        ground_truth=ground_truth,
        predictions=predictions,
        n_repeats=args.n_repeats,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Print summary
    print_summary(summary)
    
    # Save final results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = args.output_dir / f"xbench_results_{timestamp}.json"
    summary_file = args.output_dir / f"xbench_summary_{timestamp}.json"
    
    save_results(results, results_file)
    save_summary(summary, summary_file)
    
    # Also save in evaluation.json format for compatibility
    combined_output = args.output_dir / "xbench_evaluation.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': asdict(summary),
            'results': [r.to_dict() for r in results]
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved combined results to {combined_output}")
    
    print("\n✓ Evaluation complete!")
    print(f"  Results: {results_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Combined: {combined_output}")


if __name__ == "__main__":
    main()
