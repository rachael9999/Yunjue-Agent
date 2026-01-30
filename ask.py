import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from src.main import run_task


async def _run_with_timeout(question: str, run_dir: Path, task_id: str, debug: bool, timeout: int):
    if timeout and timeout > 0:
        return await asyncio.wait_for(
            run_task(question, run_dir, debug=debug, task_id=task_id),
            timeout=timeout,
        )
    return await run_task(question, run_dir, debug=debug, task_id=task_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a single question via the agent.")
    parser.add_argument("-q", "--question", required=True, help="Question to ask")
    parser.add_argument("--run_name", type=str, default=None, help="Run name (output folder)")
    parser.add_argument("--task_id", type=str, default="cli", help="Task id for logs/tools")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--timeout",
        type=int,
        default=30 * 60,
        help="Timeout in seconds for the query (default: 1800)",
    )

    args = parser.parse_args()

    run_name = args.run_name or f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("output") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    answer, _usage = asyncio.run(
        _run_with_timeout(args.question, run_dir, args.task_id, args.debug, args.timeout)
    )

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
