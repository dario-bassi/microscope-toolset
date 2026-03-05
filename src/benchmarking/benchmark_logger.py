import json
import os
from datetime import datetime
from typing import Any, Optional


class BenchmarkLogger:
    """
    Logs MCP tool calls for benchmarking untrained vs trained agents.

    Each tool call is written immediately to a JSONL file (auto-flush) with the format:
    {
        "agent_type": "untrained" | "trained",
        "run_id": "benchmark_<timestamp>",
        "user_query": "...",
        "mcp_call": {
            "tool_name": "...",
            "input_params": {...},
            "result": {...},
            "execution_time_ms": 245.5,
            "timestamp": "2026-03-02T10:30:45.123"
        }
    }

    Auto-flush ensures no data is lost even if the process crashes or you forget to call flush_query_log().
    """

    def __init__(self, agent_type: str, run_id: str):
        """
        Initialize the benchmark logger.

        Args:
            agent_type: Either "untrained" or "trained"
            run_id: Unique identifier for this benchmark run (e.g., "benchmark_20260302_103045")
        """
        self.agent_type = agent_type
        self.run_id = run_id

        # Create benchmark_logs directory if it doesn't exist
        os.makedirs("benchmark_logs", exist_ok=True)

        self.log_file = f"benchmark_logs/benchmark_{run_id}.jsonl"
        self.current_query: Optional[str] = None

    def set_query(self, query: str) -> None:
        """
        Set the current user query context.

        Call this before Claude Code processes a query, so all subsequent
        tool calls are associated with this query.

        Args:
            query: The user's original question/request
        """
        self.current_query = query

    def log_tool_call(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        result: dict[str, Any] | str,
        execution_time_ms: float
    ) -> None:
        """
        Log a single MCP tool call (auto-flushed immediately to file).

        Args:
            tool_name: Name of the MCP tool (e.g., "snap_image", "execute_python_code")
            input_params: Dictionary of input parameters to the tool
            result: The result returned by the tool (dict or error string)
            execution_time_ms: Execution time in milliseconds
        """
        if self.current_query is None:
            # Ignore if no active query set
            return

        call_entry = {
            "tool_name": tool_name,
            "input_params": input_params,
            "result": result,
            "execution_time_ms": execution_time_ms,
            "timestamp": datetime.now().isoformat()
        }

        record = {
            "agent_type": self.agent_type,
            "run_id": self.run_id,
            "user_query": self.current_query,
            "mcp_call": call_entry
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            print(f"[BenchmarkLogger] Failed to write tool call: {e}")

    def flush_query_log(self) -> None:
        """
        Optional cleanup call for query context (no longer required for data persistence).

        With auto-flush, tool calls are written immediately. This method is kept for
        backwards compatibility and can be used to mark query boundaries if needed.
        """
        # With auto-flush, there's nothing to flush
        # This is a no-op but kept for compatibility with existing code
        pass
