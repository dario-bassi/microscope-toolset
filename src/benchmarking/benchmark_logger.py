import json
import os
from datetime import datetime
from typing import Any, Optional


class BenchmarkLogger:
    """
    Logs MCP tool calls for benchmarking untrained vs trained agents.

    Each query's tool calls are buffered and flushed as a single JSON record
    with the format:
    {
        "agent_type": "untrained" | "trained",
        "run_id": "benchmark_<timestamp>",
        "user_query": "...",
        "timestamp": "2026-03-02T10:30:45.123",
        "mcp_calls": [
            {
                "tool_name": "...",
                "input_params": {...},
                "result": {...},
                "execution_time_ms": 245.5
            },
            ...
        ],
        "total_mcp_calls": 2,
        "total_execution_time_ms": 1445.5
    }
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
        self.mcp_calls = []

    def set_query(self, query: str) -> None:
        """
        Set the current user query context.

        Call this before Claude Code processes a query, so all subsequent
        tool calls are associated with this query.

        Args:
            query: The user's original question/request
        """
        # If there's a previous query with calls, flush it first
        if self.current_query is not None and self.mcp_calls:
            self.flush_query_log()

        self.current_query = query
        self.mcp_calls = []

    def log_tool_call(
        self,
        tool_name: str,
        input_params: dict[str, Any],
        result: dict[str, Any] | str,
        execution_time_ms: float
    ) -> None:
        """
        Log a single MCP tool call.

        Args:
            tool_name: Name of the MCP tool (e.g., "snap_image", "execute_python_code")
            input_params: Dictionary of input parameters to the tool
            result: The result returned by the tool (dict or error string)
            execution_time_ms: Execution time in milliseconds
        """
        call_entry = {
            "tool_name": tool_name,
            "input_params": input_params,
            "result": result,
            "execution_time_ms": execution_time_ms
        }
        self.mcp_calls.append(call_entry)

    def flush_query_log(self) -> None:
        """
        Save the current query's logs to the benchmark log file.

        Call this after Claude Code finishes processing a query.
        """
        if self.current_query is None:
            return

        total_time_ms = sum(c["execution_time_ms"] for c in self.mcp_calls)

        record = {
            "agent_type": self.agent_type,
            "run_id": self.run_id,
            "user_query": self.current_query,
            "timestamp": datetime.now().isoformat(),
            "mcp_calls": self.mcp_calls,
            "total_mcp_calls": len(self.mcp_calls),
            "total_execution_time_ms": total_time_ms
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            print(f"[BenchmarkLogger] Failed to write benchmark log: {e}")

        # Reset for next query
        self.current_query = None
        self.mcp_calls = []
