"""
Example of how to set up and run the benchmark logging system.

This shows how to initialize the MCP server with a BenchmarkLogger instance.
"""

from datetime import datetime
from src.benchmarking.benchmark_logger import BenchmarkLogger
from src.mcp_microscopetoolset.server_setup import create_mcp_server


def setup_benchmark_session(agent_type: str):
    """
    Set up a benchmark session for either untrained or trained agent.

    Args:
        agent_type: Either "untrained" or "trained"

    Returns:
        benchmark_logger: BenchmarkLogger instance to use during the session
    """
    # Create unique run ID for this benchmark
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"benchmark_{timestamp}"

    # Create benchmark logger
    benchmark_logger = BenchmarkLogger(agent_type=agent_type, run_id=run_id)

    print(f"Benchmark session created:")
    print(f"  Run ID: {run_id}")
    print(f"  Agent type: {agent_type}")
    print(f"  Log file: benchmark_logs/benchmark_{run_id}.jsonl")

    return benchmark_logger, run_id


def initialize_mcp_with_benchmark(
    database_agent,
    microscope_status,
    executor,
    viewer,
    event_cache,
    benchmark_logger,
    viewer_proxy=None
):
    """
    Initialize the MCP server with benchmark logging enabled.

    Args:
        benchmark_logger: BenchmarkLogger instance from setup_benchmark_session()
        ... other args same as create_mcp_server() ...

    Returns:
        mcp: The initialized FastMCP server with benchmarking enabled
    """
    mcp = create_mcp_server(
        database_agent=database_agent,
        microscope_status=microscope_status,
        executor=executor,
        viewer=viewer,
        event_cache=event_cache,
        viewer_proxy=viewer_proxy,
        benchmark_logger_instance=benchmark_logger  # Pass the logger here
    )
    return mcp


def run_single_query(benchmark_logger, query: str, claude_code_function):
    """
    Run a single query through Claude Code with benchmarking.

    Args:
        benchmark_logger: BenchmarkLogger instance
        query: The user query to run
        claude_code_function: Function that sends query to Claude Code and returns response

    Example:
        def ask_claude(q):
            # Your code to call Claude Code API
            # return response

        run_single_query(logger, "How do I snap an image?", ask_claude)
    """
    # Notify logger of new query
    benchmark_logger.set_query(query)

    print(f"\nRunning query: {query}")

    # Call Claude Code (this makes MCP tool calls which are logged automatically)
    response = claude_code_function(query)

    # Flush the query's logs
    benchmark_logger.flush_query_log()

    print(f"Query complete. Logs saved.")
    return response


def analyze_benchmark_results(run_id: str):
    """
    Analyze benchmark results from a completed run.

    Args:
        run_id: The benchmark run ID (e.g., "benchmark_20260302_103045")
    """
    import json
    from collections import defaultdict

    log_file = f"benchmark_logs/benchmark_{run_id}.jsonl"

    records = []
    with open(log_file, "r") as f:
        for line in f:
            records.append(json.loads(line))

    # Group by agent type
    by_agent = defaultdict(list)
    for record in records:
        by_agent[record["agent_type"]].append(record)

    # Print summary
    print(f"\nBenchmark Results for {run_id}")
    print("=" * 60)

    for agent_type in sorted(by_agent.keys()):
        runs = by_agent[agent_type]
        total_queries = len(runs)
        total_calls = sum(r["total_mcp_calls"] for r in runs)
        avg_calls = total_calls / total_queries
        avg_time = sum(r["total_execution_time_ms"] for r in runs) / total_queries

        print(f"\n{agent_type.upper()}:")
        print(f"  Queries: {total_queries}")
        print(f"  Total tool calls: {total_calls}")
        print(f"  Avg calls per query: {avg_calls:.1f}")
        print(f"  Avg execution time: {avg_time:.1f}ms")

        # Tool breakdown
        tool_counts = defaultdict(int)
        tool_times = defaultdict(float)
        for record in runs:
            for call in record["mcp_calls"]:
                tool_counts[call["tool_name"]] += 1
                tool_times[call["tool_name"]] += call["execution_time_ms"]

        print(f"\n  Tool breakdown:")
        for tool in sorted(tool_counts.keys()):
            count = tool_counts[tool]
            avg_tool_time = tool_times[tool] / count if count > 0 else 0
            print(f"    {tool}: {count} calls ({avg_tool_time:.1f}ms avg)")

    # Compare if both agent types present
    if len(by_agent) == 2:
        untrained_runs = by_agent.get("untrained", [])
        trained_runs = by_agent.get("trained", [])

        if untrained_runs and trained_runs:
            untrained_avg_calls = sum(r["total_mcp_calls"] for r in untrained_runs) / len(untrained_runs)
            trained_avg_calls = sum(r["total_mcp_calls"] for r in trained_runs) / len(trained_runs)

            improvement = ((untrained_avg_calls - trained_avg_calls) / untrained_avg_calls) * 100

            print(f"\n{'=' * 60}")
            print("COMPARISON:")
            print(f"  Untrained: {untrained_avg_calls:.1f} calls/query")
            print(f"  Trained:   {trained_avg_calls:.1f} calls/query")
            print(f"  Improvement: {improvement:.1f}% fewer calls with trained agent")


if __name__ == "__main__":
    # Example of full workflow (pseudocode)
    print("Benchmark Logging Example\n")
    print("1. Create benchmark session:")
    logger, run_id = setup_benchmark_session("untrained")

    print("\n2. Initialize MCP server with logger:")
    print(f"   mcp = initialize_mcp_with_benchmark(..., benchmark_logger=logger)")

    print("\n3. Run queries:")
    print("   for query in test_queries:")
    print("       run_single_query(logger, query, ask_claude_code)")

    print("\n4. Analyze results:")
    print(f"   analyze_benchmark_results('{run_id}')")

    print("\nSee BENCHMARK_USAGE.md for detailed instructions.")
