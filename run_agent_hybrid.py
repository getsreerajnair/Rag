# run_agent_hybrid.py
import json
import click
from agent.graph_hybrid import GraphHybrid

# ------------------------------
# Run agent on a single question
# ------------------------------
def run_agent_on_question(graph, question_dict):

    # Execute the agent with the full input dict so format_hint/id are preserved.
    # GraphHybrid.run returns the Output Contract dict directly.
    result = graph.run(question_dict)
    return result


# ------------------------------
# CLI
# ------------------------------
@click.command()
@click.option("--batch", type=str, required=True, help="Input JSONL file of questions")
@click.option("--out", type=str, required=True, help="Output JSONL file")
def main(batch, out):
    # Build LangGraph hybrid agent
    graph = GraphHybrid()

    outputs = []
    with open(batch, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                q_dict = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue

            result = run_agent_on_question(graph, q_dict)
            outputs.append(result)

    # Write outputs as JSONL
    with open(out, "w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r) + "\n")

    print(f"Processed {len(outputs)} questions. Output written to {out}")


if __name__ == "__main__":
    main()
    with open(out, "w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r) + "\n")

    print(f"Processed {len(outputs)} questions. Output written to {out}")


if __name__ == "__main__":
    main()
