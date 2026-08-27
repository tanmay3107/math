import sys
import json
import time
import random
from vector_store import VectorStore

def parse_vector(vector_str: str) -> list[float]:
    """Parse comma-separated values into a float list."""
    try:
        return [float(x.strip()) for x in vector_str.split(",")]
    except ValueError:
        raise ValueError("Vector must be comma-separated numbers (e.g., 1.0,0.5,0.0).")

def print_help():
    print("\nVectorStore CLI Commands:")
    print("  add <id> <vector_csv> [json_metadata]         - Add or update a vector")
    print("  get <id>                                      - Retrieve vector by ID")
    print("  delete <id>                                   - Delete vector by ID")
    print("  search <vector_csv> <k> [metric] [backend]   - Dense vector search (backend: exact|hnsw)")
    print("  keyword <query_text> <k>                      - BM25 metadata keyword search")
    print("  hybrid <vector_csv> <query_text> <k> [backend]- RRF hybrid search")
    print("  toggle-hnsw                                   - Enable/disable HNSW graph backend")
    print("  benchmark [num_vectors] [dim]                - Run latency & recall performance comparison")
    print("  save <filepath>                               - Save store state to JSON")
    print("  load <filepath>                               - Load store state from JSON")
    print("  help                                          - Show command menu")
    print("  exit                                          - Quit CLI session\n")

def run_benchmark(num_vectors: int = 500, dim: int = 32):
    """Run an automated performance and recall evaluation comparing Exact vs HNSW search."""
    print(f"\n[Benchmark] Generating {num_vectors} random vectors (dimension={dim})...")
    temp_store = VectorStore(use_hnsw=True)
    random.seed(42)

    for i in range(num_vectors):
        vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
        temp_store.add(f"bench_{i}", vec)

    query_vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    k = min(5, num_vectors)

    # 1. Exact Search Latency
    start_exact = time.perf_counter()
    exact_res = temp_store.search(query_vec, k=k, backend="exact")
    exact_time_ms = (time.perf_counter() - start_exact) * 1000.0

    # 2. HNSW Search Latency
    start_hnsw = time.perf_counter()
    hnsw_res = temp_store.search(query_vec, k=k, backend="hnsw")
    hnsw_time_ms = (time.perf_counter() - start_hnsw) * 1000.0

    # 3. Recall Calculation
    exact_ids = set(r["id"] for r in exact_res)
    hnsw_ids = set(r["id"] for r in hnsw_res)
    overlap = len(exact_ids.intersection(hnsw_ids))
    recall = (overlap / k) * 100.0 if k > 0 else 0.0

    print("--- Benchmark Summary ---")
    print(f"Dataset Size:        {num_vectors} vectors | Dim: {dim} | k: {k}")
    print(f"Exact Search Time:   {exact_time_ms:.3f} ms")
    print(f"HNSW Search Time:    {hnsw_time_ms:.3f} ms")
    print(f"HNSW Recall@{k}:       {recall:.1f}%\n")

def main():
    store = VectorStore(use_hnsw=True)
    print("VectorStore CLI (HNSW Graph & Hybrid Search Active)")
    print("Type 'help' for commands, 'exit' to quit.\n")

    while True:
        try:
            line = input("vectorstore> ").strip()
            if not line:
                continue
            if line.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            args_str = parts[1] if len(parts) > 1 else ""

            if cmd == "help":
                print_help()

            elif cmd == "toggle-hnsw":
                store.use_hnsw = not store.use_hnsw
                if store.use_hnsw and store.hnsw_index is None:
                    store._rebuild_hnsw()
                status = "ENABLED" if store.use_hnsw else "DISABLED"
                print(f"HNSW search backend is now {status}.")

            elif cmd == "add":
                sub_parts = args_str.split(maxsplit=2)
                if len(sub_parts) < 2:
                    print("Usage: add <id> <vector_csv> [json_metadata]")
                    continue
                doc_id, vec_csv = sub_parts[0], sub_parts[1]
                meta = json.loads(sub_parts[2]) if len(sub_parts) > 2 else {}
                vector = parse_vector(vec_csv)
                store.add(doc_id, vector, meta)
                print(f"Added vector '{doc_id}'.")

            elif cmd == "get":
                if not args_str:
                    print("Usage: get <id>")
                    continue
                res = store.get(args_str.strip())
                if res:
                    print(json.dumps(res, indent=2))
                else:
                    print(f"Vector ID '{args_str}' not found.")

            elif cmd == "delete":
                if not args_str:
                    print("Usage: delete <id>")
                    continue
                if store.delete(args_str.strip()):
                    print(f"Deleted vector '{args_str}'.")
                else:
                    print(f"Vector ID '{args_str}' not found.")

            elif cmd == "search":
                sub_parts = args_str.split()
                if len(sub_parts) < 2:
                    print("Usage: search <vector_csv> <k> [metric] [backend]")
                    continue
                vec = parse_vector(sub_parts[0])
                k = int(sub_parts[1])
                metric = sub_parts[2] if len(sub_parts) > 2 else "cosine"
                backend = sub_parts[3] if len(sub_parts) > 3 else "exact"

                results = store.search(vec, k=k, metric=metric, backend=backend)
                print(f"\nTop {len(results)} results ({metric}, {backend} backend):")
                for r in results:
                    print(f"  [{r['id']}] score: {r['score']:.4f} | metadata: {r['metadata']}")
                print()

            elif cmd == "keyword":
                sub_parts = args_str.rsplit(maxsplit=1)
                if len(sub_parts) < 2:
                    print("Usage: keyword <query_text> <k>")
                    continue
                query_text = sub_parts[0]
                k = int(sub_parts[1])
                results = store.keyword_search(query_text, k=k)
                print(f"\nTop {len(results)} BM25 results for '{query_text}':")
                for r in results:
                    print(f"  [{r['id']}] score: {r['score']:.4f} | metadata: {r['metadata']}")
                print()

            elif cmd == "hybrid":
                sub_parts = args_str.split()
                if len(sub_parts) < 3:
                    print("Usage: hybrid <vector_csv> <query_text> <k> [backend]")
                    continue
                vec = parse_vector(sub_parts[0])
                
                backend = sub_parts[-1] if sub_parts[-1] in ["exact", "hnsw"] else "exact"
                k_idx = -2 if sub_parts[-1] in ["exact", "hnsw"] else -1
                k = int(sub_parts[k_idx])
                
                text_tokens = sub_parts[1:k_idx]
                query_text = " ".join(text_tokens)

                results = store.hybrid_search(vec, query_text, k=k, backend=backend)
                print(f"\nTop {len(results)} hybrid results ({backend} backend):")
                for r in results:
                    print(f"  [{r['id']}] rrf_score: {r['rrf_score']:.4f} | metadata: {r['metadata']}")
                print()

            elif cmd == "benchmark":
                sub_parts = args_str.split()
                num_v = int(sub_parts[0]) if len(sub_parts) > 0 else 500
                dim_v = int(sub_parts[1]) if len(sub_parts) > 1 else 32
                run_benchmark(num_v, dim_v)

            elif cmd == "save":
                if not args_str:
                    print("Usage: save <filepath>")
                    continue
                store.save_to_json(args_str.strip())
                print(f"Saved store state to '{args_str.strip()}'.")

            elif cmd == "load":
                if not args_str:
                    print("Usage: load <filepath>")
                    continue
                store.load_from_json(args_str.strip())
                print(f"Loaded store state from '{args_str.strip()}'.")

            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()