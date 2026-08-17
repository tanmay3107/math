import json
import sys
from vector_store import VectorStore

def print_help():
    print("\n=================== VectorStore CLI Commands ===================")
    print("  add <id> <v1,v2,v3> [meta_json]  - Add/update vector (e.g. add doc1 1.0,0.5,0.0 {\"category\":\"tech\"})")
    print("  get <id>                         - Retrieve a vector and its metadata")
    print("  search <v1,v2,v3> <k> [metric]   - Search top-k vectors (metric: cosine|euclidean|manhattan)")
    print("  delete <id>                      - Delete a vector by ID")
    print("  save <filepath>                  - Save vector store state to a JSON file")
    print("  load <filepath>                  - Load vector store state from a JSON file")
    print("  help                             - Display command menu")
    print("  exit                             - Exit CLI")
    print("================================================================\n")

def main():
    store = VectorStore()
    print("VectorStore CLI Initialized. Type 'help' to see available commands.\n")

    while True:
        try:
            user_input = input("vectorstore> ").strip()
            if not user_input:
                continue

            parts = user_input.split(maxsplit=3)
            cmd = parts[0].lower()

            if cmd in ("exit", "quit"):
                print("Exiting VectorStore CLI. Goodbye!")
                break

            elif cmd == "help":
                print_help()

            elif cmd == "add":
                if len(parts) < 3:
                    print("Usage: add <id> <v1,v2,v3> [meta_json]")
                    continue
                vec_id = parts[1]
                try:
                    vector = [float(x) for x in parts[2].split(",")]
                except ValueError:
                    print("Error: Vector elements must be comma-separated floats (e.g. 1.0,0.5,0.0).")
                    continue

                metadata = None
                if len(parts) > 3:
                    try:
                        metadata = json.loads(parts[3])
                    except json.JSONDecodeError:
                        print("Error: Metadata must be valid JSON format (e.g. {\"key\":\"value\"}).")
                        continue

                store.add(vec_id, vector, metadata=metadata)
                print(f"Added vector '{vec_id}'.")

            elif cmd == "get":
                if len(parts) < 2:
                    print("Usage: get <id>")
                    continue
                res = store.get(parts[1])
                if res:
                    print(json.dumps(res, indent=2))
                else:
                    print(f"Vector ID '{parts[1]}' not found.")

            elif cmd == "search":
                if len(parts) < 3:
                    print("Usage: search <v1,v2,v3> <k> [metric]")
                    continue
                try:
                    query_vec = [float(x) for x in parts[1].split(",")]
                    k = int(parts[2])
                except ValueError:
                    print("Error: Query vector must be comma-separated floats and k must be an integer.")
                    continue

                metric = parts[3] if len(parts) > 3 else "cosine"
                try:
                    results = store.search(query_vec, k=k, metric=metric)
                    print(f"\nTop {len(results)} results using '{metric}':")
                    for idx, r in enumerate(results, start=1):
                        print(f"  {idx}. [{r['id']}] Score: {r['score']:.4f} | Metadata: {r['metadata']}")
                    print()
                except ValueError as err:
                    print(f"Search Error: {err}")

            elif cmd == "delete":
                if len(parts) < 2:
                    print("Usage: delete <id>")
                    continue
                if store.delete(parts[1]):
                    print(f"Deleted vector '{parts[1]}'.")
                else:
                    print(f"Vector ID '{parts[1]}' not found.")

            elif cmd == "save":
                if len(parts) < 2:
                    print("Usage: save <filepath>")
                    continue
                store.save_to_json(parts[1])
                print(f"Saved store to '{parts[1]}'.")

            elif cmd == "load":
                if len(parts) < 2:
                    print("Usage: load <filepath>")
                    continue
                try:
                    store.load_from_json(parts[1])
                    print(f"Loaded store from '{parts[1]}'.")
                except Exception as err:
                    print(f"Load Error: {err}")

            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for menu.")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break

if __name__ == "__main__":
    main()