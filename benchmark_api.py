import asyncio
import time
import random
import statistics
import httpx

BASE_URL = "http://localhost:8000"
DIM = 16
NUM_PRELOAD_VECTORS = 1000
NUM_BENCH_REQUESTS = 200
CONCURRENCY = 20

def generate_vector(dim: int = DIM) -> list[float]:
    return [round(random.uniform(-1.0, 1.0), 4) for _ in range(dim)]

async def preload_data(client: httpx.AsyncClient):
    print(f"Preloading {NUM_PRELOAD_VECTORS} vectors into vector store...")
    records = [
        {
            "id": f"bench_doc_{i}",
            "vector": generate_vector(),
            "metadata": {"category": "tech" if i % 2 == 0 else "finance", "text": f"sample document {i}"}
        }
        for i in range(NUM_PRELOAD_VECTORS)
    ]
    # Insert in batches of 200
    for i in range(0, NUM_PRELOAD_VECTORS, 200):
        batch = records[i:i+200]
        resp = await client.post(f"{BASE_URL}/vectors/batch", json={"records": batch})
        resp.raise_for_status()
    print("Preload complete.\n")

async def measure_request(client: httpx.AsyncClient, endpoint: str, payload: dict) -> float:
    start = time.perf_counter()
    resp = await client.post(f"{BASE_URL}{endpoint}", json=payload)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if resp.status_code != 200:
        raise RuntimeError(f"Request failed with status {resp.status_code}: {resp.text}")
    return latency_ms

async def run_concurrent_benchmarks(client: httpx.AsyncClient, name: str, endpoint: str, payload_fn, total_reqs: int, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def worker():
        async with semaphore:
            return await measure_request(client, endpoint, payload_fn())

    start_total = time.perf_counter()
    tasks = [asyncio.create_task(worker()) for _ in range(total_reqs)]
    latencies = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_total

    latencies.sort()
    rps = total_reqs / total_time
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(f"=== {name} Performance ===")
    print(f"Requests: {total_reqs} | Concurrency: {concurrency}")
    print(f"Throughput: {rps:.2f} req/sec | Total Time: {total_time:.3f}s")
    print(f"Latency p50: {p50:.2f} ms")
    print(f"Latency p95: {p95:.2f} ms")
    print(f"Latency p99: {p99:.2f} ms\n")

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check server availability
        try:
            res = await client.get(f"{BASE_URL}/")
            res.raise_for_status()
        except Exception as e:
            print(f"Error connecting to FastAPI server at {BASE_URL}. Ensure container or app is running.")
            return

        await preload_data(client)

        # 1. Benchmark Exact Dense Search
        await run_concurrent_benchmarks(
            client=client,
            name="Dense Search (Exact)",
            endpoint="/search/dense",
            payload_fn=lambda: {"query_vector": generate_vector(), "k": 5, "backend": "exact"},
            total_reqs=NUM_BENCH_REQUESTS,
            concurrency=CONCURRENCY
        )

        # 2. Benchmark HNSW ANN Search
        await run_concurrent_benchmarks(
            client=client,
            name="Dense Search (HNSW)",
            endpoint="/search/dense",
            payload_fn=lambda: {"query_vector": generate_vector(), "k": 5, "backend": "hnsw"},
            total_reqs=NUM_BENCH_REQUESTS,
            concurrency=CONCURRENCY
        )

        # 3. Benchmark Hybrid Search
        await run_concurrent_benchmarks(
            client=client,
            name="Hybrid Search (HNSW + BM25)",
            endpoint="/search/hybrid",
            payload_fn=lambda: {"query_vector": generate_vector(), "query_text": "sample document", "k": 5, "backend": "hnsw"},
            total_reqs=NUM_BENCH_REQUESTS,
            concurrency=CONCURRENCY
        )

if __name__ == "__main__":
    asyncio.run(main())