import asyncio
import aiohttp
import time

async def hit_api(session, question, user_id):
    url = "http://127.0.0.1:8000/ask"
    payload = {"question": question}
    
    print(f"[user {user_id}] firing request: '{question[:30]}...'")
    start_time = time.perf_counter()
    
    try:
        async with session.post(url, json=payload) as response:
            await response.json()
            elapsed = time.perf_counter() - start_time
            print(f"[user {user_id}] finished in {elapsed:.2f} seconds")
    except Exception as e:
        print(f"[user {user_id}] crashed! error: {e}")

async def main():
    # three distinct questions to prevent the model from just caching the exact same answer
    questions = [
        "what are the conditions for a problem to be in NP-Hard?",
        "explain the difference between NP Complete and NP Hard.",
        "Why is there no solution for NP Hard Problems?"
    ]
    
    print("=== starting concurrency load test ===")
    start_all = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, q in enumerate(questions):
            tasks.append(hit_api(session, q, i+1))
        
        # fire all requests at the exact same time
        await asyncio.gather(*tasks)
        
    total_time = time.perf_counter() - start_all
    print(f"=== load test complete in {total_time:.2f} seconds ===")

if __name__ == "__main__":
    asyncio.run(main())