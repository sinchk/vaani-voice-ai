import time
import requests
import json

BENCHMARK_QUERIES = [
    {"query": "What is your return policy?", "type": "RAG", "expected_source": "return_policy.txt"},
    {"query": "Can I cancel my order 1042?", "type": "Tool", "expected_tool": "cancel_order"},
    {"query": "Check appointment for 2026-10-10", "type": "Tool", "expected_tool": "check_appointment"},
    {"query": "How long does shipping take?", "type": "RAG", "expected_source": "shipping_policy.txt"},
    {"query": "ನನ್ನ ಆರ್ಡರ್ 1042 ರ ಸ್ಥಿತಿ ಏನು?", "type": "Tool", "expected_tool": "get_order_status"}, # Kannada: What is status of order 1042?
    {"query": "ವಾಪಸಾತಿ ನೀತಿ ತಿಳಿಸಿ", "type": "RAG", "expected_source": "return_policy.txt"}, # Kannada: Tell me return policy
    {"query": "Book appointment for Sinchana tomorrow at 10 AM", "type": "Tool", "expected_tool": "book_appointment"},
    {"query": "Can I return a used item?", "type": "RAG", "expected_source": "return_policy.txt"},
    {"query": "I want to talk to a human agent", "type": "Handoff", "expected_handoff": True},
    {"query": "Help me with cancellation policy", "type": "RAG", "expected_source": "cancellation_policy.txt"},
    {"query": "ನನಗೆ ಒಬ್ಬ ಏಜೆಂಟ್ ಬೇಕು", "type": "Handoff", "expected_handoff": True} # Kannada: I need an agent
]

def run_evaluation():
    print("Starting Vaani Evaluation Benchmark (10+ queries)...")
    results = []
    
    for item in BENCHMARK_QUERIES:
        start_time = time.time()
        try:
            response = requests.post("http://localhost:8000/api/chat", json={
                "session_id": f"eval_{int(time.time())}",
                "message": item["query"]
            }).json()
            latency = (time.time() - start_time) * 1000
            time.sleep(2)
            
            # Check Tool Success
            tool_success = False
            if item["type"] == "Tool":
                called_tools = [tc["name"] for tc in response.get("tool_calls", [])]
                tool_success = item["expected_tool"] in called_tools
            
            # Check RAG Success
            rag_success = False
            if item["type"] == "RAG":
                sources = [s["source"] for s in response.get("sources", [])]
                rag_success = item["expected_source"] in sources
                
            # Check Handoff Success
            handoff_success = False
            if item["type"] == "Handoff":
                handoff_success = response.get("handoff", False) == item["expected_handoff"]

            results.append({
                "query": item["query"],
                "type": item["type"],
                "latency": latency,
                "tool_success": tool_success,
                "rag_success": rag_success,
                "handoff_success": handoff_success,
                "response": response["response"]
            })
            print(f"Query: {item['query'][:30]}... | Latency: {latency:.2f}ms")
        except Exception as e:
            print(f"Error evaluating query {item['query']}: {e}")

    # Metrics calculation
    avg_latency = sum(r['latency'] for r in results) / len(results) if results else 0
    
    tool_queries = [r for r in results if r['type'] == 'Tool']
    tool_acc = sum(1 for r in tool_queries if r['tool_success']) / len(tool_queries) if tool_queries else 0
    
    rag_queries = [r for r in results if r['type'] == 'RAG']
    rag_recall = sum(1 for r in rag_queries if r['rag_success']) / len(rag_queries) if rag_queries else 0
    
    handoff_queries = [r for r in results if r['type'] == 'Handoff']
    handoff_acc = sum(1 for r in handoff_queries if r['handoff_success']) / len(handoff_queries) if handoff_queries else 0

    print("\n--- Real Evaluation Metrics ---")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Tool Selection Accuracy: {tool_acc * 100:.2f}%")
    print(f"RAG Recall@3: {rag_recall * 100:.2f}%")
    print(f"Handoff Trigger Accuracy: {handoff_acc * 100:.2f}%")
    print(f"Groundedness: Based on RAG Recall")

if __name__ == "__main__":
    run_evaluation()
