from evaluation import RetrievalCase, evaluate_retrieval
from retrieval import InMemoryVectorIndex


# Hand-authored toy vectors keep the example deterministic and offline.
# In a real system these would come from an embedding provider.
index = InMemoryVectorIndex()
index.add(
    "banff",
    "Banff National Park has mountain hiking and alpine lakes.",
    [0.98, 0.05, 0.02],
    metadata={"country": "Canada", "theme": "outdoors"},
)
index.add(
    "tokyo",
    "Tokyo has an extensive subway and rail network.",
    [0.02, 0.99, 0.03],
    metadata={"country": "Japan", "theme": "transit"},
)
index.add(
    "montreal",
    "Montreal is known for cafes, pastries, and food culture.",
    [0.03, 0.05, 0.97],
    metadata={"country": "Canada", "theme": "food"},
)

results = index.search([1.0, 0.0, 0.0], k=2)
for result in results:
    print(result.document_id, round(result.score, 3), "-", result.text)

metrics = evaluate_retrieval(
    index,
    [
        RetrievalCase(
            name="mountain hiking",
            query_embedding=[1.0, 0.0, 0.0],
            relevant_document_ids=frozenset({"banff"}),
        ),
        RetrievalCase(
            name="public transit",
            query_embedding=[0.0, 1.0, 0.0],
            relevant_document_ids=frozenset({"tokyo"}),
        ),
    ],
    k=2,
)
print("hit_rate@2:", metrics.hit_rate_at_k)
print("mrr:", metrics.mean_reciprocal_rank)
