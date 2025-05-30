# search_emojis.py

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ⚙️ Khởi tạo Qdrant và model
client = QdrantClient(host="localhost", port=6333)
collection_name = "emoji"
model = SentenceTransformer('all-MiniLM-L6-v2')

print("\n🔎 Bạn có thể tìm emoji bằng mô tả. Gõ 'exit' để thoát.")

while True:
    query = input("\nNhập mô tả emoji (VD: happy face): ")
    if query.strip().lower() == "exit":
        break

    vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=collection_name,
        query=vector,
        limit=1,
        with_payload=True
    )

    print("\n📌 Kết quả gần nhất:")
    print(results)
    for i, r in enumerate(results.points, start=1):
        desc = r.payload.get("desc", "N/A")
        path = r.payload.get("path", "N/A")
        print(f"{i}. 📝 {desc}")
        print(f"   🖼️  {path}")
