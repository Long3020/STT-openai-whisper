from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from uuid import uuid5, NAMESPACE_DNS
import os
import json

# ⚙️ Khởi tạo Qdrant
client = QdrantClient(host="localhost", port=6333)
collection_name = "emoji"

# ⚠️ Nếu cần xóa collection cũ, bỏ comment dòng này:
client.delete_collection(collection_name=collection_name)

# ✅ Tạo collection nếu chưa có
if not client.collection_exists(collection_name=collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("✅ Tạo collection 'emoji'")
else:
    print("ℹ️ Collection 'emoji' đã tồn tại")

# 🤖 Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 📂 Thư mục gốc chứa các thư mục con của JoyPixels
emoji_root = r"D:/Whisper/JoyPixels/png/labeled/128"
points = []

# 🔍 Duyệt đệ quy tất cả ảnh PNG trong các thư mục con
for root, _, files in os.walk(emoji_root):
    for file in files:
        if file.endswith(".png"):
            unicode_name = file[:-4].lower()  # bỏ .png và chuyển lowercase
            emoji_path = os.path.join(root, file)

            description = f"emoji {unicode_name}"
            embedding = model.encode(description).tolist()

            emoji_id = str(uuid5(NAMESPACE_DNS, unicode_name))  # ID hợp lệ

            points.append(
                PointStruct(
                    id=emoji_id,
                    vector=embedding,
                    payload={
                        "desc": description,
                        "unicode": unicode_name,
                        "path": emoji_path
                    }
                )
            )

# ⬆️ Up dữ liệu
if points:
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Đã lưu {len(points)} emoji vào Qdrant!")
else:
    print("⚠️ Không tìm thấy ảnh PNG nào trong thư mục.")


# Load unicode ảnh png vào file json
# 📁 Thư mục gốc chứa các thư mục con
emoji_root_folder = r"D:/Whisper/JoyPixels/png/labeled/128"
output_file = "emojis.json"

# 📦 Kết quả gom nhóm theo tên thư mục
emoji_dict = {}

# 🔁 Duyệt qua từng thư mục con
for subdir in os.listdir(emoji_root_folder):
    subdir_path = os.path.join(emoji_root_folder, subdir)
    if os.path.isdir(subdir_path):
        unicodes = [
            file[:-4].lower()
            for file in os.listdir(subdir_path)
            if file.endswith(".png")
        ]
        emoji_dict[subdir] = unicodes

# 💾 Ghi ra file JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(emoji_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Đã ghi emoji unicode theo thư mục vào {output_file}")

