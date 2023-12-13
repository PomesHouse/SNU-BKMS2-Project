# faiss index 불러오기 
import faiss
faiss_index = faiss.read_index("./faiss_index_file/faiss_index_file_students")
print(faiss_index)