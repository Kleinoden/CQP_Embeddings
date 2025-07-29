import pickle
import faiss


with open("/home/steffen88/Documents/PHD/Embedding_Queries/token_info_by_id.pkl", "rb") as f:
    token_info_by_id = pickle.load(f)


index_id = faiss.read_index("/home/steffen88/Documents/PHD/Embedding_Queries/bert_token_index.faiss")
