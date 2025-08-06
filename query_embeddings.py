import pickle
import faiss
import subprocess
import re
import shlex
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import string
punctuation_set = set(string.punctuation)

with open("/home/steffen88/Documents/PHD/Embedding_Queries/token_info_by_id.pkl", "rb") as f:
    token_info_by_id = pickle.load(f)


index_id = faiss.read_index("/home/steffen88/Documents/PHD/Embedding_Queries/bert_token_index.faiss")



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

def get_token_embedding(text, target_word):
    tokens = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    #print("tokens from bert: ", tokens)
    #print('tokens["input_ids"][0]', tokens["input_ids"][0])
    model_inputs = {k: tokens[k] for k in ['input_ids', 'attention_mask', 'token_type_ids'] if k in tokens}

    with torch.no_grad():
        output = model(**model_inputs)

    # Remove batch dim: [1, seq_len, hidden] → [seq_len, hidden]
    subword_embeddings = output.last_hidden_state.squeeze(0)
    #print("lenght of subword embeddings", len(subword_embeddings))
    offsets = tokens["offset_mapping"][0]
    #print("offsets", offsets)
    # Get token ids for mapping back to original text (optional)
    token_strings = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    #print("length of token strings: ", len(token_strings))
    
    word_embeddings = []
    word_tokens = []
    current_word = ""
    current_embeds = []
    current_start = None
    last_end = -1

    for i, (token, offset) in enumerate(zip(token_strings, offsets)):
        start, end = offset.tolist()
        # Skip special tokens
        if token in tokenizer.all_special_tokens or (start == end == 0):
            continue

        is_subword = token.startswith("##")
        is_punct = token in punctuation_set
        # New word boundary: if not a subword or there's a gap in text
        if (not is_subword and start > last_end) or is_punct:
            if current_embeds:
                word_vec = torch.stack(current_embeds).mean(dim=0)
                word_embeddings.append(word_vec)
                word_tokens.append(current_word)
                current_word = text[start:end]
                #print("current word", current_word)
                current_embeds = [subword_embeddings[i]]
            current_word = token if is_punct else text[start:end]
            current_embeds = [subword_embeddings[i]]
                
                
        else:
            # Same word or subword continuation
            if is_subword:
                current_word += token[2:]  # remove "##"
            else:
                current_word += text[start:end]
            current_embeds.append(subword_embeddings[i])

        last_end = end

    # Add last word
    if current_embeds:
        #print("current word", current_word)
        word_vec = torch.stack(current_embeds).mean(dim=0)
        word_embeddings.append(word_vec)
        word_tokens.append(current_word)
        
    target_index = None
    for i, tok in enumerate(word_tokens):
        if tok == target_word:
            target_index = i
            break

    print("\n---")
    print("text sent: ", text)
    print("word tokens: ", word_tokens)
    print("len of word tokens", len(word_tokens))
    print("len of word embeddings", len(word_embeddings))
    print("---\n")
    return word_tokens[i], word_embeddings[i]



target_token, target_embedding = get_token_embedding(
    text="Die Gesellschaft wurde als Unternehmen gegründet.",
    target_word="Gesellschaft"
)

print("target embedding: ", target_embedding)

def cqp_query(query, corpus_name="CWB_EMBEDDINGS", registry="/home/steffen88/Documents/PHD/Embedding_Queries/cwb_embedding_test/registry"):
    commands = [
        "set AutoShow on;",  # Re-enable automatic display
        f"{corpus_name};",
        query,
    ]
    
    full_query = '\n'.join(commands)
    #full_query = f'{corpus_name};\n[word="{query}"];\n'
    
    cmd = ['cqp', '-r', registry, '-c']
    
    print("Running CQP command:", ' '.join(cmd))
    print("Sending query:", repr(full_query))
    
    result = subprocess.run(cmd, input=full_query, capture_output=True, text=True)
    
    # print("Raw CQP output:\n", result.stdout)
    # print("Raw stderr:\n", result.stderr)
    # print("Return code:", result.returncode)
    
    matches = []
    for line in result.stdout.splitlines():
        m = re.match(r"^\s*(\d+):", line)
        if m:
            matches.append(int(m.group(1)))
    
    return matches




def create_faiss_subset(matches):

    matched_ids_np = np.array(matches, dtype=np.int64)


    vecs = []
    ids = []

    for tok_id in matches:
        if tok_id in token_info_by_id:
            vec = token_info_by_id[tok_id]["vector"]  # Should be a 1D np.array
            vecs.append(vec)
            ids.append(tok_id)
            
            
    vecs_np = np.array(vecs).astype("float32")
    ids_np = np.array(ids).astype("int64")

    faiss.normalize_L2(vecs_np)

    subset_index = faiss.IndexFlatIP(vecs_np.shape[1])  # IP = inner product (for cosine)
    subset_index_idmap = faiss.IndexIDMap(subset_index)
    subset_index_idmap.add_with_ids(vecs_np, ids_np)
    
    return subset_index_idmap



def return_query_faiss_subset(query):
    matches = cqp_query(query)
    subset_index_idmap=  create_faiss_subset(matches)
    return subset_index_idmap






def query_embedding_similarity(query,target_word, target_context,  threshold = 0.8):
    
    
    faiss_subset = return_query_faiss_subset(query)
    # Get embedding of word "Gesellschaft" in clarifying sentence
    token, query_vec = get_token_embedding(target_context, target_word)
    query_vec_np = query_vec.detach().cpu().numpy().reshape(1, -1)
    faiss.normalize_L2(query_vec_np)

    # Search for top 50 most similar tokens in the subset FAISS index
    D, I = faiss_subset.search(query_vec.reshape(1, -1), k=50)

    # Filter results by cosine similarity threshold (e.g., 0.8)
    filtered = [(i, float(d)) for i, d in zip(I[0], D[0]) if d >= threshold]
    enriched_results = []
    for token_id, score in filtered:
        info = token_info_by_id[int(token_id)]  # cast np.int64 to int, just to be safe
        enriched_results.append({
            "token_id": int(token_id),
            "token": info["token"],
            "sentence": info["sentence"],
            "bert_idx": info["bert_idx"],
            "sent_ind": info["sent_ind"],
            "text_id": info["text_id"],
            "similarity": score
        })
    
    
    return enriched_results







emb_matches = query_embedding_similarity('[lemma= "Gesellschaft"];',"Gesellschaft", "Die Gesellschaft wurde gegründet.")

#print("Matches based on embedding similarity: ", emb_matches)



# optionally print the top results
for result in emb_matches:
    print(f"[{result['similarity']:.3f}] {result['token']} at index {result['token_id']} from text {result['text_id']} in sentence: {result['sentence']}")