
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import sys
import os
import re
import chardet
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append('/home/steffen88/repos/annotations/Korpusaufbereitung_Zwischenverfügungen')
import GWG_shareholder_check
import string
punctuation_set = set(string.punctuation)

input_path = r'/home/steffen88/Documents/PHD/Embedding_Queries/clean_docs_mus_test'
output_path = r'/home/steffen88/Documents/PHD/Embedding_Queries/corpus_for_embedding.txt'




tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()

text = "The bank raised interest rates."



def split_sent_mus(content):
    new_text = re.sub(r"([a-z])(-\n\n?)([a-z])", r"\1\3", content)
    new_text = new_text.replace("\n", ' ')
    sentences = GWG_shareholder_check.preprocess_raw(new_text)
    sentences = GWG_shareholder_check.restore_abbreviations(sentences)
    return sentences



def get_token_embeddings(text):
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
        #print("token in loop",token)
        # Skip special tokens
        if token in tokenizer.all_special_tokens or (start == end == 0):
            continue

        is_subword = token.startswith("##")

        # New word boundary: if not a subword or there's a gap in text
        if (not is_subword and start > last_end) or token in punctuation_set:
            if current_embeds:
                word_vec = torch.stack(current_embeds).mean(dim=0)
                word_embeddings.append(word_vec)
                word_tokens.append(current_word)
                current_word = text[start:end]
                #print("current word", current_word)
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

    

    return subword_embeddings, token_strings, tokens["offset_mapping"][0], word_tokens, word_embeddings


embeddings, token_strings, offsets, word_tokens, word_embeddings = get_token_embeddings(text)

print("embeddings: " , embeddings)
print("token strings: ", token_strings)
print("offsets:", offsets)
print("word tokens:", word_tokens)
print("word embeddings:", len(word_embeddings))



embedding_vectors = []         # list of np.array of shape (768,)
corpus_token_ids = []          # list of CWB token positions (integers)
token_info_by_id = {}          # corpus_token_id -> token data

token_counter = 0
sent_counter = 0



with open(output_path, 'w', encoding="utf-8") as output_file:
    output_file.write(f"<corpus>" + '\n')
    for i,filename in enumerate(os.listdir(input_path)):
                print(f"Processing file {filename}, file  {i} out of {len(os.listdir(input_path))}")
                if filename.endswith('.txt'):
                    file_path = os.path.join(input_path, filename)
                    try:
                        with open(file_path, "rb") as f:
                            content = f.read()
                            encoding = chardet.detect(content)['encoding']
                        with open(file_path, 'r', encoding = encoding) as file:
                            content = file.read()
                        
                    except UnicodeDecodeError as e:
                        logger.exception(f"Failed to read {filename}: {e}")
                    output_file.write(f"<text id={filename[:-4]}>" + '\n')
                    sentences = split_sent_mus(content)
                    for sent_ind, sent in enumerate(sentences):
                        output_file.write(f"<s>" + '\n')
                       
                        embeddings, token_strings, offsets, word_tokens, word_embeddings = get_token_embeddings(sent)
                        for i, (vec, tok) in enumerate(zip(word_embeddings, word_tokens)):
                        
                            
                            output_file.write(tok + '\n')

                            # Save with CWB token index as key
                            embedding_vectors.append(vec.numpy())
                            corpus_token_ids.append(token_counter)

                            token_info_by_id[token_counter] = {
                                "token": tok,
                                "sentence": sent,
                                "bert_idx": i,
                                "sent_ind": sent_ind,
                                "text_id" : filename.replace(".txt",'')
                            }

                            token_counter += 1
                        sent_ind += 1
                        output_file.write(f"</s>" + '\n')
                    output_file.write(f"</text>" + '\n')
    output_file.write(f"</corpus>")
    
    
    

for num in range(10):
    print(f"token_info_by_id[{num}]: ", token_info_by_id[num]["token"])


# Convert to arrays
vecs_np = np.array(embedding_vectors).astype("float32")
ids_np = np.array(corpus_token_ids).astype("int64")

# Normalize if using cosine similarity
faiss.normalize_L2(vecs_np)

# Create FAISS index with ID mapping
index = faiss.IndexFlatIP(vecs_np.shape[1])           # or L2 depending on use
index_id = faiss.IndexIDMap(index)
index_id.add_with_ids(vecs_np, ids_np)
faiss.write_index(index_id, "/home/steffen88/Documents/PHD/Embedding_Queries/bert_token_index.faiss")






# embedding_dim = 768  # for BERT
# index = faiss.IndexFlatIP(embedding_dim)  # Use cosine similarity (after normalization)

# Normalize vectors if using cosine similarity
# embedding_matrix = np.vstack(embedding_vectors).astype("float32")
# faiss.normalize_L2(embedding_matrix)

# # Add to FAISS
# index.add(embedding_matrix)

# # Optional: Save the index
# faiss.write_index(index, "/home/steffen88/repos/CQP_Embeddings/corpus_embeddings.faiss")



# def encode_embedding_corpus(input_path, output_path):
#     nlp = spacy.load("de_core_news_sm")
#     with open(output_path, 'w', encoding="utf-8") as output_file:
#         output_file.write(f"<corpus>" + '\n')
#         for i,filename in enumerate(os.listdir(input_path)):
#             print(f"Processing file {filename}, file  {i} out of {len(os.listdir(input_path))}")
#             if i > 1000:
#                 break
#             if filename.endswith('.txt'):
#                 file_path = os.path.join(input_path, filename)
#                 try:
#                     with open(file_path, "rb") as f:
#                         content = f.read()
#                         encoding = chardet.detect(content)['encoding']
#                     with open(file_path, 'r', encoding = encoding) as file:
#                         content = file.read()
                       
#                 except UnicodeDecodeError as e:
#                     logger.exception(f"Failed to read {filename}: {e}")
             
                
#                 output_file.write(f"<text id={filename[:-4]}>" + '\n')  
#                 sentences = GWG_shareholder_check.preprocess_raw(content)
#                 sentences = GWG_shareholder_check.restore_abbreviations(sentences)
#                 #print("sentences:\n", sentences)
#                 for sent in sentences:
#                     # sent = restore_abbr_periods_per_sent(sent, reversed_dict)
#                     sent = sent.strip()
#                     if sent == '\n':
#                         continue
#                     pos_lems= [(token.text,token.tag_, token.lemma_) for token in nlp(sent) if token.text != '\n']
#                     spacy_tokens = [i[0] for i in pos_lems]
#                     sent = ' '.join(spacy_tokens)
#                     ner_tagged = ner_tag_sentence(sent,pos_lems)
#                     output_file.write(f"<s>" + '\n')
#                     for i,item in enumerate(ner_tagged):
#                         if isinstance(item, str):
#                             output_file.write(item+ "\n")
#                         else:
#                             output_file.write(item[0]+"\t"+item[1]+"\t"+item[2]+ "\n")
#                     output_file.write(f"</s>" + '\n')


#                 output_file.write(f"</text>" + '\n')
#         output_file.write(f"</corpus>")
#         print(f"We found {invalid} invalid docs")




#to do; integration of similarity search into query
"""create method that takes in a query with a place holder for embedding token
may we can give a temporary syntax like so:
query string arg:  [] [] EMBEDDING [] []
embedding arg: look up in database, bank as in river bank
similarity threshold arg: 0.8
get all sentences with embeddings that meet condition


or 

get all sentences that meet non-embedding parts of query and have embedding token be wild any token,
of those, filter for similarity at index x which is the embedding token that our method should accept 


so here is the approach I'm toying with in my head:
I calculate all the context sensitive token embeddings for my corpus and store them (probably with faiss) and encode the corpus with a embedding id p-attribute. 
I wirte a method that takes in a target token and its context, maybe as sentence. The method calculates the target tokens contextual embedding. 
The method also takes a query as an argument that has a temporary position for the target embedding.  Then the target position is just replaced by an any token. Maybe we can specifiy a pos. This query can then be used to match a subset of the coprus sentences that serve as the basis for the similarity comparison.  We find all the tokens that correspond to the target position in this subsets and do the similarity calculation on those and keep the ones that meet the threshold. Does that sound feasible? 




Summary of Pipeline

Preprocesses the entire corpus:

Compute contextual embeddings for every token (e.g., using BERT).

Store these embeddings in FAISS (for fast similarity search).

Assign each embedding an ID, and encode this ID as a p-attribute in your CWB corpus — e.g., embed_id.

Takes a query token + its context (e.g., a sentence):

Compute a contextual embedding for the token in context (important: same model, same tokenization as corpus).

This is your semantic probe vector.

Takes a CQP query pattern:

One token position in the query corresponds to the probe (e.g., [pos = "NN"] or [word]).

You rewrite this as [pos = "NN"] ->probe to label it.

Runs the CQP query:

Finds sentences/contexts where this pattern matches.

Extracts all tokens that occurred at the probe position across matches.

Looks up the embed_id of each token at the probe position.

Retrieves the actual embedding vector for those IDs from FAISS.

Compares each retrieved embedding to the probe vector:

Using cosine similarity or dot product.

Keeps only those above a threshold (or top-k).

Returns:

Matching sentences or token instances, ranked or filtered by semantic similarity.
"""