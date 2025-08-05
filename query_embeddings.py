import pickle
import faiss
import subprocess
import re
import shlex

with open("/home/steffen88/Documents/PHD/Embedding_Queries/token_info_by_id.pkl", "rb") as f:
    token_info_by_id = pickle.load(f)


index_id = faiss.read_index("/home/steffen88/Documents/PHD/Embedding_Queries/bert_token_index.faiss")


# def cqp_query(query, corpus_name=r"/home/steffen88/Documents/PHD/Embedding_Queries/cwb_embedding_test"):
#     # Run the CQP query through CCC
#     cmd = f'echo "{query};" | cqp -c {corpus_name}'
#     result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

#     matches = []
#     for line in result.stdout.splitlines():
#         # Match IDs inside [ ]
#         m = re.match(r"\[(\d+)\]", line.strip())
#         if m:
#             token_id = int(m.group(1))
#             matches.append(token_id)

#     return matches

# matches = cqp_query("und")
# print(matches)


# def cqp_query(query, corpus_name="CWB_EMBEDDINGS", registry="/home/steffen88/Documents/PHD/Embedding_Queries/cwb_embedding_test/registry"):
#     #cmd = f'echo "{query};" | cqp -r {registry} -c {corpus_name}'
#     cmd = f'echo "{query}"; | cqp -r {registry} -c {corpus_name}'

#     result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
#     print("Raw CQP output:")
#     print(result.stdout)
#     print("Raw stderr:")
#     print(result.stderr)

#     matches = []
#     for line in result.stdout.splitlines():
#         m = re.match(r"\[(\d+)\]", line.strip())
#         if m:
#             token_id = int(m.group(1))
#             matches.append(token_id)

#     return matches



# def cqp_query(query, corpus_name, registry):
#     full_cmd = f'echo {shlex.quote(query)}; | cqp -r {shlex.quote(registry)} -c {shlex.quote(corpus_name)}' 

#     print("Running shell command:")
#     print(full_cmd)  # <-- Useful for debugging

#     result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

#     print("Raw CQP output:\n", result.stdout)
#     print("Raw stderr:\n", result.stderr)

#     # Parse matches (e.g. lines like "117: some text...")
#     matches = []
#     for line in result.stdout.splitlines():
#         m = re.match(r"^\s*(\d+):", line)
#         if m:
#             matches.append(int(m.group(1)))

#     return matches



# # Example
# # token_ids = cqp_query("word = 'bank'")
# # print(token_ids)  # [12, 53, 201, ...]


# matches = cqp_query("und", corpus_name="CWB_EMBEDDINGS", registry="/home/steffen88/Documents/PHD/Embedding_Queries/cwb_embedding_test/registry")
# print(matches)



# def cqp_query(query, corpus_name, registry):
#     full_query = f'{corpus_name}; [word="{query}"];'
#     cmd = f'echo {shlex.quote(full_query)} | cqp -r {shlex.quote(registry)} -c'

#     print("Running shell command:")
#     print(cmd)

#     result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

#     print("Raw CQP output:\n", result.stdout)
#     print("Raw stderr:\n", result.stderr)

#     matches = []
#     for line in result.stdout.splitlines():
#         m = re.match(r"^\s*(\d+):", line)
#         if m:
#             matches.append(int(m.group(1)))

#     return matches

# matches = cqp_query("und", corpus_name="CWB_EMBEDDINGS", registry="/home/steffen88/Documents/PHD/Embedding_Queries/cwb_embedding_test/registry")
# print(matches)





def cqp_query(query, corpus_name, registry):
    commands = [
        "set AutoShow on;",  # Re-enable automatic display
        f"{corpus_name};",
        f'[word="{query}"];',
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

matches = cqp_query("und", corpus_name="CWB_EMBEDDINGS", registry="/home/steffen88/Documents/PHD/Embedding_Queries/cwb_embedding_test/registry")
print(matches)