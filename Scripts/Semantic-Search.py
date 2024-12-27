import chromadb, gdown
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, os
from Scripts.BM25 import load_queries, load_relevance_judgements, calculate_metrics, save_results

'''
This script cretes a collection in ChromaDB using the SentenceTransformer model 
and then evaluates the retrieval performance.
'''

def create_collection(collection_name, path_param):	
    '''
    This function creates a collection in ChromaDB using the SentenceTransformer model.
    '''
    corpus_content = []
    with open("./train.docs", "r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            if not line: continue
            corpus_content.append(line)


    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    client = chromadb.PersistentClient(path=path_param)

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        metadata={"hnsw:space": "cosine"} 
    )  
    
    chromadb_doc_ids = []
    chromadb_documents = []
    for line in corpus_content:
        parts = line.split("\t", 1)
        if len(parts) != 2: print(f"Line with invalid format: {line}")
        doc_id, content = parts[0], parts[1]

        chromadb_doc_ids.append(doc_id)
        chromadb_documents.append(f"{content}")
    
    chromadb_embeddings = model.encode(chromadb_documents, batch_size=100, show_progress_bar=True, device="cpu")     
    document_batches = get_batches(chromadb_documents)
    ids_batches = get_batches(chromadb_doc_ids)
    embedding_batches = get_batches(chromadb_embeddings)

    for i in range(len(document_batches)):
        documents = document_batches[i]
        doc_ids = ids_batches[i]
        embeddings = embedding_batches[i]

        # Add the documents, ids and embeddings to the collection
        collection.add(
            documents=documents,
            ids=doc_ids,
            embeddings=embeddings
        )

def get_batches(lista, chunk_size=100):
    '''
    It splits a list into chunks of a given size.
    '''
    return [lista[i:i + chunk_size] for i in range(0, len(lista), chunk_size)]


def connectToDB(collection_name, path_param):
    '''
    This function connects to an existing collection in ChromaDB.
    It returns the collection object, which is an object that allows to query the collection.
    '''
    client = chromadb.PersistentClient(path=path_param)
    existing_collections = client.list_collections()
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Check if the collection exists
    if collection_name in [col.name for col in existing_collections]:
        collection = client.get_collection(
            collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        )

        return collection
    else:
        print(f"{collection_name} doesn't exist! You need to create it.")
        return None


def main_run(collection, max_results=100):
    '''
    This function runs the queries on the collection and returns the run.
    '''
    queries = load_queries()
    run = {}

    for query in queries:
        query_id = query["id"]
        query_text = query["query"]

        results = collection.query(
            query_texts=[query_text],
            n_results=max_results
        )

        doc_info_list = []
        for i in range(len(results['ids'][0])): 
            doc_id = results['ids'][0][i]
            doc_content = results['documents'][0][i]  
            doc_score = results['distances'][0][i]  

            doc_info_list.append({
                "id": doc_id,
                "content": doc_content,  
                "score": float(doc_score)
            })

        run[query_id] = doc_info_list

    return run


if __name__ == "__main__":
    try:
        # It is executed only once to create the collection, it takes  time
        # create_collection("TREC-COVID_collection", "./chromadb-storage/") 

        url = "https://drive.google.com/uc?id=1tme7R3L8QYt8kzlDwzIpayixbl0_mA-U"
        file_id = "1tme7R3L8QYt8kzlDwzIpayixbl0_mA-U"
        output = "chromadb-storage.tar"

        gdown.download(url, output, quiet=False)
        destination_folder = "chromadb-storage"
        os.system(f"tar -xvf {output}")

        relevance_judgements = load_relevance_judgements()
        collection = connectToDB("TREC-COVID_collection", "./chromadb-storage/")
        run = main_run(collection)
        
        metrics = calculate_metrics(run, relevance_judgements)
        save_results(run, relevance_judgements, "results-ex5", "chromadb_results", metrics)
        print("Results saved.")

    except Exception as e:
        print("An unexpected error occurred.")
    