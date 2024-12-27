import bm25s, Stemmer, json, os
from Scripts.Indexing import parse_nf_corpus, index_corpus, load_index

'''
This script runs the BM25 algorithm for the TREC-COVID dataset.
It uses the BM25 algorithm with different configurations.
''' 

'''
Global variables for the script.
'''
max_results = 100 
stemmer = Stemmer.Stemmer("english")
bm25_variants = ["robertson", "atire", "bm25l", "bm25+", "lucene"]

preprocessing_options = [
    {"stopwords": "en", "stemming": stemmer},  
    {"stopwords": None, "stemming": stemmer}, 
    {"stopwords": "en", "stemming": None}, 
    {"stopwords": None, "stemming": None}, ]



def load_queries(file_path="./train.nontopic-titles.queries"):
    '''
    We load the queries from the file queries.jsonl.
    '''
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)  
            if len(parts) != 2:
                print(f"Line with invalid format: {line}")
                continue

            query_id, query_text = parts[0], parts[1].lower()
            queries.append({"id": query_id, "query": query_text})
    return queries


def load_relevance_judgements(file_path="./train.3-2-1.qrel"):
    '''
    We load the relevance judgements from the file train.3-2-1.qrel.
    We only consider documents with relevance judgements greater than 0.
    '''
    relevance_judgements = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    print(f"Line with invalid format: {line}")
                    continue

                query_id, _, corpus_id, score = parts

                if int(score) > 1: # 0 are not in the query, 1 are marginally relevant
                    if query_id not in relevance_judgements:
                        relevance_judgements[query_id] = {}
                    relevance_judgements[query_id][corpus_id] = int(score)
            except ValueError:
                continue

    return relevance_judgements


def retrieve_documents(retriever, query_tokenized, max_results):
    '''
    We retrieve the documents for the given query.
    We return the document IDs and the relevance scores.
    '''
    results = retriever.retrieve(query_tokenized, corpus=retriever.corpus, 
        k=max_results, return_as="tuple", show_progress=True)
    return results.documents[0], results.scores[0]


def save_results(run, relevance_judgements, dir, filename, metrics):
    '''
    We save the results of the run in a JSON file.
    '''
    output_file = f"{dir}/{filename}.json"
    metrics = calculate_metrics(run, relevance_judgements)
    
    result_data = {
        "configuration": filename,
        "metrics": {
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "micro_f1": metrics["micro_f1"]
        },
        "queries": []
    }

    for query_id, doc_info_list in run.items():
        query_data = {
            "query_id": query_id,
            "documents_info": doc_info_list,
            "precision": metrics["query_metrics"][query_id]["precision"],
            "recall": metrics["query_metrics"][query_id]["recall"],
            "f1": metrics["query_metrics"][query_id]["f1"]
        }
        
        result_data["queries"].append(query_data)
    
    os.makedirs(dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4)


def calculate_metrics(run, relevance_judgements):
    '''
    We calculate the precision, recall, and F1 metrics for the given run.
    '''
    precision_values = []
    recall_values = []
    f1_values = []
    query_metrics = {}

    global_retrieved = 0
    global_relevant = 0
    global_retrieved_and_relevant = 0

    for query_id in run.keys():

        retrieved_results = []
        for doc_info in run[query_id]:
            retrieved_results.append(doc_info['id'])

        relevant_results = relevance_judgements.get(query_id, {}).keys()
        relevant_and_retrieved = set(retrieved_results) & set(relevant_results)

        global_retrieved += len(retrieved_results)
        global_relevant += len(relevant_results)
        global_retrieved_and_relevant += len(relevant_and_retrieved)

        precision = len(relevant_and_retrieved) / len(retrieved_results) if len(retrieved_results) > 0 else 0
        recall = len(relevant_and_retrieved) / len(relevant_results) if len(relevant_results) > 0 else 0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_values.append(f1)

        precision_values.append(precision)
        recall_values.append(recall)

        query_metrics[query_id] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    macro_average_precision = sum(precision_values) / len(precision_values) if precision_values else 0
    macro_average_recall = sum(recall_values) / len(recall_values) if recall_values else 0
    macro_average_f1 = sum(f1_values) / len(f1_values) if f1_values else 0

    micro_average_precision = global_retrieved_and_relevant / global_retrieved if global_retrieved > 0 else 0
    micro_average_recall = global_retrieved_and_relevant / global_relevant if global_relevant > 0 else 0
    micro_average_f1 = (2 * (micro_average_precision * micro_average_recall) / 
                        (micro_average_precision + micro_average_recall)) if (micro_average_precision + micro_average_recall) > 0 else 0

    return {
        "macro_precision": round(macro_average_precision, 4),
        "macro_recall": round(macro_average_recall, 4),
        "macro_f1": round(macro_average_f1, 4),
        "micro_precision": round(micro_average_precision, 4),
        "micro_recall": round(micro_average_recall, 4),
        "micro_f1": round(micro_average_f1, 4),
        "query_metrics": query_metrics
    }


def print_results_summary(results_summary):
    '''
    We print the results summary for the different configurations.
    '''
    for result in results_summary:
        print(f"{result['config_file']}:")
        print(f"  Macro-Precision = {result['macro_precision']}")
        print(f"  Macro-Recall = {result['macro_recall']}")
        print(f"  Macro-F1 = {result['macro_f1']}")
        print(f"  Micro-Precision = {result['micro_precision']}")
        print(f"  Micro-Recall = {result['micro_recall']}")
        print(f"  Micro-F1 = {result['micro_f1']}")
        print("")


def create_indexes():
    """
    It creates the indexes for the different configurations.
    """
    corpus_verbatim, corpus_plaintext = parse_nf_corpus()

    for bm25_variant in bm25_variants:
        for preprocessing in preprocessing_options:

            output_dir = "index_ex2"
            filename_index = f"{bm25_variant}-{'stopwords' if preprocessing['stopwords'] else 'NONE-stopwords'}-{'stemming' if preprocessing['stemming'] else 'NONE-stemming'}-index_ex2"
            
            index_path = os.path.join(output_dir, filename_index)
            if os.path.exists(index_path):
                print(f"El índice {filename_index} ya existe. Saltando reindexación.")
                continue

            stemmer_used = preprocessing["stemming"]
            stopwords_choosen = preprocessing["stopwords"]
        
            index_corpus(bm25_variant, bm25_variant, corpus_verbatim, 
                corpus_plaintext, stemmer_used, stopwords_choosen, filename_index, output_dir)
            

def run_bm25(queries, relevance_judgements):
    '''
    We run the BM25 algorithm for the different configurations.
    '''
    results_summary = []
    for bm25_variant in bm25_variants:
        for preprocessing in preprocessing_options:

            directory = "index_ex2"
            filename_index = f"{bm25_variant}-{'stopwords' if preprocessing['stopwords'] else 'NONE-stopwords'}-{'stemming' if preprocessing['stemming'] else 'NONE-stemming'}-index_ex2"
            
            stemmer_used = preprocessing["stemming"]
            stopwords_choosen = preprocessing["stopwords"]
            retriever = load_index(directory, filename_index)
           
            '''
            We iterate over the queries and retrieve the documents for each query.
            The results are stored in the run dictionary.
            '''
            run = {}
            for query in queries:
                query_id = query["id"]
                query_string = query["query"]

                query_tokenized = bm25s.tokenize(query_string, stopwords=stopwords_choosen, 
                        stemmer=stemmer_used, show_progress=True)
                returned_docs, returned_scores = retrieve_documents(retriever, query_tokenized, max_results)

                doc_info_list = []
                for i in range(len(returned_docs)):
                    doc_info_list.append({"id": returned_docs[i]["id"], "doc_score": float(returned_scores[i])})
                    
                run[query_id] = doc_info_list

            filename = f"{bm25_variant}-{'stopwords' if stopwords_choosen else 'NONE-stopwords'}-{'stemming' if stemmer_used else 'NONE-stemming'}"
            metrics = calculate_metrics(run, relevance_judgements)
            save_results(run, relevance_judgements, "results-ex2", filename, metrics)
            results_summary.append({ "config_file": filename, **metrics })

    print("Final Summary:")
    print_results_summary(results_summary)


if __name__ == "__main__":
    try:
        create_indexes() 
        run_bm25(load_queries(), load_relevance_judgements())

    except Exception as e:
        print("An unexpected error occurred.")