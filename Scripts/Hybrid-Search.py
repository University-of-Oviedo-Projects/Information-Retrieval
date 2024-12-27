import bm25s, Stemmer
from Scripts.Indexing import parse_nf_corpus, load_index
from Scripts.BM25 import load_queries, load_relevance_judgements, calculate_metrics, save_results, retrieve_documents
from Exercise3 import compute_term_frequencies_from_corpus_tokenized, expand_query_with_llr
from Exercise5 import connectToDB

'''
In this script we will mix the results from the lexical and semantic search engines.
Then, we will compute the precision, recall and F1 score for the mixed results.
This script performs an Hybrid search using Lexical and Semantic search engines.
'''

def submit_queries_to_lexical_and_get_run(queries, stemmer, max_results=100):
    '''
    We submit the queries to the lexical search engine and return the run.
    '''
    run = {}
    corpus_verbatim, corpus_plaintext = parse_nf_corpus()
    filename_index = f"lucene-stopwords-stemming-index_ex2"
    retriever = load_index("index_ex2", filename_index)

    corpus_tokenized = bm25s.tokenize(corpus_plaintext, stopwords="en", stemmer=stemmer, show_progress=True)
    global_term_frequencies = compute_term_frequencies_from_corpus_tokenized(corpus_tokenized)

    for query in queries:
        query_id = query["id"]
        query_string = query["query"].lower()

        original_query_tokenized = bm25s.tokenize(query_string, stopwords="en", 
                stemmer=stemmer, show_progress=True)
        returned_docs_orig, _ = retrieve_documents(retriever, original_query_tokenized, 4)
        
        expanded_query = expand_query_with_llr (
            query_string, returned_docs_orig, corpus_verbatim, global_term_frequencies, stemmer, 5)

        expanded_query_tokenized = bm25s.tokenize(expanded_query, stopwords="en", 
                stemmer=stemmer, show_progress=True)
        returned_docs, _ = retrieve_documents(retriever, expanded_query_tokenized, 100)
        
        returned_ids = []; 
        for i in range(len(returned_docs)):
            returned_ids.append(str(returned_docs[i]["id"]))

        run[query_id] = returned_ids
    return run


def submit_queries_to_semantic_and_get_run(queries, collection, max_results=100):
    '''
    We submit the queries to the semantic search engine and return the run.
    '''
    run = {}

    for query in queries:
        query_id = query["id"]
        query_text = query["query"].lower()

        results = collection.query(
            query_texts=[query_text],
            n_results=max_results)

        run[query_id] = results['ids'][0]
    return run


def reciprocal_rank_fusion(ranking1, ranking2, k=100):
    '''
    We perform the reciprocal rank fusion of two rankings.
    This is a simple fusion method that combines the ranks of two rankings.
    It is used for combining the results of two search engines.
    '''
    ranking1 = {string: index for index, string in enumerate(ranking1)}
    ranking2 = {string: index for index, string in enumerate(ranking2)}

    all_docs_ids = set(ranking1.keys()).union(set(ranking2.keys()))
    rrf_values = dict()

    for doc_id in all_docs_ids:
        try:
            rank_1 = 1/(k + ranking1[doc_id])
        except:
            rank_1 = 0
        try:
            rank_2 = 1/(k + ranking2[doc_id])
        except:
            rank_2 = 0

        rrf_values[doc_id] = rank_1 + rank_2

    # We sort the dictionary by values in descending order.
    return sorted(rrf_values.items(), key=lambda x: x[1], reverse=True)


def mix_lexical_semantic_runs(lexical_run, semantic_run):
    '''
    We mix the lexical and semantic runs using the reciprocal rank fusion method.

    The process is as follows:
        1. We get the maximum number of results for each query.
        2. We get the results of the lexical and semantic search engines.
        3. We perform the reciprocal rank fusion of the two rankings.
        4. We return the mixed run.
    '''
    mixed_run = {}

    for query_id in lexical_run.keys():
      max_results = len(lexical_run[query_id])
      lexical_results = lexical_run[query_id]
      semantic_results = semantic_run[query_id]

      lexical_results_dict = {string: index for index, string in enumerate(lexical_results)}
      semantic_results_dict = {string: index for index, string in enumerate(semantic_results)}
      lexical_semantic_results = reciprocal_rank_fusion(lexical_results_dict, semantic_results_dict)

      mixed_run[query_id] = list(lexical_semantic_results)[:max_results]
      mixed_run[query_id] = [item[0] for item in mixed_run[query_id]]

      mixed_run[query_id] = [{"id": doc_id, "doc_score": score} 
            for doc_id, score in lexical_semantic_results[:max_results]]

    return mixed_run


if __name__ == "__main__":
    try:
        queries = load_queries()
        relevance_judgements = load_relevance_judgements()
        stemmer = Stemmer.Stemmer("english")
        
        collection = connectToDB("TREC-COVID_collection", "./chromadb-storage")
        
        '''
        We submit the queries to the lexical and semantic search engines.
        '''
        original_lexical_run = submit_queries_to_lexical_and_get_run(
            queries, stemmer, max_results=100)
        original_semantic_run = submit_queries_to_semantic_and_get_run(
            queries, collection, max_results=100)

        '''
        We mix the lexical and semantic runs.
        '''
        mixed_run = mix_lexical_semantic_runs(original_lexical_run, original_semantic_run)
        metrics = calculate_metrics(mixed_run, relevance_judgements)

        # Save the results to a JSON file in the specified directory
        save_results(mixed_run, relevance_judgements, "results-optional-ex", "hybrid_run_results", metrics)
        
    except Exception as e:
        print("An unexpected error occurred.")

    
