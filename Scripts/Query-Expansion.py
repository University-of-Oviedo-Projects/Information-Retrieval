import Stemmer, bm25s
from Scripts.Indexing import parse_nf_corpus, load_index

from Scripts.BM25 import (
    load_queries,  
    load_relevance_judgements, 
    retrieve_documents,
    calculate_metrics,
    save_results
)

from Exercise3 import (
    compute_term_frequencies_from_corpus_tokenized, 
    expand_query_with_llr
)

'''
This script evaluates the query expansion method using the Signed LLR method.
'''

def evaluate_query_expansion(n_values, m_values, bm25_variant="lucene"):
    '''
    We evaluate the query expansion method using the Signed LLR method.
    '''
    evaluation_results = {}
    stemmer = Stemmer.Stemmer("english")
    
    queries = load_queries()
    relevance_judgements = load_relevance_judgements()
    corpus_verbatim, corpus_plaintext = parse_nf_corpus()

    filename_index = f"{bm25_variant}-stopwords-stemming-index_ex2"
    retriever = load_index("index_ex2", filename_index)    

    corpus_tokenized = bm25s.tokenize(corpus_plaintext, stopwords="en", stemmer=stemmer, show_progress=True)
    global_term_frequencies = compute_term_frequencies_from_corpus_tokenized(corpus_tokenized)

    '''
    N value is the number of documents to retrieve in the first step of the query expansion process.
    M value is the number of terms to add to the original query.
    '''
    for n in n_values:
        for m in m_values:
            print(f"Evaluando configuración: n={n}, m={m}")
            run = {}  

            '''
            We iterate over the queries, retrieve the top N documents for each query,
            and then expand the query using M terms and the Signed LLR method.
            '''
            for query in queries:
                query_id = query["id"]
                original_query = query["query"]

                original_query_tokenized = bm25s.tokenize(original_query, stopwords="en", 
                        stemmer=stemmer, show_progress=True)
                relevant_documents, _ = retrieve_documents(retriever, original_query_tokenized, n)

                '''
                We expand the query using the Signed LLR method.

                Parameters:
                    - original_query: the original query. 
                    - original_query_tokenized: the original query tokenized.
                    - original_docs: the documents retrieved for the original query.
                    - corpus_verbatim: the corpus of documents.
                    - global_term_frequencies: the term frequencies of the corpus.
                    - stemmer: the stemmer to use.
                    - m: the number of terms to add to the original query.
                '''
                expanded_query = expand_query_with_llr (
                    original_query, relevant_documents, 
                    corpus_verbatim, global_term_frequencies, stemmer, m)

                expanded_query_tokenized = bm25s.tokenize(expanded_query, stopwords="en", 
                        stemmer=stemmer, show_progress=True)
                relevant_documents, returned_scores = retrieve_documents(retriever, expanded_query_tokenized, 100)
                
                doc_info_list = []
                for i in range(len(relevant_documents)):
                    doc_info_list.append({"id": relevant_documents[i]["id"], "doc_score": float(returned_scores[i])})
                
                run[query_id] = doc_info_list

            '''
            We save the results to a file and calculate the metrics.
            '''
            filename = f"lucene-stopwords-stemming-query_expansion_n{n}_m{m}.txt"
            metrics = calculate_metrics(run, relevance_judgements)
            save_results(run, relevance_judgements, "results-ex4", filename, metrics)
            evaluation_results[(n, m)] = metrics

    return evaluation_results


def analyze_results(evaluation_results):
    '''
    This function analyzes the results of the query expansion evaluation.
    It prints the results for each configuration and identifies the best configuration.
    '''
    print("\nAnalysis of the results:")
    best_macro_f1 = -1
    best_configuration = None

    for (n, m), metrics in evaluation_results.items():
        print(f"Configuration n={n}, m={m}: Macro F1={metrics['macro_f1']:.4f}, "
              f"Micro F1={metrics['micro_f1']:.4f}, Macro Precision={metrics['macro_precision']:.4f}, "
              f"Macro Recall={metrics['macro_recall']:.4f}")

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_configuration = (n, m)

    print("\Best configuration:")
    print(f"n={best_configuration[0]}, m={best_configuration[1]} with Macro F1={best_macro_f1:.4f}")


if __name__ == "__main__":
    n_values = range(1, 6)  # [1, 5]
    m_values = range(3, 6)  # [3, 5]

    try :
        evaluation_results = evaluate_query_expansion(n_values, m_values)
        analyze_results(evaluation_results)
    except Exception as e:
        print("An unexpected error occurred.")