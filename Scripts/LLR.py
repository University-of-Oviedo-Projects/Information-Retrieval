import Stemmer, bm25s, math
from collections import Counter
from Scripts.Indexing import parse_nf_corpus

from Scripts.BM25 import (
    load_queries,  
    load_relevance_judgements, 
    retrieve_documents,
    calculate_metrics,
    save_results
)


def compute_term_frequencies_from_corpus_tokenized(corpus_tokenized):
    """
    This function calculates the term frequencies for the entire corpus.
    """
    tmp = dict()
    for _, document in enumerate(corpus_tokenized[0]):
        freqs = dict(Counter(document))

        for token, freq in freqs.items():
            tmp[token] = tmp.get(token, 0) + freq

    inverted_vocab = {corpus_tokenized[1][key]: key for key in corpus_tokenized[1].keys()}
    total_freqs = {inverted_vocab[key]: freq for key, freq in tmp.items()}
    return total_freqs


def compute_tf_excluding_relevant(global_term_frequencies, tf_relevant):
    """
    This function computes the term frequencies for non-relevant documents.
    """
    tf_non_relevant = {}
    for term, freq in global_term_frequencies.items():
        diff = freq - tf_relevant.get(term, 0)
        if diff > 0:
            tf_non_relevant[term] = diff
    total_non_relevant = sum(tf_non_relevant.values())
    
    return tf_non_relevant, total_non_relevant


def calculate_term_frequencies_relevant(relevant_docs, stemmer):
    """
    This function calculates the term frequencies for relevant documents.
    """
    term_frequencies = Counter()
    for _, doc in enumerate(relevant_docs):
        doc_text = doc["text"]
        if not doc_text: continue

        tokens = bm25s.tokenize(doc_text, stopwords="en", stemmer=stemmer)
        id_to_term = {v: k for k, v in tokens.vocab.items()}
        
        for token_list in tokens.ids:
            for token_id in token_list:
                term = id_to_term.get(token_id, None)
                term_frequencies[term] += 1

    return term_frequencies


def x_log_x(x):
    return 0 if x == 0 else x * math.log(x)

def entropy(*counts):
    total = sum(counts)
    return x_log_x(total) - sum(x_log_x(c) for c in counts)

def log_likelihood_ratio(k11, k12, k21, k22):
    row_entropy = entropy(k11 + k12, k21 + k22)
    column_entropy = entropy(k11 + k21, k12 + k22)
    matrix_entropy = entropy(k11, k12, k21, k22)

    if row_entropy + column_entropy < matrix_entropy:
        return 0.0
    return 2.0 * (row_entropy + column_entropy - matrix_entropy)


def root_log_likelihood_ratio(k11, k12, k21, k22):
    llr = log_likelihood_ratio(k11, k12, k21, k22)
    sqrt_llr = math.sqrt(llr)
    if k11 / (k11 + k12) < k21 / (k21 + k22):
        sqrt_llr = -sqrt_llr
    return sqrt_llr


def compare_frequencies(tf_relevant, tf_non_relevant, threshold):
    total_relevant = sum(tf_relevant.values())
    total_non_relevant = sum(tf_non_relevant.values())
    scored_terms = []
    
    for term, count_relevant in tf_relevant.items():
        count_non_relevant = tf_non_relevant.get(term, 0)
        
        score = root_log_likelihood_ratio(count_relevant, total_relevant - count_relevant,
                count_non_relevant, total_non_relevant - count_non_relevant)
        
        if score >= threshold:
            scored_terms.append((term, score))
    
    scored_terms.sort(key=lambda x: x[1], reverse=True)    
    return scored_terms


def expand_query_with_llr(query, relevant_docs, corpus, global_term_frequencies, stemmer, max_terms):
    tf_relevant = calculate_term_frequencies_relevant(relevant_docs, stemmer)
    tf_non_relevant, _ = compute_tf_excluding_relevant(global_term_frequencies, tf_relevant)
    scored_terms = compare_frequencies(tf_relevant, tf_non_relevant, threshold=0.0)

    scored_terms_to_add = []
    for term, score in scored_terms:
        if term not in query and len(scored_terms_to_add) < max_terms:
            scored_terms_to_add.append((term, score))

    expanded_query = query + " " + " ".join(term for term, _ in scored_terms_to_add)
    return expanded_query


def main():
    queries = load_queries()
    relevance_judgements = load_relevance_judgements()
    corpus_verbatim, corpus_plaintext = parse_nf_corpus()

    n = 5; m = 5; run = {} # As example, n=5 and m=5
    stemmer = Stemmer.Stemmer("english")   

    corpus_tokenized = bm25s.tokenize(corpus_plaintext, stopwords="en", stemmer=stemmer, show_progress=True)
    retriever = bm25s.BM25(corpus=corpus_verbatim, method="lucene", idf_method="lucene")
    retriever.index(corpus_tokenized, show_progress=True)

    global_term_frequencies = compute_term_frequencies_from_corpus_tokenized(corpus_tokenized)

    for query in queries:
        query_id = query["id"]
        original_query = query["query"]

        original_query_tokenized = bm25s.tokenize(original_query, stopwords="en", 
                        stemmer=stemmer, show_progress=True)
        relevant_doc_ids, orig_scores = retrieve_documents(retriever, original_query_tokenized, n)

        expanded_query = expand_query_with_llr (
            original_query, relevant_doc_ids, corpus_verbatim, global_term_frequencies, stemmer, m)

        expanded_query_tokenized = bm25s.tokenize(expanded_query, stopwords="en", 
                        stemmer=stemmer, show_progress=True)
        returned_docs, returned_scores = retrieve_documents(retriever, expanded_query_tokenized, 100)

        print(f"Query: {original_query}")   
        print(f"Expanded query: {expanded_query}")
        
        doc_info_list = []
        for i in range(len(returned_docs)):
            doc_info_list.append({"id": returned_docs[i]["id"], "doc_score": float(returned_scores[i])})
        
        run[query_id] = doc_info_list

    filename = f"lucene-stopwords-stemming-query_expansion_n{n}_m{m}.txt"
    metrics = calculate_metrics(run, relevance_judgements)
    save_results(run, relevance_judgements, "results-ex3", filename, metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An unexpected error occurred")
