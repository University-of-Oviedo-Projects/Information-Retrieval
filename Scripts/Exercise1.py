import bm25s, Stemmer, os, tarfile, re

'''
This script is used to index a corpus of documents using the BM25 algorithm.
'''
def parse_nf_corpus(file_path="./train.docs"):
    '''
    This function parses a plain text corpus file and returns a list of dictionaries
    with the document id and text content. It also returns a list with the plain text.
    '''
    corpus_verbatim = []
    corpus_plaintext = []

    with open(file_path, "r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            if not line: continue
            parts = line.split("\t", 1)

            if len(parts) != 2:
                print(f"Line with invalid format: {line}")
                continue

            document = { "id": parts[0], "text": parts[1] }
            corpus_verbatim.append(document)
            corpus_plaintext.append(document["text"].lower())

    return corpus_verbatim, corpus_plaintext


def index_corpus(bm25_flavor, idf_flavor, corpus_verbatim, 
                 corpus_plaintext, stemmer_used, stopwords, filename, output_dir):
    '''
    This function indexes a corpus of documents using the BM25 algorithm.
    '''
    corpus_tokenized = bm25s.tokenize(corpus_plaintext, stopwords=stopwords, stemmer=stemmer_used, show_progress=True)

    retriever = bm25s.BM25(corpus=corpus_verbatim, method=bm25_flavor, idf_method=idf_flavor)
    retriever.index(corpus_tokenized, show_progress=True)

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, filename)
    retriever.save(index_path, corpus=corpus_verbatim)

    print("Index process succesfully complete, file saved.")
    compress_index(index_path, filename)


def compress_index(index_path, filename):
    '''
    This function compresses the index file into a tar.gz file.
    '''
    tar_filename = f"{index_path}.tar.gz"
    with tarfile.open(tar_filename, "w:gz") as tar:
        tar.add(index_path, arcname=filename)


def load_index(directory, filename):
    '''
    This function loads a BM25 index from a file.
    '''
    index_path = os.path.join(directory, filename)
    if not os.path.exists(index_path):
        print(f"The file {index_path} was not found.")
        return None
    return bm25s.BM25.load(index_path, load_corpus=True)


if __name__ == "__main__":
    try:
        corpus_file_path = "train.docs"
        corpus_verbatim, corpus_plaintext = parse_nf_corpus(corpus_file_path)

        index_corpus("lucene", "lucene", corpus_verbatim, 
                     corpus_plaintext, Stemmer.Stemmer("english"), "en", "corpus-index-lucene", "index_ex1")

        retriever = load_index("index_ex1", "corpus-index-lucene")   

        if retriever is not None:
            print("Index was succesfully loaded.")
        else:
            print("Error: index could not be loaded")

    except Exception as e:
        print("Unexpected error, aborting execution.")
