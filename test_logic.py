import os
import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# (Copying core logic from app.py for testing)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text, case_sensitive=True):
    text = re.sub(r'[^\w\s]', ' ', text)
    if not case_sensitive:
        text = text.lower()
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def create_incidence_matrix(documents):
    all_terms = sorted(list(set().union(*documents.values())))
    matrix = []
    for term in all_terms:
        row = []
        for doc_name in sorted(documents.keys()):
            if term in documents[doc_name]:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
    df = pd.DataFrame(matrix, index=all_terms, columns=sorted(documents.keys()))
    return df

# Test Data
docs = {
    "doc1.txt": preprocess_text("Komputer dan komputasi."),
    "doc2.txt": preprocess_text("Kecerdasan buatan.")
}

print("Tokens Doc 1:", docs["doc1.txt"])
print("Tokens Doc 2:", docs["doc2.txt"])

df = create_incidence_matrix(docs)
print("\nIncidence Matrix:")
print(df)

# Check Boolean AND
def AND_op(a, b): return [x & y for x, y in zip(a, b)]
v1 = df.loc["komputer"].values.tolist()
v2 = df.loc["komputasi"].values.tolist()
res = AND_op(v1, v2)
print(f"\nSearch 'komputer AND komputasi': {res}")

if res == [1, 0]:
    print("Test Logic Passed!")
else:
    print("Test Logic Failed!")
