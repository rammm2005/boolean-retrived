import os
import re
import pandas as pd
import numpy as np
import streamlit as st
from collections import defaultdict

def vector_to_binary_string(vector):
    return "".join(map(str, vector))

def preprocess_text(text, case_sensitive=True):
    text = re.sub(r'[^\w\s]', ' ', text)
    if not case_sensitive:
        text = text.lower()
    return text.split()

def get_documents(folder_path, case_sensitive=True):
    documents = {}
    if not os.path.exists(folder_path):
        return documents
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text = file.read()
                tokens = preprocess_text(text, case_sensitive=case_sensitive)
                documents[filename] = tokens
    return documents

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
    return pd.DataFrame(matrix, index=all_terms, columns=sorted(documents.keys()))

def create_inverted_index(documents):
    inverted_index = {}
    for doc_name, tokens in sorted(documents.items()):
        for position, term in enumerate(tokens):
            if term not in inverted_index:
                inverted_index[term] = {}
            if doc_name not in inverted_index[term]:
                inverted_index[term][doc_name] = {"frequency": 0, "positions": []}
            inverted_index[term][doc_name]["frequency"] += 1
            inverted_index[term][doc_name]["positions"].append(position)
    return inverted_index

def AND_op(a, b):
    return [x & y for x, y in zip(a, b)]

def OR_op(a, b):
    return [x | y for x, y in zip(a, b)]

def NOT_op(a):
    return [1 - x for x in a]

def get_vector_from_matrix(term, df):
    if term in df.index:
        return df.loc[term].values.tolist()
    return [0] * len(df.columns)

def get_vector_from_inverted(term, inverted_index, doc_list):
    vector = []
    for doc in doc_list:
        if term in inverted_index and doc in inverted_index[term]:
            vector.append(1)
        else:
            vector.append(0)
    return vector

class BooleanEvaluator:
    def __init__(self, df, inverted_index, doc_names, mode, case_sensitive=True):
        self.df = df
        self.inverted_index = inverted_index
        self.doc_names = doc_names
        self.mode = mode
        self.case_sensitive = case_sensitive
        self.steps = []
        self.term_info = []
        self.expansion_steps = []
        self.term_vectors = {}

    def get_term(self, term):
        return term if self.case_sensitive else term.lower()

    def get_vector(self, term):
        processed_term = self.get_term(term)
        if processed_term not in self.term_vectors:
            if self.mode == "Incidence Matrix":
                v = get_vector_from_matrix(processed_term, self.df)
            else:
                v = get_vector_from_inverted(processed_term, self.inverted_index, self.doc_names)
            self.term_vectors[processed_term] = v
            self.term_info.append(f"Vektor ({processed_term}): {vector_to_binary_string(v)}")
        return self.term_vectors[processed_term]

    def evaluate(self, tokens, raw_query):
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        output_queue = []
        operator_stack = []
        
        expanded_query_tokens = []
        for token in tokens:
            if token in ["(", ")", "AND", "OR", "NOT"]:
                expanded_query_tokens.append(token)
            else:
                v = self.get_vector(token)
                expanded_query_tokens.append(vector_to_binary_string(v))

        self.expansion_steps.append(f"Query: {raw_query}")
        self.expansion_steps.append(f"Ekspansi: {' '.join(expanded_query_tokens)}")

        for token in tokens:
            if token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if operator_stack: operator_stack.pop()
            elif token in precedence:
                while (operator_stack and operator_stack[-1] != '(' and 
                       precedence[operator_stack[-1]] >= precedence[token]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            else:
                output_queue.append(token)
        
        while operator_stack:
            output_queue.append(operator_stack.pop())
            
        eval_stack = []
        step_num = 1
        for token in output_queue:
            if token == 'AND':
                v2 = eval_stack.pop()
                v1 = eval_stack.pop()
                res = AND_op(v1, v2)
                s1, s2, sr = map(vector_to_binary_string, [v1, v2, res])
                self.steps.append(f"Langkah {step_num}: {s1} AND {s2} -> {sr}")
                eval_stack.append(res)
                step_num += 1
            elif token == 'OR':
                v2 = eval_stack.pop()
                v1 = eval_stack.pop()
                res = OR_op(v1, v2)
                s1, s2, sr = map(vector_to_binary_string, [v1, v2, res])
                self.steps.append(f"Langkah {step_num}: {s1} OR {s2} -> {sr}")
                eval_stack.append(res)
                step_num += 1
            elif token == 'NOT':
                v = eval_stack.pop()
                res = NOT_op(v)
                s, sr = map(vector_to_binary_string, [v, res])
                self.steps.append(f"Langkah {step_num}: NOT {s} -> {sr}")
                eval_stack.append(res)
                step_num += 1
            else:
                eval_stack.append(self.get_vector(token))
        
        return eval_stack[0] if eval_stack else [0] * len(self.doc_names)

st.set_page_config(page_title="Boolean Retrieval System", layout="wide")
st.title("Sistem Temu Kembali Informasi (STKI)")
st.markdown("Implementasi model Boolean Retrieval untuk pencarian dokumen berdasarkan kata kunci (Tanpa Stemming).")

folder_path = "documents"
st.sidebar.header("⚙ Konfigurasi")
case_sensitive = st.sidebar.checkbox("Case Sensitive (Beda)", value=True)

documents = get_documents(folder_path, case_sensitive=case_sensitive)
doc_names = sorted(list(documents.keys()))

if not documents:
    st.error(f"Dokumen tidak ditemukan di folder '{folder_path}'.")
else:
    df_incidence = create_incidence_matrix(documents)
    inverted_index = create_inverted_index(documents)

    st.header("1. TF Biner (Term Frequency Biner)")
    tf_biner_cols = st.columns(3)
    for i, term in enumerate(df_incidence.index):
        v = df_incidence.loc[term].values.tolist()
        with tf_biner_cols[i % 3]:
            st.markdown(f"**TFbiner({term})** = `{vector_to_binary_string(v)}`")

    st.header("2. Pencarian Dokumen")
    mode = st.radio("Metode Retrieval:", ("Incidence Matrix", "Inverted Index"), horizontal=True)
    query = st.text_input("Query Boolean (Contoh: A AND (B OR NOT C))", value="komputer AND (AI OR NOT robot)")
    
    if query:
        raw_tokens = re.findall(r'\(|\)|[\w]+', query)
        processed_query_tokens = [t.upper() if t.upper() in ["AND", "OR", "NOT"] or t in ["(", ")"] else t for t in raw_tokens]
        evaluator = BooleanEvaluator(df_incidence, inverted_index, doc_names, mode, case_sensitive=case_sensitive)
        
        try:
            result_vector = evaluator.evaluate(processed_query_tokens, query)
            
            st.subheader("Proses Ekspansi & Perhitungan")
            with st.container():
                st.info(f"Metode: {mode} | Prioritas: ( ) > NOT > AND > OR")
                
                exp_text = "\n".join(evaluator.expansion_steps)
                info_text = "\n".join(evaluator.term_info)
                calc_text = "\n".join(evaluator.steps) if evaluator.steps else f"Vektor Hasil: {vector_to_binary_string(result_vector)}"
                
                full_process = f"--- EKSPANSI QUERY ---\n{exp_text}\n\n--- VEKTOR TERM ---\n{info_text}\n\n--- LANGKAH PERHITUNGAN ---\n{calc_text}"
                st.code(full_process, language="text")

            binary_res = vector_to_binary_string(result_vector)
            relevant_docs = [doc_names[i] for i, val in enumerate(result_vector) if val == 1]
            
            st.success(f"**Hasil Akhir (Biner):** `{binary_res}`")
            doc_output = ", ".join(relevant_docs) if relevant_docs else "Tidak ada dokumen yang relevan"
            st.markdown(f"**Dokumen Relevan:** `{doc_output}`")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

    st.header("3. Struktur Data")
    with st.expander("Detail Incidence Matrix & Inverted Index"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Incidence Matrix")
            st.dataframe(df_incidence, use_container_width=True)
        with col2:
            st.subheader("Inverted Index")
            st.write("Format: <Idj, fij, [posisi]>")
            for term, docs in inverted_index.items():
                st.markdown(f"**{term}**")
                entry_strings = []
                for doc_name, info in docs.items():
                    doc_id_match = re.search(r'\d+', doc_name)
                    doc_id_str = f"Id{doc_id_match.group()}" if doc_id_match else doc_name
                    entry_strings.append(f"<{doc_id_str}, {info['frequency']}, {info['positions']}>")
                st.write(", ".join(entry_strings))

    st.header("4. Koleksi Dokumen")
    with st.expander("Isi Dokumen"):
        for name in doc_names:
            st.markdown(f"### 📄 {name}")
            with open(os.path.join(folder_path, name), "r", encoding="utf-8") as f:
                st.text(f.read())
            st.divider()
