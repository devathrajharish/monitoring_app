import os
import json
import streamlit as st
import pandas as pd
import hashlib
import warnings
from typing import List, Dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
import openai
# import warnings
# warnings.filterwarnings("ignore", message="huggingface/tokenizers: The current process just got forked")

# -------------------- Thread Limiting for macOS Stability --------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="huggingface/tokenizers: The current process just got forked")

# -------------------- API and Environment Setup --------------------
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
embeddings = OpenAIEmbeddings()
openai.api_key = openai_api_key

llm = ChatOpenAI(temperature=0, model="gpt-4o")
embeddings = OpenAIEmbeddings()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="AI Log Analyzer", layout="wide")
st.title("🛠️ AI Log Analyzer - Pipeline Fixer")
st.markdown("Analyze failed pipelines, group errors, and get contextual remediations.")

# -------------------- Upload Logs --------------------
logs_file = st.file_uploader("📁 Upload pipeline failure logs (.json)", type="json")

if logs_file:
    logs_data = json.load(logs_file)
    if not all('error_message' in entry for entry in logs_data):
        st.error("🚫 Some entries in logs.json are missing 'error_message'. Please check the format.")
        st.stop()

    df = pd.DataFrame(logs_data)

    st.markdown("### 🧾 Uploaded Logs Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("🔍 Analyze Failures"):
        with st.spinner("Analyzing failures using GPT-4o + FAISS..."):
            error_messages = [entry['error_message'] for entry in logs_data if 'error_message' in entry]
            error_embeddings = embedding_model.encode(error_messages)
            n_clusters = min(4, len(error_messages))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(error_embeddings)

            grouped = {}
            for idx, label in enumerate(labels):
                if label not in grouped:
                    grouped[label] = []
                grouped[label].append(logs_data[idx])

            docs = [
                Document(page_content="Null value errors are often resolved by checking schema definitions and adding null checks.", metadata={"source": "runbook"}),
                Document(page_content="Access Denied errors usually relate to missing RBAC roles or key permissions.", metadata={"source": "runbook"}),
                Document(page_content="Data type mismatches are resolved by casting or schema adjustments.", metadata={"source": "runbook"}),
                Document(page_content="File not found errors suggest verifying file paths or ingestion timing.", metadata={"source": "runbook"})
            ]
            vectorstore = FAISS.from_documents(docs, embeddings)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

            result = []
            for label, group in grouped.items():
                example_error = group[0]['error_message']
                query = f"Suggest a fix for: {example_error}"
                fix = qa_chain.run(query)

                pipeline_names = [entry['pipeline_name'] for entry in group]
                result.append({
                    "Error Category": example_error[:80],
                    "Affected Pipelines": ", ".join(pipeline_names),
                    "Recommended Fix": fix,
                    "Feedback": "👍 / 👎"
                })

            grouped_error_df = pd.DataFrame(result)

            st.markdown("### 📌 Grouped Failures with Fixes")
            st.dataframe(grouped_error_df, use_container_width=True)

            st.markdown("### 💬 Ask GPT-4o about a failure group")
            with st.form("gpt_query_form"):
                query = st.text_input("Ask something like: 'What is the fix for null column errors?'")
                submitted = st.form_submit_button("Ask GPT-4o")

            if submitted and query.strip() != "":
                with st.spinner("Thinking..."):
                    try:
                        response = qa_chain.run(query)
                        st.success("GPT-4o Response:")
                        st.markdown(response)
                    except Exception as e:
                        st.error(f"⚠️ GPT-4o failed: {str(e)}")

            st.markdown("### 🔁 Feedback on Recommendations")
            for i, row in grouped_error_df.iterrows():
                key = hashlib.md5(row["Error Category"].encode()).hexdigest()
                feedback = st.radio(f"Feedback for: {row['Error Category'][:60]}", ["👍", "👎"], key=key)
                # In real scenario, save feedback to a DB or JSON

            st.download_button("📥 Download Grouped Results as JSON", json.dumps(result, indent=2), "grouped_fixes.json")
else:
    st.info("Upload a logs.json file to start analysis.")




# import os
# import json
# import streamlit as st
# import pandas as pd
# import hashlib
# from typing import List, Dict
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.schema import Document
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# import numpy as np
# import os
# import openai
# import warnings
# warnings.filterwarnings("ignore", message="huggingface/tokenizers: The current process just got forked")


# # 🧠 Prevent OpenMP/OpenBLAS crashes
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"



# # -------------------- API and Environment Setup --------------------
# # openai_api_key = "sk---kDxPwA"  # Use st.secrets in production
# # os.environ["OPENAI_API_KEY"] = openai_api_key

# openai_api_key = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = openai_api_key
# embeddings = OpenAIEmbeddings()
# openai.api_key = openai_api_key

# llm = ChatOpenAI(temperature=0, model="gpt-4o")
# embeddings = OpenAIEmbeddings()
# embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# st.set_page_config(page_title="AI Log Analyzer", layout="wide")
# st.title("🛠️ AI Log Analyzer - Pipeline Fixer")
# st.markdown("Analyze failed pipelines, group errors, and get contextual remediations.")

# # -------------------- Upload Logs --------------------
# logs_file = st.file_uploader("📁 Upload pipeline failure logs (.json)", type="json")

# if logs_file:
#     logs_data = json.load(logs_file)
#     df = pd.DataFrame(logs_data)

#     st.markdown("### 🧾 Uploaded Logs Preview")
#     st.dataframe(df.head(), use_container_width=True)

#     # -------------------- Analyze Failures --------------------
#     if st.button("🔍 Analyze Failures"):
#         with st.spinner("Analyzing failures using GPT-4o + FAISS..."):
#             # Step 1: Embed all error messages
#             error_messages = [entry.get('error_message', '') for entry in logs_data if 'error_message' in entry]
#            # error_messages = [entry['error_message'] for entry in logs_data]
#             error_embeddings = embedding_model.encode(error_messages)

#             # Step 2: Cluster errors (e.g., 4 clusters)
#             n_clusters = min(4, len(error_messages))
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             labels = kmeans.fit_predict(error_embeddings)

#             # Step 3: Group errors
#             grouped = {}
#             for idx, label in enumerate(labels):
#                 if label not in grouped:
#                     grouped[label] = []
#                 grouped[label].append(logs_data[idx])

#             # Step 4: Create vector store from documentation/runbooks (mocked)
#             docs = [
#                 Document(page_content="Null value errors are often resolved by checking schema definitions and adding null checks.", metadata={"source": "runbook"}),
#                 Document(page_content="Access Denied errors usually relate to missing RBAC roles or key permissions.", metadata={"source": "runbook"}),
#                 Document(page_content="Data type mismatches are resolved by casting or schema adjustments.", metadata={"source": "runbook"}),
#                 Document(page_content="File not found errors suggest verifying file paths or ingestion timing.", metadata={"source": "runbook"})
#             ]
#             vectorstore = FAISS.from_documents(docs, embeddings)
#             qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), chain_type="stuff")

#             # Step 5: Prepare results for UI
#             result = []
#             for label, group in grouped.items():
#                 example_error = group[0]['error_message']
#                 query = f"Suggest a fix for: {example_error}"
#                 fix = qa_chain.run(query)

#                 pipeline_names = [entry['pipeline_name'] for entry in group]
#                 result.append({
#                     "Error Category": example_error[:80],
#                     "Affected Pipelines": ", ".join(pipeline_names),
#                     "Recommended Fix": fix,
#                     "Feedback": "👍 / 👎"
#                 })

#             grouped_error_df = pd.DataFrame(result)

#             st.markdown("### 📌 Grouped Failures with Fixes")
#             st.dataframe(grouped_error_df, use_container_width=True)

#             # -------------------- Q&A Section --------------------
#             st.markdown("### 💬 Ask GPT-4o about a failure group")
#             query = st.text_input("Ask something like: 'What is the root cause of access denied errors?'")
#             if st.button("Ask GPT-4o"):
#                 with st.spinner("Generating answer..."):
#                     response = qa_chain.run(query)
#                     st.markdown(f"🧠 GPT-4o says: {response}")

#             # -------------------- Feedback Loop --------------------
#             st.markdown("### 🔁 Feedback on Recommendations")
#             for i, row in grouped_error_df.iterrows():
#                 key = hashlib.md5(row["Error Category"].encode()).hexdigest()
#                 feedback = st.radio(f"Feedback for: {row['Error Category'][:60]}", ["👍", "👎"], key=key)
#                 # In real scenario, save feedback to a DB or JSON

#             # -------------------- Export --------------------
#             st.download_button("📥 Download Grouped Results as JSON", json.dumps(result, indent=2), "grouped_fixes.json")
# else:
#     st.info("Upload a logs.json file to start analysis.")
