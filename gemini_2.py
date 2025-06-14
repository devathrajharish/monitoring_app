import os
import streamlit as st
import openai
import json
import difflib
from datetime import datetime, date
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import re # For email validation
import hashlib # For creating unique keys for feedback

# ----------------------- Configuration -----------------------
# 🔐 IMPORTANT: In a production environment, use st.secrets!
# For local testing, you can keep it here or set it as an environment variable.
# # openai_api_key = st.secrets["OPENAI_API_KEY"] 
# openai_api_key = os.getenv("OPEN_API_KEY")  # 🔐 Use st.secrets in production 
# # os.environ["OPENAI_API_KEY"] = openai_api_key
# embeddings = OpenAIEmbeddings()
# openai.api_key = openai_api_key

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key
embeddings = OpenAIEmbeddings()
openai.api_key = openai_api_key

# --- File Paths ---
LOGS_FILE = "logs.json"
FEEDBACK_LOG_FILE = "feedback_log.json"
KNOWLEDGE_BASE_FILE = "knowledge_base.json"

# ----------------------- Data Loading Helpers -----------------------
def load_json_file(file_path, default_value=None):
    """Loads a JSON file, returning default_value if not found or malformed."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return default_value if default_value is not None else []
    except json.JSONDecodeError:
        st.error(f"Error: {file_path} is malformed. Please check its content. Returning empty list.")
        return default_value if default_value is not None else []

def save_json_file(file_path, data):
    """Saves data to a JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving to {file_path}: {e}")

pipeline_logs = load_json_file(LOGS_FILE, [])

# ----------------------- Log Analysis Functions -----------------------
def get_failed_pipelines_today(logs):
    """Returns a list of logs for pipelines that failed today."""
    today_str = date.today().isoformat()
    failed_today = []
    for log in logs:
        log_date = log.get("timestamp", "")
        # Check if timestamp starts with today's date (assuming ISO formatYYYY-MM-DDTHH:MM:SS)
        if log_date.startswith(today_str) and log.get("status", "").lower() == "failed":
            failed_today.append(log)
    return failed_today

def get_pipeline_errors(logs, pipeline_name=None):
    """Returns a dictionary of pipeline names to their error messages."""
    errors = {}
    for log in logs:
        name = log.get("pipeline_name", "unknown")
        if log.get("status", "").lower() == "failed" and log.get("error_message"):
            if pipeline_name and name != pipeline_name:
                continue # Skip if specific pipeline requested and it doesn't match
            if name not in errors:
                errors[name] = []
            errors[name].append(log["error_message"])
    return errors

def build_pipeline_context(logs, limit=10):
    """Builds a formatted summary of recent pipeline logs for LLM context."""
    if not logs:
        return "No pipeline logs available."

    # Sort logs by timestamp (most recent first)
    sorted_logs = sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    summary = "### 🗂 Recent Pipeline Execution Summary:\n"
    for log in sorted_logs[:limit]: # Limit to most recent logs for context
        name = log.get("pipeline_name", "unknown")
        status = log.get("status", "unknown")
        error = log.get("error_message", "No error")
        timestamp = log.get("timestamp", "unknown")
        summary += f"- **{name}** | Status: `{status}` | Time: {timestamp}"
        if status.lower() == "failed":
            summary += f" | ❌ Error: `{error}`"
        summary += "\n"
    summary += "\n(Only showing up to the last 10 log entries for brevity. You can ask for more details.)"
    return summary

# ----------------------- Fuzzy Matching Helper -----------------------
def fuzzy_match_errors(search_text, logs):
    """Performs fuzzy matching on error messages to find relevant pipelines."""
    matched_pipelines = []
    threshold = 0.6 # Adjust threshold for stricter/looser matching
    for log in logs:
        if log["status"].lower() == "failed" and log.get("error_message"):
            score = difflib.SequenceMatcher(None, search_text.lower(), log["error_message"].lower()).ratio()
            if score > threshold:
                matched_pipelines.append(log["pipeline_name"])
    return list(set(matched_pipelines)) # Return unique pipeline names

# ----------------------- Vectorstore for RAG -----------------------
def load_and_index_vectorstore():
    """Loads knowledge base from JSON and re-indexes the FAISS vectorstore."""
    with st.spinner("Loading and indexing knowledge base..."):
        try:
            kb_data = load_json_file(KNOWLEDGE_BASE_FILE, [])
            documents = []
            for entry in kb_data:
                error_text = str(entry.get('error', ''))
                fix_text = str(entry.get('fix', ''))
                if error_text and fix_text:
                    content = f"Error: {error_text}\nFix: {fix_text}"
                    # Use both error and fix content for embedding, and store error/fix for metadata
                    documents.append(Document(page_content=content, metadata={"error": error_text, "fix": fix_text}))
            
            if documents:
                embeddings = OpenAIEmbeddings()
                return FAISS.from_documents(documents, embeddings)
            else:
                st.info("Knowledge base is empty. Vector store not created.")
                return None
        except Exception as e:
            st.error(f"Failed to load or index knowledge base: {e}")
            return None

def get_context_from_rag(query):
    """
    Retrieves the most relevant fix from the FAISS vectorstore based on a query.
    Returns the fix string if a sufficiently similar and relevant fix is found,
    otherwise returns an empty string.
    """
    if st.session_state.vectorstore is None:
        st.info("Knowledge Base not loaded for RAG search.") # Diagnostic
        return ""
    try:
        # Use similarity_search_with_score to get distances (lower is better for FAISS L2)
        # We fetch top 1 to get the single most similar document
        docs_with_score = st.session_state.vectorstore.similarity_search_with_score(query, k=1) 
        
        # Define a relevance threshold for the distance. This value needs tuning.
        # A distance below this threshold implies sufficient similarity.
        # Starting with a stricter value (e.g., 0.5) to avoid overly generic matches.
        relevance_distance_threshold = 0.5 # TUNING POINT: Adjust this value!

        if docs_with_score:
            top_doc, score = docs_with_score[0]
            st.info(f"RAG search found: '{top_doc.page_content[:50]}...' with score {score:.4f}. Threshold: {relevance_distance_threshold}") # Diagnostic
            
            # Check if the score indicates high similarity AND if the document contains a "Fix:"
            if score < relevance_distance_threshold and "Fix:" in top_doc.page_content:
                # Extract and return only the fix part from the page_content
                # Assuming the page_content is structured as "Error: ...\nFix: ..."
                fix_match = re.search(r"Fix:\s*(.*)", top_doc.page_content, re.DOTALL)
                if fix_match:
                    st.info(f"RAG returning specific fix: {fix_match.group(1).strip()[:50]}...") # Diagnostic
                    return fix_match.group(1).strip()
                else:
                    # Fallback if "Fix:" found but regex failed (shouldn't happen with standard structure)
                    # This part still returns the whole content if regex fails, which is less ideal.
                    return top_doc.page_content.replace(top_doc.metadata.get("error", "") + "\n", "").replace("Error:", "").strip()
        st.info("RAG found no sufficiently relevant fix within threshold.") # Diagnostic
        return "" # No sufficiently relevant fix found
    except Exception as e:
        st.warning(f"RAG search failed: {e}") # Diagnostic
        return ""

# ----------------------- UI Setup -----------------------
st.set_page_config(page_title="LogBot - GPT-4o", page_icon="💬")
st.title("💬 LogBot - GPT-4o Log Chatbot")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "email" not in st.session_state:
    st.session_state.email = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_and_index_vectorstore()
if "last_selected_error" not in st.session_state:
    st.session_state.last_selected_error = ""
# New state variable to store details of the last LLM response in general chat for feedback
if "last_llm_response_details" not in st.session_state:
    st.session_state.last_llm_response_details = None 


page = st.sidebar.selectbox("📂 Navigate", ["Chat with LogBot", "Knowledge Base", "Feedback Insights"])

# ----------------------- Email Validation -----------------------
def validate_email(email):
    """Basic regex for email validation."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# ----------------------- Chat Page -----------------------
if page == "Chat with LogBot":
    if not st.session_state.email:
        email_input = st.text_input("Enter your email to start chatting:", key="email_input")
        if st.button("Start Chat"):
            if email_input.strip() and validate_email(email_input.strip()):
                st.session_state.email = email_input.strip()
                st.success(f"Welcome, {st.session_state.email}! You can now chat with LogBot.")
                st.rerun() # Rerun to clear the email input and show chat interface
            else:
                st.error("Please enter a valid email address.")
    else:
        st.write(f"Logged in as: **{st.session_state.email}**")
        st.markdown("---")

        # Section for predefined error types and specific feedback
        st.subheader("Quick Fix for Known Errors")
        predefined_errors = [
            "Select", # Default option
            "Null value in non-nullable column",
            "FileNotFoundError",
            "Token expired",
            "Schema mismatch",
            "OutOfMemoryError",
            "Permission denied",
            "Database connection failed",
            "API limit exceeded",
            "Invalid credentials"
        ]

        selected_error = st.selectbox("Choose an error type to find related pipelines:", predefined_errors)
        force_llm_fallback = st.checkbox("Always ask LLM for suggested fix (even if Knowledge Base has an answer)", key="predefined_force_llm_fallback")


        if selected_error != "Select":
            st.session_state.last_selected_error = selected_error
            
            with st.spinner(f"Searching for pipelines with '{selected_error}' errors..."):
                matched = fuzzy_match_errors(selected_error, pipeline_logs)
                if matched:
                    st.markdown(f"### Pipelines that failed due to: `{selected_error}`")
                    st.markdown("\n".join([f"- {p}" for p in matched]))
                else:
                    st.info("No pipelines matched for the selected error type in recent logs.")

            rag_fix = get_context_from_rag(selected_error) # Now returns only the fix or ""
            source = "Knowledge Base"
            final_fix = "" # Initialize final_fix

            if rag_fix and not force_llm_fallback: # rag_fix is only non-empty if it's relevant
                final_fix = rag_fix
            elif not rag_fix and not force_llm_fallback: # If no relevant RAG fix and no LLM fallback forced
                final_fix = "Solution not present in Knowledge Base."
                source = "Knowledge Base"
            else: # Either rag_fix exists but LLM fallback is forced, or rag_fix is empty but LLM fallback is forced
                try:
                    with st.spinner("Asking LogBot for a suggested fix..."):
                        system_message = "You are a helpful assistant that recommends resolutions for data pipeline failures. Use the provided known solutions and your own expertise. Be concise and actionable."
                        user_message = f"Error Type: {selected_error}\n\n"
                        if rag_fix: # Pass the (potentially relevant) RAG fix as additional context
                            user_message += f"Here is a known fix from our knowledge base (prioritize this if directly relevant):\n{rag_fix}\n\n"
                        user_message += "Suggest a possible resolution in a clear and concise format. If no specific fix is found in the KB, provide a general but accurate solution based on the error type."

                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_message}
                            ],
                            temperature=0.3, # Lower temperature for more deterministic/factual responses
                            max_tokens=300
                        )
                        final_fix = response['choices'][0]['message']['content']
                        source = "GPT-4o"
                        if rag_fix:
                            source += " (with KB context)" 
                except Exception as e:
                    st.error(f"GPT Error: {e}")
                    final_fix = "Could not generate a fix at this time."
                    source = "Error"
            
            st.markdown(f"#### 💡 Suggested Fix ({source})")
            st.markdown(final_fix)

            # Only show feedback if a fix was suggested (not "Solution not present" or error message)
            if final_fix != "Solution not present in Knowledge Base." and final_fix != "Could not generate a fix at this time.":
                st.markdown("---")
                st.markdown("#### Provide Feedback on this Suggestion")
                feedback = st.radio("Was this suggestion helpful?", ["👍 Yes", "👎 No"], horizontal=True, key=f"feedback_radio_{selected_error}")
                auto_save = st.checkbox("Automatically save helpful fixes to knowledge base", key=f"auto_save_checkbox_{selected_error}")
                user_fix = ""
                if feedback == "👎 No":
                    user_fix = st.text_area("Optional: Describe how you fixed it yourself (this can be saved to the knowledge base)", placeholder="Steps or code you used...", key=f"user_fix_textarea_{selected_error}")

                if st.button("Submit Feedback", key=f"submit_feedback_button_{selected_error}"):
                    feedback_entries = load_json_file(FEEDBACK_LOG_FILE, [])
                    feedback_entry = {
                        "error": selected_error,
                        "suggested_fix": final_fix,
                        "feedback_type": "positive" if feedback == "👍 Yes" else "negative",
                        "user_fix": user_fix if feedback == "👎 No" else None,
                        "timestamp": datetime.now().isoformat(),
                        "email": st.session_state.email
                    }
                    feedback_entries.append(feedback_entry)
                    save_json_file(FEEDBACK_LOG_FILE, feedback_entries)
                    st.success("✅ Feedback submitted! Thank you.")

                    # --- Knowledge Base Update Logic for Predefined Errors ---
                    if (feedback == "👍 Yes" and auto_save and final_fix.strip()) or \
                       (feedback == "👎 No" and user_fix.strip()):
                        
                        kb_data = load_json_file(KNOWLEDGE_BASE_FILE, [])
                        
                        # Check if an entry for this error already exists (case-insensitive for error name)
                        existing_entry_index = -1
                        for i, entry in enumerate(kb_data):
                            if entry.get('error', '').lower() == selected_error.lower():
                                existing_entry_index = i
                                break

                        new_fix_content = final_fix.strip() if feedback == "👍 Yes" else user_fix.strip()
                        new_source = "GPT-4o (User Confirmed)" if feedback == "👍 Yes" else "User Submitted"
                        
                        if existing_entry_index != -1:
                            # Update existing entry
                            kb_data[existing_entry_index]['fix'] = new_fix_content
                            kb_data[existing_entry_index]['source'] = new_source
                            kb_data[existing_entry_index]['timestamp'] = datetime.now().isoformat()
                            st.info(f"Updated existing knowledge base entry for '{selected_error}'.")
                        else:
                            # Add new entry
                            new_kb_entry = {
                                "error": selected_error,
                                "fix": new_fix_content,
                                "source": new_source,
                                "timestamp": datetime.now().isoformat()
                            }
                            kb_data.append(new_kb_entry)
                            st.info(f"Added new knowledge base entry for '{selected_error}'.")
                        
                        save_json_file(KNOWLEDGE_BASE_FILE, kb_data)
                        # Re-index the vectorstore immediately after KB update
                        st.session_state.vectorstore = load_and_index_vectorstore()
                        st.success("Knowledge Base updated and re-indexed!")
                    
                    st.rerun() # Rerun to clear feedback options and show new state

        st.markdown("---")
        st.subheader("General Chat with LogBot")
        
        # Display chat history and potentially feedback options after each bot message
        for i, msg in enumerate(st.session_state.chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # If this is an assistant message AND it's the most recent one AND we have its details for feedback
                if msg["role"] == "assistant" and i == len(st.session_state.chat_history) - 1 and st.session_state.last_llm_response_details:
                    # Crucial check: Ensure this displayed message is indeed the one we want feedback on
                    # (to avoid showing feedback for old messages on rerun if a new interaction hasn't happened yet)
                    if st.session_state.last_llm_response_details["response_content"] == msg["content"]:
                        
                        # Only show feedback if a fix was actually suggested (not a "solution not present" message)
                        if st.session_state.last_llm_response_details["response_content"] != "Solution not present in Knowledge Base." and \
                           st.session_state.last_llm_response_details["response_content"] != "Could not generate a fix at this time.":
                            
                            # Generate a unique key for the feedback widgets to avoid conflicts
                            # Combining user query and timestamp of the response
                            # FIX: Encode the string before hashing
                            unique_chat_key = hashlib.md5(f"{st.session_state.last_llm_response_details['user_query']}-{st.session_state.last_llm_response_details['timestamp']}".encode('utf-8')).hexdigest()
                            
                            st.markdown("---")
                            st.markdown("#### Provide Feedback on this Suggestion")
                            feedback_chat = st.radio("Was this suggestion helpful?", ["👍 Yes", "👎 No"], horizontal=True, key=f"feedback_chat_radio_{unique_chat_key}")
                            auto_save_chat = st.checkbox("Automatically save helpful fixes to knowledge base", key=f"auto_save_chat_checkbox_{unique_chat_key}")
                            user_fix_chat = ""
                            if feedback_chat == "👎 No":
                                user_fix_chat = st.text_area("Optional: Describe how you fixed it yourself (this can be saved to the knowledge base)", placeholder="Steps or code you used...", key=f"user_fix_chat_textarea_{unique_chat_key}")

                            if st.button("Submit Feedback for this chat", key=f"submit_feedback_chat_button_{unique_chat_key}"):
                                feedback_entries_chat = load_json_file(FEEDBACK_LOG_FILE, [])
                                
                                # The 'error' for general chat feedback will be the user's original query (or stack trace)
                                error_for_feedback = st.session_state.last_llm_response_details["user_query"] 
                                suggested_fix_for_feedback = st.session_state.last_llm_response_details["response_content"]

                                feedback_entry_chat = {
                                    "error": error_for_feedback,
                                    "suggested_fix": suggested_fix_for_feedback,
                                    "feedback_type": "positive" if feedback_chat == "👍 Yes" else "negative",
                                    "user_fix": user_fix_chat if feedback_chat == "👎 No" else None,
                                    "timestamp": datetime.now().isoformat(),
                                    "email": st.session_state.email
                                }
                                feedback_entries_chat.append(feedback_entry_chat)
                                save_json_file(FEEDBACK_LOG_FILE, feedback_entries_chat)
                                st.success("✅ Feedback submitted for this chat! Thank you.")

                                # --- Knowledge Base Update Logic for General Chat ---
                                if (feedback_chat == "👍 Yes" and auto_save_chat and suggested_fix_for_feedback.strip()) or \
                                   (feedback_chat == "👎 No" and user_fix_chat.strip()):
                                    
                                    kb_data = load_json_file(KNOWLEDGE_BASE_FILE, [])
                                    
                                    # Use the user's query as the 'error' key for KB lookup
                                    # This handles both simple questions and full stack traces.
                                    error_kb_key = error_for_feedback 
                                    
                                    existing_entry_index_chat = -1
                                    # Check if an entry with this exact 'error_kb_key' already exists
                                    for kbi, entry_kb in enumerate(kb_data):
                                        if entry_kb.get('error', '').lower() == error_kb_key.lower():
                                            existing_entry_index_chat = kbi
                                            break

                                    new_fix_content_chat = suggested_fix_for_feedback.strip() if feedback_chat == "👍 Yes" else user_fix_chat.strip()
                                    new_source_chat = "GPT-4o (User Confirmed via Chat)" if feedback_chat == "👍 Yes" else "User Submitted via Chat"
                                    
                                    if existing_entry_index_chat != -1:
                                        # Update existing entry
                                        kb_data[existing_entry_index_chat]['fix'] = new_fix_content_chat
                                        kb_data[existing_entry_index_chat]['source'] = new_source_chat
                                        kb_data[existing_entry_index_chat]['timestamp'] = datetime.now().isoformat()
                                        st.info(f"Updated existing knowledge base entry for '{error_kb_key}'.")
                                    else:
                                        # Add new entry
                                        new_kb_entry_chat = {
                                            "error": error_kb_key,
                                            "fix": new_fix_content_chat,
                                            "source": new_source_chat,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        kb_data.append(new_kb_entry_chat)
                                        st.info(f"Added new knowledge base entry for '{error_kb_key}'.")
                                    
                                    save_json_file(KNOWLEDGE_BASE_FILE, kb_data)
                                    st.session_state.vectorstore = load_and_index_vectorstore() # Re-index after update
                                    st.success("Knowledge Base updated and re-indexed!")
                                
                                st.session_state.last_llm_response_details = None # Clear after feedback to avoid showing it again
                                st.rerun() # Rerun to remove feedback widgets and clean up
                            st.markdown("---") # Visual separator after feedback options block for current response

        user_input = st.chat_input("Ask about logs, pipelines, or paste errors...")
        if user_input:
            st.session_state.last_llm_response_details = None
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("LogBot is thinking..."):
                try:
                    rag_fix_chat = get_context_from_rag(user_input)
                    final_chat_reply = ""
                    source_chat_reply = "Knowledge Base"

                    if rag_fix_chat:
                        final_chat_reply = rag_fix_chat
                    else:
                        system_message_chat = (
                            "You are a highly intelligent and accurate Log Analysis Chatbot. Your primary goal is to assist users with data pipeline logs and errors. "
                            "Leverage the provided `logs.json` data for specific queries about pipeline status, failures, and statistics. "
                            "If a user provides an error stack trace or error message, analyze it carefully, consult the knowledge base for known fixes, and provide a precise, actionable solution. "
                            "Always aim for accuracy and be concise."
                        )
                        messages_for_llm = [{"role": "system", "content": system_message_chat}]
                        if any(keyword in user_input.lower() for keyword in ["log", "pipeline", "status", "failure", "error", "summary", "how many", "failed today"]):
                            log_context = build_pipeline_context(pipeline_logs)
                            messages_for_llm.append({"role": "system", "content": log_context})

                        messages_for_llm.extend(st.session_state.chat_history)

                        response = openai.ChatCompletion.create(
                            model="gpt-4o",
                            messages=messages_for_llm,
                            temperature=0.3,
                            max_tokens=1000
                        )
                        final_chat_reply = response['choices'][0]['message']['content']
                        source_chat_reply = "GPT-4o"

                    assistant_reply = final_chat_reply if final_chat_reply else "Solution not present in Knowledge Base."
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})

                    # ---- Enhanced Per-Fix Feedback Block ----
                    suggestions = re.split(r"\n(?=\s*(?:-|\*|\d+\.|💡|###|####|Fix\s*:))", assistant_reply.strip())

                    for idx, suggestion in enumerate(suggestions):
                        if suggestion.strip():
                            with st.chat_message("assistant"):
                                st.markdown(suggestion.strip())

                                feedback_key = hashlib.md5(
                                    f"{user_input}-{suggestion.strip()}-{idx}".encode("utf-8")
                                ).hexdigest()

                                st.markdown("##### 🙋 Feedback on this suggestion")
                                feedback_option = st.radio(
                                    "Was this helpful?",
                                    ["👍 Yes", "👎 No"],
                                    horizontal=True,
                                    key=f"feedback_radio_{feedback_key}"
                                )
                                user_fix = ""
                                if feedback_option == "👎 No":
                                    user_fix = st.text_area(
                                        "Optional: How did you resolve it instead?",
                                        key=f"user_fix_{feedback_key}",
                                        placeholder="Describe your fix or workaround..."
                                    )

                                if st.button("Submit Feedback", key=f"submit_feedback_{feedback_key}"):
                                    feedback_entries = load_json_file(FEEDBACK_LOG_FILE, [])
                                    feedback_entry = {
                                        "error": user_input,
                                        "suggested_fix": suggestion.strip(),
                                        "feedback_type": "positive" if feedback_option == "👍 Yes" else "negative",
                                        "user_fix": user_fix if feedback_option == "👎 No" else None,
                                        "timestamp": datetime.now().isoformat(),
                                        "email": st.session_state.email
                                    }
                                    feedback_entries.append(feedback_entry)
                                    save_json_file(FEEDBACK_LOG_FILE, feedback_entries)
                                    st.success("✅ Feedback submitted!")

                                    if (feedback_option == "👍 Yes" and suggestion.strip()) or (feedback_option == "👎 No" and user_fix.strip()):
                                        kb_data = load_json_file(KNOWLEDGE_BASE_FILE, [])
                                        existing_entry = next(
                                            (e for e in kb_data if e.get("error", "").lower() == user_input.lower()), None
                                        )
                                        new_fix = suggestion.strip() if feedback_option == "👍 Yes" else user_fix.strip()
                                        source = "User Feedback via Chat"
                                        if existing_entry:
                                            existing_entry["fix"] = new_fix
                                            existing_entry["source"] = source
                                            existing_entry["timestamp"] = datetime.now().isoformat()
                                        else:
                                            kb_data.append({
                                                "error": user_input,
                                                "fix": new_fix,
                                                "source": source,
                                                "timestamp": datetime.now().isoformat()
                                            })
                                        save_json_file(KNOWLEDGE_BASE_FILE, kb_data)
                                        st.session_state.vectorstore = load_and_index_vectorstore()
                                        st.success("Knowledge base updated with your feedback!")
                                    st.rerun()
                except Exception as e:
                    st.error(f"LogBot encountered an error: {e}")
                    st.session_state.chat_history.append({"role": "assistant", "content": "An error occurred. Please try again later."})








        # # Chat input at the bottom
        # user_input = st.chat_input("Ask about logs, pipelines, or paste errors...")
        # if user_input:
        #     # Clear previous LLM response details before a new interaction starts
        #     st.session_state.last_llm_response_details = None 

        #     # Append user message to chat history
        #     st.session_state.chat_history.append({"role": "user", "content": user_input})
            
        #     # Display user message immediately
        #     with st.chat_message("user"):
        #         st.markdown(user_input)

        #     with st.spinner("LogBot is thinking..."):
        #         try:
        #             # Attempt RAG search first to see if a direct KB fix is available
        #             rag_fix_chat = get_context_from_rag(user_input) # Returns direct fix or ""
                    
        #             final_chat_reply = ""
        #             source_chat_reply = "Knowledge Base"

        #             if rag_fix_chat: # If a relevant RAG fix was found
        #                 final_chat_reply = rag_fix_chat
        #             else: # No relevant RAG fix found, proceed to LLM
        #                 system_message_chat = "You are a highly intelligent and accurate Log Analysis Chatbot. Your primary goal is to assist users with data pipeline logs and errors. "\
        #                                    "Leverage the provided `logs.json` data for specific queries about pipeline status, failures, and statistics. "\
        #                                    "If a user provides an error stack trace or error message, analyze it carefully, consult the knowledge base for known fixes, and provide a precise, actionable solution. "\
        #                                    "Always aim for accuracy and be concise. If you don't have enough information, ask clarifying questions."

        #                 messages_for_llm = [{"role": "system", "content": system_message_chat}]
                        
        #                 # Add relevant context from logs based on user query keywords
        #                 log_context_for_llm = ""
        #                 if any(keyword in user_input.lower() for keyword in ["log", "pipeline status", "failure", "error", "summary", "how many", "which pipelines", "failed today", "list errors"]):
        #                     log_context_for_llm = build_pipeline_context(pipeline_logs)
        #                     messages_for_llm.append({"role": "system", "content": f"Here is a summary of recent pipeline executions from logs.json:\n{log_context_for_llm}"})

        #                 # Add full chat history to maintain conversation flow and provide context to LLM
        #                 # Ensure the most recent user message is always the last, possibly augmented by initial RAG if we decided to pass it
        #                 messages_for_llm.extend(st.session_state.chat_history) 
                        
        #                 # If a relevant KB fix was found, we could pass it to the LLM for augmentation/validation,
        #                 # but if we want to *prioritize* KB, we return it directly.
        #                 # Since `rag_fix_chat` is empty here, we don't pass it as direct context.
        #                 # The LLM will rely on its general knowledge and the system message.

        #                 response = openai.ChatCompletion.create(
        #                     model="gpt-4o",
        #                     messages=messages_for_llm,
        #                     temperature=0.3, # Keep temperature low for factual/accurate responses
        #                     max_tokens=1000
        #                 )
        #                 final_chat_reply = response['choices'][0]['message']['content']
        #                 source_chat_reply = "GPT-4o"
                        
        #                 # If LLM produces a very generic response or nothing specific, and no KB fix was found,
        #                 # we might want to explicitly state "Solution not present".
        #                 # This heuristic is simple: if the LLM's reply is short or generic AND no KB fix was found.
        #                 # This part might need fine-tuning. For now, let's just use the LLM's reply as is.
        #                 # Removed the heuristic for now to allow LLM to always attempt a general answer if KB is empty.
        #                 pass

        #             # After determining final_chat_reply
        #             # If RAG found nothing, and LLM was not invoked (because it's only invoked if RAG fails), 
        #             # OR if RAG found nothing and LLM also produced a very unhelpful general response (optional future refinement)
        #             if not rag_fix_chat and source_chat_reply == "Knowledge Base": # This condition is met if get_context_from_rag returned ""
        #                 assistant_reply = "Solution not present in Knowledge Base."
        #             else:
        #                 assistant_reply = final_chat_reply

        #             st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
                    
        #             # Store details of this LLM response for potential feedback
        #             st.session_state.last_llm_response_details = {
        #                 "user_query": user_input, # Store the exact user query for feedback context
        #                 "response_content": assistant_reply,
        #                 "timestamp": datetime.now().isoformat()
        #             }

        #         except Exception as e:
        #             st.error(f"OpenAI Error: {e}")
        #             assistant_reply = "I'm sorry, I encountered an error and cannot respond right now. Please try again later."
        #             st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        #             st.session_state.last_llm_response_details = None # If error, no feedback is possible for this turn

        #     # Rerun the app to display the newly added assistant message and the feedback options
        #     # The loop at the top of this section will render the updated chat history including feedback.
        #     st.rerun()


# ----------------------- Knowledge Base Page -----------------------
elif page == "Knowledge Base":
    st.subheader("🧠 Current Knowledge Base")
    
    kb_data = load_json_file(KNOWLEDGE_BASE_FILE, [])

    if not kb_data:
        st.info("No knowledge base entries found. Add fixes from the 'Chat with LogBot' page or manually add them here.")
    else:
        st.write("Edit or delete knowledge base entries:")
        
        # Search filter for the knowledge base
        search_term = st.text_input("Search Knowledge Base (by error or fix content)", key="kb_search").lower()
        
        filtered_kb_data = [
            entry for entry in kb_data 
            if search_term in entry.get('error', '').lower() or 
               search_term in entry.get('fix', '').lower()
        ]

        # Display and allow editing/deleting entries
        for i, entry in enumerate(filtered_kb_data):
            # Using a unique hash for expanders and inputs
            unique_id_kb = f"{hashlib.md5(entry.get('error', '').encode('utf-8')).hexdigest()}_{i}" # Ensure encoding here too for consistency
            with st.expander(f"**Error**: {entry.get('error', 'N/A')} (Source: {entry.get('source', 'N/A')})", expanded=False):
                st.text_input(f"Error Name", value=entry.get('error', ''), key=f"edit_error_{unique_id_kb}")
                edited_fix = st.text_area(f"Fix Content", value=entry.get('fix', ''), key=f"edit_fix_{unique_id_kb}", height=100)
                st.markdown(f"**Source**: {entry.get('source', 'N/A')} | **Last Updated**: {entry.get('timestamp', 'N/A')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Update Entry", key=f"update_kb_{unique_id_kb}"):
                        # Find the original entry in the full kb_data list based on its original error name
                        for kb_idx, kb_entry in enumerate(kb_data):
                            if kb_entry.get('error', '') == entry.get('error', ''): # Match original error name
                                kb_data[kb_idx]['error'] = st.session_state[f"edit_error_{unique_id_kb}"]
                                kb_data[kb_idx]['fix'] = edited_fix
                                kb_data[kb_idx]['source'] = "Manually Edited"
                                kb_data[kb_idx]['timestamp'] = datetime.now().isoformat()
                                break
                        save_json_file(KNOWLEDGE_BASE_FILE, kb_data)
                        st.session_state.vectorstore = load_and_index_vectorstore() # Re-index after update
                        st.success("Knowledge base updated and re-indexed!")
                        st.rerun() # Rerun to reflect changes
                with col2:
                    if st.button(f"Delete Entry", key=f"delete_kb_{unique_id_kb}"):
                        # Create a new list excluding the entry to be deleted
                        kb_data = [e for e in kb_data if not (e.get('error', '') == entry.get('error', '') and e.get('fix', '') == entry.get('fix', ''))] # More robust deletion criteria
                        save_json_file(KNOWLEDGE_BASE_FILE, kb_data)
                        st.session_state.vectorstore = load_and_index_vectorstore() # Re-index after deletion
                        st.success("Entry deleted and knowledge base re-indexed!")
                        st.rerun() # Rerun to reflect changes
            st.markdown("---") # Separator between entries
            
        st.markdown("### Add New Knowledge Base Entry")
        new_error_name = st.text_input("New Error Name:", key="new_error_name")
        new_fix_content = st.text_area("New Fix Content:", key="new_fix_content", height=100)
        
        if st.button("Add New Entry to KB"):
            if new_error_name and new_fix_content:
                kb_data = load_json_file(KNOWLEDGE_BASE_FILE, [])
                
                # Check for existing entry (case-insensitive) to prevent duplicates
                if any(entry.get('error', '').lower() == new_error_name.lower() for entry in kb_data):
                    st.warning("An entry with this error name already exists. Please update the existing one or use a different name.")
                else:
                    new_entry = {
                        "error": new_error_name,
                        "fix": new_fix_content,
                        "source": "Manually Added",
                        "timestamp": datetime.now().isoformat()
                    }
                    kb_data.append(new_entry)
                    save_json_file(KNOWLEDGE_BASE_FILE, kb_data)
                    st.session_state.vectorstore = load_and_index_vectorstore() # Re-index after adding
                    st.success("New entry added and knowledge base re-indexed!")
                    st.rerun() # Rerun to clear inputs and show updated KB
            else:
                st.error("Please provide both an error name and fix content.")


# ----------------------- Feedback Insights -----------------------
elif page == "Feedback Insights":
    st.subheader("📊 User Feedback")
    feedback_entries = load_json_file(FEEDBACK_LOG_FILE, [])

    if not feedback_entries:
        st.info("No feedback available yet.")
    else:
        # Calculate positive and negative feedback counts
        positives = sum(1 for entry in feedback_entries if entry.get("feedback_type") == "positive")
        negatives = sum(1 for entry in feedback_entries if entry.get("feedback_type") == "negative")
        
        st.metric("👍 Helpful Fixes", positives)
        st.metric("👎 Not Helpful Fixes", negatives)
        st.divider()
        st.markdown("#### Recent Feedback Entries")
        
        # Display latest 10 feedback entries
        for entry in reversed(feedback_entries[-10:]): # Show latest 10, most recent first
            with st.expander(f"Feedback for '{entry.get('error', 'N/A')}' on {entry.get('timestamp', 'N/A').split('T')[0]}"): # Show only date in expander title
                st.json(entry)

# ----------------------- Footer -----------------------
st.markdown("---")
st.caption("Built with ❤️ using GPT-4o and LangChain RAG")
