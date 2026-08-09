import streamlit as st
from google import genai

from rag import (
    extract_text,
    split_text,
    create_vector_store,
    search
)

from prompts import (
    qa_prompt,
    summary_prompt,
    flashcard_prompt,
    mcq_prompt
)


# ==========================
# Gemini API
# ==========================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="📚 Educational QA Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ==========================
# Custom CSS
# ==========================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.block-container {
    padding-top: 2rem;
}

h1 {
    color: #1f4e79;
    text-align: center;
}

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================
# Session State
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "text" not in st.session_state:
    st.session_state.text = ""


# ==========================
# Sidebar
# ==========================

st.sidebar.title("🤖 AI Educational Chatbot")
st.sidebar.markdown("---")

st.sidebar.info("""
📚 Upload a PDF and ask questions.

Powered by:

- Gemini AI
- FAISS
- Sentence Transformers
- PyMuPDF
""")

st.sidebar.markdown("---")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()


# ==========================
# Title
# ==========================

st.title("📚 Educational QA Chatbot")

st.success("Upload any PDF and start chatting!")


# ==========================
# Upload PDF
# ==========================

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)


if uploaded_file:

    st.success(
        f"📄 Uploaded PDF: {uploaded_file.name}"
    )

    # ==========================
    # Process PDF
    # ==========================

    if st.session_state.vector_store is None:

        with st.spinner("📖 Reading PDF..."):

            text = extract_text(uploaded_file)

            chunks = split_text(text)

            vector_store = create_vector_store(chunks)

            st.session_state.vector_store = vector_store
            st.session_state.chunks = chunks
            st.session_state.text = text

        st.success(
            f"✅ PDF Loaded ({len(chunks)} chunks)"
        )


    # ==========================
    # Study Tools
    # ==========================

    text = st.session_state.text

    st.markdown("## 📚 Study Tools")

    col1, col2, col3 = st.columns(3)


    # ==========================
    # Summary
    # ==========================

    with col1:

        if st.button("📝 Summary"):

            with st.spinner("Generating Summary..."):

                prompt = summary_prompt(text)

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt
                )

                st.subheader("📄 Summary")

                st.write(response.text)


    # ==========================
    # Flashcards
    # ==========================

    with col2:

        if st.button("🎴 Flashcards"):

            with st.spinner("Generating Flashcards..."):

                prompt = flashcard_prompt(text)

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt
                )

                st.subheader("🎴 Flashcards")

                st.write(response.text)


    # ==========================
    # MCQs
    # ==========================

    with col3:

        if st.button("❓ MCQs"):

            with st.spinner("Generating MCQs..."):

                prompt = mcq_prompt(text)

                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt
                )

                st.subheader("❓ MCQs")

                st.write(response.text)


    # ==========================
    # Display Chat History
    # ==========================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])


    # ==========================
    # Chat Input
    # ==========================

    question = st.chat_input(
        "💬 Ask a question about the uploaded PDF..."
    )


    if question:

        # Display user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # ==========================
        # Search PDF + Gemini
        # ==========================

        with st.spinner("🔍 Searching PDF..."):

            context = search(
                question,
                st.session_state.vector_store,
                st.session_state.chunks,
                k=3
            )

            prompt = qa_prompt(
                context,
                question
            )

            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )

            answer = response.text


        # ==========================
        # Save Assistant Response
        # ==========================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):

            st.markdown(answer)


else:

    st.info("⬆️ Upload a PDF to begin.")


# ==========================
# Footer
# ==========================

st.markdown("---")

st.caption("❤️ Developed by Sinchana S")