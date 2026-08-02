import streamlit as st
import fitz
from google import genai

# ==========================
# Gemini API
# ==========================
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

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

.main{
    background-color:#f5f7fa;
}

h1{
    color:#1f4e79;
    text-align:center;
}

.block-container{
    padding-top:2rem;
}

.stChatMessage{
    border-radius:12px;
    padding:10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Chat History
# ==========================
if "messages" not in st.session_state:
    st.session_state.messages=[]

# ==========================
# Sidebar
# ==========================
st.sidebar.title("🤖 AI Educational Chatbot")
st.sidebar.markdown("---")

st.sidebar.info("""
📚 Ask questions from any PDF.

Powered by:
- Gemini AI
- Streamlit
- PyMuPDF
""")

st.sidebar.markdown("---")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages=[]
    st.rerun()

# ==========================
# Title
# ==========================
st.title("📚 Educational QA Chatbot")

st.success("Upload any PDF and start chatting!")

# ==========================
# Upload PDF
# ==========================
uploaded_file=st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf=fitz.open(stream=uploaded_file.read(),filetype="pdf")

    text=""

    for page in pdf:
        text+=page.get_text()

    st.success("✅ PDF Loaded Successfully!")

    # Show old chat
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question=st.chat_input("Ask your question...")

    if question:

        st.session_state.messages.append(
            {
                "role":"user",
                "content":question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        prompt=f"""
Answer ONLY from the uploaded PDF.

If answer is not available, say:

Sorry, I couldn't find this in the uploaded PDF.

PDF:

{text}

Question:

{question}
"""

        with st.spinner("🤖 Thinking..."):

            response=client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )

        answer=response.text

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":answer
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