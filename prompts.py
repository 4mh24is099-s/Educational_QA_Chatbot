def qa_prompt(context, question):
    return f"""
You are an educational assistant.

Your primary source is the uploaded PDF.

Instructions:
1. Read the context carefully.
2. If the answer is fully available in the context, answer ONLY from the context.
3. If the context contains partial information, first explain everything available from the context, then clearly add any missing details under the heading "Additional Explanation".
4. If the context is completely empty or unrelated, reply:
   "This topic is not available in the uploaded PDF."
5. Keep the answer simple and suitable for students.
6. Use bullet points whenever appropriate.
7. Do not invent information that contradicts the context.

Context:
{context}

Question:
{question}

Answer:
"""


def summary_prompt(context):
    return f"""
You are an educational assistant.

Summarize the following content into clear, easy-to-understand bullet points for students.

Context:
{context}
"""


def flashcard_prompt(context):
    return f"""
You are an educational assistant.

Create 10 flashcards from the following content.

Format:

Q:
A:

Context:
{context}
"""


def mcq_prompt(context):
    return f"""
You are an educational assistant.

Generate 10 multiple-choice questions from the following content.

Each question should have:
A)
B)
C)
D)

Mention the correct answer after each question.

Context:
{context}
"""