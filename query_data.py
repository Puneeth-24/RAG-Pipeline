import argparse

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # get the query from the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # prepare the db
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=gemini_embeddings)

    # search the db
    result = db.similarity_search_with_relevance_scores(query_text, k=3)

    context_text = "\n-------\n".join([doc.page_content for doc, _score in result])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text).lower()

    print(prompt)
    model = GoogleGenerativeAI(model="gemini-2.5-flash")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in result]
    formatted_response = f"Response : {response_text} \nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
