"""
Vector RAG Evaluation
Evaluates the ChromaDB-based RAG system using RAGAS metrics.
"""

# Standard library imports
import asyncio
from pathlib import Path

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from litellm import completion
from openai import AsyncOpenAI

# RAGAS imports
from ragas import experiment
from ragas.dataset import Dataset
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# Local imports
from embeddings import create_embeddings
from chromadb_utils import (
    get_chroma_client,
    get_or_create_collection,
    retrieve_from_vector_db
)
import os

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


class VectorRAGClient:
    """ChromaDB-based RAG implementation."""

    def __init__(self, collection_name="documents", model="openai/@openrouter-09cb28/openai/gpt-4.1-mini"):
        self.model = model
        self.collection_name = collection_name

        # Initialize ChromaDB
        self.client = get_chroma_client()
        self.collection = get_or_create_collection(self.client, collection_name)

    def retrieve(self, query: str, n_results: int = 3):
        """Retrieve relevant documents from vector database."""
        # Create embedding for the query
        query_embedding = create_embeddings([query])[0]

        # Retrieve from vector database
        results = retrieve_from_vector_db(query_embedding, self.collection, n_results=n_results)

        # Combine retrieved documents
        if results['documents'] and len(results['documents'][0]) > 0:
            retrieved_docs = results['documents'][0]
            return retrieved_docs
        else:
            return []

    def query(self, query: str, n_results: int = 3):
        """RAG pipeline: retrieve + generate."""
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, n_results=n_results)

        if not retrieved_docs:
            return {
                "answer": "No relevant documents found.",
                "logs": "No documents retrieved"
            }

        context = "\n\n".join(retrieved_docs)

        # Generate answer using LLM
        prompt = f"""<context> {context} </context>

Based on the above context, answer the user query: {query}"""

        messages = [
            {"role": "system", "content": "You are a helpful RAG assistant. Answer questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = completion(
                model=self.model,
                messages=messages,
                api_key=os.getenv("PORTKEY_API_KEY"),
                api_base=os.getenv("PORTKEY_API_BASE"),
            )
            answer = response.choices[0].message.content.strip()

            return {
                "answer": answer,
                "logs": f"Retrieved {len(retrieved_docs)} documents from ChromaDB"
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "logs": f"Error occurred: {str(e)}"
            }


# Initialize RAG client
rag_client = VectorRAGClient()

# Create LLM for metrics
client = AsyncOpenAI()
llm = llm_factory("gpt-4o-mini", client=client)

# Define evaluation metrics
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response}\nGrading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)

context_relevance_metric = DiscreteMetric(
    name="context_relevance",
    prompt="Check if the response is based on the context and answers the question. Return 'relevant' if the response uses the context to answer, 'irrelevant' if not.\nResponse: {response}",
    allowed_values=["relevant", "irrelevant"],
)


@experiment()
async def run_experiment(row):
    """Run RAG experiment on a single row."""
    response = rag_client.query(row["query"])

    # Score correctness
    correctness_score = await correctness_metric.ascore(
        llm=llm,
        response=response.get("answer", ""),
        grading_notes=row["grading_notes"]
    )

    # Score context relevance
    relevance_score = await context_relevance_metric.ascore(
        llm=llm,
        response=response.get("answer", "")
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "correctness": correctness_score.value,
        "relevance": relevance_score.value,
        "log_file": response.get("logs", ""),
    }
    return experiment_view


if __name__ == "__main__":
    # Create test dataset with Harry Potter questions
    # These are based on the first 20 pages we indexed
    samples = [
        {
            "query": "Who are the Dursleys?",
            "grading_notes": "- The Dursleys are a family living at Privet Drive - Mr. Dursley works at Grunnings - They are related to Harry Potter"
        },
        {
            "query": "What does Mr. Dursley do for work?",
            "grading_notes": "- Mr. Dursley is the director of Grunnings - Grunnings makes drills"
        },
        {
            "query": "Where do the Dursleys live?",
            "grading_notes": "- The Dursleys live at number four, Privet Drive"
        },
        {
            "query": "How are the Potters related to the Dursleys?",
            "grading_notes": "- Mrs. Potter is Mrs. Dursley's sister - The Dursleys pretend they don't have a sister"
        },
        {
            "query": "What is unusual happening in the story?",
            "grading_notes": "- Strange and mysterious things are happening - There are mentions of magical events - People in cloaks and owls flying during the day"
        }
    ]

    dataset_file = DATASETS_DIR / "harry_potter_rag_test.csv"
    pd.DataFrame(samples).to_csv(dataset_file, index=False)
    print(f"Created test dataset at: {dataset_file}")

    # Load dataset
    df = pd.read_csv(dataset_file)
    dataset = Dataset.from_pandas(
        df,
        name="harry_potter_queries",
        backend="local/csv",
        root_dir=str(DATASETS_DIR)
    )

    # Run the experiment
    print("\nRunning RAG evaluation experiment...")
    print("=" * 70)
    asyncio.run(run_experiment.arun(dataset, name="vector_rag_evaluation"))
    print("\nEvaluation complete!")
