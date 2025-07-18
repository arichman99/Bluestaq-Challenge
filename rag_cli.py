import click
import logging
import time
import torch
import psutil
import re
import numpy as np
import json
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.dataclasses import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz

# setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# initialize document store
document_store = InMemoryDocumentStore()

# sample corpus
with open("corpus.json", "r") as f:
	corpus_data = json.load(f)
corpus = [Document(content=doc["content"], meta=doc["meta"]) for doc in corpus_data]

# initialize document embedder
document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L12-v2",
    device=ComponentDevice.from_str("cuda" if torch.cuda.is_available() else "cpu")
)
document_embedder.warm_up()
embedded_documents = document_embedder.run(corpus)["documents"]
document_store.write_documents(embedded_documents)

# initialize retrieval pipeline
def create_retrieval_pipeline():
    pipeline = Pipeline()
    pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=document_store, top_k=10))
    pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L12-v2",
        device=ComponentDevice.from_str("cuda" if torch.cuda.is_available() else "cpu")
    ))
    pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=document_store, top_k=10))
    pipeline.add_component("retriever", DocumentJoiner(join_mode="merge", weights=[0.6, 0.4]))
    pipeline.connect("bm25.documents", "retriever.documents")
    pipeline.connect("dense.documents", "retriever.documents")
    pipeline.connect("query_embedder.embedding", "dense.query_embedding")
    return pipeline

retrieval_pipeline = create_retrieval_pipeline()

# initialize generation pipeline
def create_generation_pipeline():
    pipeline = Pipeline()
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model = model.to(device)
    generator = HuggingFaceLocalGenerator(
        model="google/flan-t5-base",
        task="text2text-generation",
        generation_kwargs={
            "max_new_tokens": 100,
            "num_beams": 5,
            "no_repeat_ngram_size": 2,
            "do_sample": False,
            "truncation": True,
        },
        huggingface_pipeline_kwargs={
            "model": model,
            "tokenizer": tokenizer,
            "device": 0 if device == "cuda" else -1
        }
    )
    pipeline.add_component("generator", generator)
    return pipeline

generation_pipeline = create_generation_pipeline()

# initialize sentence transformer for semantic similarity
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

def reformulate_query(query):
    query = query.strip().rstrip('?')
    if not query.lower().startswith(("what", "where", "who", "how", "when", "why")):
        logger.info("Query does not start with a question word. Reformulating.")
        if len(query.split()) == 1:
            return f"What is the capital city of the country related to {query}?"
        # Match "starts with" followed by a letter (optionally in quotes)
        starts_with_match = re.search(r"starts with\s+['‘’]?([a-zA-Z])['‘’]?", query.lower())
        if starts_with_match:
            letter = starts_with_match.group(1).upper()
            return f"Which European country has a capital city starting with the letter {letter}?"
        # Match "ends with" followed by a letter (optionally in quotes)
        ends_with_match = re.search(r"ends with\s+['‘’]?([a-zA-Z])['‘’]?", query.lower())
        if ends_with_match:
            letter = ends_with_match.group(1).lower()
            return f"Which European country has a capital city ending with the letter {letter}?"
        return f"What is {query}?"
    return query

def normalize_answer(answer, expected):
    norm_answer = answer.strip().rstrip('.').lower()
    norm_expected = expected.strip().rstrip('.').lower()
    # check string containment
    if norm_answer in norm_expected or norm_expected in norm_answer:
        return True
    # check fuzzy matching
    if fuzz.ratio(norm_answer, norm_expected) > 75:
        return True
    # check semantic similarity
    embeddings = similarity_model.encode([answer, expected], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity > 0.65

def check_guardrails(query):
    forbidden_patterns = [
        r"\b(politics|violence|hate\s*speech|religion)\b",
        r"\b(terrorism|war|discrimination)\b"
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, query.lower()):
            return {"error": f"Query contains prohibited content matching pattern: {pattern}"}
    return None

def run_rag(query):
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu = process.cpu_percent(interval=None)

    original_query = query
    if not query.strip():
        return {"error": "Please provide a non-empty query."}

    guardrail_result = check_guardrails(query)
    if guardrail_result:
        return guardrail_result

    query = reformulate_query(query)
    logger.info(f"Reformulated query: {query}")
    out = retrieval_pipeline.run({"bm25": {"query": query}, "query_embedder": {"text": query}})
    docs = sorted(out["retriever"]["documents"], key=lambda x: x.score, reverse=True)[:10]
    logger.info(f"Retrieved documents: {[(d.meta['title'], d.score) for d in docs]}")
    if not docs:
        return {"error": "No relevant documents found."}

    context = "\n".join(f"{d.content}" for d in docs)
    prompt = (
        "Using the provided passages, answer the question concisely and accurately. "
        "For queries about capital cities, respond with: 'The capital of [Country] is [Capital].' "
        "For queries about landmarks, respond with: '[Landmark] is in [City], [Country].' "
        "For queries requiring reasoning (e.g., capitals starting or ending with a letter), respond with: 'The capital of [Country] is [Capital].' "
        "For descriptive queries, provide a concise sentence including relevant details. "
        "If no relevant information is found, respond with: 'I cannot find an answer based on the provided information.' "
        "Do not include document titles, brackets, or placeholder text in the answer.\n\n"
        f"Passages:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    logger.info(f"Prompt: {prompt}")
    gen_out = generation_pipeline.run({"generator": {"prompt": prompt}})
    answer = gen_out["generator"]["replies"][0].strip().rstrip('.')
    
    # fallback if answer contains placeholders
    if "[Capital]" in answer or "[Country]" in answer or "[Landmark]" in answer:
        answer = "I cannot find an answer based on the provided information."
        logger.warning(f"Fallback triggered for query: {query}")

    sources = [(doc.meta["title"], doc.meta["source"], doc.score) for doc in docs if any(term.lower() in answer.lower() for term in doc.content.split())]
    
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    end_cpu = process.cpu_percent(interval=None) / psutil.cpu_count()
    return {
        "query": original_query,
        "reformulated_query": query,
        "documents": [(d.meta["title"], d.content, d.meta["source"], d.score) for d in docs],
        "answer": answer,
        "sources": sources,
        "latency_s": time.time() - start_time,
        "memory_mb": end_memory,
        "cpu_percent": end_cpu
    }

def evaluate_model():
    test_queries = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Berlin", "Berlin is the capital of Germany."),
        ("Capital of Spain", "The capital of Spain is Madrid."),
        ("Which capital begins with B?", "Bern is the capital of Switzerland."),
        ("What country's capital is Helsinki?", "Finland's capital is Helsinki."),
        ("Name the capital of Portugal.", "The capital of Portugal is Lisbon."),
        ("Eiffel Tower", "The Eiffel Tower is in Paris, France."),
        ("Where is the Colosseum located?", "The Colosseum is located in Rome, Italy."),
        ("Tell me about Big Ben", "Big Ben is the nickname for the huge bell in London, England."),
        ("What and where is Sagrada Família?", "The Sagrada Família is an unfinished basilica in Barcelona, Spain."),
        ("Which city is the seat of government of Italy?", "Rome is the seat of government of Italy."),
        ("Name a famous landmark in Athens", "The Acropolis is a famous landmark in Athens, Greece."),
        ("Country whose capital starts with 'A'?", "Austria's capital is Vienna."),
        ("Which European capital ends with 'o'?", "Rome ends with 'o'."),
        ("Describe Denmark in one sentence.", "Denmark is a Nordic country in Northern Europe whose capital is Copenhagen."),
    ]

    results = []
    for query, expected in test_queries:
        result = run_rag(query)
        if "error" in result:
            results.append({"query": query, "success": False, "error": result["error"]})
        else:
            success = normalize_answer(result["answer"], expected)
            results.append({
                "query": query,
                "success": success,
                "answer": result["answer"],
                "expected": expected,
                "latency_s": result["latency_s"],
                "memory_mb": result["memory_mb"],
                "cpu_percent": result["cpu_percent"]
            })
    accuracy = sum(1 for r in results if r["success"]) / len(results) * 100
    avg_latency = sum(r["latency_s"] for r in results if "latency_s" in r) / len([r for r in results if "latency_s" in r])
    avg_memory = sum(r["memory_mb"] for r in results if "memory_mb" in r) / len([r for r in results if "memory_mb" in r])
    avg_cpu = sum(r["cpu_percent"] for r in results if "cpu_percent" in r) / len([r for r in results if "cpu_percent" in r])
    click.echo(f"\n=================== Evaluation Results ===================")
    click.echo(f"Accuracy: {accuracy:.2f}%")
    click.echo(f"Average Latency: {avg_latency:.2f} seconds")
    click.echo(f"Average Memory Usage: {avg_memory:.2f} MB")
    click.echo(f"Average CPU Usage: {avg_cpu:.2f}%")
    click.echo("==========================================================")
    for r in results:
        click.echo(f"\nQuery: {r['query']}")
        if r["success"] or "error" not in r:
            click.echo(f"Answer: {r['answer']} (Expected: {r['expected']})")
            click.echo(f"Latency: {r['latency_s']:.2f} seconds")
            click.echo(f"Memory Usage: {r['memory_mb']:.2f} MB")
            click.echo(f"CPU Usage: {r['cpu_percent']:.2f}%")
        else:
            click.echo(f"Error: {r['error']}")
            click.echo("----------------------------------------------------------")
    return results

@click.command()
@click.option('--evaluate', is_flag=True, help='Run evaluation on test set.')
def cli_main(evaluate):
    if evaluate:
        evaluate_model()
    else:
        click.echo("\nWelcome to Adam's RAG CLI!")
        click.echo("This tool retrieves relevant documents using BM25 and dense retrieval, generating answers with google/flan-t5-base.")
        click.echo("\nUsage:")
        click.echo("  - Enter a query to get a response.")
        click.echo("  - Type 'exit' or 'quit' to stop, or press Ctrl+C.")
        click.echo("  - Run evaluation with: python rag_cli.py --evaluate")
        click.echo("Example queries: 'What is the capital of France?', 'Berlin', 'Eiffel Tower'")
        click.echo("=================== Start Query ===================")
        while True:
            try:
                query = input("Enter your query: ")
                if query.lower() in ["exit", "quit"]:
                    click.echo("=================== Exiting ===================")
                    break
                result = run_rag(query)
                if "error" in result:
                    click.echo(f"Error: {result['error']}")
                else:
                    click.echo("\n=================== Query ===================")
                    click.echo(f"\nQuery: {result['query']}")
                    if result['query'] != result['reformulated_query']:
                        click.echo(f"Reformulated Query: {result['reformulated_query']}")
                    click.echo("\n=================== Retrieved Documents ===================")
                    for i, (title, content, source, score) in enumerate(result['documents'], 1):
                        click.echo(f"{i}. [{score:.3f}] {title} — {source}\n   {content}")
                    click.echo("\n=================== Generated Response ===================")
                    click.echo(f"Answer: {result['answer']}")
                    if result.get("sources"):
                        click.echo("\n=================== Sources ===================")
                        for i, (title, source, score) in enumerate(result["sources"], 1):
                            click.echo(f"{i}. [{score:.3f}] {title} — {source}")
                    click.echo("\n=================== Performance Metrics ===================")        
                    click.echo(f"Response time: {result['latency_s']:.2f} seconds")
                    click.echo(f"Memory Usage: {result['memory_mb']:.2f} MB")
                    click.echo(f"CPU Usage: {result['cpu_percent']:.2f}%")
                    click.echo("=================== End Query ===================")
            except KeyboardInterrupt:
                click.echo("\n=================== Exiting ===================")
                break
            except Exception as e:
                click.echo(f"Unexpected error: {str(e)}")
                logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    cli_main()
