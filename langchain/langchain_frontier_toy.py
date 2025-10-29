import os
import argparse
import json
from pathlib import Path
from typing import Optional, List

# ---- Enforce offline mode (Frontier compute nodes typically lack egress) ----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- Torch / Transformers ----
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# ---- LangChain core / community / huggingface ----
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

# BM25 fallback retriever (no embeddings needed)
from langchain_community.retrievers import BM25Retriever


# ------------------------------ Utilities ------------------------------

def detect_device(device_arg: str) -> str:
    """
    Resolve device selection:
    - "auto": use "cuda" if available (ROCm build on Frontier exposes CUDA device), else "cpu"
    - explicit: "cuda" or "cpu"
    """
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg in {"cuda", "cpu"}:
        return device_arg
    raise ValueError(f"Unsupported device: {device_arg}")


def resolve_dtype(dtype_arg: str):
    if dtype_arg.lower() in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_arg.lower() in {"fp16", "float16"}:
        return torch.float16
    if dtype_arg.lower() in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def ensure_toy_docs(docs_dir: Path) -> None:
    """
    Create a tiny doc set if the user didn't supply one.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    samples = {
        "frontier_overview.txt": (
            "Frontier is an exascale supercomputer at OLCF. "
            "Each node has an AMD EPYC CPU and four AMD Instinct MI250X GPUs, "
            "with HPE Slingshot interconnect and a Lustre parallel filesystem."
        ),
        "rag_validation.txt": (
            "This file is used to validate a small-scale RAG pipeline on a single node. "
            "We test LangChain retrieval, vector store indexing, and end-to-end QA."
        ),
    }
    for name, text in samples.items():
        p = docs_dir / name
        if not p.exists():
            p.write_text(text, encoding="utf-8")


# ------------------------------ Model Builders ------------------------------

def build_local_llm_pipeline(
    llm_model_dir: Path,
    device: str,
    dtype,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> HuggingFacePipeline:
    """
    Loads a local causal LLM and wraps it as a LangChain LLM via HuggingFacePipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        str(llm_model_dir), local_files_only=True, use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    # If CPU or single device, place explicitly
    if device == "cpu":
        model = model.to("cpu")

    hf_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0),
        # trust_remote_code not needed if using standard models; ensure offline safety:
        # trust_remote_code=False
    )
    return HuggingFacePipeline(pipeline=hf_pipe)


def build_embeddings_or_none(emb_model_dir: Optional[Path]) -> Optional[HuggingFaceEmbeddings]:
    """
    Try to build a local embeddings model. Return None if unavailable, so we can fall back to BM25.
    """
    if emb_model_dir is None:
        return None
    if not emb_model_dir.exists():
        return None
    try:
        return HuggingFaceEmbeddings(
            model_name=str(emb_model_dir),
            cache_folder=str(emb_model_dir),
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"[WARN] Could not initialize embeddings from {emb_model_dir}: {e}")
        return None


# ------------------------------ Data & Indexing ------------------------------

def load_and_split_docs(docs_dir: Path):
    """
    Load .txt (and .md) files and split into chunks for RAG.
    """
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=False,
        use_multithreading=False,
    )
    docs = loader.load()
    if not docs:
        print(f"[INFO] No .txt docs found in {docs_dir}, creating toy docs.")
        ensure_toy_docs(docs_dir)
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def build_retriever(
    split_docs,
    emb: Optional[HuggingFaceEmbeddings],
    faiss_index_dir: Path,
    k: int = 4,
):
    """
    Build a retriever using FAISS+embeddings if available; else BM25 (no embeddings).
    """
    if emb is not None:
        faiss_index_dir.mkdir(parents=True, exist_ok=True)
        vs = FAISS.from_documents(split_docs, emb)
        # Persist FAISS index (CPU index by default)
        vs.save_local(str(faiss_index_dir))
        retriever = vs.as_retriever(search_kwargs={"k": k})
        print("[INFO] Using FAISS+HF embeddings retriever.")
        return retriever
    else:
        print("[INFO] Embeddings unavailable -> Falling back to BM25 retriever.")
        bm25 = BM25Retriever.from_documents(split_docs)
        bm25.k = k
        return bm25


# ------------------------------ Chains ------------------------------

def build_qa_chain(llm: HuggingFacePipeline):
    """
    Simple zero-RAG QA chain (no retrieval).
    """
    prompt = ChatPromptTemplate.from_template(
        "You are a concise scientific assistant. Answer the user's question.\n\nQuestion: {query}\nAnswer:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain


def build_rag_chain(llm: HuggingFacePipeline, retriever):
    """
    Classic RAG (retrieve -> synthesize) using LCEL.
    """
    system_prompt = (
        "You are a scientific assistant. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    prompt = ChatPromptTemplate.from_template(system_prompt)

    def join_docs(docs) -> str:
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever | (lambda docs: join_docs(docs)),
         "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm_model_dir", type=str, required=True,
                    help="Path to local HF causal LLM directory (no internet).")
    ap.add_argument("--emb_model_dir", type=str, default="",
                    help="Path to local HF embeddings directory (optional; fallback to BM25 if missing).")
    ap.add_argument("--docs_dir", type=str, default="./toy_docs",
                    help="Directory of small local text documents for RAG.")
    ap.add_argument("--faiss_index_dir", type=str, default="./toy_index",
                    help="Where to persist FAISS index (if embeddings available).")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=4, help="Top-k for retriever.")
    args = ap.parse_args()

    device = detect_device(args.device)
    dtype = resolve_dtype(args.dtype)

    print("Building pipeline")

    llm = build_local_llm_pipeline(
        llm_model_dir=Path(args.llm_model_dir),
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # --- QA chain (no retrieval) ---
    qa_chain = build_qa_chain(llm)
    qa_query = "What is an exascale supercomputer and why is interconnect bandwidth important?"
    qa_answer = qa_chain.invoke({"query": qa_query})
    print("\n=== Zero-RAG QA (local LLM) ===")
    print(f"Q: {qa_query}\nA: {qa_answer}\n")

    # --- RAG pipeline (retrieval + synthesis) ---
    docs_dir = Path(args.docs_dir)
    ensure_toy_docs(docs_dir)
    split_docs = load_and_split_docs(docs_dir)

    print("Building Embeddings")
    emb = build_embeddings_or_none(Path(args.emb_model_dir)) if args.emb_model_dir else None
    retriever = build_retriever(
        split_docs=split_docs,
        emb=emb,
        faiss_index_dir=Path(args.faiss_index_dir),
        k=args.top_k,
    )

    print("Building RAG Chain")
    rag_chain = build_rag_chain(llm, retriever)
    rag_query = "Summarize Frontier's node architecture and storage in two sentences."
    rag_answer = rag_chain.invoke(rag_query)
    print("=== RAG (local docs + local LLM) ===")
    print(f"Q: {rag_query}\nA: {rag_answer}\n")

    # Structured JSON dump for simple logging on Frontier
    out = {
        "device": device,
        "dtype": str(dtype),
        "qa": {"query": qa_query, "answer": qa_answer},
        "rag": {"query": rag_query, "answer": rag_answer},
        "retriever": "FAISS+HFEmbeddings" if emb is not None else "BM25",
        "docs_dir": str(docs_dir.resolve()),
        "faiss_index_dir": str(Path(args.faiss_index_dir).resolve()),
    }
    Path("toy_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[OK] Wrote toy_results.json")

if __name__ == "__main__":
    main()           
