# eval/eval_ragas.py
import argparse
import uuid
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall,
)
from ragas.run_config import RunConfig


# ---------- 参数 ----------
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="GWM")
ap.add_argument("--input-dir", default=None)
ap.add_argument("--out-dir", default=None)
ap.add_argument("--rag-api", default="http://localhost:8002/ask")
ap.add_argument("--vllm-chat", default="http://localhost:8000/v1")
ap.add_argument("--vllm-emb", default="http://localhost:8001/v1")
ap.add_argument("--chat-model", default="qwen3.5-4b")
ap.add_argument("--emb-model", default="qwen3-embedding-0.6b")
ap.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
args, _ = ap.parse_known_args()

ds = args.dataset
EXCEL_DIR = Path(args.input_dir or f"./data/test/{ds}")
OUT_DIR = Path(args.out_dir or f"./ragas_results/{ds}")

print(f"输入目录: {EXCEL_DIR}")
print(f"输出目录: {OUT_DIR}")
print(f"RAG API : {args.rag_api}  (mode={args.mode})")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 评估模型（只建一次，复用） ----------
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(
    model=args.chat_model,
    openai_api_base=args.vllm_chat,
    openai_api_key="not-needed",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
))
evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
    model=args.emb_model,
    openai_api_base=args.vllm_emb,
    openai_api_key="not-needed",
    check_embedding_ctx_length=False,
))


# ---------- 主流程 ----------
excel_files = sorted(EXCEL_DIR.glob("*.xlsx"))
print(f"\n共发现 {len(excel_files)} 个 Excel 文件")

for excel_path in excel_files:
    name = excel_path.stem
    print(f"\n===== 处理: {excel_path.name} =====")

    # 1. 读 Excel
    df = pd.read_excel(excel_path)
    df = df[["query", "answer"]].rename(
        columns={"query": "question", "answer": "ground_truth"}
    )
    df = df.dropna(subset=["question", "ground_truth"]).reset_index(drop=True)
    print(f"共 {len(df)} 条测试数据")

    # 2. 调 RAG 接口，收集 answer + contexts
    answers, contexts = [], []
    for i, q in enumerate(df["question"]):
        try:
            r = requests.post(
                args.rag_api,
                json={
                    "question": q,
                    "mode": args.mode,
                    "session_id": f"eval-{name}-{i}",
                },
                timeout=120,
            ).json()
            answers.append(r["answer"])
            contexts.append([s["document"] for s in r.get("sources", [])])
        except Exception as e:
            print(f"[{i}] 请求失败: {e}")
            answers.append("")
            contexts.append([])
        print(f"[{i+1}/{len(df)}] done")

    df["answer"] = answers
    df["contexts"] = contexts

    # 保存中间产物
    #mid_path = OUT_DIR / f"{name}_{args.mode}_rag_answers.xlsx"
    #df.to_excel(mid_path, index=False)
    #print(f"中间产物已保存: {mid_path}")

    # 3. 构造 Ragas 数据集
    dataset = Dataset.from_pandas(
        df[["question", "answer", "contexts", "ground_truth"]]
    )

    # 4. 跑评估
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        run_config=RunConfig(max_workers=2, timeout=180),
    )

    print(result)
    out_path = out_path = OUT_DIR / f"{name}_{args.mode}_ragas_result.xlsx"
    result.to_pandas().to_excel(out_path, index=False)
    print(f"结果已保存到 {out_path}")

print("\n全部完成")