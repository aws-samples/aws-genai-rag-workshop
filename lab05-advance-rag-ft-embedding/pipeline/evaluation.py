import logging
import argparse
import os
import tarfile
import json
import pandas as pd
import pathlib

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def shap_ire_results(df, prefix):
    ire_response = dict()
    for key, value in df.to_dict("records")[0].items():
        print(key)
        if (key != "epoch") and (key != "steps"):
            ire_response[f"{prefix}-{key}"] = {
                "value": value,
                "standard_deviation": "NaN"
            }
    return ire_response


def evaluate_top_hit(dataset, embeddings, top_k=5):
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']

    docs = [Document(metadata=dict(id_=id_), page_content=text) for id_, text in corpus.items()] 

    db = FAISS.from_documents(docs, embeddings)

    eval_results = []
    for query_id, query in queries.items():
        retrieved_docs = db.similarity_search(query, top_k)
        retrieved_ids = [doc.metadata['id_'] for doc in retrieved_docs]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            'is_hit': is_hit,
            'retrieved': retrieved_ids,
            'expected': expected_id,
            'query': query_id,
        }
        eval_results.append(eval_result)

    return eval_results


def evaluate_sentence_transformers(
    dataset,
    model,
    output_path,
    name,
):
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']

    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=name)
    return evaluator(model, output_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id",
                        type=str,
                        default="sentence-transformers/msmarco-bert-base-dot-v5")
    parser.add_argument("--model-file", type=str, default="model.tar.gz")
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--test-data-path",
                        type=str,
                        default="/opt/ml/processing/input/data")
    parser.add_argument("--test-file",
                        type=str,
                        default="val_dataset.json")

    args, _ = parser.parse_known_args()

    # load test data.  this should be an argument
    logger.debug("Load test data...")

    test_data = os.path.join(args.test_data_path, args.test_file)

    with open(test_data, 'r+') as f:
        test_dataset = json.load(f)

    logger.debug("Load base model...")
    
    base_embeddings = HuggingFaceEmbeddings(model_name=args.base_model_id)

    eval_results = evaluate_top_hit(test_dataset, base_embeddings)

    df_base = pd.DataFrame(eval_results)
    base_top_hits = df_base['is_hit'].mean()

    logger.info(f"base model top hits: {base_top_hits}")

    # load fine tuned model
    logger.debug("Extracting the model...")

    model_file = os.path.join(args.model_path, args.model_file)
    file = tarfile.open(model_file)
    file.extractall(args.model_path)

    file.close()

    logger.debug("Load fine tuned model...")
    
    finetuned_embeddings = HuggingFaceEmbeddings(model_name=args.model_path)

    eval_results = evaluate_top_hit(test_dataset, finetuned_embeddings)

    df_finetuned = pd.DataFrame(eval_results)
    feintuned_top_hits = df_finetuned['is_hit'].mean()

    logger.info(f"finetuned model top hits: {feintuned_top_hits}")

    logger.info("Evalute using InformationRetrievalEvaluator from sentence_transformers...")

    base_model = SentenceTransformer(args.base_model_id)
    finetuned_model = SentenceTransformer(args.model_path)

    tmp_path = "/tmp/results"
    pathlib.Path(tmp_path).mkdir(parents=True, exist_ok=True)

    evaluate_sentence_transformers(test_dataset,
                                   base_model,
                                   output_path=tmp_path,
                                   name='base')

    evaluate_sentence_transformers(test_dataset,
                                   finetuned_model,
                                   output_path=tmp_path,
                                   name='finetuned')

    df_st_base = pd.read_csv(f"{tmp_path}/Information-Retrieval_evaluation_base_results.csv")
    df_st_finetuned = pd.read_csv(f"{tmp_path}/Information-Retrieval_evaluation_finetuned_results.csv")

    base_ire = shap_ire_results(df_st_base, "1-base")
    finetuned_ire = shap_ire_results(df_st_base, "0-finetuned")

    report_dict = {
        "multiclass_classification_metrics": {
            "confusion_matrix": {
              "0": {
                "0": 1180,
                "1": 510
              },
              "1": {
                "0": 268,
                "1": 138
              }
            },
            "1-base-top-hits": {
                "value": base_top_hits,
                "standard_deviation": "NaN"
            },
            "0-finetuned-top-hits": {
                "value": feintuned_top_hits,
                "standard_deviation": "NaN"
            },
        }
    }

    report_dict["multiclass_classification_metrics"] = {**report_dict["multiclass_classification_metrics"],
                                                        **base_ire,
                                                        ** finetuned_ire}
    logger.info(report_dict)

    output_dir = "/opt/ml/processing/evaluation"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))