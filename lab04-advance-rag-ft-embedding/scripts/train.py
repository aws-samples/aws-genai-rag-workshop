import argparse
from sentence_transformers import SentenceTransformer
import json

from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import losses

output_dir = "/opt/ml/model/"


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--train_data_file",
        type=str, 
        default="/opt/ml/input/data/train/train_dataset.json",
        help="Path to training dataset."
    )
    parser.add_argument(
        "--valid_data_file",
        type=str,
        default="/opt/ml/input/data/valid/val_dataset.json",
        help="Path to validation dataset."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train for"
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=50,
        help="Number of steps for evaluation"
    )
    args, _ = parser.parse_known_args()

    return args


def load_dataset(file_path, batch_size, data_type="train"):

    with open(file_path, 'r+') as f:
        data = json.load(f)

    corpus = data['corpus']
    queries = data['queries']
    relevant_docs = data['relevant_docs']

    if data_type == "train":
        datasets = []
        for query_id, query in queries.items():
            node_id = relevant_docs[query_id][0]
            text = corpus[node_id]
            dataset = InputExample(texts=[query, text])
            datasets.append(dataset)

        return DataLoader(datasets, batch_size=batch_size)

    else:
        return InformationRetrievalEvaluator(queries, corpus, relevant_docs)


def training_function(args):

    model = SentenceTransformer(args.model_id)

    # define loss function
    loss = losses.MultipleNegativesRankingLoss(model)

    # load training dataset
    loader = load_dataset(args.train_data_file, args.batch_size)

    # load evaluator
    evaluator = load_dataset(args.valid_data_file, args.batch_size, data_type="valid")

    warmup_steps = int(len(loader) * args.epochs * 0.1)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps,
    )

    print("=========== Training Complete ================")


def main():
    args = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()