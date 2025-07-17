import ast
from itertools import product
import json
import logging

import numpy as np
from openai import OpenAI
import pulp
from sklearn.metrics import average_precision_score, f1_score

from fin_rag.train_diff_agg import compute_metrics

logger = logging.getLogger(__name__)

def pad_and_stack(predictions_list, labels_list, pad_label=-100):
    """
    Given a list of predictions and labels arrays (each with shape (1, seq_len, ...)),
    pad each to the maximum sequence length along the sequence dimension and then stack.

    predictions_list: list of numpy arrays of shape (1, seq_len, num_labels)
    labels_list: list of numpy arrays of shape (1, seq_len)
    pad_label: the value with which to pad the labels (e.g. -100)

    Returns:
        final_preds: numpy array of shape (n_examples, max_seq_len, num_labels)
        final_labels: numpy array of shape (n_examples, max_seq_len)
    """
    # Determine maximum sequence length across all examples.
    max_seq_len = max(arr.shape[1] for arr in predictions_list)

    padded_preds = []
    padded_labels = []
    for pred, lab in zip(predictions_list, labels_list):
        # pred has shape (1, seq_len, num_labels) and lab has shape (1, seq_len)
        seq_len = pred.shape[1]
        pad_len = max_seq_len - seq_len

        # Pad predictions with zeros; padding only on the sequence dimension.
        if pad_len > 0:
            padded_pred = np.pad(pred, ((0,0), (0, pad_len), (0,0)), mode='constant', constant_values=0)
            padded_lab = np.pad(lab, ((0,0), (0, pad_len)), mode='constant', constant_values=pad_label)
        else:
            padded_pred = pred
            padded_lab = lab

        padded_preds.append(padded_pred)
        padded_labels.append(padded_lab)

    # Concatenate along the batch (first) axis.
    final_preds = np.concatenate(padded_preds, axis=0)
    final_labels = np.concatenate(padded_labels, axis=0)
    return final_preds, final_labels


def main():
    
    client = OpenAI()

    
    expert_annotated_results = []
    with open('fin_rag/annotation/annotated_result/all/setting2/expert.jsonl', 'r') as file:
        for line in file:
            data = json.loads(line)
            key = list(data.keys())[0]
            id_text_label = [
                key,
                data[key]['text'],
                data[key]['tokens'],
                data[key]['binary_labels']
            ]
            expert_annotated_results.append(id_text_label)

    predictions_list = []
    labels_list = []
    for expert_element in expert_annotated_results:
        ls = expert_element[2]
        
        query_string = (
            "Analyze the significance of tokens from a financial report paragraph and return only "
            "the most important ones in their original order. Ensure that each output token exactly "
            "matches one from the input list, including punctuation. Provide no additional text, explanation, or formatting.\n\n"
            f"{ls}"
        )

        
        # Retry loop until valid output is received
        while True:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query_string}
                ]
            )
            output = completion.choices[0].message.content
            
            try:
                ls2 = ast.literal_eval(output)
                break  # Exit the loop if parsing succeeds
            except Exception as e:
                logger.warning("Parsing failed, retrying: %s", e)

        result = [1 if item in ls2 else 0 for item in ls]

        pred_np = np.zeros((1, len(result), 2), dtype=float)
        for i, r in enumerate(result):
            pred_np[0, i, int(r)] = 1.0

        lab_np  = np.array(expert_element[3], dtype=int)[None, :]

        predictions_list.append(pred_np)
        labels_list.append(lab_np)
    
    predictions, labels = pad_and_stack(predictions_list, labels_list, pad_label=-100)
    expert_metrics = compute_metrics((predictions, labels))
    logger.info("Expert Metrics: %s", expert_metrics)
    


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
