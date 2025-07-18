import argparse
import json
import logging
import os
import pickle
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertForTokenClassification

from fin_rag.train_diff_agg import compute_metrics

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    'ignore',
    message="Some weights of .* were not initialized from the model checkpoint"
)

logging.getLogger("transformers").setLevel(logging.ERROR)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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



def evaluate_dataset(annotated_results, model, retriever, tokenizer1, tokenizer2, all_doc_embeddings, cik_to_name, final_texts, args, device):
    """
    Evaluate the model on the given dataset using weighted multi-paragraph highlighting and return metrics.
    """
    predictions_list = []
    labels_list = []


    count = 0
    # Iterate over all examples in the dataset
    for element in annotated_results:
        count += 1

        query_firm_name = cik_to_name[element[0].split('_')[2]]
        concat_text = f"{query_firm_name} {element[1]}"
        
        # Embed the query text
        query_embedding = embed_texts_contriever2(concat_text, retriever, tokenizer1, device)
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        
        query_embedding = query_embedding.to(device)  # Shape: (1, embedding_dim)

        # Compute similarities (dot product)
        similarities = torch.matmul(all_doc_embeddings, query_embedding.t()).squeeze()  # Shape: (num_docs,)

        # Retrieve top-k documents
        topk_values, topk_indices = torch.topk(similarities, 1)
        retrieval_weights = torch.softmax(topk_values, dim=0)
        
        # Tokenize training element
        # 1) Tokenize the list of words properly:
        encoded_A = tokenizer2(
            element[2],              # e.g. ["This","is","a","sentence"]
            is_split_into_words=True,         # <-- very important
            add_special_tokens=False,         # we’ll add SEP manually
            truncation=True,
            max_length=250,
            return_tensors="pt",
        )

        input_ids_A = encoded_A["input_ids"].to(device)         # [1, seq_len_A]
        attention_mask_A = encoded_A["attention_mask"].to(device)

        raw_labels = [
            lbl if lbl is not None else -100 
            for lbl in element[3]
        ]

        # build aligned labels for A
        word_ids = encoded_A.word_ids(batch_index=0)  # a list of len(seq_len_A)
        labels_A = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # special token or padding
                labels_A.append(-100)
            elif word_idx != prev_word_idx:
                # first sub-token of a word → use the true label
                labels_A.append(raw_labels[word_idx])
            else:
                # continuation sub-token → ignore
                labels_A.append(-100)
            prev_word_idx = word_idx

        labels_A = torch.tensor(labels_A, dtype=torch.long, device=device).unsqueeze(0)

        # Manually insert one SEP between them (if any):
        sep_id = tokenizer2.sep_token_id
        sep_tensor = torch.tensor([[sep_id]], device=device)             # [1,1]
        mask_sep   = torch.ones_like(sep_tensor)

        # Iterate over the retrieval paragraphs and compute predictions
        paragraph_logits_list = []
        labels = labels_A
        for idx in topk_indices:
            # Get the text of the retrieval paragraph
            retrieval_text = final_texts[idx.item()]
            encoded_B = tokenizer2(
                retrieval_text,
                add_special_tokens=False,
                truncation=True,
                max_length=250,
                return_tensors="pt",
            )
            input_ids_B = encoded_B["input_ids"].to(device)       # [1, seq_len_B]
            attention_mask_B   = encoded_B["attention_mask"].to(device)
            

            input_ids      = torch.cat([input_ids_A, sep_tensor, input_ids_B], dim=1)
            attention_mask = torch.cat([attention_mask_A, mask_sep, attention_mask_B],    dim=1)

            b_len = input_ids_B.size(1)
            pad_labels = [-100] * (1 + b_len)
            labels = torch.cat([labels_A, torch.tensor(pad_labels, device=device).unsqueeze(0)], dim=1)

            #forward pass
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits                        # [1, seq_len, 2]

            sep_index = (input_ids == sep_id).nonzero(as_tuple=True)[1].item()
            logits = logits[:, :sep_index, :]   # [1, seq_len_A, num_labels]
            labels = labels[:, :sep_index]      # [1, seq_len_A]

            paragraph_logits_list.append(logits)

        # Now, compute a weighted sum of logits for the top 10 retrieval paragraphs.
        # Make sure the logits have the same shape (they correspond to the query tokens)
        weighted_logits = sum(w * l for w, l in zip(retrieval_weights, paragraph_logits_list))

        # Store the raw logits and true labels (convert to numpy)
        predictions_list.append(weighted_logits.detach().cpu().numpy())
        true_labels = labels.cpu().numpy()
        labels_list.append(true_labels.reshape(1, -1))

    # Pad and stack sequences
    predictions, labels = pad_and_stack(predictions_list, labels_list, pad_label=-100)

    # Compute and return metrics
    metrics = compute_metrics((predictions, labels))
    return metrics


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def embed_texts_contriever(text, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    emb = mean_pooling(model(**inputs)[0], inputs['attention_mask'])
    
    return emb

def embed_texts_contriever2(text, model, tokenizer, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        emb = mean_pooling(model(**inputs)[0], inputs['attention_mask'])
    
    return emb



def main():

    # Argument parser setup
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs_range', type=int, default=25)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--training_label', type=str, default='naive', choices=['naive', 'loose', 'strict'])
    parser.add_argument('--valid_label', type=str, default='naive', choices=['naive', 'loose', 'strict'])
    parser.add_argument('--testing_label', type=str, default='naive', choices=['naive', 'loose', 'strict'])

    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cik_to_name = {}
    
    directory1 = "/home/ythsiao/output"
    firm_list = os.listdir(directory1)

    for firm in firm_list:
        directory2 = os.path.join(directory1, firm)
        directory3 = os.path.join(directory2, "10-K")
        tenK_list = os.listdir(directory3)
        tenK_list = sorted(tenK_list)


        for tenK in tenK_list:
            if tenK[:4] == "2022":
                file_path = os.path.join(directory3, tenK)
                with open(file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        if ('company_name' in data) and ('cik' in data):
                            firm_name = data['company_name']
                            cik = data['cik']
                            cik_to_name[cik] = firm_name
                        
    
    # Load annotated results
    training_annotated_results = []
    if args.training_label == "naive":
        with open('fin_rag/annotation/annotated_result/all/setting2/train.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['naive_aggregation']['label'],
                    data['highlight_probs']
                ]
                training_annotated_results.append(id_text_label)
    elif args.training_label == "loose":
        with open('fin_rag/annotation/annotated_result/all/setting2/train.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['loose_aggregation']['label'],
                    data['highlight_probs']
                ]
                training_annotated_results.append(id_text_label)
    elif args.training_label == "strict":
        with open('fin_rag/annotation/annotated_result/all/setting2/train.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['strict_aggregation']['label'],
                    data['highlight_probs']
                ]
                training_annotated_results.append(id_text_label)

    validation_annotated_results = []
    if args.valid_label == "naive":
        with open('fin_rag/annotation/annotated_result/all/setting2/valid.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['naive_aggregation']['label'],
                    data['highlight_probs']
                ]
                validation_annotated_results.append(id_text_label)
    elif args.valid_label == "loose":
        with open('fin_rag/annotation/annotated_result/all/setting2/valid.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['loose_aggregation']['label'],
                    data['highlight_probs']
                ]
                validation_annotated_results.append(id_text_label)
    elif args.valid_label == "strict":
        with open('fin_rag/annotation/annotated_result/all/setting2/valid.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['strict_aggregation']['label'],
                    data['highlight_probs']
                ]
                validation_annotated_results.append(id_text_label)

    testing_annotated_results = []
    if args.testing_label == "naive":
        with open('fin_rag/annotation/annotated_result/all/setting2/test.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['naive_aggregation']['label'],
                    data['highlight_probs']
                ]
                testing_annotated_results.append(id_text_label)
    elif args.testing_label == "loose":
        with open('fin_rag/annotation/annotated_result/all/setting2/test.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['loose_aggregation']['label'],
                    data['highlight_probs']
                ]
                testing_annotated_results.append(id_text_label)
    elif args.testing_label == "strict":
        with open('fin_rag/annotation/annotated_result/all/setting2/test.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                id_text_label = [
                    data['sample_id'],
                    data['text'],
                    data['tokens'],
                    data['strict_aggregation']['label'],
                    data['highlight_probs']
                ]
                testing_annotated_results.append(id_text_label)

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

    # Load paragraph information
    with open('paragraph_encodings/para_info_contriever_firm.pkl', 'rb') as f:
        para_info = pickle.load(f)


    # Extract embeddings and texts
    final_embeddings = np.vstack([item[2] for item in para_info]).astype('float32')
    final_texts = [item[1] for item in para_info]
    final_ids = [item[3] for item in para_info]

    # Initialize tokenizer and retriever model
    tokenizer1 = AutoTokenizer.from_pretrained('facebook/contriever')
    tokenizer2 = AutoTokenizer.from_pretrained('bert-base-uncased')
    retriever = AutoModel.from_pretrained('facebook/contriever')
    retriever.to(device).eval()

    # Initialize classification model
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device).train()

    # Setup optimizer and loss criterion
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(retriever.parameters()),
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Convert embeddings to tensors
    all_doc_embeddings = torch.tensor(final_embeddings).to(device)  # Shape: (num_docs, embedding_dim)

    best_val_f1      = 0.0
    patience         = 5
    patience_counter = 0

    for epoch in range(args.epochs_range):

        model.train()
        retriever.train()
        
        count = 0
        for training_element in training_annotated_results:
            
            count += 1

            query_firm_name = cik_to_name[training_element[0].split('_')[2]]
            concat_text = f"{query_firm_name} {training_element[1]}"
            
            # Embed the query text
            query_embedding = embed_texts_contriever(concat_text, retriever, tokenizer1, device)
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
            query_embedding = query_embedding.to(device)  # Shape: (1, embedding_dim)

            # Compute similarities (dot product)
            similarities = torch.matmul(all_doc_embeddings, query_embedding.t()).squeeze()  # Shape: (num_docs,)

            # Retrieve top-k documents
            topk_values, topk_indices = torch.topk(similarities, args.top_k)
            retrieval_weights = torch.softmax(topk_values, dim=0)
            
            # Tokenize training element
            # 1) Tokenize the list of words properly:
            encoded_A = tokenizer2(
                training_element[2],              # e.g. ["This","is","a","sentence"]
                is_split_into_words=True,         # <-- very important
                add_special_tokens=False,         # we’ll add SEP manually
                truncation=True,
                max_length=250,
                return_tensors="pt",
            )

            input_ids_A = encoded_A["input_ids"].to(device)         # [1, seq_len_A]
            attention_mask_A = encoded_A["attention_mask"].to(device)

            raw_labels = [
                lbl if lbl is not None else -100 
                for lbl in training_element[3]
            ]

            # build aligned labels for A
            word_ids = encoded_A.word_ids(batch_index=0)  # a list of len(seq_len_A)
            labels_A = []
            prev_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    # special token or padding
                    labels_A.append(-100)
                elif word_idx != prev_word_idx:
                    # first sub-token of a word → use the true label
                    labels_A.append(raw_labels[word_idx])
                else:
                    # continuation sub-token → ignore
                    labels_A.append(-100)
                prev_word_idx = word_idx

            labels_A = torch.tensor(labels_A, dtype=torch.long, device=device).unsqueeze(0)

            # Manually insert one SEP between them (if any):
            sep_id = tokenizer2.sep_token_id
            sep_tensor = torch.tensor([[sep_id]], device=device)             # [1,1]
            mask_sep   = torch.ones_like(sep_tensor)

            # Iterate over the retrieval paragraphs and compute predictions
            paragraph_logits_list = []
            labels = labels_A
            for idx in topk_indices:
                # Get the text of the retrieval paragraph
                retrieval_text = final_texts[idx.item()]
                encoded_B = tokenizer2(
                    retrieval_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=250,
                    return_tensors="pt",
                )
                input_ids_B = encoded_B["input_ids"].to(device)       # [1, seq_len_B]
                attention_mask_B   = encoded_B["attention_mask"].to(device)
                

                input_ids      = torch.cat([input_ids_A, sep_tensor, input_ids_B], dim=1)
                attention_mask = torch.cat([attention_mask_A, mask_sep, attention_mask_B],    dim=1)

                b_len = input_ids_B.size(1)
                pad_labels = [-100] * (1 + b_len)
                labels = torch.cat([labels_A, torch.tensor(pad_labels, device=device).unsqueeze(0)], dim=1)

                #forward pass
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask)
                logits = outputs.logits                        # [1, seq_len, 2]

                sep_index = (input_ids == sep_id).nonzero(as_tuple=True)[1].item()
                logits = logits[:, :sep_index, :]   # [1, seq_len_A, num_labels]
                labels = labels[:, :sep_index]      # [1, seq_len_A]

                paragraph_logits_list.append(logits)

            # Now, compute a weighted sum of logits for the top 10 retrieval paragraphs.
            # Make sure the logits have the same shape (they correspond to the query tokens)
            weighted_logits = sum(w * l for w, l in zip(retrieval_weights, paragraph_logits_list))
            
            #compute the loss exactly as BertForTokenClassification does
            loss = criterion(
                weighted_logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # At evaluation time (after model training epoch, for example)
        model.eval()
        retriever.eval()

        training_metrics = evaluate_dataset(training_annotated_results, model, retriever, tokenizer1, tokenizer2, all_doc_embeddings, cik_to_name, final_texts, args, device)
        validation_metrics = evaluate_dataset(validation_annotated_results, model, retriever, tokenizer1, tokenizer2, all_doc_embeddings, cik_to_name, final_texts, args, device)
        testing_metrics = evaluate_dataset(testing_annotated_results, model, retriever, tokenizer1, tokenizer2, all_doc_embeddings, cik_to_name, final_texts, args, device)
        expert_metrics = evaluate_dataset(expert_annotated_results, model, retriever, tokenizer1, tokenizer2, all_doc_embeddings, cik_to_name, final_texts, args, device)

        logger.info("Train for %d epochs:", epoch + 1)
        logger.info("Training Metrics: %s", training_metrics)
        logger.info("")
        logger.info("Validation Metrics: %s", validation_metrics)
        logger.info("")
        logger.info("Testing Metrics: %s", testing_metrics)
        logger.info("")
        logger.info("Expert Metrics: %s", expert_metrics)
        logger.info("")

        val_f1 = validation_metrics["f1"]

        # Check for improvement
        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            # Save the model’s state dict (or full model) as the best so far
            torch.save(model.state_dict(), f"model_checkpoints/best_model_(End_to_End)_{args.top_k}_{args.training_label}_{args.valid_label}_{args.testing_label}.pt")
            logger.info("▶ New best model (val F1 = %.4f), saving to best_model.pt", val_f1)
        else:
            patience_counter += 1
            logger.info("No improvement in val F1 for %d/%d epochs", patience_counter, patience)
            if patience_counter >= patience:
                logger.info("⏹ Early stopping (no improvement for %d epochs).", patience)
                return

        logger.info("Epoch %d complete — best val F1: %.4f", epoch + 1, best_val_f1)
        logger.info("")



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    main()
