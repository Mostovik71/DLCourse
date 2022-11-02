from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup, AdamW
from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
import random
from sklearn.metrics import accuracy_score
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, models
from sklearn.feature_extraction.text import CountVectorizer


def load_models(path_to_bert_model, path_to_cos_model):
    model = torch.load(path_to_bert_model)
    raw_model = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(raw_model, do_lower_case=True)

    ## Step 1: use an existing language model
    word_embedding_model = models.Transformer('distilroberta-base')
    # word_embedding_model = models.Transformer('stsb-roberta-large')
    ## Step 2: use a pool function over the token embeddings
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    model_cos = torch.load(path_to_cos_model)

    return model, model_cos, tokenizer


def convert_to_dataset(data: pd.DataFrame, tokenizer) -> TensorDataset:
    input_ids = []
    attention_masks = []
    token_type_ids = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(row["query"], row["name"], max_length=300,
                                             pad_to_max_length=True,
                                             return_attention_mask=True, return_tensors='pt', truncation=True)
        input_ids.append(encoded_dict['input_ids'])
        token_type_ids.append(encoded_dict["token_type_ids"])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    input_ids.to(dtype=torch.long)
    token_type_ids.to(dtype=torch.long)
    attention_masks.to(dtype=torch.long)

    return TensorDataset(input_ids, attention_masks, token_type_ids)


def infer(dataloader, model):
    embs = []

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        input_ids, attention_masks, token_type_ids = batch

        input_ids = input_ids.to(dtype=torch.long)
        token_type_ids = token_type_ids.to(dtype=torch.long)
        attention_masks = attention_masks.to(dtype=torch.long)
        with torch.no_grad():
            m = (model(input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_masks)).logits
            embs.append(torch.nn.Softmax()(m))
    return embs


def mesure_cos(target_company, all_comp_emb, model_cos, n=10):
    target_emb = model_cos.encode(target_company)[0]
    output = pd.DataFrame(columns=['company_name', 'sim_score'])

    comp_labels = []
    comp_emb = []
    for comp, emd_dict in all_comp_emb.items():
        comp_labels.append(comp)
        comp_emb.append(emd_dict["emb"])

    out = cosine_similarity([target_emb], comp_emb)
    for i, score in enumerate(out[0]):
        output.loc[len(output)] = [comp_labels[i], score]

    output = output.sort_values(by=['sim_score'], ascending=False)

    print(output.head(10))

    best = output.iloc[:n + 1]
    top_n = []
    for i, b in best.iterrows():
        top_n.append(b["company_name"])

    return top_n


def load_companies_db(model_cos, db_path):
    all_comp = json.load(open(db_path))  # Сюда нужно вставить ссылку на данные с гугл - диска.

    for sent, d in tqdm(all_comp.items(), total=len(all_comp)):
        emm = model_cos.encode([sent])
        all_comp[sent] = {"emb": emm[0]}

    return all_comp


def get_duplicate(target_company, db_path, model_bert_path, model_cos_path):
    model, model_cos, tokenizer = load_models(model_bert_path, model_cos_path)
    all_comp = load_companies_db(model_cos, db_path)

    output = mesure_cos(target_company, all_comp, model_cos)
    target_company = [target_company[0] for _ in range(len(output))]
    df = pd.DataFrame({'query': target_company,
                       'name': output})
    test = convert_to_dataset(df, tokenizer)
    test_dataloader = DataLoader(test, sampler=SequentialSampler(test), batch_size=1)
    emn = infer(test_dataloader, model)
    ems = [i.detach().cpu().numpy() for i in emn]
    em = [(ems[i][0][1]) for i, k in enumerate(ems)]
    df['is_duplicate_score'] = pd.Series(em)
    df = df.sort_values('is_duplicate_score', ascending=False)
    print(df)


if __name__ == '__main__':
    target = "API"
    db_path = "../all_comp.json"
    model_bert_path = "../model_bert_best.pth"
    model_cos_path = "../model_cos_5_64"
    get_duplicate(target, db_path, model_bert_path, model_cos_path)