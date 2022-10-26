from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import pandas as pd
from tqdm import tqdm

model = torch.load('C:/Users/mosto/PycharmProjects/GlubOb/model.pth', map_location='cpu')
raw_model = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(raw_model, do_lower_case=True)

df = pd.DataFrame({'query': ['your_query'],
                   'name': ['company_name']})


def convert_to_dataset(data: pd.DataFrame) -> TensorDataset:
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


test = convert_to_dataset(df)
test_dataloader = DataLoader(test, sampler=SequentialSampler(test), batch_size=1)


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


emn = infer(test_dataloader, model)
ems = [i.detach().cpu().numpy() for i in emn]
em = [(ems[i][0][1]) for i, k in enumerate(ems)]
df['is_duplicate_score'] = pd.Series(em)
df.sort_values('is_duplicate_score', ascending=False)
print(df)
