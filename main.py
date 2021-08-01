from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
import numpy as np
bert_config = read_json(configs.embedder.bert_embedder)
bert_config['metadata']['variables']['BERT_PATH'] = r'C:\Users\Максим\PycharmProjects\BERT_embeddings\Files\models\multi_cased_L-12_H-768_A-12_pt'

m = build_model(bert_config)

texts = ['Hi, i want my embedding.', 'And mine too, please!']
tokens, token_embs, subtokens, subtoken_embs, sent_max_embs, sent_mean_embs, bert_pooler_outputs = m(texts)
print(tokens)
print(np.array(token_embs))
print(np.array(token_embs)[0].shape)
print(subtokens)
print(subtoken_embs)
print(np.array(subtoken_embs)[0].shape)
print(sent_max_embs)
print(sent_mean_embs)
print(bert_pooler_outputs)

