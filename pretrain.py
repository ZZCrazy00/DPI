import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModel
# 用预训练模型获取初始特征

# esm2_t6_8M_UR50D
# esm2_t12_35M_UR50D
# esm2_t30_150M_UR50D
# esm2_t33_650M_UR50D
target_model = 'esm2_t6_8M_UR50D'
tokenizer_ = AutoTokenizer.from_pretrained("facebook/{}".format(target_model))
model_ = AutoModel.from_pretrained("facebook/{}".format(target_model))
data_name = "BindingDB"
target = pd.read_excel("dataset/{}/targets.xlsx".format(data_name))['Fasta'].values.tolist()
for seq in tqdm.tqdm(target):
    outputs = model_(**tokenizer_(seq, return_tensors='pt'))
    seq_feature = outputs.pooler_output[0].detach().numpy().tolist()
    seq_feature = [str(f) for f in seq_feature]
    with open("dataset/{}/{}/targets_feature.txt".format(data_name, target_model), 'a+') as file:
        for l in seq_feature:
            line = ','.join(seq_feature)
        file.write(line + '\n')

# PubChem10M_SMILES_BPE_60k
# PubChem10M_SMILES_BPE_120k
# PubChem10M_SMILES_BPE_240k
# PubChem10M_SMILES_BPE_450k
drug_model = 'PubChem10M_SMILES_BPE_450k'
tokenizer = AutoTokenizer.from_pretrained('seyonec/{}'.format(drug_model))
model = AutoModel.from_pretrained('seyonec/{}'.format(drug_model))
data_name = "BindingDB"
drug = pd.read_excel("dataset/{}/drugs.xlsx".format(data_name))['SMILES'].values.tolist()
for seq in tqdm.tqdm(drug):
    if len(seq) > 512:
        outputs = model(**tokenizer(seq[:512], return_tensors='pt'))
    else:
        outputs = model(**tokenizer(seq, return_tensors='pt'))
    seq_feature = outputs.pooler_output[0].detach().numpy().tolist()
    seq_feature = [str(f) for f in seq_feature]
    with open("dataset/{}/{}/drugs_feature.txt".format(data_name, drug_model), 'a+') as file:
        for l in seq_feature:
            line = ','.join(seq_feature)
        file.write(line + '\n')
