import torch
import sentencepiece as spm
from torch.utils.data import Dataset

from utils import subsequent_mask
from vocab import Vocab

def read_corpus(
    file_path, 
    source, 
    sp_model_path,
):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            if source == 'tgt':
                subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data

class NMTDataset(Dataset):
    def __init__(
        self, 
        src_data, 
        tgt_data, 
        src_vocab,
        tgt_vocab,
    ):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        src_sent = self.src_data[idx] 
        tgt_sent = self.tgt_data[idx]

        src_ids = [self.src_vocab.word2id(x) for x in src_sent]
        tgt_ids = [self.tgt_vocab.word2id(x) for x in tgt_sent]

        item = {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_words": src_sent,
            "tgt_words": tgt_sent,
        }

        return item
    
    def generate_batch(self, item_list):

        src_ids = [x["src_ids"] for x in item_list]
        tgt_ids = [x["tgt_ids"] for x in item_list]
        src_words = [x["src_words"] for x in item_list]
        tgt_words = [x["tgt_words"] for x in item_list]
        src_pad_id = self.src_vocab.pad_id
        tgt_pad_id = self.tgt_vocab.pad_id

        max_src_len = max(len(x) for x in src_ids)
        src_ids = [x + [src_pad_id for _ in range(max_src_len - len(x))] for x in src_ids]
        src_ids = torch.LongTensor(src_ids)
        src_mask = (src_ids != src_pad_id)

        max_tgt_len = max(len(x) for x in tgt_ids)
        tgt_ids = [x + [tgt_pad_id for _ in range(max_tgt_len - len(x))] for x in tgt_ids]
        tgt_ids = torch.LongTensor(tgt_ids)
        new_tgt_ids = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]
        tgt_mask = self.make_std_mask(new_tgt_ids, tgt_pad_id)
        tgt_ids = new_tgt_ids
        ntokens = (labels != tgt_pad_id).data.sum()

        batch = {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "labels": labels,
            "src_words": src_words,
            "tgt_words": tgt_words,
            "ntokens": ntokens,
        }

        return batch
    
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        subseq_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        mask = tgt_mask & subseq_mask
        
        return mask
    

def test_dataset():
    import os, sys, json
    from torch.utils.data import DataLoader
    os.chdir(sys.path[0])

    src_file_path = "../data/zh_en_data/train.zh"
    src_sp_model_path = "../data/src.model"

    tgt_file_path  ="../data/zh_en_data/train.en"
    tgt_sp_model_path = "../data/tgt.model"

    src_data = read_corpus(src_file_path, "src", src_sp_model_path)
    tgt_data = read_corpus(tgt_file_path, "tgt", tgt_sp_model_path)

    vocab_path = "../data/vocab.json"
    vocab = json.load(open(vocab_path))

    src_word2id = vocab["src_word2id"]
    tgt_word2id = vocab["tgt_word2id"]
    src_vocab = Vocab(src_word2id)
    tgt_vocab = Vocab(tgt_word2id)

    dataset = NMTDataset(
        src_data,
        tgt_data,
        src_vocab,
        tgt_vocab,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=dataset.generate_batch,
    )

    for batch in dataloader:
        print(batch.keys())
    
if __name__ == "__main__":
    test_dataset()



 

