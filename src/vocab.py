import os


class Vocab(object):
    def __init__(self, word2id):
        self.word2idx = word2id
        self.idx2word = {v: k for k, v in word2id.items()}
        self.pad_id = word2id["<pad>"]
        self.unk_id = word2id["<unk>"]
        self.start_id = word2id["<s>"]
        self.end_id = word2id["</s>"]
    
    def word2id(self, word):
        return self.word2idx.get(word, self.unk_id)

    def id2word(self, idx):
        return self.idx2word[idx]
    
    def __len__(self):
        
        return len(self.word2idx)

    def __repr__(self):
        return f"Vocabulary[size={len(self)}]"
    


