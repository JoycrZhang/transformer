import os
import sys
import json
import torch
import sacrebleu
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader

from vocab import Vocab
from model import Transformer
from utils import subsequent_mask
from dataset import read_corpus, NMTDataset


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    
    # detokenize the subword pieces to get full sentences
    detokened_refs = [''.join(pieces).replace('▁', ' ') for pieces in references]
    detokened_hyps = [''.join(hyp).replace('▁', ' ') for hyp in hypotheses]

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(detokened_hyps, [detokened_refs])

    return bleu.score


def greedy_decode(model, src, src_mask, max_len, start_id, end_id):
    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_id).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            ys, memory, subsequent_mask(ys.size(1)).type_as(src.data), src_mask
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).fill_(next_word).type_as(src.data)], dim=1
        )
        if next_word == end_id:
            break
        
    return ys


def beam_search(model, src, src_mask, beam_size, max_len, start_id, end_id, tgt_word2id, device):
    """
    src: (1, src_len)
    """
    model.eval()
    tgt_vocab_size = len(tgt_word2id)
    memory = model.encode(src.to(device), src_mask.to(device))
    ys = torch.zeros(1, 1).fill_(start_id).type_as(src.data)
    completed_hypotheses = []
    hypotheses = [[start_id]]
    hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float)
    ys = torch.LongTensor(hypotheses)
    t = 0
    with torch.no_grad():
        while len(completed_hypotheses) < beam_size and t < max_len:
            t += 1 
            exp_memory = memory.expand(len(hypotheses), memory.size(1), memory.size(2)).to(device)
            exp_src_mask = src_mask.expand(len(hypotheses), src_mask.size(1)).to(device)
            tgt_mask = subsequent_mask(ys.size(1)).type_as(src.data).to(device)
            ys = ys.to(device)

            out = model.decode(
                ys, exp_memory, tgt_mask, exp_src_mask
            )
            log_prob = model.generator(out[:, -1])
            log_prob = log_prob.detach().cpu()

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_prob) + log_prob).view(-1) # 求score应该用概率相乘，这里求和是因为log_p_t是log(p), log(p)相加等价于概率相乘
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, tgt_vocab_size, rounding_mode="floor")
            hyp_word_ids = top_cand_hyp_pos % tgt_vocab_size

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                new_hyp_ids = hypotheses[prev_hyp_id] + [hyp_word_id]
                if hyp_word_id == end_id:
                    completed_hypotheses.append(
                        {"word_ids": new_hyp_ids[1:-1], "score": cand_new_hyp_score}
                    )
                else:
                    new_hypotheses.append(new_hyp_ids)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)
                
            if len(completed_hypotheses) == beam_size:
                break

            hypotheses = new_hypotheses
            ys = torch.LongTensor(hypotheses)
            hyp_scores = torch.tensor(new_hyp_scores)
    
    if len(completed_hypotheses) == 0:
        completed_hypotheses.append(
            {"word_ids": hypotheses[0][1:], "score": hyp_scores[0].item()}
        )
    
    completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x["score"], reverse=True)

    tgt_id2word = {v: k for k, v in tgt_word2id.items()}
    results = []
    for hyp in completed_hypotheses:
        words = [tgt_id2word[x] for x in hyp["word_ids"]]
        text = " ".join(words).replace("▁", "")
        results.append({"words": words, "text": text, "score": hyp["score"]})
    
    return results



def test():
    batch_size = 1
    vocab_path = "../data/vocab.json"
    vocab_data = json.load(open(vocab_path))
    src_word2id = vocab_data["src_word2id"]
    tgt_word2id = vocab_data["tgt_word2id"]
    src_vocab = Vocab(src_word2id)
    tgt_vocab = Vocab(tgt_word2id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取测试数据
    src_sp_model_path = "../data/src.model"
    tgt_sp_model_path = "../data/tgt.model"
    test_src_file_path = "../data/zh_en_data/test.zh"
    test_tgt_file_path = "../data/zh_en_data/test.en"
    test_src_data = read_corpus(test_src_file_path, 'src', src_sp_model_path)
    test_tgt_data = read_corpus(test_tgt_file_path, 'tgt', tgt_sp_model_path)

    test_dataset = NMTDataset(
        src_data=test_src_data,
        tgt_data=test_tgt_data,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.generate_batch,
        # num_workers=8,
        # pin_memory=True,
    )
    model_dir = "../outputs/20230708-183125/"
    model_name = "zh_en_model__final.pth"
    model_path = os.path.join(model_dir, model_name)
    model = Transformer.load(model_path)
    model = model.to(device)

    src_texts = []
    grt_texts = []
    pred_texts = []

    beam_size = 10
    start_id = tgt_vocab.start_id
    end_id = tgt_vocab.end_id

    hypotheses = []

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):

        preds = beam_search(
            model=model, 
            src=batch["src_ids"], 
            src_mask=batch["src_mask"], 
            beam_size=beam_size, 
            max_len=130, 
            start_id=start_id, 
            end_id=end_id, 
            tgt_word2id=tgt_word2id,
            device=device,
        )

        hypotheses.append(preds[0]["words"])
        
        src_texts.append(batch["src_words"])
        grt_texts.append(batch["tgt_words"])
        pred_texts.append(preds)

    
    bleu_score = compute_corpus_level_bleu_score(test_tgt_data, hypotheses)
    print('Corpus BLEU: {}'.format(bleu_score), file=sys.stderr)
    
    df = pd.DataFrame()
    df["src"] = src_texts
    df["tgt"] = grt_texts
    df["pred"] = pred_texts
    save_name = "test_result.csv"
    save_path = os.path.join(model_dir, save_name)
    df.to_csv(save_path, index=False)

    return df


if __name__ == "__main__":
    os.chdir(sys.path[0])
    df = test()
    

