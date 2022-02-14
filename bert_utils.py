
from fastai.text import *
from fastai.callbacks import *
import glob
from configuration_bertabs import BertAbsConfig
from modeling_bertabs import BertAbsConfig, BertAbs, build_predictor
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.nn import functional as F
import os
os.system("pip install --quiet transformers==2.8.0")
from transformers import BertTokenizer, BertModel
import logging
import pandas as pd
import torch
logging.getLogger().setLevel(100)

actual_df = pd.read_csv("NewsSummaryDataset.csv")


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
args = Namespace(
    adam_b1=0.9,
    adam_b2=0.999,
    alpha=0.95,
    batch_size=1,
    beam_size=5,
    block_size=512,
    block_trigram=True,
    data_path="./NewsSummaryDataset.csv",
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    max_length=200, 
    min_length=50,
    model_name="temp",
    subset=400,
    train_pct=0.8
)

def encode_text(text, tokenizer, symbols, is_summary=False):
    if is_summary:
        encoded = [tokenizer.encode(s)[1:-1] for s in text]
    else:
        encoded = [tokenizer.encode(s) for s in text]
        
    flattened = [item for sublist in encoded for item in sublist]
    
    if is_summary:
        return [symbols['BOS']] + flattened + [symbols['EOS']] 
    
    return flattened

def create_seg_embs(encoded_text, tokenizer):
    segment_embeddings = []
    sentence_num = 0 
    for item in encoded_text:
        segment_embeddings.append(sentence_num % 2)
        if item == tokenizer.sep_token_id:
            sentence_num += 1
            
    return segment_embeddings

def pad(encoded_text, seq_length, tokenizer, symbols, is_summary=False):
    if len(encoded_text) > seq_length:
        if is_summary:
            encoded_text = encoded_text[:seq_length]
        else:
            sent_sep_idxs = [idx for idx, t in enumerate(encoded_text) if t == tokenizer.sep_token_id and idx < seq_length]
            last_sent_sep_idx = min(max(sent_sep_idxs)+1 if (len(sent_sep_idxs) > 0) else seq_length, seq_length)
            encoded_text = encoded_text[:last_sent_sep_idx]
    
    if len(encoded_text) < seq_length:
        encoded_text.extend([tokenizer.pad_token_id] * (seq_length - len(encoded_text)))
    
    
    if is_summary:
        encoded_text += [tokenizer.pad_token_id]

    return encoded_text

def create_mask(text_tensor):
    mask = torch.zeros_like(text_tensor)
    mask[text_tensor != tokenizer.pad_token_id] = 1 
    
    return mask

def collate_function(data, tokenizer, symbols, block_size, training):
    encoded_stories = [encode_text(story, tokenizer, symbols) for _, story, summary in data]
    # print('Encoded Stories :',encoded_stories)
    encoded_summaries = [encode_text(summary, tokenizer, symbols, True) for _, story, summary in data]
    # print('Encoded Summaries :',encoded_summaries)
    story_segembs = [create_seg_embs(s, tokenizer) for s in encoded_stories]
    # print('Story Segment Embeds: ',story_segembs)
        
    padded_stories = torch.tensor([pad(s, block_size, tokenizer, symbols) for s in encoded_stories]).long()
    padded_summaries = torch.tensor([pad(s, block_size, tokenizer, symbols, True) for s in encoded_summaries]).long()
    padded_segembs = torch.tensor([pad(s, block_size, tokenizer, symbols) for s in story_segembs]).long()
    
    stories_mask = create_mask(padded_stories)
    summaries_mask = create_mask(padded_summaries)
    # print('See padded stuff: ')
    # print([padded_stories, padded_summaries, padded_segembs, stories_mask, summaries_mask])
    # [padded_stories, padded_summaries, padded_segembs, stories_mask, summaries_mask]
    if training:
        return [padded_stories, padded_summaries, padded_segembs, stories_mask, summaries_mask], padded_summaries[:,1:]
    else:
        Batch = namedtuple("Batch", ["document_names", "batch_size", "src", "segs", "mask_src", "tgt_str"])
        names = [name for name, _, _ in data]
        summaries = [" ".join(summary_list) for _, _, summary_list in data]
        batch = Batch(
            document_names=names,
            batch_size=len(encoded_stories),
            src=padded_stories.to(args.device),
            segs=padded_segembs.to(args.device),
            mask_src=stories_mask.to(args.device),
            tgt_str=summaries,
        )
        
        return batch

def summaries_loss_fn(inputs, targs):
    loss_fn = FlattenedLoss(
        partial(
            LabelSmoothingCrossEntropy, eps=0.1, reduction='mean', ignore_index=tokenizer.pad_token_id
        )
    )
    loss = loss_fn(inputs, targs)
    return loss

def seq2seq_acc(out, targ, pad_idx=-1):    
    return (out.argmax(2)==targ).float().mean()


def load_model(pretrained=False, path=None): 
    config = BertAbsConfig(max_pos=args.block_size)
    if pretrained:    
        if path:
            model = BertAbs.from_pretrained(
                "remi/bertabs-finetuned-cnndm-extractive-abstractive-summarization", 
                state_dict=torch.load(path, map_location=torch.device(args.device)), 
                config=config) 
        else: 
            model = BertAbs.from_pretrained(
                "remi/bertabs-finetuned-cnndm-extractive-abstractive-summarization", 
                config=config
            )
    else:
        model = BertAbs(args=config)

    return model.to(args.device)


def format_summary(translation):
    raw_summary, _, _ = translation
    summary = (raw_summary.replace("[unused0]", "")
                          .replace("[unused3]", "")
                          .replace("[PAD]", "")
                          .replace("[unused1]", "")
                          .replace(r" +", " ")
                          .replace(" [unused2] ", ". ")
                          .replace("[unused2]", "")
                          .replace(" .", ".")
                          .replace(" ,", ",")
                          .replace(" ?", "?")
                          .replace(" !", "!")
                          .strip())
    return summary
def summarise_articles(input_list, model):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)

    symbols={"BOS": tokenizer.vocab["[unused0]"], "EOS": tokenizer.vocab["[unused1]"]}

    iterator = DataLoader(
        input_list, 
        sampler=SequentialSampler(input_list), 
        batch_size=min(len(input_list), args.batch_size), 
        collate_fn=partial(
            collate_function, 
            tokenizer=tokenizer, 
            symbols=symbols, 
            block_size=args.block_size, 
            training=False
        ),
        pin_memory=False
    )
    
    summaries = []
    predictor = build_predictor(args, tokenizer, symbols, model)
    for batch in progress_bar(iterator):
        batch_data = predictor.translate_batch(batch)
        translations = predictor.from_batch(batch_data)
        summaries.extend([format_summary(t) for t in translations])
            
    return summaries
  
# model = load_model(pretrained=True, path=f"./models/{args.model_name}_encdec_weights.pth")


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)

# symbols={"BOS": tokenizer.vocab["[unused0]"], "EOS": tokenizer.vocab["[unused1]"]}

# config = BertAbsConfig(max_pos=args.block_size)
# model = BertAbs.from_pretrained(
#     'remi/bertabs-finetuned-cnndm-extractive-abstractive-summarization', 
#     config=config
# )

# data = SummarisationDataset(args.data_path, subset=args.subset)

# train_ds, test_ds = train_test_split(data, test_size=1-args.train_pct)
# valid_ds, test_ds = train_test_split(test_ds, test_size=0.1)

# [padded_stories, padded_summaries, padded_segembs, stories_mask, summaries_mask], labels = collate_function(
#     [train_ds[1], train_ds[2]], tokenizer, symbols, args.block_size, training=True
# )
# preds = model(padded_stories, padded_summaries, padded_segembs, stories_mask, summaries_mask)

# loss = summaries_loss_fn(preds, labels)
# model = load_model(pretrained=True)

# data = DataBunch.create(
#     train_ds, 
#     valid_ds, 
#     bs=args.batch_size, 
#     collate_fn=partial(collate_function, tokenizer=tokenizer, symbols=symbols, block_size=args.block_size, training=True), 
# )

# learn = Learner(
#     data, 
#     model, 
#     opt_func=partial(Adam, lr=2e-3, betas=(args.adam_b1, args.adam_b2)),
#     loss_func=summaries_loss_fn,
#     metrics=[seq2seq_acc],
#     callback_fns=ShowGraph
# )
# learn.path = Path('.')
# learn = learn.split([model.bert, model.decoder])

# learn.freeze_to(-1)
# learn.fit_one_cycle(
#     1, 
#     max_lr=5e-3, 
#     moms=(0.8, 0.7),
#     wd=0.1,
#     callbacks=[
#         SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=f"{args.model_name}_dec")
#     ]
# )

# learn.unfreeze()
# learn.fit_one_cycle(
#     1, 
#     max_lr=5e-3, 
#     moms=(0.8, 0.7),
#     wd=0.1,
#     callbacks=[
#         SaveModelCallback(learn, every='improvement', monitor='valid_loss', name=f"{args.model_name}_encdec")
#     ]
# )

# learn = learn.load(f"{args.model_name}_encdec")
# torch.save(learn.model.state_dict(), f"./models/{args.model_name}_encdec_weights.pth")
# model = load_model(pretrained=True, path=f"./models/{args.model_name}_encdec_weights.pth")



# input_list = []
# for file in glob.glob(f"{args.stories_folder}/*.txt"):
#     text = open(file).read()
#     text = re.sub('\n', '', text)
#     text = re.split('(?<=\w[!\?\.])', text) 
#     tup = ('', text, [''])
#     input_list.append(tup)

# summaries = summarise_articles(input_list, tokenizer, symbols, model)