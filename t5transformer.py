# !pip install --quiet transformers==4.5.0
# !pip install --quiet pytorch-lightning==1.2.7

import json
import pandas as pd
import numpy as np
import torch
torch.cuda.empty_cache()
# path for data
from pathlib import Path
# dataset and dataloader for functions
from torch.utils.data import Dataset, DataLoader
# lightning for data class
import pytorch_lightning as pl
# leveraging the model checkpoints
from pytorch_lightning.callbacks import ModelCheckpoint
# we can visualize performance of model
from pytorch_lightning.loggers import TensorBoardLogger
# splitting the data
from sklearn.model_selection import train_test_split
# color formatting in ANSII code for output in terminal
from termcolor import colored
# wraps the paragraph into a single line or string
import textwrap
# installing multiple utilities
# including optimizer , tokenizer and generation module
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
# showing bars for processes in notebook
from tqdm.auto import tqdm

# random pseudo numbers
pl.seed_everything(42) 

actual_df = pd.read_csv('NewsSummaryDataset.csv')
df = actual_df[['summary', 'text']]

# dropping out the Not Available values
df = df.dropna()
df=df.head(500)
# using sklearn utility, splitting the data into 10:1 ratio
train_df, test_df = train_test_split(df, test_size=0.1)
# let's check the shape of our data
train_df.shape, test_df.shape 

# final_container = container
 # class for creating the dataset which extends from pytorch 
class NewsSummaryDataset(Dataset):
   # init it , create a constructor
     def __init__(
         self,
         # data in the form of a dataframe
         data: pd.DataFrame,
         # a tokenizer
         tokenizer: T5Tokenizer,
         # max token length of input sequence
         text_max_token_len: int = 512,
         # same for the summary but less length
         summary_max_token_len: int = 128
     ):
         # saving all
         self.tokenizer = tokenizer
         self.data = data
         self.text_max_token_len = text_max_token_len
         self.summary_max_token_len = summary_max_token_len
     # length method
     def __len__(self):
         return len(self.data)
     # getting the items method  
     def __getitem__(self, index: int):
       # data row from data at current index
         data_row = self.data.iloc[index]
         # get the full text
         text = data_row['text']
         # encoding the text
         text_encoding = tokenizer(
             text,
             # setting max length
             max_length=self.text_max_token_len,
             # for same length
             padding='max_length',
             # cutting longer sequences
             truncation=True,
             # masking unwanted words
             return_attention_mask=True,
             # special tokens for start and end
             add_special_tokens=True,
             # return pytorch tensors
             return_tensors='pt'
         )
         # same is done with summary encoding
         summary_encoding = tokenizer(
             data_row['summary'],
             truncation=True,
             return_attention_mask=True,
             add_special_tokens=True,
             max_length=self.summary_max_token_len,
             padding='max_length',
             return_tensors='pt'
         )
         # creating the actual labels
         labels = summary_encoding['input_ids'] 
         labels[labels == 0] = -100 # to make sure we have correct labels for T5 text generation
         return dict(
             # data
             text=text,
             # task
             summary=data_row['summary'],
             # easy batching
             text_input_ids=text_encoding['input_ids'].flatten(),
             # masking
             text_attention_mask=text_encoding['attention_mask'].flatten(),
             # again flatten
             labels=labels.flatten(),
             labels_attention_mask=summary_encoding['attention_mask'].flatten()
         ) 

   # data module for pytorch lightning
class NewsSummaryDataModule(pl.LightningDataModule):
     def __init__(
         self,
         # pass in train data
         train_df: pd.DataFrame,
         # pass in test data
         test_df: pd.DataFrame,
         # tokenizer
         tokenizer: T5Tokenizer,
         # batch_size
         batch_size: int = 1,
         # length of sequence
         text_max_token_len: int = 128,
         # length of output sequence
         summary_max_token_len: int = 64
     ):
         super().__init__()
         # storing the data in class objects
         self.train_df = train_df
         self.test_df = test_df
         self.batch_size = batch_size
         self.tokenizer = tokenizer
         self.text_max_token_len = text_max_token_len
         self.summary_max_token_len = summary_max_token_len
     # automatically called by the trainer  
     def setup(self, stage=None):
         self.train_dataset = NewsSummaryDataset(
             self.train_df,
             self.tokenizer,
             self.text_max_token_len,
             self.summary_max_token_len
         )
         self.test_dataset = NewsSummaryDataset(
             self.test_df,
             self.tokenizer,
             self.text_max_token_len,
             self.summary_max_token_len
         )
     # for train data
     def train_dataloader(self):
         return DataLoader(
             self.train_dataset,
             batch_size=self.batch_size,
             shuffle=True,
             num_workers=2
         )
   # for test data
     def test_dataloader(self):
         return DataLoader(
             self.test_dataset,
             batch_size=self.batch_size,
             shuffle=True,
             num_workers=2
         )
     # valid data
     def val_dataloader(self):
         return DataLoader(
             self.test_dataset,
             batch_size=self.batch_size,
             shuffle=True,
             num_workers=2
         ) 

# create lightning module for summarizatio
class NewsSummaryModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return output.loss, output.logits
    def training_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_size):
        input_ids = batch['text_input_ids']
        attention_mask = batch['text_attention_mask']
        labels = batch['labels']
        labels_attention_mask = batch['labels_attention_mask']
        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
        )
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001) 

def summarizeText(text):
     text_encoding = tokenizer(
         text,
         max_length=512,
         padding='max_length',
         truncation=True,
         return_attention_mask=True,
         add_special_tokens=True,
         return_tensors='pt'
     )
     generated_ids = trained_model.model.generate(
         input_ids=text_encoding['input_ids'],
         attention_mask=text_encoding['attention_mask'],
         max_length=150,
         num_beams=2,
         repetition_penalty=2.5,
         length_penalty=1.0,
         early_stopping=True
     )
     preds = [
             tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
             for gen_id in generated_ids
     ]
     return "".join(preds) 

# leveraging the base T5 transformer
MODEL_NAME = 't5-base'
# instantiate the tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

N_EPOCHS = 1
BATCH_SIZE = 3
# call the data module
data_module = NewsSummaryDataModule(train_df, test_df, tokenizer)

model = NewsSummaryModel()

checkpoint_callback = ModelCheckpoint(
     dirpath='checkpoints',
     filename='best-checkpoint',
     save_top_k=1,
     verbose=True,
     monitor='val_loss',
     mode='min'
 )
#  logger = TensorBoardLogger("lightning_logs", name='news-summary')
trainer = pl.Trainer(
  #  logger=logger,
    checkpoint_callback=checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=1,
    
)
 
trainer.fit(model, data_module)
trained_model = NewsSummaryModel.load_from_checkpoint(
     trainer.checkpoint_callback.best_model_path
 )
trained_model.freeze()

