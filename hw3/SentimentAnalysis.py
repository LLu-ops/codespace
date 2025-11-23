# Module for sentiment analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import torch
from torch import nn
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

SENTIMENT_DICT = {0:'positive',1:'negative',2:'neutral'}


class TextSentiment:
	'''Clean and visualize text and sentiment data.'''

	def __init__(self):
		self.df = np.nan

	def read_data(self, file_path):
		self.df = pd.read_csv(file_path,encoding='ISO-8859-1',header=None)
		self.df.rename(columns={0:'sentiment', 1:'text'},inplace=True)

	def drop_duplicates(self):		
		self.df.drop_duplicates(subset=['text'],keep='first',inplace=True)

	def make_labels(self):
		self.df['label'] = 2
		self.df['label'] = self.df.apply(lambda x: 0 if x['sentiment']=='positive' else x['label'], axis=1)
		self.df['label'] = self.df.apply(lambda x: 1 if x['sentiment']=='negative' else x['label'], axis=1)

	def clean_text(self):
		self.df['text'] = self.df['text'].replace(r'\n', ' ', regex=True)

	def plot_word_cloud(self, sentiment=np.nan):
		if sentiment != sentiment:
			text = " ".join([x for x in self.df.text])
		else:
			text = " ".join([x for x in self.df.text[self.df.sentiment==sentiment]])

		wordcloud = WordCloud(background_color='white').generate(text)
		plt.figure(figsize=(8,6))
		plt.imshow(wordcloud,interpolation='bilinear')
		plt.axis('off')
		plt.show()

	def plot_counts(self):
		sns.countplot(self.df.sentiment)


class Model:
	'''Model for sentiment classification.'''

	def __init__(self):
		self.model = np.nan
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.base_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=3)
		self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

		self._apply_lora_to_bert(self.base_model, lora_dim=8)
		self.model = self.base_model  # assign the actual model
		self.model.to(self.device) 
		for name, param in self.base_model.named_parameters():
			if "lora_" in name:
				param.requires_grad = True
			else:
				param.requires_grad = False
		
		self.epochs = 20




	def train_test_split(self, df, train_pct=0.8):
		train_set, test_set = train_test_split(df, test_size=1-train_pct)
		train_df = train_set[['text','label']]
		test_df = test_set[['text','label']]
		return train_df, test_df

	def train(self, train_df, batch_size=4, accum_steps=4):

		
		encodings = self.tokenizer(
			train_df['text'].tolist(),
			truncation=True,
			padding=True,
			return_tensors='pt'
		)
		labels = torch.tensor(train_df['label'].tolist(), dtype=torch.long)

		dataset = list(zip(encodings['input_ids'], encodings['attention_mask'], labels))
		train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		optimizer = torch.optim.Adam(
			self.model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08
		)
		total_steps = len(train_loader) * self.epochs
		scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
		)

		self.model.to(self.device)
		self.model.train()

		for epoch in range(self.epochs):
			running_loss = 0.0
			optimizer.zero_grad()

			for step, (input_ids, attention_mask, labels_tensor) in enumerate(train_loader):
				input_ids = input_ids.to(self.device)
				attention_mask = attention_mask.to(self.device)
				labels_tensor = labels_tensor.to(self.device)

				# Forward pass
				outputs = self.model(
					input_ids=input_ids, attention_mask=attention_mask, labels=labels_tensor
				)
				loss = outputs.loss / accum_steps  # normalize loss
				loss.backward()

				# Gradient accumulation step
				if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
					optimizer.step()
					scheduler.step()
					optimizer.zero_grad()

				running_loss += loss.item() * accum_steps

			avg_loss = running_loss / len(train_loader)
			print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")


        
	def predict(self, test_df):
		self.model.eval()
		y_pred = []
		y_true = []

		for text, label in zip(test_df['text'], test_df['label']):
			enc = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
			input_ids = enc['input_ids'].to(self.device)
			attention_mask = enc['attention_mask'].to(self.device)

			with torch.no_grad():
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				predicted = int(torch.argmax(outputs.logits, dim=1).item())

			y_pred.append(predicted)
			y_true.append(label)

		# print("y_pred length:", len(y_pred), "y_true length:", len(y_true))  # sanity check
		return y_pred, y_true



	def plot_confusion_matrix(self, y_pred, y_true):
		mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
		df_cm = pd.DataFrame(mat, range(3), range(3))
		sns.heatmap(df_cm, annot=True) 
		plt.ylabel('True')
		plt.xlabel('Predicted')
		plt.show()

	def report_eval_stats(self, y_pred, y_true):
		return sklearn.metrics.classification_report(y_true, y_pred,target_names=['positive','neutral','negative'])

	def classify(self, text):
		df = pd.DataFrame({'text':[text], 'label':[0]})
		result, _ = self.predict(df)   # result is a list of predicted class indices
		pos = result[0]                # directly take the first element
		return SENTIMENT_DICT[pos]

	def _apply_lora_to_bert(self, model, lora_dim):
		
		for name, module in model.named_modules():  
			if isinstance(module, nn.Linear):
				if "attention.self.query" in name or "attention.self.value" in name:
					
					name_struct = name.split(".")
					parent = model
					for struct in name_struct[:-1]: 
						parent = getattr(parent, struct)
					
					
					orig_linear = getattr(parent, name_struct[-1])
					lora_linear = LoRA_Linear(orig_linear.weight, orig_linear.bias, lora_dim)
					setattr(parent, name_struct[-1], lora_linear)

		print(model)


class LoRA_Linear(nn.Module):
	def __init__(self, weight, bias, lora_dim):
		super().__init__()
		row, column = weight.shape

        # restore Linear
		if bias is None:
			self.linear = nn.Linear(column, row, bias=False)
			self.linear.load_state_dict({"weight": weight})	
		else:
			self.linear = nn.Linear(column, row)
			self.linear.load_state_dict({"weight": weight, "bias": bias})

        # create LoRA weights (with initialization)
		self.lora_right = nn.Parameter(torch.zeros(column, lora_dim))
		nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))
		self.lora_left = nn.Parameter(torch.zeros(lora_dim, row))
	def forward(self, input):
		x = self.linear(input)
		y = input @ self.lora_right @ self.lora_left
		return x + y