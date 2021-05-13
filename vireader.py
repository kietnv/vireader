import torch
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
from underthesea import sent_tokenize
from transformers import XLMRobertaForQuestionAnswering, XLMRobertaTokenizer
import re

class ViReader():
  def __init__(self,model_path="./model"):
    self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    self.model = XLMRobertaForQuestionAnswering.from_pretrained(model_path)
    self.sentence_model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    
  def pre_processing(text):
      text = text.lower()
      text = text.replace("\n","")
      text = re.sub(' +', ' ', text)
      return text
  
  def predict(self, context, question):
    #input: context, question
    #output: answer
    context = pre_processing(text) 
    question = pre_processing(text)

    #Find top 5 sentences
    corpus = sent_tokenize(context)
    if len(corpus)>5:
      corpus_embeddings = self.sentence_model.encode(corpus, convert_to_tensor=True)
      query_embedding = self.sentence_model.encode(question, convert_to_tensor=True)
      cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
      cos_scores = cos_scores.cpu()
      top_results = torch.topk(cos_scores, k=5)
      sentence_new_context = []
      for score, idx in zip(top_results[0], top_results[1]):
        sentence_new_context.append(corpus[idx])
      context = ''
      for pa in corpus:
        if pa in sentence_new_context: context+= pa+' '
      context = context[:-1]
    
    #predict answer
    encoding = self.tokenizer(question, context, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    start_scores, end_scores = self.model(input_ids, attention_mask=attention_mask, output_attentions=False)[:2]

    all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
    answer = self.tokenizer.convert_tokens_to_ids(answer.split())
    answer = self.tokenizer.decode(answer)
    return answer
