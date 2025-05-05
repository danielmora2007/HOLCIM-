

from array import array
import torch
from tqdm.notebook import tqdm
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from textwrap import wrap
import pickle

import pandas as pd

df= pd.read_excel(r'Base ME COLOMBIA (2).xlsx') #IMPORTAMOS LOS DATOS, aqui entrenamos los correos de colombia, puedes poner datos de otro pais.
df.head()
df["reviwe"] = df["Breve descripción"]  + " " + df["Descripción"]     #UNIMOS LAS DOS COLUMNAS EN UNA NUEVA

df.drop('País', axis=1, inplace=True)    #BORRAMOS VARIAS COLUMNAS
df.drop('Breve descripción', axis=1, inplace=True)
df.drop('Correo electrónico', axis=1, inplace=True)
df.drop('Descripción', axis=1, inplace=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN=300 #MAXIMA NUMERO DE PALABRAS 

################## Codificación de las etiquetas ##############################################33
possible_labels = df.Clasificacion.unique() #SE ESTABLESE LAS CATEGORIAS


# SE ENUMERA CADA CATEGORIA
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict


#NUEVA COLUMNA DONDE VA EL NUMERO DE LA CATEGRORIA CORRESPONDIENTE
df['label'] = df.Clasificacion.replace(label_dict)
df


#División de entrenamiento y validación
#Debido a que las etiquetas están desequilibradas, 
#dividimos el conjunto de datos de forma estratificada, usándolo como las etiquetas de clase.

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.label.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'
df["reviwe"]=df["reviwe"].apply(str)                                 
df.groupby(['Clasificacion', 'label', 'data_type']).count()


#TOKENIZADOR DEL MODELO PREENTRENADO
tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', 
                                          do_lower_case=True)
                                          
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].reviwe.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    truncation=True,
    max_length=256, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].reviwe.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)



# SE IMPORTAL EL MODELO PREENTRANADO BERT EN ESPAÑOL
model = BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

modelo=model.to(device)                                                      
#Cargadores de datos
#DataLoadercombina un conjunto de datos y una muestra, y proporciona una iteración sobre el conjunto de datos dado.
#Usamos RandomSamplerpara entrenamiento y SequentialSamplerpara validación.
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


batch_size = 3

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

#Optimizador y programador
                                   
from transformers import AdamW, get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
                  
epochs = 3 #NUMERO DE CICLOS

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

#Métricas de rendimiento
#Usaremos la puntuación f1 y la precisión por clase como métricas de rendimiento.
from sklearn.metrics import f1_score

# COMO ES DESBALANCEADO LAS CATEGORIAS TOCA USAR F1 COMO METRICA
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')


#################### Bucle de entrenamiento
import random
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals
    


from ipywidgets import IntProgress
from tqdm import tqdm
import torch



################################################### en este ciclo se hace el entrenamiento, este ciclo se demora mucho, no correrlo ya que 
################################################### el modelo esta en la carpeta y solo hay que cargarlo.
for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
         
        
    #torch.save(model.state_dict(), f'data_volume/finetuned_BERT_epoch_{epoch}.model')
    torch.save(model.state_dict(), f'finetuned_BERT_epoch_{epoch}.model') #se guarda el modelo entrenado 

    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')

#################################################################################################################

################ Cargar y evaluar el modelo ##############################33


model = BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.to(device)

#CARGAMOS EL MODELO DE DONDE SE ENTRA EL ARCHIVO EN ESTE CASO LLAMADO "finetuned_BERT_epoch_1.model"
model.load_state_dict(torch.load('finetuned_BERT_epoch_3.model', map_location=torch.device('cuda')))

def CATEGORIA_PREDICHA(review_text):
  encoding_review = tokenizer.encode_plus(
      review_text,
      max_length = MAX_LEN,
      truncation = True,
      add_special_tokens = True,
      return_token_type_ids = False,
      padding='max_length',
      return_attention_mask = True,
      return_tensors = 'pt'
      )
  
  input_ids = encoding_review['input_ids'].to(device)
  attention_mask = encoding_review['attention_mask'].to(device)
  output = model(input_ids, attention_mask)
  predictions = torch.sigmoid(output.logits).cpu().detach().numpy().tolist()
  return possible_labels[int(np.argmax(predictions, axis=1))]


############## CICLO PARA HALLAR TODAS las categorias predichas
def asf3():
 asx=[]
 for i in range(0, len(df)):
  asx.append(CATEGORIA_PREDICHA( df["reviwe"][i]))
 return asx

data = {'CATEGORIA_PREDICHA': asf3()}
df2 = pd.DataFrame(data, columns = ['CATEGORIA_PREDICHA'])
df2.to_excel('CATEGORIA_PREDICHA_colombia.xlsx')

##############################################################
#FUNCION PARA HALLAR LA SEGUNDA PROBABILIDA MAS ALTA

def SEGUNDA_PROBA(review_text):
  encoding_review = tokenizer.encode_plus(
      review_text,
      max_length = MAX_LEN,
      truncation = True,
      add_special_tokens = True,
      return_token_type_ids = False,
      padding='max_length',
      return_attention_mask = True,
      return_tensors = 'pt'
      )
  
  input_ids = encoding_review['input_ids'].to(device)
  attention_mask = encoding_review['attention_mask'].to(device)
  output = model(input_ids, attention_mask)
  predictions = torch.sigmoid(output.logits).cpu().detach().numpy().tolist()
  return sorted(predictions[0])[-2]


############## CICLO PARA HALLAR TODAS LAS 2 PROBABBILIDADES DEL ARCHIVO ME MEXICO
def asf2():
 asx=[]
 for i in range(0, len(df)):
  asx.append(SEGUNDA_PROBA( df["reviwe"][i]))
 return asx

data = {'SEGUNDA_PROBA': asf2()}
df2 = pd.DataFrame(data, columns = ['SEGUNDA_PROBA'])
df2.to_excel('SEGUNDA_PROBA_colombia.xlsx')