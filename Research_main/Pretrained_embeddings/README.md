###### *upcoming link for downloading train, dev, test data (every file .pkl contains an object which has the following fields: user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_review_target, rating_score_target)*

#### --------------- Train ---------------
python training.py   
--epochs **integer** (epochs to train)  
--batch_size **integer** (samples per batch)  
--hidden_units **integer**  
--shuffle **boolean** True/False (shuffle data at each epoch)  
--pretrained **boolean** True/False (continue training by loading a pretrained model)  
--mode **string** (only when pretrained=True give the path from the model to be loaded)  
--encoder_input_dropout **float** (dropout percentage)  
--dropout_lstm_output **float** (dropout percentage)  
--dropout_after_linear **float** (dropout percentage)  
--episodes **integer** (number of recurrencies in DMN)  
--number_of_layers_encoder **integer** (number of layers of LSTM encoder)  
--number_of_layers_decoder **integer** (number of layers of LSTM decoder)  
--parallel **boolean** True/False (if training will by parallel and distributed)  

Example:
---
```console
(conda_env) ~$ python training.py --epochs 100 --batch_size 32 --hidden_units 300 shuffle True --pretrained False --mode no_path --encoder_input_dropout 0.2 --dropout_lstm_output 0.2 --dropout_after_linear 0.4 --episodes 3 --number_of_layers_encoder 2 --number_of_layers_decoder 2 --parallel True
```
---



#### --------------- Test ---------------
python inference.py  
--epoch **integer** (load a model from a specific epoch)  
--batch_size **integer** (samples per batch)  
--mode **string** (test/dev validate on test or development set)

Example:
---
```console
(conda_env) ~$ python inference.py --epochs 100 --batch_size 32 --mode test
```
---


#### --------------- Download pretrained embeddings ---------------
Google official site: https://code.google.com/archive/p/word2vec/

#### --------------- Preprocessing ---------------
##### Input:
REAL_Dataset
##### Outputs
1. preprocessing.make_sorted_dataset() --> tensors (take from the pretained model) representing the reviews, ready to fed the model.  
The ouput is an object (saved .pkl format) having the following fields (for every sample):  
----> user_reviews  
----> business_reviews  
----> neighbourhood_reviews    
----> product_ratings  
----> text_review_target  
----> rating_score_target  
