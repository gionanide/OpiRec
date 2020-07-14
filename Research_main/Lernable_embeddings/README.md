
upcoming link for downloading train, dev, test data (every file .pkl contains an object which has the following fields: user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_review_target, rating_score_target)

#### --------------- Train ---------------
python training.py   
--epochs integer (epochs to train)  
--batch_size integer (samples per batch)  
--hidden_units integer  
--shuffle boolean True/False (shuffle data at each epoch)  
--pretrained boolean True/False (continue training by loading a pretrained model)  
--mode string (only when pretrained=True give the path from the model to be loaded)  
--encoder_input_dropout float (dropout percentage)  
--dropout_lstm_output float (dropout percentage)  
--dropout_after_linear float (dropout percentage)  
--episodes integer (number of recurrencies in DMN)  
--number_of_layers_encoder integer (number of layers of LSTM encoder)  
--number_of_layers_decoder integer (number of layers of LSTM decoder)  
--parallel True/False (if training will by parallel and distributed)  



    
#### --------------- Test ---------------
python inference.py  
--epoch integer (load a model from a specific epoch)  
--batch_size integer (samples per batch)  
--mode string (test/dev validate on test or development set)  

#### --------------- Preprocessing ---------------
##### Input:
REAL_Dataset
##### Outputs:
1. preprocessing.make_vocabularies() --> generate vocabularies
2. preprocessing.encode_sequences() --> generate all the reviews as sequences of indices, ready to fed the model
The ouput in an object (saved .pkl format) having the following fields (for every sample):
----> user_reviews
----> business_reviews
----> neighbourhood_reviews
----> product_ratings
----> text_review_target
----> rating_score_target
