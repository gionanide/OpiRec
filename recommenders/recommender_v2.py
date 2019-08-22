#!usr/bin/python
import keras
import sys
#insert the path to shared file first and then import the scripts
sys.path.insert(0, '/media/data/gionanide/shared_python_scripts')
import bahdanau_attention

'''

here we are implementing our model, which is consist of three model, the archtecture is as follows:

user model: input --> all users revies, recall that in the first two models their common review is not in those sets, but it is used as the ground truth

product model: input --> all products reviews

neighbourhood model: input --> all users neighbourhood reviews

'''
def lstm_joint_network(user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, max_review_length, normalize_reviews, hidden_size):
    
        #just to check
        #print(train_user.shape)
        #print('train user shape',train_user.shape[1])
        
        #-----------------------------------------> ENCODER <-----------------------------------------------------------
        
        
        #-------------------------------------------------------------------------------------------------------------------------------------------> Inputs
        #make the user model, initialize it with the inputs
        user_model_inputs = keras.layers.Input(shape=(normalize_reviews, 300), name='user_model_inputs')
        
        
        product_model_inputs = keras.layers.Input(shape=(normalize_reviews, 300), name='product_model_inputs')
        
        
        neighbourhood_model_inputs = keras.layers.Input(shape=(normalize_reviews, 300), name='neighbourhood_model_inputs')
        
        
        #-------------------------------------------------------------------------------------------------------------------------------------------> LSTM layers
        #define the LSTM's dimensionality of the output space, first input (300) as our length of the review, 
        #-- None in the number of rows because for every user we have different number of reviews
        
        #--------------------------------> Bidirectional keras.layers.Bidirectional(keras.layers.LSTM())
        
        user_lstm_layer = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0, name='user_lstm_layer')
        #-- use --
        user_lstm_hidden_sequence, user_lstm_h, user_lstm_c = user_lstm_layer(user_model_inputs) 
        print('User lstm hidden sequence ------>',user_lstm_hidden_sequence.shape)
        print('User lstm hidden states h ------>',user_lstm_h.shape)
  
        product_lstm_layer = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0, name='product_lstm_layer')  
        #-- use --   
        product_lstm_hidden_sequence, product_lstm_h, product_lstm_c = product_lstm_layer(product_model_inputs)
        print('Product lstm hidden sequence ------>',product_lstm_hidden_sequence.shape)
        print('Product lstm hidden states h ------>',product_lstm_h.shape)
        
        neighbourhood_lstm_layer = keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0, name='neighbourhood_lstm_layer')  
        #-- use --   
        neighbourhood_lstm_hidden_sequence, neighbourhood_lstm_h, neighbourhood_lstm_c = neighbourhood_lstm_layer(neighbourhood_model_inputs) 
        print('Neighbourhood lstm hidden sequence ------>',neighbourhood_lstm_hidden_sequence.shape)
        print('Neighbourhood lstm hidden states h ------>',neighbourhood_lstm_h.shape)
        

        #-------------------------------------------------------------------------------------------------------------------------------------------> Attention layers
        
        user_attention = bahdanau_attention.BahdanauAttention(300)
        user_attention_output, user_attention_weights = user_attention([user_lstm_h, user_lstm_hidden_sequence])
        print('User attention ------>',user_attention_output.shape)
        print('User attention weights ------>',user_attention_weights.shape)
        
        
        
        user_attention_output = keras.layers.Reshape((1, 300))(user_attention_output)
        print('User attention reshaped ------>',user_attention_output.shape)
        

        
        
        
        #product_attention = bahdanau_attention.BahdanauAttention(300)
        #product_attention_output, product_attention_weights = product_attention([product_lstm_h, product_lstm_hidden_sequence])
        #print('Product attention ------>',product_attention_output.shape)
        #print('Product attention weights ------>',product_attention_weights.shape)
        
        neighbourhood_attention = bahdanau_attention.BahdanauAttention(300)
        neighbourhood_attention_output, neighbourhood_attention_weights = neighbourhood_attention([neighbourhood_lstm_h, neighbourhood_lstm_hidden_sequence])
        print('Neighbourhood attention ------>',neighbourhood_attention_output.shape)
        print('Neighbourhood attention weights ------>',neighbourhood_attention_weights.shape)
        
        
        
        neighbourhood_attention_output = keras.layers.Reshape((1, 300))(neighbourhood_attention_output)
        print('Neighbourhood attention reshaped ------>',neighbourhood_attention_output.shape)
        
        
        
        
        
        #-----------------------------------------> DECODER <-------------------------------------------------------------------------
        
        
        #-------------------------------------------------------------------------------------------------------------------------------------------> Concatenate output
        #concatenate all outputs (user, neighbourhood) 
        merged_hidden_sequence = keras.layers.Concatenate(axis=1,name='concat')([user_attention_output, neighbourhood_attention_output])
        print('\n')
        print('Merge user, neighbourhood attention outputs ------>',merged_hidden_sequence.shape)
        
        
        
        #concatenate all outputs (user, product hidden states, neighbourhood) 
        merged_hidden_sequence_all = keras.layers.Concatenate(axis=1,name='concat1')([merged_hidden_sequence, product_lstm_hidden_sequence])
        print('\n')
        print('Merge user, product, neighbourhood attention outputs ------>',merged_hidden_sequence_all.shape)
        
        
        

        overall_attention = bahdanau_attention.BahdanauAttention(300)
        overall_attention_output, overall_attention_weights = overall_attention([product_lstm_h, merged_hidden_sequence_all])
        print('Overall attention ------>',overall_attention_output.shape)
        print('Overall attention weights ------>',overall_attention_weights.shape)
        
        
        
        
        
        #--------------------------------------------------------------------------------------------------------------------> Classify
        dense_prepare = keras.layers.Dense(1024, activation='relu', name='prepare')(overall_attention_output)
        print('Dense prepare ------>',dense_prepare.shape)
        
        
        
        dense_output = keras.layers.Dense(max_review_length, activation='relu',name='classifyyyy')(dense_prepare)
        print('Dense classify ------>',dense_output.shape)
        
 #----------------------------------------------------------------------------------------------------------------------------------------------------------------------> Full model
        full_model = keras.models.Model(inputs=[user_model_inputs, product_model_inputs, neighbourhood_model_inputs], outputs=[dense_output])
        
        
        
        print(full_model.summary())
        
        
        keras.utils.vis_utils.plot_model(full_model, show_shapes=True, show_layer_names=True, to_file='recommender.png')
        
        
        return full_model
