## Implemented tasks

```javascript

          * Decoder is running word by word
          * Beam search 
          
```
Implementation procedure:


## Decoder

          * Decoder start generating words given a random tensor as first input
          * From the first prediction we take the k (beam depth) words with the highest probability
                  * For every of the k words we repeat the following procedure:
                                    * give this word as input to the Decoder
                                    * take the k words with the highest probability
                                    * we keep only the k paths with the higest joint probability
                                    
                                    
                                    
#### Procedure insights

         We are using Priority Queues (PQ) to implement the aforementioned procedure.
         
         We can analyze this is three stages.
         
###### Stage 1

         - The decoder given a random input makes a prediction. We keep the k words with the highest 
         - probability (if we use a greedy search we are keeping the word with the highest probability). 
         - We save the k predictions in a Priority Queue, global one, where the first element (root) is the 
         - element with the lowest probability, so the element which is going out first if we insert another 
         - with higher probability (key).
         
         
```python

input: tuple ( user_inputs(user_reviews,300), product_inputs(product_reviews,300), neighbourhood_inputs(neighbourhood_reviews,300) )

output: tensor(1, vocabulary_size) #we choose the k words with the highest probability

- assume k = 3

[probability], [word_id]
  
[ [0.15], [6]           ************* global PQ:  [ [0.15], [6]
  [0.48], [3]                                       [0.18], [8]
  [0.18], [8] ]                                     [0.48], [3] ]


```
         
         
###### Stage 2
         
         - Then we feed the Decoder with the first word of the k predictions that we kept. And we make 
         - another  Priority Queue, a temporary one, to keep the k words with the highest probability 
         - which generated when  we fed the model with the first of the k words that we initialy kept. 
         - We are keeping paths and joint probabilities because each word is stem from a previous word
         - so we are having a sequence of words.
         
         
```python

input to the decoder, the first of the k words.
input: [word_id] = [6]

output: tensor(1, vocabulary_size) #we choose the k words with the highest probability

temporary PQ:  [ [0.089], [6, 80]
                 [0.022], [6, 3]                  
                 [0.018], [6, 230] ]                  


```

         
###### Stage 3

         - As a last step we are iterating all the first k words and we feed the Decoder for each 
         - one of them. For every iteration the Decoder generates predictions and we keep the k most 
         - probable. Recall that the temporary PQ (Priority Queue) has k words. So for every word 
         - generated we check if its probability is higher than the minimum probability of the PQ. 
         - If it is, we erase the element with the minimum probability and we insert the other. 
         - We do this iteratively until we iterate all the first k generated words.
         
         
```python

#I iterate only the first word of the first k words so I have to iterate k-1 more.

input to the decoder, the second of the k words.
input: [word_id] = [3]

temporary PQ:  [ [0.015], [6, 230]            predictions:  [ [0.090], [3, 3]
                 [0.022], [6, 3]                              [0.002], [3, 780] 
                 [0.089], [6, 80] ]                           [0.008], [6, 90] ] 
                 
                 
#check if an element from the prediction has higher probability from the lowest probability of the temporary queue, if this is True insert it in the Queue.

temporary PQ after update:  [ [0.022], [6, 3]         
                              [0.089], [6, 80]                              
                              [0.090], [3, 3] ] 
                              
                              
#we repeat this precedure until we iterate all the elements from the global queue and when we finish we assign the temporary queue to the global one


```

          
###### Final stage

         - After all this we have conclude to a PQ which has the k most probable paths until now. 
         - If we continue this procedure for all the words we have the k most probable sentence predictions. 
         - Stage 1,2,3 is only for the prediction of one word given k words. So before we go again in the 
         - Stage 1 to predict the next k words for every word in k words we have to replace the global 
         - PQ with the temporary one.
         
         
```python

new global PQ = [ [0.022], [6, 3]         
                  [0.089], [6, 80]                              
                  [0.090], [3, 3] ] 
                  
#start again the whole procedure using this as a global queue.


```
