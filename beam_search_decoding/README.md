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
         
###### Stage 2
         
         - Then we feed the Decoder with the first word of the k predictions that we kept. And we make 
         - another  Priority Queue, a temporary one, to keep the k words with the highest probability 
         - which generated when  we fed the model with the first of the k words that we initialy kept. 
         - We are keeping paths and joint probabilities because each word is stem from a previous word
         - so we are having a sequence of words.
         
###### Stage 3

         - As a last step we are iterating all the first k words and we feed the Decoder for each 
         - one of them. For every iteration the Decoder generates predictions and we keep the k most 
         - probable. Recall that the temporary PQ (Priority Queue) has k words. So for every word 
         - generated we check if its probability is higher than the minimum probability of the PQ. 
         - If it is, we erase the element with the minimum probability and we insert the other. 
         - We do this iteratively until we iterate all the first k generated words.
          
###### Final stage

         - After all this we have conclude to a PQ which has the k most probable paths until now. 
         - If we continue this procedure for all the words we have the k most probable sentence predictions. 
         - Stage 1,2,3 is only for the prediction of one word given k words. So before we go again in the 
         - Stage 1 to predict the next k words for every word in k words we have to replace the global 
         - PQ with the temporary one.
