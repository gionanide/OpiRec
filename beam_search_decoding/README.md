## Implemented tasks

```javascript

          * Decoder is running word by word
          * Beam search 
          
```
Implementation procedure:


#### Decoder

          * Decoder start generating words given a random tensor as first input
          * From the first prediction we take the k (beam depth) words with the highest probability
                  * For every of the k words we repeat the following procedure:
                                    * give this word as input to the Decoder
                                    * take the k words with the highest probability
                                    * we keep only the k paths with the higest joint probability
          

