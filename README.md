# Opinion_Recommendation

#### Abstract

###### 
Recommendation aims at providing a service or a product to a consumer based on consumer’s special interests. By doing so, a personalized recommendation is provided, which is a feature of utmost importance for a recommendation system. Here, we are interesting in the opinion recommendation task. That is, to consistently generate a text review and a rating score that a certain user would give to a certain product, which has never seen before. The mathematical modeling of natural language, as well as the recent progress in the specific field, suggests that deep learning methods are justified. As a consequence, deep neural networks will be employed for modeling the text reviews with the objective of capturing user and product profiles. As a first step, a vector representation of the text is derived using the Skip-Gram model. Such a representation, depicts the semantic of the words. Providing the property that two words with similar meaning have small distance between their vector representations the semantics is illustrated. Besides the aforementioned, Skip-Gram model provides a dense vector representation of a word, which enforces generalization power. Taking into account the extracted text representation, Long Short-Term Memory networks (LSTMs) are used to model user and product profile. The recurrent nature of the foresaid networks can model sequential data, as the existing reviews are. An attention mechanism is used to extract the most informative reviews yielding to a better vector representation of the user under consideration. Higher level of abstraction and latent interactions between user and product will be obtained using the Dynamic Memory Network (DMN). The core of the previously mentioned network consists of a recurrent attention mechanism. This property (i.e., recurrent attention) provides relevant features using a query
among product reviews, where the query consists of the derived user vector representation. A reasoning about retrieved facts, that is text reviews, is considered, modeling the interaction between the user and the product. Having all the information captured in a vector representation, the decoding procedure consists of a standard LSTM to generate the text review in natural language. The rating score prediction is an average between the weighted sum of all the previous ratings for the specific product and model prediction.


###### File structure:
#### ----- Research_main -----
folder contains the source code of opinion recommendation task.

#### ---- Worksheet -----
folder contains previous approaches and variations of the main procedure (deprecated).
