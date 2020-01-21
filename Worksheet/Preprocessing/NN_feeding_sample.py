#!usr/bin/python


#make a sample to feed the NN as an object
class Sample(object):


        
        #def __init__(self, userId, businessId, rating_review, text_review, estimation_rating, user_text_reviews, business_text_reviews, neighbourhood_text_reviews):
        
        
        def __init__(self):
        
        
                self.userId = '' # userId, which we want to predict his text review and rating
                self.businessId = '' #businessd, to which the text review and rating is given
                self.rating_review = '' #rating review ground truth
                self.text_review = '' #text review ground truth
                self.estimation_rating = '' #estimated rating review from matrix factorization procedure
                self.user_text_reviews = '' #all the reviews that the user made, except the one to the businessId
                self.user_text_reviews_embeddings = '' #embedding reviews
                self.business_text_reviews = '' #all the businessed reviews, except the one from the userId
                self.business_text_reviews_embeddings = ''
                self.neighbourhood_text_reviews = '' #all the user's neighbourhood reviews
                self.neighbourhood_text_reviews_embeddings = ''
                self.role = '' #represents if the specific review is for training or testing
        
                '''
                #we initialize an object
                self.userId = userId
                self.businessId = businessId
                self.rating_review = rating_review
                self.text_review = text_review
                self.estimation_rating = estimation_rating
                self.user_text_reviews = user_text_reviews
                self.business_text_reviews = business_text_reviews
                self.neighbourhood_text_reviews = neighbourhood_text_reviews
                '''
                
                
        #----------------------------------------------> Setters
        def set_userId(self, userId):
        
                self.userId = userId
                
                
        def set_businessId(self, businessId):
        
                self.businessId = businessId
                
                
        def set_ratingReview(self, rating_review):
        
                self.rating_review = rating_review
                
                
        def set_textReview(self, text_review):
        
                self.text_review = text_review   
                
                
        def set_estimationRating(self, estimation_rating):

                self.estimation_rating = estimation_rating
                
                
        def set_userTextReviews(self, user_text_reviews):
        
                self.user_text_reviews = user_text_reviews 
                
                
        def set_userTextReviewsEmbeddings(self, user_text_reviews_embeddings):
        
                self.user_text_reviews_embeddings = user_text_reviews_embeddings
                
                
        def set_businessTextReviews(self, business_text_reviews):
        
                self.business_text_reviews = business_text_reviews
                
                
        def set_businessTextReviewsEmbeddings(self, business_text_reviews_embeddings):
        
                self.business_text_reviews_embeddings = business_text_reviews_embeddings
                
                
        def set_neighbourhoodTextReviews(self, neighbourhood_text_reviews):
        
                self.neighbourhood_text_reviews = neighbourhood_text_reviews
                
                
        def set_neighbourhoodTextReviewsEmbeddings(self, neighbourhood_text_reviews_embeddings):
        
                self.neighbourhood_text_reviews_embeddings = neighbourhood_text_reviews_embeddings
                
                
        def set_role(self, role):
        
                self.role = role
                
                
                
                
                
        #----------------------------------------------------------------------> Getters
        def get_userId(self):
        
                return self.userId
                
                
        def get_businessId(self):
        
                return self.businessId
                
                
        def get_ratingReview(self):
        
                return self.rating_review
                
                
        def get_textReview(self):
        
                return self.text_review   
                
                
        def get_estimationRating(self):

                return self.estimation_rating
                
                
        def get_userTextReviews(self):
        
                return self.user_text_reviews 
                
                
        def get_userTextReviewsEmbeddings(self):
        
                return self.user_text_reviews_embeddings
                
                
        def get_businessTextReviews(self):
        
                return self.business_text_reviews
                
                
        def get_businessTextReviewsEmbeddings(self):
        
                return self.business_text_reviews_embeddings
                
                
        def get_neighbourhoodTextReviews(self):
        
                return self.neighbourhood_text_reviews
                
                
        def get_neighbourhoodTextReviewsEmbeddings(self):
        
                return self.neighbourhood_text_reviews_embeddings
                
                
        def get_role(self, role):
        
                return self.role               
                
                
                
        #--------------------------------------------------------------> General functions
        
        def print_fields(self):
        
                #print all the fields
                print('userId: ',self.userId)
                print('businessId: ',self.businessId)
                print('rating review: ',self.rating_review)
                print('text_review: ',self.text_review)
                print('estimated rating: ',self.estimation_rating)
                print('all users reviews: ',self.user_text_reviews)
                print('all users reviews embeddings: ',self.user_text_reviews_embeddings)
                print('all business reviews: ',self.business_text_reviews)
                print('all business reviews embeddings: ',self.business_text_reviews_embeddings)
                print('all neighbourhood reviews: ',self.neighbourhood_text_reviews) 
                print('all neighbourhood reviews embeddings: ',self.neighbourhood_text_reviews_embeddings)
                print('role: ',self.role)
                
                
        #return a dictionary with all the fields, make a json format
        def make_field_dictionary(self,user_model,product_model,user_neighbourhood_model):
        
                #if we need all the fields
                sample_dictionary = {'userId': self.userId, 'businessId': self.businessId, 'rating_review': self.rating_review, 'text_review': self.text_review, 'rating_estimation': self.estimation_rating, 'role': self.role, 'user_text_reviews': user_model, 'business_text_reviews': product_model, 'neighbourhood_text_reviews': user_neighbourhood_model}
                
                return sample_dictionary
                
                
                
        def make_field_dictionary_only_embeddings(self,user_model_embeddings,product_model_embeddings,user_neighbourhood_model_embeddings, text_review_embedding):
                
                #if we need only the mebeddings
                sample_dictionary = {'userId': self.userId, 'businessId': self.businessId, 'rating_review': self.rating_review, 'text_review': self.text_review, 'text_review_embedding': text_review_embedding, 'rating_estimation': self.estimation_rating, 'role': self.role, 'user_text_reviews_embeddings': user_model_embeddings, 'business_text_reviews_embeddings': product_model_embeddings, 'neighbourhood_text_reviews_embeddings': user_neighbourhood_model_embeddings}
                
                return sample_dictionary
                
                
                
        def make_field_dictionary_only_review_embeddings(self,text_review_embedding):
        
                sample_dictionary = {'userId': self.userId, 'businessId': self.businessId, 'rating_review': self.rating_review, 'text_review': self.text_review, 'rating_estimation': self.estimation_rating, 'role': self.role, 'text_review_embedding': text_review_embedding}
                
                
                
                return sample_dictionary
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
