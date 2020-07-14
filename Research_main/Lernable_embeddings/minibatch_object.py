#!usr/bin/python

class Minibatch:
    def __init__(self, user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews):
        #initialize properties
        self.user_reviews = user_reviews
        self.product_reviews = product_reviews
        self.neighbourhood_reviews = neighbourhood_reviews
        self.product_ratings = product_ratings
        self.text_reviews = text_reviews
        self.rating_reviews = rating_reviews
