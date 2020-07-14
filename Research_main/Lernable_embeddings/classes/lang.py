#!usr/bin/python

class Lang:
    def __init__(self):
        self.eos_token = 1
        self.sos_token = 2
        self.word2index = {"eos":1, "sos":2}
        self.word2count = {}
        self.index2word = {1:"eos", 2:"sos"}
        self.n_words = 3
        
    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words+=1
        else:
            self.word2count[word]+=1
