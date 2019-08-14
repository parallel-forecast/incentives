# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:59:26 2019

@author: Jakey
"""

# Currently this is not simulating one question I'm interested in: the trade-off between which
# questions forecasters choose to answer, depending on their values?

# TO-DO: better ways of analysing the outcome

# TIP: try setting self.points for forecasters to a function of self-values and see how that influences the leaderboard

import numpy as np
import math

# Parameters
def sample_epistemics():
    return np.random.lognormal()

def sample_values():
    return np.random.lognormal()

def sample_UB_accuracy():
    return 1

def sample_importance():
    return 10*np.random.lognormal()

# Functions
def scoring_function(prediction, outcome):
    return math.log(abs( 1-outcome - prediction))

def imp_weighted_score(importance, score):
    return importance*score


# Forecaster class
class forecaster:
    def __init__(self, epistemics, values):
        self.epistemics = epistemics
        self.values = values
        self.points = 0 
        self.history = []
    
    def vote_power(self):
        if self.points > 0:
            return math.log(self.points, 5)
        else:
            return 0.1
      
    def sample_opinion(self, question):
        # What vote will a forecaster assign to a question?
        opinion = np.random.normal(question.importance, 1/self.values)
        
        if opinion >= question.points:
            vote = self.vote_power()
        else:
            vote = -self.vote_power()
          
        return opinion, vote 
  
    def predict(self, question):
        valid_prediction = False
        while valid_prediction == False:
            prediction = np.random.normal(question.outcome, 0.05*(1/self.epistemics))
            if 0 < prediction < 1:
                valid_prediction = True
        return prediction

forecasters = [forecaster(sample_epistemics(), sample_values()) for i in range(50)]

# Question class
class Q:
    def __init__(self, UB_accuracy, importance):
        self.UB_accuracy = UB_accuracy
        self.importance = importance
        self.points = 1
        self.outcome = np.random.binomial(1,0.5)
    
    def score(self, prediction):
        return scoring_function(prediction, self.outcome) 
        
rounds = 10
Qs_per_round = 10
Qs = []
run_simulation = True

# Iterate over rounds
if run_simulation:
    for r in range(rounds):
        
        # Generate new questions
        new_Qs = [Q(sample_UB_accuracy(), sample_importance()) for i in range(Qs_per_round)]
        Qs.append( new_Qs )
        
        for q in Qs[r]:
            # Forecasters vote on questions
            for f in forecasters:
                q.points += f.sample_opinion(q)[1] #the second output is the vote
                
            # Forecasters forecast and get points
            for f in forecasters:
                forecast = f.predict(q)
                score = q.score(forecast)
                f.points += imp_weighted_score(q.points,score)
                f.history.append( [q, {"forecast": forecast, "score": score, "imp_weighted_score": imp_weighted_score} ] )
                
leaderboard = {f.points:[f, f.values, f.epistemics] for f in forecasters}

for i in sorted (leaderboard) : 
    print ((i, leaderboard[i]), end ="\n\n")
    
for i in sorted (leaderboard) : 
    print (("points", i, "values", leaderboard[i][1], "epistemics", leaderboard[i][2]), end ="\n\n")

#
