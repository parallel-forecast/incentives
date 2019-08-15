# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:59:26 2019

@author: Jakey
"""

# Currently this is not simulating one question I'm interested in: the trade-off between which
# questions forecasters choose to answer, depending on their values?

# TODO: better ways of analysing the outcome

# TODO: give forecasters utility functions which they optimise, instead of just randomly sampling their forecasts (e.g. that way they
# hijack non-proper scoring functions).

# TODO: use more sensible functions for sampling parameters, opinions, forecasts, etc.

# TIP: try setting self.points for forecasters to a function of self-values and see how that influences the leaderboard

# What are the hypothesis this simulation is testing?
#   1) trading off attention across different questions
#   when is the equilibrium for important questions to dominate?
#       need questions to have importance parameter, and "fun" parameter
#       need agents to have different preferences in their upvoting
#   2) can you get high points by being right on trivial questions?
#       need mechanism whereby they trade-off which questions they answer
#   are you going to win by being right on important questions?
#   3) something something efficient frontier of values and epistemics




import numpy as np
import math

# Parameters
def sample_epistemics():
    return np.random.lognormal()

def sample_values():
    return np.random.lognormal()

def sample_humour():
    return np.random.lognormal()

def sample_UB_accuracy():
    return 1

def sample_importance():
    return 10*np.random.lognormal()

def sample_fun():
    return 10*np.random.lognormal()

# Functions
def scoring_function(prediction, outcome):
    return math.log(abs( 1-outcome - prediction)/0.5)

def imp_weighted_score(importance, score):
    return importance*score

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


# Forecaster class
class forecaster:
    def __init__(self, epistemics, values, humour, ID):
        self.epistemics = epistemics
        self.values = values
        self.humour = humour
        self.points = 0
        self.history = []
        self.id = ID

    def __str__(self):
        return "Forecaster {}, with epistemics: {}, values: {}, humour: {}, points: {}".format(
                np.round(self.id,1), np.round(self.epistemics,1), np.round(self.values,1), 
                np.round(self.humour,1), np.round(self.points,1))

    def vote_power(self):
        if self.points > 0:
            return math.log(self.points, 5)
        else:
            return 0.1

    def sample_opinion(self, question):
        # What vote will a forecaster assign to a question?
        importance_opinion = np.random.normal(question.importance, 10/self.values)
        fun_opinion = np.random.normal(question.fun, 10/self.humour)
        w = importance_opinion / (importance_opinion+fun_opinion)
        opinion = w*importance_opinion + (1-w)*fun_opinion

        story = np.random.binomial(1,0.003)

        if opinion >= question.points:
            vote = self.vote_power()
        else:
            vote = -self.vote_power()

        # Occasionally describe what users are doing
        story = np.random.binomial(1,0.003)
        if story:
            print(self, ", faced with\n", question, "\nhad importance opinion: ",
                  importance_opinion, ", fun opinion: ", fun_opinion, "\nand decided to vote: ", vote, "\n\n")

        return opinion, vote

    def predict(self, question, just_considering_it=False):
        valid_prediction = False
        while valid_prediction == False:
            prediction = np.random.normal(question.outcome, 0.05*(1/self.epistemics))
            if 0 < prediction < 1:
                valid_prediction = True

        if not just_considering_it: #instead actually predicting and registering the score
            question.predictions[self.id] = prediction

        return prediction

    def choose_Qs(self, available_Qs):

        # Create dict of question_index:points
        point_dict = { i:self.EV_of_predicting(available_Qs[i]) for i in range(len(available_Qs)) }
        # Pick the curret top half of questions to forecast
        point_ordering = sorted(point_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

        chosen_Qs = [available_Qs[Q_id] for Q_id, points in point_ordering[0:( math.floor( len(available_Qs)/2 ) )] ]

        return chosen_Qs

    def EV_of_predicting(self, question):
        # The expected value of predicting on this question for this forecaster.
        # Should take into account: edge over crowd + question points
        # TODO
        # Figure out what would have been predicted
        p = self.predict(question, just_considering_it=True)
        # Figure out score in each possible world
        EV = (  p   * imp_weighted_score(question.importance, question.score(p, False)) +
              (1-p) * imp_weighted_score(question.importance, question.score(p, True))    )
        return EV
#        return question.points


forecasters = [forecaster(sample_epistemics(), sample_values(), sample_humour(), i) for i in range(50)]

# Question class
class Q:
    def __init__(self, UB_accuracy, importance, fun):
        self.UB_accuracy = UB_accuracy
        self.importance = importance
        self.fun = fun
        self.points = 1
        self.outcome = np.random.binomial(1,0.5)
        self.predictions = {}

    def __str__(self):
        return "Question, with importance: {}, fun: {}, points: {}, and {} predictions".format(
                np.round(self.importance,1), np.round(self.fun,1), np.round(self.points,1), len(self.predictions))

    def score(self, prediction, flip_outcome=False):
        outcome = abs(self.outcome - int(flip_outcome))
        return scoring_function(prediction, outcome)

rounds = 10
Qs_per_round = 10
Qs = []
run_simulation = True

# Iterate over rounds
if run_simulation:
    for r in range(rounds):

        # Generate new questions
        new_Qs = [Q(sample_UB_accuracy(), sample_importance(), sample_fun()) for i in range(Qs_per_round)]
        Qs.append( new_Qs ) # I think this saves a pointer rather than copy, but not sure. Otherwise should replace new_Qs with Qs[r] below

        for f in forecasters:

            # Vote on Qs
            for q in new_Qs:
                q.points += f.sample_opinion(q)[1] #index 1 as the second output is the vote

            # Choose Qs to predct
            chosen_Qs = f.choose_Qs( new_Qs )
            for q in chosen_Qs:
                f.predict( q )

        print("Round ", r, " question ranking:\n\n")

        # Create dict of question_index:points
        point_dict = { i:new_Qs[i].points for i in range(len(new_Qs)) }
        # Pick the curret top half of questions to forecast
        point_ordering = sorted(point_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

        for Q_id, points in point_ordering:
            print( ("Question_id", Q_id, "Points", np.round(new_Qs[Q_id].points),
                    "Importance", np.round(new_Qs[Q_id].importance), "fun", np.round(new_Qs[Q_id].fun),
                    "Number of predictions", len(new_Qs[Q_id].predictions)) )
        print("\n\n")

        # TODO DISPLAY GRAPHS
        #   1) Occasionally, histogram over forecaster sampled opinions about a question's 
        #       fun and importance, along with a vertical line showing true values
        #   2) Occasionally the following correlations: question points x importance, 
        #       question points x fun, importance x number of  predictions, fun x number of predictions
        #   3) For each of those correlations, after the final round is finished, 
        #       display a line graph showing how the round correlation coefficient evolves over time over rounds


        #Award points
        for q in new_Qs:
            for f_id in q.predictions:
                prediction = q.predictions[f_id]
                score = q.score(prediction)
                imp_score = imp_weighted_score(q.points,score)
                f = forecasters[f_id]
                f.points += imp_score
                f.history.append( [q, {"forecast": prediction, "score": score, "imp_weighted_score": imp_score} ] )

leaderboard = {f.points:[f, f.values, f.epistemics, f.humour] for f in forecasters}
order = sorted(leaderboard)

for i in order :
    print (("forecaster_id", leaderboard[i][0].id, "points", np.round(i,1),
            "values", np.round(leaderboard[i][1],1), "epistemics", np.round(leaderboard[i][2],1),
            "humour", np.round(leaderboard[i][3],1)),    end ="\n\n")

#DISPLAY GRAPHS
