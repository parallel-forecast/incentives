# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:59:26 2019

@author: Jakey

Notes by Dave:
    did some ugly stuff... forecaster data is saved within the
    forecaster class, wheras question data is saved in a global
    container. consider generating questions and forecasters prior
    to loop, will save CPU time and will make code nicer, but
    will cost some memory.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt     # for plotting
from scipy.optimize import minimize
from scipy.stats import truncnorm
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

optimisation_mode = True
if optimisation_mode:
    print("Running in optimisation mode, where forecasters explore prediction space"
          "to maximise points instead of just being truthful.")

# Functions
def scoring_function(prediction, outcome, function="log"):
    if function == "log":
        return math.log(abs( 1-outcome - prediction)/0.5)
    elif function == "linear":
        return abs(prediction-outcome)

def imp_weighted_score(importance, score):
    return importance*score

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def linreg (x,y) :      # takes 

    # error handling
    if (len(x) != len(y)) :
        print("incompatible vector sizes in linear regression, exiting")
        return 0
    
    if (len(x) == 0 || len(y) == 0)
        print("linear regression failure, vector size is 0")
        retirn 0

    # special cases
    if (len(y) == 1)
        slope = y

        return y

    # calculate linear regression
    xavg = np.mean(x)
    yavg = np.mean(y)

    a = 0
    b = 0

    n = len(x)

    for k in range n :
        a += (y[k] - yavg) * (x[k] - xavg)
        b += (x[k] - xavg)**2

    slope = a / b

    return slope


def corr (x,y) :     # calculate correlation coeficient, takes np.arrays
    
    # error handling
    if len(x) != len (y) :
        print("incompatible vector sizes in correlation function, exiting function")
        return 0
    
    if (len(x) == 0 || len(y) == 0)
        print("correlation failure, vector size is 0")
        return 0

    # calculate parts
    n = len(x)
    xysum = np.sum(x*y)
    xsum = np.sum(x)
    ysum = np.sum(y)
    xsum2 = np.sum(x**2)
    ysum2 = np.sum(y**2)

    # calculate correlation
    corr = (n*xysum - xsum*ysum)/np.sqrt((n*xsum2-xsum**2)*(n*ysum2-ysum**2))

    return corr

# Forecaster class
class forecaster:
    def __init__(self, epistemics, values, humour, ID):
        self.epistemics = epistemics
        self.values = values
        self.humour = humour
        self.points = 0
        self.history = []
        self.archive = []       # for saving data
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
        w = self.values / (self.values + self.humour)
        opinion = w*importance_opinion + (1-w)*fun_opinion
        
        self.archive.append([importance_opinion,fun_opinion])

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
#        valid_prediction = False
#        
#        while valid_prediction == False:
#            prediction = np.random.normal(question.outcome, 0.05*(1/self.epistemics))
#            if 0 < prediction < 1:
#                valid_prediction = True
        
        # TODO add proper Bayesian belief forming process
        # TODO fix mean and std of this one
        true_belief = truncnorm.rvs(0.001,0.999)
        
     #   strategic_prediction = minimize(-scoring_function, true_belief, method='nelder-mead', options={'xtol': 1e-8, 'disp': True}, bounds=(0.001,0.999))

        if not just_considering_it: #instead actually predicting and registering the score with the forecaster
            question.predictions[self.id] = true_belief

        return true_belief

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
        true_belief = self.predict(question, just_considering_it=True)
        # Figure out score in each possible world
        
        def EV(strategic_prediction, just_considering=True):
            p = true_belief
            
            q = strategic_prediction[0]
            EV = (  p   * imp_weighted_score(question.importance, question.score(q, False)) +
                  (1-p) * imp_weighted_score(question.importance, question.score(q, True))    )
            return EV
        
        def neg_EV(strategic_prediction):
            # minimising the negative because that's how scipy works ¯\_(ツ)_/¯
            return -EV(strategic_prediction)
        
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        if optimisation_mode:
            prediction = minimize(neg_EV, np.array([true_belief]), method='TNC', options={'xtol': 1e-8, 'disp': True}, bounds=((0.001,0.999),))
        else:
            prediction = {"x": true_belief}
            
        return EV(prediction["x"], False) #question.points


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
        return scoring_function(prediction, outcome, function="linear")

rounds = 10
Qs_per_round = 10
Qs = []
run_simulation = True

# Iterate over rounds

qarchive = []

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
        
        # saving data for later
        book = []
        
        for Q_id in range (len(new_Qs)) :
            vals = [Q_id, np.round(new_Qs[Q_id].points) ,np.round(new_Qs[Q_id].importance), np.round(new_Qs[Q_id].fun)]
            
            book.append(vals)

        qarchive.append(book)
        

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

        # Some intermediate graphs

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

# Scatter plots and correlations 

qarchive = np.array(qarchive)   # transform into numpy array for easier maniplulation

blockid = 9

qblock = qarchive[blockid]

pts = qblock[:,1]
imps = qblock[:,2]
funs = qblock[:,3]

corr_imppts = corr(imps,pts)

fig2, ax = plt.subplots()
ax.plot(imps,pts,'bo')
ax.set(xlabel='importance')
ax.set(ylabel='points')
ax.set(title='scatter plot with r = {}'.format(corr_imppts))

fig2.savefig("scatterplot.png",dpi=600)

# histograms

# slice data
blockid = 9     # 0 indexed question set selector 
qid = 9         # 0 indexed question selector 

trueimp = imps[qid]     # warning, must be before imps are cleared
truefun = funs[qid]

# get containers
imps = []
funs = []

# fill containers with data
for f in forecasters :
    imps.append(f.archive[blockid*Qs_per_round+qid][0])
    funs.append(f.archive[blockid*Qs_per_round+qid][1])

nbins = 10      # arbitrary, sort of...

# make histograms

fig1, (ax1, ax2) = plt.subplots(nrows = 2)

n,f1,patches = ax1.hist(imps,nbins,edgecolor='black')
n,f2,patches = ax2.hist(funs,nbins,edgecolor='black')

ax1.axvline(x=trueimp,color='r',linestyle='dashed',linewidth=1)
ax2.axvline(x=truefun,color='r',linestyle='dashed',linewidth=1)

ax1.set(xlabel="importance")
ax1.set(ylabel="fequency")

ax2.set(xlabel="fun")
ax2.set(ylabel="frequency")

fig1.subplots_adjust(hspace=0.45)

fig1.savefig("histogram.png",dpi=600)

# plt.show() # uncomment for showing plot at end of run


