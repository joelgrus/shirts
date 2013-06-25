from PIL import Image
import PIL.ImageOps

from collections import defaultdict
from glob import glob
from random import shuffle, seed
import numpy as np
import pylab as pl
import pandas as pd
import re
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression


# this is the size of all the Target.com images
STANDARD_SIZE = (138,138)
HALF_SIZE = (STANDARD_SIZE[0]/2,STANDARD_SIZE[1]/2)

def img_to_array(filename):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

# my files are set up like "images/girls/gapkids/image1.jpg" and "images/boys/oldnavy/image1.jpg"
girls_files = glob('images/girls/*/*')
boys_files = glob('images/boys/*/*')

process_file = img_to_array

raw_data = [(process_file(filename),'girl',filename) for filename in girls_files] + \
           [(process_file(filename),'boy',filename) for filename in boys_files]

# randomly order the data
seed(0)
shuffle(raw_data)

# pull out the features and the labels
data = np.array([cd for (cd,_y,f) in raw_data])
labels = np.array([_y for (cd,_y,f) in raw_data])

# find the principal components
N_COMPONENTS = 10
pca = RandomizedPCA(n_components=N_COMPONENTS, random_state=0)
X = pca.fit_transform(data)
y = [1 if label == 'boy' else 0 for label in labels]

def image_from_component(component):
    """takes one of the principal components and turns it into an image"""
    hi = max(component)
    lo = min(component)
    n = len(component) / 3
    def rescale(x):
        return int(255 * (x - lo) / (hi - lo))
    d = [(rescale(component[3 * i]),
          rescale(component[3 * i + 1]),
          rescale(component[3 * i + 2])) for i in range(n)]
    im = Image.new('RGB',STANDARD_SIZE)
    im.putdata(d)
    return im

# write out each eigenshirt and the shirts that 
for i,component in enumerate(pca.components_):
    img = image_from_component(component)
    img.save(str(i) + "_eigenshirt.png")
    reverse_img = PIL.ImageOps.invert(img)
    reverse_img.save(str(i) + "_inverted_eigenshirt.png")

    ranked_shirts = sorted(enumerate(X),
           key=lambda (a,x): x[i])
    most_i = ranked_shirts[-1][0]
    least_i = ranked_shirts[0][0]
    ranked_shirts.sort(key=lambda (a,x): abs(x[i]))
    no_i = ranked_shirts[0][0]

    Image.open(raw_data[most_i][2]).save(str(i) + "_most.png")
    Image.open(raw_data[least_i][2]).save(str(i) + "_least.png")
    Image.open(raw_data[no_i][2]).save(str(i) + "_none.png")

def reconstruct(shirt_number):
    """this was my attempt to reconstruct shirts from their first 10 principal components
    but they don't look like much of anything"""
    components = pca.components_
    eigenvalues = X[shirt_number]
    eigenzip = zip(eigenvalues,components)
    N = len(components[0])    
    r = [int(sum([w * c[i] for (w,c) in eigenzip]))
                     for i in range(N)]
    d = [(r[3 * i], r[3 * i + 1], r[3 * i + 2]) for i in range(len(r) / 3)]
    img = Image.new('RGB',STANDARD_SIZE)
    img.putdata(d)
    print raw_data[shirt_number][2]
    img.save('reconstruct.png')

#find and reconstruct the monkey shirt:
monkey_index = [i for (i,(cd,_y,f)) in enumerate(raw_data) if '243A637' in f]
reconstruct(282)
    
#
# and now for some predictive modeling

# split the data into a training set and a test set
train_split = int(len(data) * 4.0 / 5.0)

X_train = X[:train_split]
X_test = X[train_split:]
y_train = y[:train_split]
y_test = y[train_split:]

# if you wanted to use a different model, you'd specify that here
clf = LogisticRegression(penalty='l2')
clf.fit(X_train,y_train)

print "score",clf.score(X_test,y_test)
    
# and now some qualitative results

# first, let's find the model score for every shirt in our dataset
probs = zip(clf.decision_function(X),raw_data)

girliest_girl_shirt = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'girl' else 1,p))[0]
girliest_boy_shirt = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'boy' else 1,p))[0]
boyiest_girl_shirt = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'girl' else 1,-p))[0]
boyiest_boy_shirt = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'boy' else 1,-p))[0]
most_androgynous_shirt = sorted(probs,key=lambda (p,(cd,g,f)): abs(p))[0]

# and let's look at the most and least extreme shirts
cd = zip(X,raw_data)
least_extreme_shirt = sorted(cd,key=lambda (x,(d,g,f)): sum([abs(c) for c in x]))[0]
most_extreme_shirt =  sorted(cd,key=lambda (x,(d,g,f)): sum([abs(c) for c in x]),reverse=True)[0]

least_interesting_shirt = sorted(cd,key=lambda (x,(d,g,f)): max([abs(c) for c in x]))[0]
most_interesting_shirt =  sorted(cd,key=lambda (x,(d,g,f)): min([abs(c) for c in x]),reverse=True)[0]

# and now let's look at precision-recall
probs = zip(clf.decision_function(X_test),raw_data[train_split:])
num_boys = len([c for c in y_test if c == 1])
num_girls = len([c for c in y_test if c == 0])
lowest_score = round(min([p[0] for p in probs]),1) - 0.1
highest_score = round(max([p[0] for p in probs]),1) + 0.1
INTERVAL = 0.1

# first do the girls
score = lowest_score
while score <= highest_score:
    true_positives  = len([p for p in probs if p[0] <= score and p[1][1] == 'girl'])
    false_positives = len([p for p in probs if p[0] <= score and p[1][1] == 'boy'])
    positives = true_positives + false_positives
    if positives > 0:
        precision = 1.0 * true_positives / positives
        recall = 1.0 * true_positives / num_girls
        print "girls",score,precision,recall
    score += INTERVAL

# then do the boys
score = highest_score
while score >= lowest_score:
    true_positives  = len([p for p in probs if p[0] >= score and p[1][1] == 'boy'])
    false_positives = len([p for p in probs if p[0] >= score and p[1][1] == 'girl'])
    positives = true_positives + false_positives
    if positives > 0:
        precision = 1.0 * true_positives / positives
        recall = 1.0 * true_positives / num_boys
        print "boys",score,precision,recall
    score -= INTERVAL

# now do both
score = lowest_score
while score <= highest_score:
    girls  = len([p for p in probs if p[0] <= score and p[1][1] == 'girl'])
    boys = len([p for p in probs if p[0] <= score and p[1][1] == 'boy'])
    print score, girls, boys
    score += INTERVAL


