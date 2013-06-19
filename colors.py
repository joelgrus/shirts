from PIL import Image
from collections import defaultdict
from glob import glob
from random import shuffle, seed
import numpy as np
import pylab as pl
import re
from sklearn.linear_model import LogisticRegression

# if we do nothing, there are 256 * 256 * 256 = 16M possible RGB colors
# we don't want that many colors, so we'll use this many buckets for each of R, G, and B
NUM_BUCKETS = 3 # this means there will be 3 * 3 * 3 = 27 possible colors

bucket_size = 256 / NUM_BUCKETS

# these are the possible values for r, g, and b
quanta = [bucket_size * i for i in range(NUM_BUCKETS)]
colors = [(r,g,b)
           for r in quanta
           for g in quanta
           for b in quanta]

def quantize(rgb):
    """map a tuple (r,g,b) each between 0 and 255
    to our discrete color buckets"""

    r,g,b = rgb
    r = max([q for q in quanta if q <= r])
    g = max([q for q in quanta if q <= g])
    b = max([q for q in quanta if q <= b])
    return (r,g,b)

# my files are set up like "images/girls/gapkids/image1.jpg" and "images/boys/oldnavy/image1.jpg"
# if yours are set up differently you should change this
girls_files = glob('images/girls/*/*')
boys_files = glob('images/boys/*/*')

def color_dist(image_file):
    """given an image file, return its vector of colors
    (using the quantized colors defined previously)"""
    img = Image.open(image_file)
    num_pixels = img.size[0] * img.size[1]
    color_counts = defaultdict(int)
    for (c,rgb) in img.getcolors(num_pixels):
        color_counts[quantize(rgb)] += c

    # simplest possible is to return 1 if the image contains a color, 0 if it doesn't
    result = [(1 if color_counts[c] > 0 else 0) for c in colors]

    # another possibility would be to return the fraction of pixels that are that color
    # result = [1.0 * color_counts[c] / num_pixels for c in colors]
    return np.array(result)

# now map each file into a tuple (features,label,filename)
# filename is just to make it easier to inspect the results later
data = ([(color_dist(g),0,g) for g in girls_files] +
        [(color_dist(b),1,b) for b in boys_files])

# randomly order the data
seed(0)
shuffle(data)

# pull out the features and the labels
X = np.array([cd for (cd,_y,f) in data])
y = np.array([_y for (cd,_y,f) in data])

# and split it into a training set and a test set
train_split = int(len(data) * 4.0 / 5.0)

X_train = X[:train_split]
X_test = X[train_split:]
y_train = y[:train_split]
y_test = y[train_split:]

# if you wanted to use a different model, you'd specify that here
clf = LogisticRegression(penalty='l2')
clf.fit(X_train,y_train)

print "score",clf.score(X_test,y_test)

# because we rounded down, the bucket "(0,0,0)" contains all colors where the r,g,b are between 0 and BUCKET_SIZE
# as the representative, let's take the *center* of that cube,
# and pair it with the regression coefficient for that color
# that will allow us to see the relative importances of different colors
features = sorted([((r + bucket_size / 2,
                     g + bucket_size / 2,
                     b + bucket_size / 2),i)
                   for ((r,g,b),i) in zip(colors,clf.coef_[0])],key=lambda ci: ci[1],reverse=True)


# super hacky way to get some HTML of the importances
html = """<html><body><table>"""
for ((r,g,b),p) in features:
    pct = round(100.0 * p,2)
    width = str(max(1,abs(pct))) + "px"
    html += '<tr>'
    if pct >= 0:    
        html += '<td width="50%"></td><td width="50%"><div style="width:' + width + ';background-color:rgb(' + str(r) + "," + str(g) + "," + str(b) + ')">' + str(pct) + "</td></div>"
    else:
        html += '<td width="50%"><div style="text-align:right;float:right;width:' + width + ';background-color:rgb(' + str(r) + "," + str(g) + "," + str(b) + ')">' + str(pct) + '</td></div><td width="50%"></td>'
    html += '</tr>'
html += "</table></body></html>"

f = open("html.html","w")
f.write(html)
f.close()

# and now some qualitative results

# first, let's find the model score for every shirt in our dataset
probs = zip(clf.decision_function(X),data)

girliest_girl_shirt = sorted(probs,key=lambda (p,(cd,g,f)): (g,p))[0]
girliest_boy_shirt = sorted(probs,key=lambda (p,(cd,g,f)): (-g,p))[0]
boyiest_girl_shirt = sorted(probs,key=lambda (p,(cd,g,f)): (g,-p))[0]
boyiest_boy_shirt = sorted(probs,key=lambda (p,(cd,g,f)): (-g,-p))[0]
most_androgynous_shirt = sorted(probs,key=lambda (p,(cd,g,f)): abs(p))[0]
blandest = sorted(probs,key=lambda (p,(cd,g,f)): sum(cd))[0]
coloriest = sorted(probs,key=lambda (p,(cd,g,f)): -sum(cd))[0]

# and now let's look at precision-recall
probs = zip(clf.decision_function(X_test),data[train_split:])
num_boys = len([c for c in y_test if c == 1])
num_girls = len([c for c in y_test if c == 0])
lowest_score = round(min([p[0] for p in probs]),1) - 0.1
highest_score = round(max([p[0] for p in probs]),1) + 0.1
INTERVAL = 0.1

# first do the girls
score = lowest_score
while score <= highest_score:
    true_positives  = len([p for p in probs if p[0] <= score and p[1][1] == 0])
    false_positives = len([p for p in probs if p[0] <= score and p[1][1] == 1])
    positives = true_positives + false_positives
    if positives > 0:
        precision = 1.0 * true_positives / positives
        recall = 1.0 * true_positives / num_girls
        print "girls",score,precision,recall
    score += INTERVAL

# then do the boys
score = highest_score
while score >= lowest_score:
    true_positives  = len([p for p in probs if p[0] >= score and p[1][1] == 1])
    false_positives = len([p for p in probs if p[0] >= score and p[1][1] == 0])
    positives = true_positives + false_positives
    if positives > 0:
        precision = 1.0 * true_positives / positives
        recall = 1.0 * true_positives / num_boys
        print "boys",score,precision,recall
    score -= INTERVAL

# now do both
score = lowest_score
while score <= highest_score:
    girls  = len([p for p in probs if p[0] <= score and p[1][1] == 0])
    boys = len([p for p in probs if p[0] <= score and p[1][1] == 1])
    print score, girls, boys
    score += INTERVAL
