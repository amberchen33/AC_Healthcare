
# coding: utf-8

# ## Model Building

# In[18]:

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Basic operations
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Machine learning
import lightgbm as lgb


# Sklearn packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score 

import lightgbm as lgb




# In[19]:

test_final = pd.read_csv('/Volumes/Transcend/gwu_course/ml1/project/data_test.csv')


# In[20]:

train = pd.read_csv('/Volumes/Transcend/gwu_course/ml1/project/data_cleaned.csv')


# In[21]:

test_final_set = set(test_final.columns)
train_set = set(train.columns)


# In[22]:

combine_set = test_final_set.intersection(train_set)
#print(combine_set)
print(train_set - combine_set)


# In[23]:

train.head()


# In[28]:

print(pd.get_dummies(train['AMONTH']).head(5))


# In[41]:

dummy_list = ['HOSP_LOCTEACH']
table = pd.get_dummies(train['AMONTH'])

for i in dummy_list:
    table = pd.concat([table,pd.get_dummies(train[i])],  axis = 1)
print(table.head(5))  

undummy = ['']


# In[31]:

pd.get_dummies(train['HOSP_LOCTEACH'])


# In[32]:

pd.get_dummies(train['HOSP_REGION'])


# In[33]:

pd.get_dummies(train['PAY1'])


# In[34]:

pd.get_dummies(train['DQTR'])


# In[35]:

pd.get_dummies(train["NPR"])


# In[37]:

pd.get_dummies(["HOSP_CONTROL"])


# In[38]:

pd.


# In[ ]:




# In[ ]:




# In[ ]:




# In[24]:

label = train[["RACE", "ASOURCE", "ATYPE", "TOTCHG", "ZIPINC_QRTL"]]
label.head()


# In[25]:

label.describe()


# In[26]:

print(train.shape)
train = train.drop(["RACE", "ASOURCE", "ATYPE", "TOTCHG", "ZIPINC_QRTL"],1)
print(train.shape)


# # auto-encoder

# In[27]:

import numpy as np
import sklearn.preprocessing as prep
import pandas as pd
import tensorflow as tf
import os

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant*np.sqrt(6.0/(fan_in +fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out), minval=low, maxval=high,
                             dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale = 0.1 ):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale

        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale*tf.random_normal((n_input,)),
            self.weights['w1']), self.weights['b1']
        ))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),self.weights['b2'])

        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints' + "/checkpoint"))
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    def transform(self, X):
        return  self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    def check_restore_parameters(self):
        """ Restore the previously trained parameters if there are any. """
        if self.ckpt and self.ckpt.model_checkpoint_path:
            print("Loading parameters for the Autoencoder...")
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        else:
            # print('1',self.ckpt)
            # print('2',self.ckpt.model_checkpoint_path)
            print("Initializing fresh parameters for the Autoencoder...")

    def check_save(self):
        print('Save sess ......')
        self.saver.save(self.sess, os.path.join('checkpoints','encoder'))
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train, X_test, y_train, y_test = train_test_split(train, label.RACE, test_size=0.3)
n_samples = int(len(X_train))
train_epochs = 100000

batch_size = 8192
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=106, n_hidden=50, transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.00000001),
                                               scale=0.01)
autoencoder.check_restore_parameters()

for epoch in range(train_epochs):
# while True:
    avg_cost = 0
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples*batch_size

    if epoch % display_step == 0:
        print("Epoch: {0}, cost= {1}".format(epoch,avg_cost))
    if epoch % 20 == 0:
        autoencoder.check_save()
        print('Save ......')
        tmp_test = np.array(autoencoder.transform(X_test))
        np.save(file='auto_test.npy', arr=tmp_test)
        tmp_train = np.array(autoencoder.transform(X_train))
        np.save(file='auto_train.npy', arr=tmp_train)

print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))


# In[ ]:




# ## RACE

# In[6]:

X_train, X_test, y_train, y_test = train_test_split(train, label.RACE, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[43]:

clf = RandomForestClassifier()
clf.fit(X_train,y_train)
print("Train score: {}".format(clf.score(X_train,y_train)))
print("Test score: {}".format(clf.score(X_test, y_test)))
print(confusion_matrix(y_test, clf.predict(X_test)))


# In[44]:

def rf_para_search(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    clf = RandomForestClassifier( max_depth=30)
    param_grid = {"max_depth": [10,20,30,40,50]
    }
    # set different n_estimators in the 
    CV_clf = GridSearchCV(estimator= clf, param_grid=param_grid, cv=5)
    CV_clf.fit(X_train, y_train)
    params = CV_clf.best_params_
    print('best para:', params)
    return params

params = rf_para_search(X_train, y_train)
clf = RandomForestClassifier(max_depth=params["max_depth"])
clf.fit(X_train,y_train)
print("Train score: {}".format(clf.score(X_train,y_train)))
print("Test score: {}".format(clf.score(X_test, y_test)))
print(confusion_matrix(y_test, clf.predict(X_test)))


# ## ASOURCE

# In[47]:

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=5)
scores.mean()


# In[48]:

X_train, X_test, y_train, y_test = train_test_split(train, label.ASOURCE, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf2 = RandomForestClassifier(max_depth=30)
clf2.fit(X_train,y_train)
print("Train score: {}".format(clf.score(X_train,y_train)))
print("Test score: {}".format(clf.score(X_test, y_test)))
print(confusion_matrix(y_test, clf.predict(X_test)))


# In[50]:

X_train, X_test, y_train, y_test = train_test_split(train, label.ASOURCE, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf2 = RandomForestClassifier(max_depth=30)
clf2.fit(X_train,y_train)
print("Train score: {}".format(clf.score(X_train,y_train)))
print("Test score: {}".format(clf.score(X_test, y_test)))
print(confusion_matrix(y_test, clf.predict(X_test)))


# In[49]:

def rf_para_search(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    clf = RandomForestClassifier( max_depth=30)
    param_grid = {"max_depth": [10,20,30,40,50]
    }
    # set different n_estimators in the 
    CV_clf = GridSearchCV(estimator= clf, param_grid=param_grid, cv=5)
    CV_clf.fit(X_train, y_train)
    params = CV_clf.best_params_
    print('best para:', params)
    return params

params = rf_para_search(X_train, y_train)
clf2 = RandomForestClassifier(max_depth=params["max_depth"])
clf2.fit(X_train,y_train)
print("Train score: {}".format(clf2.score(X_train,y_train)))
print("Test score: {}".format(clf2.score(X_test, y_test)))
print(confusion_matrix(y_test, clf2.predict(X_test)))


# In[51]:

scores = cross_val_score(clf2, X_train, y_train, cv=5)
scores.mean()


# ## ATYPE

# In[53]:

X_train, X_test, y_train, y_test = train_test_split(train, label.ATYPE, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf3 = RandomForestClassifier(max_depth=30)
clf3.fit(X_train,y_train)
print("Train score: {}".format(clf3.score(X_train,y_train)))
print("Test score: {}".format(clf3.score(X_test, y_test)))
print(confusion_matrix(y_test, clf3.predict(X_test)))


# In[54]:

scores = cross_val_score(clf3, X_train, y_train, cv=5)
scores.mean()


#  ## ZIPINC_QRTL

# In[10]:

X_train, X_test, y_train, y_test = train_test_split(train, label.ZIPINC_QRTL, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf5 = RandomForestClassifier(max_depth=30)
clf5.fit(X_train,y_train)
print("Train score: {}".format(clf5.score(X_train,y_train)))
print("Test score: {}".format(clf5.score(X_test, y_test)))
print(confusion_matrix(y_test, clf5.predict(X_test)))


# In[13]:

scores = cross_val_score(clf5, X_train, y_train, cv=5)
scores.mean()


# ## Light GBM

# In[6]:

X_train, X_test, y_train, y_test = train_test_split(train, label.TOTCHG, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape,
      y_test.shape)


# In[7]:

rounds = 1000

param  = {
    'objective': 'regression',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'num_boost_round': rounds
}


# In[8]:

data_lgb = lgb.Dataset(X_train, y_train)
lgbm = lgb.train(param, data_lgb)
pred = lgbm.predict(X_test)
pred


# In[14]:

r2_score(y_test, pred)


# In[29]:

label.ZIPINC_QRTL=label.ZIPINC_QRTL.astype('category')
label.ZIPINC_QRTL.value_counts()


# In[25]:

X_train, X_test, y_train, y_test = train_test_split(train, label.ZIPINC_QRTL, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[60]:

rounds = 1000

param  = {
    'objective': 'multiclass',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'num_boost_round': rounds,
    'num_class': 5
}


# In[61]:

data_lgb = lgb.Dataset(X_train, y_train)
lgbm = lgb.train(param, data_lgb)
pred = lgbm.predict(X_test)
pred


# In[63]:

pred1.head()


# In[62]:

pred1 = pd.DataFrame(pred)


# In[58]:

max_ = pred1.max(1)


# In[ ]:




# ## Stochastic Gradient Descent - SGD

# In[13]:

from sklearn.linear_model import SGDClassifier


# In[14]:

clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X_train, y_train)


# In[22]:

y_pred =clf.predict(X_test)
print(y_pred)
np.unique(y_pred)


# In[21]:

acc_sgd = round(clf.score(X_train, y_train) * 100, 2)
acc_sgd

