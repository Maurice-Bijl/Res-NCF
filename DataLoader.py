
import numpy as np
import pandas as pd
from time import time
import scipy.sparse as sp
import argparse

class DataSet:

    def __init__(self):
        self.pos_per_user = None
        self.nUsers = 0
        self.nItems = 0
        self.nClicks = 0

        self.userids = {}
        self.itemids = {}

        self.rUserids = {}
        self.rItemids = {}

        self.pos_per_user = {}

        ##path variables
        self.save_path = "data/"
        self.raw_path = "raw_data/"
        self.file_name = "combined_data_"
        self.number_files = 4

    def set_paths(self,  path = "/data", raw = "raw_data/",filename = "combined_data_", number_f = 4):
        self.raw_path = raw
        self.file_name = filename
        self.number_files = number_f
        self.save_path = path

    def read_raw_data(self, userMin, itemMin, userSample =15_000, itemSample = 6_000):

        t1 = time() # set timer
        np.random.seed (777)

        ## load datasets as pandas dataframe
        df = pd.read_csv (self.raw_path + self.file_name + "1.txt" ,header = None,
                       names = ['Cust_Id', 'Rating',"Date"])
        if self.number_files > 1:
            for i in range(2, self.number_files + 1):
                df = df.append(pd.read_csv (self.raw_path + self.file_name + str(i) + ".txt",header = None,
                        names = ['Cust_Id', 'Rating',"Date"]))

        df ['Rating'] = df ['Rating'].astype (float)
        df.index = np.arange (0, len (df)) ## reset the index

        movie_count = df.isnull ().sum () [1] ## number of movies
        cust_count = df ['Cust_Id'].nunique () - movie_count ## number of users
        rating_count = df ['Cust_Id'].count () - movie_count ## number of ratings

        print ("\nLoading raw data done in %s seconds.\nnumber of movies: %d.\nnumber of users: %d."
               "\nnumber of ratings: %d." % (round(time ()- t1,2), movie_count, cust_count, rating_count))

        ## create an array for the movie ids
        df_nan = pd.DataFrame(pd.isnull(df.Rating))
        df_nan = df_nan[df_nan['Rating']==True]
        df_nan = df_nan.reset_index()

        movie_np = np.empty (rating_count)
        movie_id = 1
        start = 0
        end = 0
        for i, j in zip (df_nan ['index'] [1:], df_nan ['index'] [:-1]):
            temp = np.full ((1, i - j - 1), movie_id)
            end += temp.shape [1]
            movie_np [start:end] = movie_id
            start += temp.shape [1]
            movie_id += 1

        movie_np [end:] = movie_id

        ## add movie ID to df
        df = df [pd.notnull (df ['Rating'])]
        df ['Movie_Id'] = movie_np.astype (int)
        df ['Cust_Id'] = df ['Cust_Id'].astype (int)

        ## save in right order
        cols = ['Cust_Id', 'Movie_Id', 'Rating', 'Date']
        df = df [cols]
        df.sort_values (by = ['Cust_Id'])
        df ['Date'] = pd.to_datetime (df ['Date'])
        df ['Date'] = pd.to_numeric (df ['Date'])
        df['Rating'] = 1

        print ('\nDataset Examples: ')
        print (df.iloc [::10_000_000, :])

        if userSample == 0: ## 0 is default value for the whole user population
            user_df = df [['Cust_Id', 'Rating']].groupby ('Cust_Id').sum ()
            user_tresh = user_df [user_df ['Rating'] >= userMin].reset_index ()
            user_tresh = user_tresh ['Cust_Id'].values
            df = df [df ['Cust_Id'].isin (user_tresh)]
        else:
            user_df = df [['Cust_Id', 'Rating']].groupby ('Cust_Id').sum ()
            user_tresh = user_df [user_df ['Rating'] >= userMin].reset_index ()
            user_tresh = user_tresh ['Cust_Id'].values
            user_sample = np.random.choice (user_tresh, userSample, replace = False)
            df = df [df ['Cust_Id'].isin (user_sample)]

        cust_count_after = df ['Cust_Id'].nunique ()

        if itemSample == 0:
            item_df = df [['Movie_Id', 'Rating']].groupby ('Movie_Id').sum ()
            item_tresh = item_df [item_df ['Rating'] >= itemMin].reset_index ()
            item_tresh = item_tresh ['Movie_Id'].values
            df = df [df ['Movie_Id'].isin (item_tresh)]

        else:
            item_df = df [['Movie_Id', 'Rating']].groupby ('Movie_Id').sum ()
            item_tresh = item_df [item_df ['Rating'] >= itemMin].reset_index ()
            item_tresh = item_tresh ['Movie_Id'].values
            item_sample = np.random.choice (item_tresh, itemSample, replace = False)
            df = df [df ['Movie_Id'].isin (item_sample)]

        movie_count_after = df ['Movie_Id'].nunique ()

        while cust_count != cust_count_after and movie_count != movie_count_after:
            cust_count = df ['Cust_Id'].nunique ()

            user_df = df [['Cust_Id', 'Rating']].groupby ('Cust_Id').sum ()
            user_tresh = user_df [user_df ['Rating'] >= userMin].reset_index ()
            user_tresh = user_tresh ['Cust_Id'].values
            df = df [df ['Cust_Id'].isin (user_tresh)]

            movie_count = df ['Movie_Id'].nunique ()
            item_df = df [['Movie_Id', 'Rating']].groupby ('Movie_Id').sum ()
            item_tresh = item_df [item_df ['Rating'] >= itemMin].reset_index ()
            item_tresh = item_tresh ['Movie_Id'].values
            df = df [df ['Movie_Id'].isin (item_tresh)]

            cust_count_after = df ['Cust_Id'].nunique ()
            movie_count_after = df ['Movie_Id'].nunique ()

        ## save dataframe for later use
        np.savetxt(self.save_path + "nx_U" + str(userSample)+"_I"+ str(itemSample) +"_minU" + str(userMin) + "_minI" +
                   str(itemMin) + ".txt",df.values,fmt = '%s')
        print ('\nReshaping and saving for later use is done in %s seconds. Thanks for waiting! :D' %
                   (round (time () - t1, 2)))
        return df

    def loadClicks(self, userMin, itemMin, userSample =15_000, itemSample = 6_000):

        try:
            df = pd.read_csv (self.save_path + "nx_U" + str(userSample)+"_I"+ str(itemSample) +"_minU" +
                              str(userMin) + "_minI" + str(itemMin) + ".txt", delimiter = " ", header = None,
                              names = ['Cust_Id', 'Movie_Id', 'Rating', 'Date'])
            print ("\nLoading clicks from %s, userMin = %d  itemMin = %d " %
                   (self.save_path + "nx_U" + str(userSample)+"_U"+ str(itemSample) +"_minU" + str(userMin) +
                    "_minI" + str(itemMin) + ".txt", userMin, itemMin))
            df ['Cust_Id'] = df ['Cust_Id'].astype (int)
            df ['Movie_Id'] = df ['Movie_Id'].astype (int)

        except FileNotFoundError:
            print ("\nLoading clicks from raw data, userMin = %d  itemMin = %d " % (userMin, itemMin))
            df = self.read_raw_data (userMin, itemMin,userSample, itemSample)

        df ["Rating"] = 1
        user_df = df [['Cust_Id', 'Rating']].groupby ('Cust_Id').sum()
        item_df = df [['Movie_Id', 'Rating']].groupby ('Movie_Id').sum()

        uCounts = user_df.to_dict()['Rating']
        iCounts = item_df.to_dict()['Rating']
        nRead = len(df)
        df ['Date'] = df ['Date'].astype (int)
        print ("\n  First pass: #users = %d, #items = %d, #clicks = %d\n" % (len (uCounts), len (iCounts), nRead))

        for uName, iName, _, time in df.values:
            try:
                tmp = int (time)
            except:
                continue

            if uCounts [uName] < userMin:
                continue

            if iCounts [iName] < itemMin:
                continue

            self.nClicks += 1

            if iName not in self.itemids:
                self.rItemids [self.nItems] = iName
                self.itemids [iName] = self.nItems
                self.nItems += 1

            if uName not in self.userids:
                self.rUserids [self.nUsers] = uName
                self.userids [uName] = self.nUsers
                self.nUsers += 1
                self.pos_per_user [self.userids [uName]] = []
            self.pos_per_user [self.userids [uName]].append ((self.itemids [iName], int (time)))

        print ("  Sorting clicks for each users ")

        for u in range (self.nUsers):
            sorted (self.pos_per_user [u], key = lambda d: d [1])
        sparsity = 1-(float(self.nClicks)/(float(self.nUsers) * float(self.nItems)))
        print ("\n \"nUsers\": %d,\"nItems\":%d, \"nClicks\":%d, \"sparsity\": %f\n"
               % (self.nUsers, self.nItems, self.nClicks, round(sparsity,4)))

        self.val_per_user = []
        self.test_per_user = []
        self.train_per_user = {}
        self.test_negative_per_user = {}
        mat = sp.dok_matrix ((self.nUsers, self.nItems), dtype = np.float32)
        np.random.seed (2017)
        for u in range (self.nUsers):

            if len (self.pos_per_user [u]) < 3:
                item_test = -1
                item_valid = -1
                continue

            item_test = self.pos_per_user [u] [-1] [0]
            self.pos_per_user [u].pop ()
            item_valid = self.pos_per_user [u] [-1] [0]
            self.pos_per_user [u].pop ()
            self.train_per_user [u] = [e [0] for e in self.pos_per_user [u]]
            self.test_per_user.append ([u, item_test])
            self.val_per_user.append ([u, item_valid])

            for item in self.train_per_user [u]:
                mat [u, item] = 1.0

            self.test_negative_per_user [u] = []
            for i in range (100):
                neg_item_id = np.random.randint (0, self.nItems)
                while neg_item_id in self.train_per_user [u] or neg_item_id == item_test \
                        or neg_item_id == item_valid or neg_item_id in self.test_negative_per_user [u]:
                    neg_item_id = np.random.randint (0, self.nItems)
                self.test_negative_per_user [u].append (neg_item_id)
        self.trainMatrix = mat
        self.testRatings = self.test_per_user
        self.validRatings = self.val_per_user

        self.testNegatives = []
        for u in range (self.nUsers):
            if u in self.train_per_user:
                self.testNegatives.append ([e for e in self.test_negative_per_user [u]])


if __name__ == '__main__':
    users = 5000
    items = 2000

    dataset = DataSet ()
    dataset.loadClicks (10, 10, userSample = users, itemSample = items)