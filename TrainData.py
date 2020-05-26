
import numpy as np

def get_train_pair(train, num_negatives):
    user_input, item_pos, item_neg, labels = [], [], [], []
    num_items = train.shape [1]
    for (u, i) in train.keys ():
        # positive instance
        user_input.append (u)
        item_pos.append (i)
        j = np.random.randint (num_items)
        while (u, j) in train.keys ():
            j = np.random.randint (num_items)
        item_neg.append (j)
        labels.append (1)

        user_input.append (u)
        item_pos.append (j)
        item_neg.append (i)
        labels.append (0)

        # negative instances

        for cnt in range (num_negatives - 1):
            user_input.append (u)
            j = np.random.randint (num_items)
            while (u, j) in train.keys ():
                j = np.random.randint (num_items)
            item_pos.append (j)
            item_neg.append (i)
            labels.append (0)

    return user_input, item_pos, item_neg, labels

def get_train_point(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_items = train.shape [1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u,j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels