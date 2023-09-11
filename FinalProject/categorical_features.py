import sys
import json
from pyspark import SparkContext, StorageLevel
import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import numpy as np
from ast import literal_eval

def read_dataset(sc, filepath):
    return sc.parallelize(sc.textFile(filepath).map(lambda x: x.split(',')).collect()[1:]).persist(StorageLevel.MEMORY_AND_DISK)

def one_hot_encode_categories(sc, ohe, df_original, train=True):
    orig_list = df_original.values.tolist()
    orig_rdd = sc.parallelize(orig_list)

    if train:
        one_hot_categs_rdd = orig_rdd.map(lambda x: (x[16], \
            list(np.sum(ohe.transform(literal_eval(x[14])), axis=0))) if x is not None \
                else list(np.sum(ohe.transform(literal_eval(['None'])), axis=0)))

        indices = [4, 5, 6, 7, 8, 9, 10, 11]
        new_rdd = orig_rdd.map(lambda x: (x[16], [val for i, val in enumerate(x) if i in indices])).join(one_hot_categs_rdd) \
            .map(lambda x: 0)
        print(new_rdd.take(2))
    else:
        one_hot_categs_rdd = orig_rdd.map(lambda x: (x[15], \
            list(np.sum(ohe.transform(literal_eval(x[13])), axis=0))) if x is not None \
                else list(np.sum(ohe.transform(literal_eval(['None'])), axis=0)))

        indices = [4, 5, 6, 7, 8, 9, 10, 16]
        new_rdd = orig_rdd.map(lambda x: (x[15], [val for i, val in enumerate(x) if i != 16]))
        print(new_rdd.take(5))

if __name__ == "__main__":
    sc = SparkContext('local[*]', 'competition')
    sc.setLogLevel('ERROR')

    train_filename = 'new_train.csv'
    test_filename = 'new_test.csv'

    business_json_filename = 'data/business.json'
    business_list = []

    with open('data/business.json') as business_file:
        for line in business_file:
            business_list.append(json.loads(line))

    business_df = pd.DataFrame(business_list)

    train_df = pd.read_csv(train_filename)
    test_df = pd.read_csv(test_filename)

    test_df_with_labels = pd.read_csv('y_test.csv')
    test_df_with_labels['user_business_pair'] = tuple(zip(test_df_with_labels['user_id'], test_df_with_labels[' business_id']))

    final_test_df = pd.merge(test_df, test_df_with_labels[['user_business_pair', ' rating']], on="user_business_pair")

    all_city_states = business_df['city']+'-'+business_df['state']
    unique_city_states = list(set(all_city_states.values))

    categories_split = business_df['categories'].str.split(', ')
    all_categories_list = []
    count = 0
    for categories in categories_split.values:
        if categories is not None:
            all_categories_list.extend(categories)
        else:
            count += 1
            all_categories_list.append('None')

    all_categories = list(set(all_categories_list))

    le_cs = LabelEncoder()
    le_cs.fit(unique_city_states)
    print(le_cs.classes_)

    ohe_cat = LabelBinarizer()
    ohe_cat.fit(all_categories)
    print(ohe_cat.classes_)

    train_df['city_state_le'] = le_cs.transform(train_df['city_state'])
    final_test_df['city_state_le'] = le_cs.transform(final_test_df['city_state'])

    one_hot_encode_categories(sc, ohe_cat, train_df)