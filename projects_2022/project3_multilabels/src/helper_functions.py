import pandas as pd
import Levenshtein
import os
import numpy as np


def get_consistent_labels(labels, word_label, job_value):
    temp = 1000
    for i in labels.job_type.values.tolist():
        dist = Levenshtein.distance(i, word_label)
        if dist < temp:
            temp = dist
            final = i
    return final, temp



def consistent_labels_main(list_data, labels):
    consistent_job_values =[]
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for j in list_data:
        job_value = j.replace('[', '').replace(']', '')
        job_value = job_value.split(',')
        final_job_value = []
        for i in job_value:
            consistent_label, temp = get_consistent_labels(labels, i, job_value)
            if temp > 10 and len(job_value) != 1:
                count1 += 1
                pass
            
            elif temp <= 10:
                count2 += 1
                final_job_value.append(consistent_label)
                
            elif temp > 12 and len(job_value) == 1 :
                count3 += 1
                pass
            else:
                count4 += 1
                final_job_value.append(consistent_label)
        consistent_job_values.append(final_job_value)
    #print(count1, count2, count3, count4)
    return consistent_job_values

def scarce_train_data(df):
    appe = []
    for i in df.clean_job_labels.values.tolist():
        for j in i:
            appe.append(j)

    check = {'unique_label':appe}
    check_df = pd.DataFrame.from_dict(check)
    scarce_data = check_df.unique_label.value_counts().index[check_df.unique_label.value_counts().values < 150]
    return scarce_data


def remove_scarce_data(df, scarce_data):
    for i in scarce_data:
        df['reduced_job_labels'] = df['clean_job_labels'].apply(lambda row: [val for val in row if val != i])
    df = df[df["reduced_job_labels"].str.len() != 0]
    for i in [4,5]:
        df = df[df["reduced_job_labels"].str.len() != i]
    return df

def get_processed_data_EN():
    test = pd.read_csv('/content/drive/MyDrive/MLC/en_test.csv', encoding= 'utf8')
    test = test.loc[:, ~test.columns.str.contains('^Unnamed')]

    train = pd.read_csv('/content/drive/MyDrive/MLC/en_train.csv', encoding= 'utf8')
    train = train.loc[:, ~train.columns.str.contains('^Unnamed')]

    labels = pd.read_csv('/content/drive/MyDrive/MLC/en_labels.csv', encoding= 'utf8')

    test_labels = test.job.values.tolist() 
    train_labels = train.job.values.tolist()
    test_consistent_labels = consistent_labels_main(test_labels, labels)
    test["clean_job_labels"] = test_consistent_labels
    train_consistent_labels = consistent_labels_main(train_labels, labels)
    train["clean_job_labels"] = train_consistent_labels
    scarce_data = scarce_train_data(train)
    train_final = remove_scarce_data(train, scarce_data)
    test_final = remove_scarce_data(test, scarce_data)
    return train_final, test_final, labels


def get_labelCount_distributions(train_final, test_final):
    lens = []
    lens_tst = []
    for ls in train_final['reduced_job_labels']:
        lens.append(len(ls))
    for ls in test_final['reduced_job_labels']:
        lens_tst.append(len(ls))

    xlabs = np.arange(1,8)
    tr_cs = np.zeros(7)
    tst_cs = np.zeros(7)
    for i in range(len(lens)):
        tr_cs[lens[i]-1] += 1
    for i in range(len(lens_tst)):
        tst_cs[lens_tst[i]-1] += 1
    
    return xlabs, tr_cs / sum(tr_cs), tst_cs / sum(tst_cs)

def create_encodings(consistent_labels, labels):
    job_labels = labels.job_type.values.tolist()
    encodings = []
    for label in consistent_labels:
        encoding = [0]*len(job_labels)
        for job in label:
            index = job_labels.index(job)
            encoding[index] = 1
        encodings.append(encoding)
    return encodings 

def get_final_TrainTest(train_final,test_final, labels):
  test_encodings = create_encodings(test_final.reduced_job_labels.tolist(), labels)
  train_encodings = create_encodings(train_final.reduced_job_labels.tolist(), labels)

  test_final["encodings"] = test_encodings
  train_final["encodings"] = train_encodings

  final_test = test_final[['job_description', 'encodings']].copy()
  final_train  = train_final[['job_description', 'encodings']].copy()
  return final_train, final_train
