
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression as lr_sklearn


def logistic_regression(feats_train, labels_train, feats_valid, labels_valid, solver, info):
    # print (labels_train[0:10000])
    # print (np.sum(labels_train[0:10000]))
    
    print ("training logistic regression model ...")
    # C_list = [5e-2, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7]
    # 5e-7 and liblinear solver is the best for unigram feature
    C_list = [2, 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    for C in C_list:
        print ("solver:",solver, "C:", C)
        lr_model = lr_sklearn(C=C, max_iter=1e8, solver=solver, class_weight="balanced")
        lr_model.fit(feats_train, labels_train)
        # print ("prediction of validation set:")
        predict = lr_model.predict(feats_valid)
        predict1 = lr_model.predict(feats_train)
        # print (predict)
        # predict_prob = lr_model.predict_proba(feats_valid)
        # array-like, shape = [n_samples, n_classes]
        # print (predict_prob)
        print ("confusion matrix:")
        print (confusion_matrix(labels_valid, predict))
        print ("classification report for validation:")
        print (classification_report(labels_valid, predict))
        acc = accuracy_score(labels_valid, predict)
        print ("logistic regression score for",solver,"solver and",info,"feature:", acc)
        
        print ("classification report for training:")
        print (classification_report(labels_train, predict1))
        acc = accuracy_score(labels_train, predict1)
        print ("logistic regression score for",solver,"solver and",info,"feature:", acc)


pkl_in = open("../data_new/id_20TopicProb.pickle", "rb")
id_prob_train = pickle.load(pkl_in)
pkl_in = open("../data_new/id_20TopicProb_val.pickle", "rb")
id_prob_val = pickle.load(pkl_in)
pkl_in = open("/home/nayeon/fakenews/data_new/new_ids.pickle", "rb")
ids = pickle.load(pkl_in)
ids_train = ids["train"]
ids_val = ids["val"]
pkl_in = open("/home/nayeon/fakenews/data_new/train_labels.pickle", "rb")
train_labels = pickle.load(pkl_in)
pkl_in = open("/home/nayeon/fakenews/data_new/val_labels.pickle", "rb")
val_labels = pickle.load(pkl_in)

num_topic = 20

train_dict={'topic12': 0.497595758671708, 'topic14': 0.549786729188441, 'topic5': 0.5189536544203557, 
            'topic4': 0.6308458838183671, 'topic18': 0.3318474653478233, 'topic10': 0.5799339588450851, 
            'topic17': 0.5397430217102348, 'topic2': 0.4214750830564784, 'topic19': 0.5764001224150745, 
            'topic16': 0.4830931405000764, 'topic8': 0.4835419711507911, 'topic9': 0.3611635022423404, 
            'topic6': 0.5038865546218487, 'topic1': 0.5839241887101173, 'topic13': 0.6224742268041237, 
            'topic15': 0.7333370056185965, 'topic7': 0.4894125476435356, 'topic3': 0.43104445944642683, 
            'topic0': 0.5767219822686974, 'topic11': 0.309614340032591}
# train_dict ={'topic21': 0.6126881162428646, 'topic28': 0.5374195733152726, 'topic14': 0.46882739313018196, 
#              'topic24': 0.5497616732292879, 'topic11': 0.40701247904637733, 'topic9': 0.612192909662852, 
#              'topic19': 0.5116888193901485, 'topic18': 0.4047873319830602, 'topic29': 0.5020934304346722, 
#              'topic5': 0.4489138438880707, 'topic2': 0.4860188289138246, 'topic13': 0.39885486158533384, 
#              'topic12': 0.467175572519084, 'topic27': 0.5801799159088686, 'topic23': 0.5174908757176702, 
#              'topic3': 0.5338271227855834, 'topic22': 0.6145303891000726, 'topic8': 0.5984643032184284, 
#              'topic4': 0.5654829335566173, 'topic1': 0.5410391230728099, 'topic6': 0.4586455010755879, 
#              'topic0': 0.508942216347941, 'topic25': 0.7604540461369462, 'topic15': 0.23722531109346043, 
#              'topic10': 0.43528201611520656, 'topic16': 0.6344753916758981, 'topic20': 0.3672214182344428, 
#              'topic17': 0.4693062114057392, 'topic7': 0.522230008564977, 'topic26': 0.323719165085389}

fake_list = []
for i in range(0, num_topic):
    key = "topic"+str(i)
    prob = train_dict[key]
    fake_list.append(prob)

fake_list = np.array(fake_list)
# print (fake_list)

# train_gold = []
train_feature = {}
for id_ in tqdm(ids_train):
#     train_gold.append(train_labels[id_]["label"])
    distri = id_prob_train[id_]
    topic_feat = fake_list * distri
    # print (topic_feat)
    train_feature[id_] = topic_feat

# train_feature = np.array(train_feature)
# train_gold = np.array(train_gold)

# val_gold = []
val_feature = {}
for id_ in tqdm(ids_val):
#     val_gold.append(val_labels[id_]["label"])
    distri = id_prob_val[id_]
    # print (distri)
    topic_feat = fake_list * distri
    # print (topic_feat)
    val_feature[id_] = topic_feat

# val_feature = np.array(val_feature)
# val_gold = np.array(val_gold)

# solver_list = ["liblinear", "lbfgs"]
# for solver in solver_list:
#     logistic_regression(train_feature, train_gold, val_feature, val_gold, solver, "topic info")

# print (train_feature)
# print (val_feature)

# print ("dumping id_20TopicFeat_train ...")
# pkl_out = open("../data_new/id_20TopicFeat_train.pickle", "wb")
# pickle.dump(train_feature, pkl_out, protocol=4)

# print ("dumping id_20TopicFeat_val ...")
# pkl_out = open("../data_new/id_20TopicFeat_val.pickle", "wb")
# pickle.dump(val_feature, pkl_out, protocol=4)


