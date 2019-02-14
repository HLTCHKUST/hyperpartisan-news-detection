
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
    C_list = [10, 5, 3, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
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

val_dict = {'topic2': 0.5248146035367941, 'topic5': 0.4794010889292196, 'topic18': 0.42011956297670583, 
            'topic12': 0.5493765359060708, 'topic7': 0.3547763666482606, 'topic16': 0.4197146137444645, 
            'topic14': 0.5188277291570704, 'topic1': 0.47497295702625625, 'topic17': 0.47810351067680057, 
            'topic4': 0.5448794545582344, 'topic10': 0.5019173721148976, 'topic3': 0.48085211180889054, 
            'topic19': 0.5507082152974504, 'topic8': 0.42895783611774063, 'topic9': 0.4596589462967676, 
            'topic13': 0.39256360078277885, 'topic6': 0.46929927349425826, 'topic0': 0.4961753456899088, 
            'topic15': 0.5188866799204771, 'topic11': 0.46153846153846156}
train_dict={'topic12': 0.497595758671708, 'topic14': 0.549786729188441, 'topic5': 0.5189536544203557, 
            'topic4': 0.6308458838183671, 'topic18': 0.3318474653478233, 'topic10': 0.5799339588450851, 
            'topic17': 0.5397430217102348, 'topic2': 0.4214750830564784, 'topic19': 0.5764001224150745, 
            'topic16': 0.4830931405000764, 'topic8': 0.4835419711507911, 'topic9': 0.3611635022423404, 
            'topic6': 0.5038865546218487, 'topic1': 0.5839241887101173, 'topic13': 0.6224742268041237, 
            'topic15': 0.7333370056185965, 'topic7': 0.4894125476435356, 'topic3': 0.43104445944642683, 
            'topic0': 0.5767219822686974, 'topic11': 0.309614340032591}
# train_dict_top1 = {'topic12': 0.36961610486891383, 'topic2': 0.23783482142857143, 'topic18': 0.09375, 
#                    'topic10': 0.7752053771471247, 'topic4': 0.7301541976013707, 'topic6': 0.3878827790582812, 
#                    'topic1': 0.5985979884181651, 'topic17': 0.3522727272727273, 'topic7': 0.39856616934702577, 
#                    'topic8': 0.31410825199645076, 'topic0': 0.600515813764083, 'topic15': 0.9519017911048501, 
#                    'topic14': 0.6720715683533688, 'topic16': 0.4487758497741859, 'topic9': 0.06512267828479706, 
#                    'topic5': 0.464312736443884, 'topic13': 0.7130625248904818, 'topic3': 0.24795486600846262, 
#                    'topic19': 0.5935334872979214, 'topic11': 0.6666666666666666}
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

# order train_dict in topic
fake_list = []
value = train_dict.items()


train_gold = []
train_feature = []
for id_ in tqdm(ids_train):
    train_gold.append(train_labels[id_]["label"])
    distri = id_prob_train[id_]
    # print (distri)
    indices = list(range(len(distri)))
    indices.sort(key=lambda x: distri[x], reverse=True)
    # top3
    # print (indices[0:3])
    item_feat = []
    topic_top3 = indices[0:3]
    for topic_num in topic_top3:
        key = "topic"+str(topic_num)
        prob = train_dict[key]
        item_feat.append(prob)
    train_feature.append(item_feat)

train_feature = np.array(train_feature)
train_gold = np.array(train_gold)

val_gold = []
val_feature = []
for id_ in tqdm(ids_val):
    val_gold.append(val_labels[id_]["label"])
    distri = id_prob_val[id_]
    # print (distri)
    indices = list(range(len(distri)))
    indices.sort(key=lambda x: distri[x], reverse=True)
    # top3
    # print (indices[0:3])
    item_feat = []
    topic_top3 = indices[0:3]
    for topic_num in topic_top3:
        key = "topic"+str(topic_num)
        prob = train_dict[key]
        item_feat.append(prob)
    val_feature.append(item_feat)

val_feature = np.array(val_feature)
val_gold = np.array(val_gold)

solver_list = ["liblinear", "lbfgs"]
for solver in solver_list:
    logistic_regression(train_feature, train_gold, val_feature, val_gold, solver, "topic info")