from os import listdir
from os.path import isfile, join
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn import tree
import math
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize

import warnings
warnings.filterwarnings('ignore')


class RandomBalanceBoost:

    data_dict = {}
    minority_dict = {}
    majority_dict = {}
    synthetic = []
    h_for_classifiers = []
    betha_for_classifiers = []
    classifiers = []

    def build_minority_majority_dict(self):
        minority_1 = []
        majority_1 = []
        minority_2 = []
        majority_2 = []
        minority_3 = []
        majority_3 = []
        minority_4 = []
        majority_4 = []
        minority_5 = []
        majority_5 = []

        for i, label in np.ndenumerate(self.data_dict['arr_5_1_train_label']):
            if label == True:
                minority_1.append(self.data_dict['arr_5_1_train'][i])
            else:
                majority_1.append(self.data_dict['arr_5_1_train'][i])

        for i, label in np.ndenumerate(self.data_dict['arr_5_2_train_label']):
            if label == True:
                minority_2.append(self.data_dict['arr_5_2_train'][i])
            else:
                majority_2.append(self.data_dict['arr_5_2_train'][i])

        for i, label in np.ndenumerate(self.data_dict['arr_5_3_train_label']):
            if label == True:
                minority_3.append(self.data_dict['arr_5_3_train'][i])
            else:
                majority_3.append(self.data_dict['arr_5_3_train'][i])

        for i, label in np.ndenumerate(self.data_dict['arr_5_4_train_label']):
            if label == True:
                minority_4.append(self.data_dict['arr_5_4_train'][i])
            else:
                majority_4.append(self.data_dict['arr_5_4_train'][i])

        for i, label in np.ndenumerate(self.data_dict['arr_5_5_train_label']):
            if label == True:
                minority_5.append(self.data_dict['arr_5_5_train'][i])
            else:
                majority_5.append(self.data_dict['arr_5_5_train'][i])

        self.majority_dict = {
            'majority_1': majority_1,
            'majority_2': majority_2,
            'majority_3': majority_3,
            'majority_4': majority_4,
            'majority_5': majority_5,
        }

        self.minority_dict = {
            'minority_1': minority_1,
            'minority_2': minority_2,
            'minority_3': minority_3,
            'minority_4': minority_4,
            'minority_5': minority_5,
        }

    def random_balance(self, majority_name, minority_name):

        majority = self.majority_dict[majority_name]
        minority = self.minority_dict[minority_name]

        total_size = len(majority) + len(minority)
        new_majority_size = random.randint(2, total_size-2)
        new_minority_size = total_size - new_majority_size

        if new_majority_size < len(majority):
            majority_items = random.choices(majority, k=int(new_majority_size))
            num_of_new_minority_size = new_minority_size - len(minority)
            n = (num_of_new_minority_size / len(minority)) * 100
            self.synthetic = self.smote(self, n, minority, 5)
            minority_items = []
            for item in range(num_of_new_minority_size):
                minority_items.append(self.synthetic[item])

            minority_items = minority_items + minority
            random_balance_data = minority_items + majority_items

            t1 = np.ones((1, len(minority_items)), dtype=bool)
            t2 = np.zeros((1, len(majority_items)), dtype=bool)
            random_balance_label = np.concatenate((t1[0], t2[0]), axis=0)

            return random_balance_data, random_balance_label

        else:
            minority_items = random.choices(minority, k=int(new_minority_size))
            num_of_new_majority_size = new_majority_size - len(majority)
            n = (num_of_new_majority_size/ len(majority))*100
            self.synthetic = self.smote(self, n, majority, 5)
            majority_items = []
            for item in range(num_of_new_majority_size):
                majority_items.append(self.synthetic[item])

            majority_items = majority_items + majority
            random_balance_data = minority_items + majority_items

            t1 = np.ones((1, len(minority_items)), dtype=bool)
            t2 = np.zeros((1, len(majority_items)), dtype=bool)
            random_balance_label = np.concatenate((t1[0], t2[0]), axis=0)

            return random_balance_data, random_balance_label

    def smote(self, n, t, k):
        if n < 100:
            num_of_new_samples = (n//100)*len(t)
            n = 100
            centroids = random.choices(t, k=int(num_of_new_samples))
            sample = centroids
        n = int(n/100)
        numattrs = t[0].shape
        sample = t
        new_index = 0
        synthetic = []
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(sample)
        distances, nnarray = nbrs.kneighbors(sample)
        synthetic = self.populate(self, n+1, nnarray, sample, k, numattrs)  # ' n+1 is for smote problem
        return synthetic

    def populate(self, n, nnarray, sample, k, numattrs):
        synthetic = []
        new_index = 0
        while n != 0:
            for i, data in enumerate(nnarray):
                index = random.randint(1, k)
                chosen_data = sample[data[index]]
                dif = sample[i] - chosen_data
                gap = np.random.rand(1, numattrs[0])
                synthetic.append(sample[i] + gap[0] * dif)
            new_index += 1
            n -= 1

        return synthetic

    def random_balance_boost_c45(self, t, majority_name, minority_name):
        data, labels = self.random_balance(self, majority_name, minority_name)
        data, labels = shuffle(data, labels)
        #data = normalize(np.asarray(data), norm='l2', axis=0)
        weight = np.full((len(data)), 1/len(data))

        self.h_for_classifiers = []
        self.betha_for_classifiers = []
        self.classifiers = []

        for i in range(t):
            previous_data = data
            previous_labels = labels
            data, labels = self.random_balance(self, majority_name, minority_name)
            index = 0
            pre_weight = weight
            for sample, label in zip(data, labels):
                update_weight = True
                inner_index = 0
                for old_sample, old_label in zip(previous_data, previous_labels):
                    if np.array_equal(sample, old_sample):
                        if label == old_label:
                            update_weight = False
                            weight[index] = pre_weight[inner_index]
                            inner_index += 1
                            break
                    inner_index += 1
                if update_weight:
                    weight[index] = 1 / len(data)

                index += 1
            weak_classifier = tree.DecisionTreeClassifier(criterion='entropy')
            weak_classifier = weak_classifier.fit(data, labels, sample_weight=weight)
            predicted_labels = weak_classifier.predict_proba(data)
            error = np.full((len(labels)), 0, dtype=float)
            h = np.full((len(labels)), 0, dtype=float)
            for j, label_t in enumerate(labels):
                if label_t:
                    error[j] = (weight[j] *  (1 - predicted_labels[j][1] + predicted_labels[j][0]))
                else:
                    error[j] = (weight[j] *  ( 1 - predicted_labels[j][0] + predicted_labels[j][1]))


                if label_t:
                    h[j] = 1 - predicted_labels[j][0] + predicted_labels[j][1]
                else:
                    h[j] = 1 - predicted_labels[j][1] + predicted_labels[j][0]

            error_value = np.sum(error)
            betha = error_value / (1 - error_value)
            if betha == 0:
                betha = 0.001
            for k, w in enumerate(weight):
                w = w * (betha ** (0.5*(h[k])))
                weight[k] = w

            z = np.sum(weight)
            if z!=0:
                weight = weight / z

            labels_for_f1 = weak_classifier.predict(data)
            f1 = f1_score(labels, labels_for_f1)
            if f1 > 0.5:
                self.h_for_classifiers.append(h)
                self.betha_for_classifiers.append(betha)
                self.classifiers.append(weak_classifier)
            else:
                i -= 1

    def random_balance_boost_stump(self, t, majority_name, minority_name):
        data, labels = self.random_balance(self, majority_name, minority_name)
        weight = np.full((len(data)), 1/len(data))

        self.h_for_classifiers = []
        self.betha_for_classifiers = []
        self.classifiers = []

        for i in range(t):
            previous_data = data
            previous_labels = labels
            data, labels = self.random_balance(self, majority_name, minority_name)
            index = 0
            pre_weight = weight
            for sample, label in zip(data, labels):
                update_weight = True
                inner_index = 0
                for old_sample, old_label in zip(previous_data, previous_labels):
                    if np.array_equal(sample, old_sample):
                        if label == old_label:
                            update_weight = False
                            weight[index] = pre_weight[inner_index]
                            inner_index += 1
                            break
                    inner_index += 1
                if update_weight:
                    weight[index] = 1 / len(data)

                index += 1
            weak_classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
            weak_classifier = weak_classifier.fit(data, labels, sample_weight=weight)
            predicted_labels = weak_classifier.predict_proba(data)
            error = np.full((len(labels)), 0, dtype=float)
            h = np.full((len(labels)), 0, dtype=float)
            for j, label_t in enumerate(labels):
                if label_t:
                    error[j] = (weight[j] *  (1 - predicted_labels[j][1] + predicted_labels[j][0]))
                else:
                    error[j] = (weight[j] *  ( 1 - predicted_labels[j][0] + predicted_labels[j][1]))


                if label_t:
                    h[j] = 1 - predicted_labels[j][0] + predicted_labels[j][1]
                else:
                    h[j] = 1 - predicted_labels[j][1] + predicted_labels[j][0]

            error_value = np.sum(error)
            betha = error_value / (1 - error_value)
            if betha == 0:
                betha = 0.001
            for k, w in enumerate(weight):
                w = w * (betha ** (0.5*(h[k])))
                weight[k] = w

            z = np.sum(weight)
            if z!=0:
                weight = weight / z

            labels_for_f1 = weak_classifier.predict(data)
            f1 = f1_score(labels, labels_for_f1)
            if f1 > 0.5:
                self.h_for_classifiers.append(h)
                self.betha_for_classifiers.append(betha)
                self.classifiers.append(weak_classifier)
            else:
                i -= 1

    def random_balance_boost_logistic_regresion(self, t, majority_name, minority_name):
        data, labels = self.random_balance(self, majority_name, minority_name)
        data, labels = shuffle(data, labels)
        data = normalize(data, norm='l2', axis=0)
        weight = np.full((len(data)), 1/len(data))

        self.h_for_classifiers = []
        self.betha_for_classifiers = []
        self.classifiers = []

        for i in range(t):
            previous_data = data
            previous_labels = labels
            data, labels = self.random_balance(self, majority_name, minority_name)
            index = 0
            pre_weight = weight
            for sample, label in zip(data, labels):
                update_weight = True
                inner_index = 0
                for old_sample, old_label in zip(previous_data, previous_labels):
                    if np.array_equal(sample, old_sample):
                        if label == old_label:
                            update_weight = False
                            weight[index] = pre_weight[inner_index]
                            inner_index += 1
                            break
                    inner_index += 1
                if update_weight:
                    weight[index] = 1 / len(data)

                index += 1
            weak_classifier = LogisticRegression()
            weak_classifier = weak_classifier.fit(data, labels, sample_weight=weight)
            predicted_labels = weak_classifier.predict_proba(data)
            error = np.full((len(labels)), 0, dtype=float)
            h = np.full((len(labels)), 0, dtype=float)
            for j, label_t in enumerate(labels):
                if label_t:
                    error[j] = (weight[j] *  (1 - predicted_labels[j][1] + predicted_labels[j][0]))
                else:
                    error[j] = (weight[j] *  ( 1 - predicted_labels[j][0] + predicted_labels[j][1]))


                if label_t:
                    h[j] = 1 - predicted_labels[j][0] + predicted_labels[j][1]
                else:
                    h[j] = 1 - predicted_labels[j][1] + predicted_labels[j][0]

            error_value = np.sum(error)
            betha = error_value / (1 - error_value)
            if betha == 0:
                betha = 0.001
            for k, w in enumerate(weight):
                w = w * (betha ** (0.5*(h[k])))
                weight[k] = w

            z = np.sum(weight)
            if z!=0:
                weight = weight / z

            labels_for_f1 = weak_classifier.predict(data)
            f1 = f1_score(labels, labels_for_f1)
            if f1 > 0.5:
                self.h_for_classifiers.append(h)
                self.betha_for_classifiers.append(betha)
                self.classifiers.append(weak_classifier)
            else:
                i -= 1


    def test_random_balance_boost(self, data_number):


        data_name = 'arr_5_' + str(data_number+1) + '_test'
        data_label_name = 'arr_5_' + str(data_number+1) + '_test_label'

        test_data = self.data_dict[data_name]
        test_label = self.data_dict[data_label_name]
        max_values = np.full((len(test_data), 2), 0, dtype=float)
        for i, classifier in enumerate(self.classifiers):
            predicted_data = classifier.predict_proba(test_data)
            predicted_data = math.log(1 / self.betha_for_classifiers[i]) * predicted_data
            max_values = np.maximum(max_values, predicted_data)

        final_predicted_lables = []
        trues = 0
        falses = 0
        for j, data in enumerate(max_values):
            if data[0] > data[1]:
                final_predicted_lables.append(False)

            else:
                final_predicted_lables.append(True)

        f1 = f1_score(test_label, final_predicted_lables)
        print('f1-score in '+ data_name + ' is : ' + str(f1))



    def read_data_set(self, directory_wich_contains_the_files):
        files = [f for f in listdir(directory_wich_contains_the_files) if isfile(join(directory_wich_contains_the_files, f))]
        data = []
        for file in files:
            temp = open(join(directory_wich_contains_the_files, file), 'r')
            data = temp.readlines()
            arr_name = 'arr_' + file.split('.')[0]

            my_data = []
            my_label = []
            for sample in data:
                sample = sample.split(',')
                temp = [float(i) for i in sample[0:11]]
                my_data.append(temp)
                if sample[-1].__contains__('neg'):
                    label = False
                else:
                    label = True
                my_label.append(label)

            if arr_name.__contains__('tra'):
                if arr_name.__contains__('5-1'):
                    arr_5_1_train = np.asarray(my_data)
                    arr_5_1_train_label = np.asarray(my_label)
                elif arr_name.__contains__('5-2'):
                    arr_5_2_train = np.asarray(my_data)
                    arr_5_2_train_label = np.asarray(my_label)
                elif arr_name.__contains__('5-3'):
                    arr_5_3_train = np.asarray(my_data)
                    arr_5_3_train_label = np.asarray(my_label)
                elif arr_name.__contains__('5-4'):
                    arr_5_4_train = np.asarray(my_data)
                    arr_5_4_train_label = np.asarray(my_label)
                elif arr_name.__contains__('5-5'):
                    arr_5_5_train = np.asarray(my_data)
                    arr_5_5_train_label = np.asarray(my_label)

            if arr_name.__contains__('tst'):
                if arr_name.__contains__('5-1'):
                    arr_5_1_test = np.asarray(my_data)
                    arr_5_1_test_label = np.asarray(my_label)
                elif arr_name.__contains__('5-2'):
                    arr_5_2_test = np.asarray(my_data)
                    arr_5_2_test_label = np.asarray(my_label)
                elif arr_name.__contains__('5-3'):
                    arr_5_3_test = np.asarray(my_data)
                    arr_5_3_test_label = np.asarray(my_label)
                elif arr_name.__contains__('5-4'):
                    arr_5_4_test = np.asarray(my_data)
                    arr_5_4_test_label = np.asarray(my_label)
                elif arr_name.__contains__('5-5'):
                    arr_5_5_test = np.asarray(my_data)
                    arr_5_5_test_label = np.asarray(my_label)

        data_dict = {
            'arr_5_1_train':arr_5_1_train,
            'arr_5_1_test':arr_5_1_test,
            'arr_5_1_train_label': arr_5_1_train_label,
            'arr_5_1_test_label': arr_5_1_test_label,
            'arr_5_2_train': arr_5_2_train,
            'arr_5_2_test': arr_5_2_test,
            'arr_5_2_train_label': arr_5_2_train_label,
            'arr_5_2_test_label': arr_5_2_test_label,
            'arr_5_3_train': arr_5_3_train,
            'arr_5_3_test': arr_5_3_test,
            'arr_5_3_train_label': arr_5_3_train_label,
            'arr_5_3_test_label': arr_5_3_test_label,
            'arr_5_4_train': arr_5_4_train,
            'arr_5_4_test': arr_5_4_test,
            'arr_5_4_train_label': arr_5_4_train_label,
            'arr_5_4_test_label': arr_5_4_test_label,
            'arr_5_5_train': arr_5_5_train,
            'arr_5_5_test': arr_5_5_test,
            'arr_5_5_train_label': arr_5_5_train_label,
            'arr_5_5_test_label': arr_5_5_test_label,
        }

        self.data_dict = data_dict



def adaboost_dt_c45(data_train, label_train, data_test, label_test, num_of_base_classifiers, data_name):
    weak_classifier = tree.DecisionTreeClassifier(criterion='entropy')
    ensemble = AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=num_of_base_classifiers)
    ensemble.fit(data_train, label_train)
    predicted_labels = ensemble.predict(data_test)
    f1 = f1_score(label_test, predicted_labels)
    print('f1-score in ' + data_name + ' is : ' + str(f1))


def adaboost_dt_stump(data_train, label_train, data_test, label_test, num_of_base_classifiers, data_name):
    weak_classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
    ensemble = AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=num_of_base_classifiers)
    ensemble.fit(data_train, label_train)
    predicted_labels = ensemble.predict(data_test)
    f1 = f1_score(label_test, predicted_labels)
    print('f1-score in ' + data_name + ' is : ' + str(f1))


def adaboost_logistic_regresion(data_train, label_train, data_test, label_test, num_of_base_classifiers, data_name):
    weak_classifier = LogisticRegression()
    ensemble = AdaBoostClassifier(base_estimator=weak_classifier, n_estimators=num_of_base_classifiers)
    ensemble.fit(data_train, label_train)
    predicted_labels = ensemble.predict(data_test)
    f1 = f1_score(label_test, predicted_labels)
    print('f1-score in ' + data_name + ' is : ' + str(f1))


def bagging_dt_c45(data_train, label_train, data_test, label_test, num_of_base_classifiers, data_name):
    weak_classifier = tree.DecisionTreeClassifier(criterion='entropy')
    ensemble = BaggingClassifier(base_estimator=weak_classifier, n_estimators=num_of_base_classifiers)
    ensemble.fit(data_train, label_train)
    predicted_labels = ensemble.predict(data_test)
    f1 = f1_score(label_test, predicted_labels)
    print('f1-score in ' + data_name + ' is : ' + str(f1))


def bagging_dt_stump(data_train, label_train, data_test, label_test, num_of_base_classifiers, data_name):
    weak_classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
    ensemble = BaggingClassifier(base_estimator=weak_classifier, n_estimators=num_of_base_classifiers)
    ensemble.fit(data_train, label_train)
    predicted_labels = ensemble.predict(data_test)
    f1 = f1_score(label_test, predicted_labels)
    print('f1-score in ' + data_name + ' is : ' + str(f1))


def bagging_logistic_regresion(data_train, label_train, data_test, label_test, num_of_base_classifiers, data_name):
    weak_classifier = LogisticRegression()
    ensemble = BaggingClassifier(base_estimator=weak_classifier, n_estimators=num_of_base_classifiers)
    ensemble.fit(data_train, label_train)
    predicted_labels = ensemble.predict(data_test)
    f1 = f1_score(label_test, predicted_labels)
    print('f1-score in ' + data_name + ' is : ' + str(f1))




rb = RandomBalanceBoost
rb.read_data_set(rb,'winequality-red-8_vs_6-5-fold')
x = rb.data_dict
rb.build_minority_majority_dict(rb)

for i in range(5):
    minority_name = 'minority_' + str(i+1)
    majority_name = 'majority_' + str(i+1)
    rb.random_balance_boost_c45(rb, 100, minority_name=minority_name, majority_name=majority_name)
    rb.test_random_balance_boost(rb, i)

    train_data_name = 'arr_5_' + str(i+1) + '_train'
    test_data_name = 'arr_5_' + str(i + 1) + '_test'
    train_label_name = 'arr_5_' + str(i + 1) + '_train_label'
    test_label_name = 'arr_5_' + str(i + 1) + '_test_label'

    train_data = x[train_data_name]
    train_label = x[train_label_name]
    test_data = x[test_data_name]
    test_label = x[test_label_name]

    adaboost_dt_c45(train_data, train_label, test_data, test_label, 100, test_data_name)
    bagging_dt_c45(train_data, train_label, test_data, test_label, 100, test_data_name)


print('salam')

