import numpy
import warnings
from ratings import Ratings
from group_collection import GroupCollection


class RecommendationSystem:
    def __init__(self):
        self.rating_threshold = 4
        self.recommendations_size = 50

        self.precision_metrics = []
        self.recall_metrics = []

        self.ratings = Ratings()
        self.ratings.factorize()

        self.group_collection = GroupCollection(self.ratings.train_ratings, self.ratings.test_ratings)
        self.group_collection.create(self.ratings.users_number, disjoint=False)

    def train(self):  # TODO: Refactor this

        all_items = numpy.arange(len(self.ratings.train_ratings.T))

        for group in self.group_collection.groups:
            group_items = sorted(list(set(all_items) - set(group.items)))
            group_ratings = self.ratings.train_ratings[group.user_ids, :]
            aggregated_ratings = self.aggregate_by_average(group_ratings)

            s_g = []
            for item in group_items:
                s_g.append(aggregated_ratings[item] - self.ratings.ratings_mean - self.ratings.item_biases[item])

            # creating matrix A : contains rows of [item_factors of items in watched_list + '1' vector]
            A = numpy.zeros((0, self.ratings.factors_number))  # 3 is the number of features here = K

            for item in group_items:
                A = numpy.vstack([A, self.ratings.item_factors[item]])

            v = numpy.ones((len(group_items), 1))
            A = numpy.c_[A, v]
            W = self.get_weight_matrix(group, group_items)

            factor_n_bias = numpy.dot(self.make(A, W), numpy.dot(numpy.dot(A.T, W), s_g))

            group.factors = factor_n_bias[:-1]
            group.bias = factor_n_bias[-1]

            self.generate_recommendations(group)

        return

    def evaluate(self):

        for group in self.group_collection.groups:
            group.generate_actual_recommendations(self.ratings.test_ratings, self.rating_threshold)
            (precision, recall, tp, fp) = group.evaluate()

            self.precision_metrics.append(precision)
            self.recall_metrics.append(recall)

    def get_weight_matrix(self, group, items):  # TODO: Check if correct
        weights = []  # list containing ratings weights

        for item in items:
            rated_items = numpy.nonzero(self.ratings.train_ratings[:, item])  # list of users who have rated this movie
            users_with_ratings = numpy.intersect1d(rated_items, group.user_ids)  # group members rated this movie
            rated_items_std = numpy.std(rated_items)  # std item-rating
            weights += [len(users_with_ratings) / float(len(group.user_ids)) * 1 / (1 + rated_items_std)]

        weight_matrix = numpy.diag(weights)  # diagonal weight matrix

        return weight_matrix

    def make(self, A, W):
        var = numpy.dot(numpy.dot(A.T, W), A)
        identity = numpy.identity(self.ratings.factors_number + 1)

        return numpy.linalg.inv(var + self.ratings.lambda_parameter * self.ratings.lambda_parameter * identity)

    def generate_recommendations(self, group):
        # predict ratings for all candidate items
        group_ratings = {}

        for idx, item in enumerate(group.items):
            cur_rating = self.predict_group_rating(group, item)
            if cur_rating > self.rating_threshold:
                group_ratings[item] = cur_rating

        # sort and filter to keep top 'num_recos_wbf' recommendations
        group_ratings = sorted(group_ratings.items(), key=lambda x: x[1], reverse=True)[:self.recommendations_size]

        group.recommendation_list = numpy.array([rating[0] for rating in group_ratings])

    def predict_group_rating(self, group, item):
        temp = numpy.dot(group.factors.T, self.ratings.item_factors[item])
        return self.ratings.ratings_mean + group.bias + self.ratings.item_biases[item] + temp

    def get_evaluation_metrics(self):
        precision = numpy.nanmean(numpy.array(self.precision_metrics))
        recall = numpy.nanmean(numpy.array(self.recall_metrics))

        return precision, recall

    @staticmethod
    def aggregate_by_average(arr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            arr[arr == 0] = numpy.nan

            return numpy.nanmean(arr, axis=0)
