import numpy
import warnings
from ratings import Ratings
from group_collection import GroupCollection


class Recommender:
    def __init__(self):
        self.rating_threshold = 4
        self.recommendations_size = 50

        self.ratings = Ratings()
        self.ratings.factorize()

        self.group_collection = GroupCollection(self.ratings.train_ratings, self.ratings.test_ratings)
        self.group_collection.create(self.ratings.users_number, disjoint=False)

        self.train()
        self.evaluate()

    def train(self):

        aggregator = self.average

        for group in self.group_collection.groups:
            all_movies = numpy.arange(len(self.ratings.train_ratings.T))
            watched_items = sorted(list(set(all_movies) - set(group.candidate_items)))

            group_ratings = self.ratings[group.members, :]
            aggregated_rating = aggregator(group_ratings)

            s_g = []
            for j in watched_items:
                s_g.append(aggregated_rating[j] - self.ratings.ratings_mean - self.ratings.item_biases[j])

            # creating matrix A : contains rows of [item_factors of items in watched_list + '1' vector]
            A = numpy.zeros((0, self.ratings.factors_number))  # 3 is the number of features here = K

            for item in watched_items:
                A = numpy.vstack([A, self.ratings.item_factors[item]])

            v = numpy.ones((len(watched_items), 1))
            A = numpy.c_[A, v]
            W = self.get_weight_matrix(group, watched_items)

            factor_n_bias = numpy.dot(self.make(A, W), numpy.dot(numpy.dot(A.T, W), s_g))

            group.factors = factor_n_bias[:-1]
            group.bias = factor_n_bias[-1]

            self.generate_recommendations(group)

        return

    def evaluate(self):
        return

    def get_weight_matrix(self, grp, watched_items):
        weighted_matrix = []

        for item in watched_items:
            rated = numpy.argwhere(self.ratings[:, item] != 0)  # list of users who have rated this movie
            watched = numpy.intersect1d(rated, grp)  # list of group members who have watched this movie
            std_dev = numpy.std(
                filter(lambda a: a != 0, self.ratings[:, item]))  # std deviation for the rating of the item
            weighted_matrix += [
                len(watched) / float(len(grp.members)) * 1 / (1 + std_dev)]  # list containing diagonal elements

        w = numpy.diag(weighted_matrix)  # diagonal weight matrix

        return w

    def make(self, A, W):
        var = numpy.dot(numpy.dot(A.T, W), A)
        identity = numpy.identity(self.ratings.factors_number + 1)

        return numpy.linalg.inv(var + self.ratings.lambda_parameter * identity)

    def generate_recommendations(self, group):
        # predict ratings for all candidate items
        group_candidate_ratings = {}

        for idx, item in enumerate(group.items):
            cur_rating = self.predict_group_rating(group, item, 'wbf')
            if cur_rating > self.rating_threshold:
                group_candidate_ratings[item] = cur_rating

        # sort and filter to keep top 'num_recos_wbf' recommendations
        group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)[
                                  :self.recommendations_size]

        group.recommendation_list = numpy.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])

    def predict_group_rating(self, group):
        group_candidate_ratings = {}

        for idx, item in enumerate(group.items):
            cur_rating = self.predict_group_rating(group, item, 'bf')

            if cur_rating > self.rating_threshold:
                group_candidate_ratings[item] = cur_rating

        # sort and filter to keep top 'num_recos_bf' recommendations
        group_candidate_ratings = sorted(group_candidate_ratings.items(), key=lambda x: x[1], reverse=True)[
                                  :self.cfg.num_recos_bf]
        group.reco_list_bf = np.array([rating_tuple[0] for rating_tuple in group_candidate_ratings])

    @staticmethod
    def average(arr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            arr[arr == 0] = numpy.nan

            return numpy.nanmean(arr, axis=0)
