import numpy
import pandas

train_file = '../../data-sets/movie-lens-100k/u1.base'
test_file = '../../data-sets/movie-lens-100k/u1.test'


class Ratings:
    def __init__(self):
        self.train_ratings = None
        self.test_ratings = None
        self.ratings_mean = 0

        # TODO: Move
        self.learning_rate = 0.05
        self.lambda_parameter = 0.1
        self.maximum_iterations = 5
        self.factors_number = 15

        self.users_number = 0
        self.items_number = 0

        self.load_data()

        self.user_biases = numpy.zeros(self.users_number)
        self.item_biases = numpy.zeros(self.items_number)

        self.user_factors = numpy.random.uniform(-1, 1, (self.users_number, self.factors_number))
        self.item_factors = numpy.random.uniform(-1, 1, (self.items_number, self.factors_number))

    def load_data(self):
        separator = '\t'
        headers = ['user_id', 'item_id', 'rating', 'timestamp']

        train_csv = pandas.read_csv(train_file, sep=separator, names=headers)
        test_csv = pandas.read_csv(test_file, sep=separator, names=headers)

        self.users_number = max(train_csv.user_id.unique())
        self.items_number = max(train_csv.item_id.unique())

        self.train_ratings = self.convert_to_matrix(train_csv)
        self.test_ratings = self.convert_to_matrix(test_csv)

    def convert_to_matrix(self, data_set):
        matrix = numpy.zeros((self.users_number, self.items_number))

        for row in data_set.itertuples(index=False):
            matrix[row.user_id - 1, row.item_id - 1] = row.rating

        return matrix

    def predict_rating_for_user_and_item(self, user, item):
        prediction = self.ratings_mean + self.user_biases[user] + self.item_biases[item]
        prediction += self.user_factors[user, :].dot(self.item_factors[item, :].T)

        return prediction

    def factorize(self):
        self.ratings_mean = numpy.mean(self.train_ratings[numpy.where(self.train_ratings != 0)])

        ratings_row, ratings_column = self.train_ratings.nonzero()
        total_ratings = len(ratings_row)

        try:
            for iteration in range(self.maximum_iterations):

                rating_indexes = numpy.arange(total_ratings)
                numpy.random.shuffle(rating_indexes)

                for index in rating_indexes:
                    user = ratings_row[index]
                    item = ratings_column[index]

                    prediction = self.predict_rating_for_user_and_item(user, item)
                    error = self.train_ratings[user][item] - prediction

                    self.user_factors[user] += self.get_user_factors_update(item, user, error)
                    self.item_factors[item] += self.get_item_factors_update(item, user, error)

                    self.user_biases[user] += self.get_user_biases_update(user, error)
                    self.item_biases[item] += self.get_item_biases_update(item, error)

        except FloatingPointError:
            print("Error ...")

    # PRIVATE METHODS BELOW
    def get_user_factors_update(self, item, user, error):
        return self.learning_rate * (self.get_user_loss(item, error) - self.get_user_regularization(user))

    def get_item_factors_update(self, item, user, error):
        try:
            return self.learning_rate * (self.get_item_loss(user, error) - self.get_item_regularization(item))
        except Exception as w:
            print(w)

    def get_user_biases_update(self, user, error):
        return self.learning_rate * (error - self.lambda_parameter * self.user_biases[user])

    def get_item_biases_update(self, item, error):
        return self.learning_rate * (error - self.lambda_parameter * self.item_biases[item])

    def get_user_regularization(self, user):
        return self.lambda_parameter * self.user_factors[user]

    def get_item_regularization(self, item):
        return self.lambda_parameter * self.item_factors[item]

    def get_user_loss(self, item, error):
        try:
            return error * self.item_factors[item]
        except Warning as w:
            print(w)

    def get_item_loss(self, user, error):
        try:
            return error * self.user_factors[user]
        except Exception as w:
            print(w)
