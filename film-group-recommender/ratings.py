import pandas
from numpy import zeros

train_file = '../../data-sets/movie-lens-100k/u1.test'
test_file = '../../data-sets/movie-lens-100k/u1.test'


class Ratings:
    def __init__(self):
        self.train = None
        self.test = None

        self.load_data()

    def load_data(self):
        separator = '\t'
        headers = ['user_id', 'item_id', 'rating', 'timestamp']

        train_csv = pandas.read_csv(train_file, sep=separator, names=headers)
        test_csv = pandas.read_csv(test_file, sep=separator, names=headers)

        self.train = self.convert_to_matrix(train_csv)
        self.test = self.convert_to_matrix(test_csv)

    @staticmethod
    def convert_to_matrix(data_set):
        total_users_count = max(data_set.user_id.unique())
        total_items_count = max(data_set.item_id.unique())

        matrix = zeros(total_users_count, total_items_count)

        for row in data_set.itertuples(index=False):
            matrix[row.user_id - 1, row.item_id - 1] = row.rating

        return matrix
