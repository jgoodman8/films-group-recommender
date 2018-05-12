from recommendation_system import RecommendationSystem

if __name__ == "__main__":
    rs = RecommendationSystem()

    rs.train()
    rs.evaluate()
    (precision, recall) = rs.get_evaluation_metrics()

    print('Precision: ', precision)
    print('Recall: ', recall)

