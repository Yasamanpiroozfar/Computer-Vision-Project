import pickle

import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = np.asarray(false_alarm_rate_range, dtype=float)

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")
        self.train_embeddings = np.asarray(self.train_embeddings)
        self.train_labels = np.asarray(self.train_labels)
        self.test_embeddings = np.asarray(self.test_embeddings)
        self.test_labels = np.asarray(self.test_labels)

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self):
        if len(self.train_embeddings) == 0 or len(self.test_embeddings) == 0:
            raise RuntimeError("Call prepare_input_data before running the evaluation.")

        self.classifier.fit(self.train_embeddings, self.train_labels)
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(
            self.test_embeddings
        )
        prediction_labels = np.asarray(prediction_labels)
        similarities = np.asarray(similarities, dtype=float)

        similarity_thresholds = np.empty(len(self.false_alarm_rate_range), dtype=float)
        identification_rates = np.empty(len(self.false_alarm_rate_range), dtype=float)
        achieved_false_alarm_rates = np.empty(len(self.false_alarm_rate_range), dtype=float)
        unknown_mask = self.test_labels == UNKNOWN_LABEL

        for index, false_alarm_rate in enumerate(self.false_alarm_rate_range):
            threshold = self.select_similarity_threshold(similarities, false_alarm_rate)
            open_set_labels = prediction_labels.copy()
            open_set_labels[similarities < threshold] = UNKNOWN_LABEL

            similarity_thresholds[index] = threshold
            identification_rates[index] = self.calc_identification_rate(open_set_labels)
            achieved_false_alarm_rates[index] = np.mean(similarities[unknown_mask] >= threshold)

        # Report all performance measures.
        evaluation_results = {
            "false_alarm_rates": self.false_alarm_rate_range.copy(),
            "achieved_false_alarm_rates": achieved_false_alarm_rates,
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }
        return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        similarity = np.asarray(similarity, dtype=float)
        unknown_similarities = similarity[np.asarray(self.test_labels) == UNKNOWN_LABEL]
        if len(unknown_similarities) == 0:
            raise ValueError("The test set contains no unknown samples for FAR calibration.")
        if not 0.0 <= false_alarm_rate <= 1.0:
            raise ValueError("false_alarm_rate must be in [0, 1].")

        if false_alarm_rate == 0.0:
            return float(np.nextafter(np.max(unknown_similarities), np.inf))
        if false_alarm_rate == 1.0:
            return float(np.min(unknown_similarities))

        # Compute the allowed number of false alarms.
        num_unknown = len(unknown_similarities)
        allowed_false_alarms = int(np.floor(false_alarm_rate * num_unknown))
        if allowed_false_alarms == 0:
            return float(np.nextafter(np.max(unknown_similarities), np.inf))

        # Convert it to a percentile.
        percentile = 100.0 * (num_unknown - allowed_false_alarms) / (num_unknown - 1)
        threshold = np.percentile(unknown_similarities, percentile, method="higher")

        # Adjust the threshold in case of ties.
        if np.mean(unknown_similarities >= threshold) > false_alarm_rate:
            threshold = np.nextafter(threshold, np.inf)
        return float(threshold)

    def calc_identification_rate(self, prediction_labels):
        prediction_labels = np.asarray(prediction_labels)
        if prediction_labels.shape != np.asarray(self.test_labels).shape:
            raise ValueError("prediction_labels and test_labels must have the same shape.")

        known_mask = np.asarray(self.test_labels) != UNKNOWN_LABEL
        if not np.any(known_mask):
            raise ValueError("The test set contains no known samples.")
        return float(np.mean(prediction_labels[known_mask] == np.asarray(self.test_labels)[known_mask]))
