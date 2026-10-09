import os
import pickle

import cv2
import numpy as np

from cvproj_exc.config import Config


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    # Predict embedding from a given face image.
    def predict(self, face: np.ndarray) -> np.ndarray:
        if face is None or face.size == 0:
            raise ValueError("FaceNet received an empty face image.")
        if face.shape[:2] != (224, 224):
            face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize face image using the preprocessing of the provided ONNX model.
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face_rgb -= np.asarray((131.0912, 103.8827, 91.4953), dtype=np.float32)

        # Forward pass through the provided neural network.
        reshaped = np.moveaxis(face_rgb, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward()).astype(np.float64)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise RuntimeError("FaceNet returned a zero embedding.")
        return embedding / norm

    embedding_dimensionality = 128
    """Dimensionality of the extracted embeddings."""


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=3, max_distance=1.044035, min_prob=2.0 / 3.0):
        self.facenet = FaceNet()
        self.num_neighbours = max(1, int(num_neighbours))
        self.max_distance = float(max_distance)
        self.min_prob = float(min_prob)

        self.labels = []
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float64)

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceRecognizer saving: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        print("FaceRecognizer loading: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)
        self.labels = list(self.labels)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float64)

    # Train face identification with a new face with labeled identity.
    def partial_fit(self, face: np.ndarray, label):
        color_embedding, grayscale_embedding = self._extract_color_and_grayscale(face)
        self.embeddings = np.vstack((self.embeddings, color_embedding, grayscale_embedding))
        self.labels.extend((label, label))

    # Predict the identity for a new face.
    def predict(self, face: np.ndarray) -> tuple[str, float, float]:
        if len(self.labels) == 0 or len(self.embeddings) == 0:
            raise RuntimeError("FaceRecognizer cannot predict with an empty gallery.")
        if len(self.labels) != len(self.embeddings):
            raise RuntimeError("Gallery labels and embeddings have inconsistent lengths.")

        query_color, query_gray = self._extract_color_and_grayscale(face)
        distances, sample_labels = self._fused_gallery_distances(query_color, query_gray)

        k = min(self.num_neighbours, len(distances))
        nearest_indices = np.argsort(distances, kind="stable")[:k]
        nearest_labels = [sample_labels[index] for index in nearest_indices]
        nearest_distances = distances[nearest_indices]

        vote_counts = {}
        for label in nearest_labels:
            vote_counts[label] = vote_counts.get(label, 0) + 1
        largest_vote = max(vote_counts.values())
        candidates = [label for label, count in vote_counts.items() if count == largest_vote]
        predicted_label = min(
            candidates,
            key=lambda label: min(
                distance
                for distance, neighbour_label in zip(nearest_distances, nearest_labels)
                if neighbour_label == label
            ),
        )

        posterior_probability = vote_counts[predicted_label] / k
        predicted_distance = min(
            distance
            for distance, neighbour_label in zip(nearest_distances, nearest_labels)
            if neighbour_label == predicted_label
        )

        if predicted_distance > self.max_distance or posterior_probability < self.min_prob:
            output_label = "unknown"
        else:
            output_label = predicted_label

        return output_label, float(posterior_probability), float(predicted_distance)

    def _extract_color_and_grayscale(self, face: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        color_embedding = self.facenet.predict(face)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_three_channels = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        grayscale_embedding = self.facenet.predict(gray_three_channels)
        return color_embedding, grayscale_embedding

    def _fused_gallery_distances(
        self, query_color: np.ndarray, query_gray: np.ndarray
    ) -> tuple[np.ndarray, list]:
        labels = list(self.labels)

        # Combine color and grayscale distances.
        paired = len(labels) % 2 == 0 and all(
            labels[index] == labels[index + 1] for index in range(0, len(labels), 2)
        )
        if paired:
            gallery_pairs = self.embeddings.reshape(-1, 2, FaceNet.embedding_dimensionality)
            color_distances = np.linalg.norm(gallery_pairs[:, 0, :] - query_color, axis=1)
            gray_distances = np.linalg.norm(gallery_pairs[:, 1, :] - query_gray, axis=1)
            return 0.5 * (color_distances + gray_distances), labels[::2]

        color_distances = np.linalg.norm(self.embeddings - query_color, axis=1)
        gray_distances = np.linalg.norm(self.embeddings - query_gray, axis=1)
        return 0.5 * (color_distances + gray_distances), labels


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=200):
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.embedding_dimensionality), dtype=np.float64)

        # Number of cluster centers for k-means clustering.
        self.num_clusters = int(num_clusters)
        # Cluster centers.
        self.cluster_center = np.empty(
            (self.num_clusters, FaceNet.embedding_dimensionality), dtype=np.float64
        )
        # Cluster index associated with the different samples.
        self.cluster_membership = np.empty((0,), dtype=int)

        # Maximum number of iterations and stopping tolerance for k-means clustering.
        self.max_iter = int(max_iter)
        self.tolerance = 1e-6
        self.random_state = 0
        self.objective_history = []

        # Load face clustering from pickle file if available.
        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )
        self.embeddings = np.asarray(self.embeddings, dtype=np.float64)
        self.cluster_center = np.asarray(self.cluster_center, dtype=np.float64)
        self.cluster_membership = np.asarray(self.cluster_membership, dtype=int)

    def partial_fit(self, face: np.ndarray):
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack((self.embeddings, embedding))

    def fit(self):
        num_samples = len(self.embeddings)
        if self.num_clusters < 2:
            raise ValueError("k-means requires num_clusters >= 2.")
        if num_samples < self.num_clusters:
            raise ValueError("The number of embeddings must be at least num_clusters.")

        rng = np.random.default_rng(self.random_state)
        initial_indices = rng.choice(num_samples, size=self.num_clusters, replace=False)
        centers = self.embeddings[initial_indices].copy()
        previous_membership = None
        self.objective_history = []

        for _ in range(self.max_iter):
            squared_distances = np.sum(
                (self.embeddings[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            membership = np.argmin(squared_distances, axis=1)

            new_centers = centers.copy()
            for cluster_index in range(self.num_clusters):
                members = self.embeddings[membership == cluster_index]
                if len(members) > 0:
                    new_centers[cluster_index] = np.mean(members, axis=0)
                else:
                    # Handle empty cluster.
                    assigned_distance = squared_distances[
                        np.arange(num_samples), membership
                    ]
                    replacement_index = int(np.argmax(assigned_distance))
                    new_centers[cluster_index] = self.embeddings[replacement_index]
                    membership[replacement_index] = cluster_index

            objective = float(
                np.sum(
                    (self.embeddings - new_centers[membership])
                    * (self.embeddings - new_centers[membership])
                )
            )
            self.objective_history.append(objective)
            center_shift = float(np.linalg.norm(new_centers - centers))
            labels_unchanged = (
                previous_membership is not None
                and np.array_equal(membership, previous_membership)
            )
            centers = new_centers
            previous_membership = membership.copy()

            if labels_unchanged or center_shift <= self.tolerance:
                break

        self.cluster_center = centers
        self.cluster_membership = previous_membership
        return self.cluster_membership

    def predict(self, face: np.ndarray) -> tuple[int, np.ndarray]:
        if (
            len(self.cluster_center) == 0
            or len(self.cluster_membership) == 0
            or not np.all(np.isfinite(self.cluster_center))
        ):
            raise RuntimeError("FaceClustering must be fitted before prediction.")
        embedding = self.facenet.predict(face)
        distances = np.linalg.norm(self.cluster_center - embedding, axis=1)
        best_cluster = int(np.argmin(distances))
        return best_cluster, distances
