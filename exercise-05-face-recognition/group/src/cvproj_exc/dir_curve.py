import matplotlib.pyplot as plt
import numpy as np

from cvproj_exc.classifier import NearestNeighborClassifier
from cvproj_exc.config import Config
from cvproj_exc.evaluation import OpenSetEvaluation


def main():
    # The range of the false alarm rate in logarithmic space to draw DIR curves.
    false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)

    # Pickle files containing embeddings and corresponding class labels for the
    # training and the test dataset.
    train_data_file = Config.EVAL_TRAIN_DATA
    test_data_file = Config.EVAL_TEST_DATA

    # We use a nearest neighbor classifier for this evaluation.
    classifier = NearestNeighborClassifier()

    # Prepare a new evaluation instance and feed training and test data into this evaluation.
    evaluation = OpenSetEvaluation(
        classifier=classifier, false_alarm_rate_range=false_alarm_rate_range
    )
    evaluation.prepare_input_data(train_data_file, test_data_file)

    # Run the evaluation and retrieve the performance measures on the test dataset.
    results = evaluation.run()

    # Plot and save the DIR curve.
    plt.semilogx(
        results["false_alarm_rates"],
        results["identification_rates"],
        markeredgewidth=1,
        linewidth=3,
        linestyle="--",
        color="blue",
    )
    plt.grid(True)
    plt.axis(
        [
            false_alarm_rate_range[0],
            false_alarm_rate_range[len(false_alarm_rate_range) - 1],
            0,
            1,
        ]
    )
    plt.xlabel("False alarm rate")
    plt.ylabel("Identification rate")

    output_dir = Config.PROJECT_DIR.joinpath("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath("dir_curve.png")
    
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    print("DIR curve saved to {}".format(output_file))
    
    if "agg" in plt.get_backend().lower():
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()
