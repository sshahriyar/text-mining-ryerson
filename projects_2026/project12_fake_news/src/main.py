from TrainAndEvaluateClassifier import TrainAndEvaluateClassifer
from TrainAndEvaluateLSTM import TrainAndEvaluateLSTM
from ReinforcementLearning import train_q_learning


def run_single_experiment():
    print("=== Running Single Model (Baseline) ===")

    data = TrainAndEvaluateClassifer(
        NumberOfArtificialArticlesTouse=50,
        UseAll=1
    )

    accuracy = TrainAndEvaluateLSTM(data)

    print(f"\nBaseline Accuracy (50 synthetic): {accuracy:.4f}")


def run_rl_optimization():
    print("\n=== Running Q-Learning Optimization ===")

    best = train_q_learning()

    print(f"\nOptimal number of synthetic articles: {best}")


if __name__ == "__main__":
    
    print("Starting Project...\n")

    # Baseline run
    run_single_experiment()

    # RL optimization
    run_rl_optimization()

    print("\n=== Done ===")