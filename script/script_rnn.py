# script_rnn.py
import os
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# --- Setup Project Root ---
# Assuming this script is in a project structure like:
# project_root/
#   code/
#     stage_4_code/
#       Dataset_Loader.py
#       Method_RNN.py
#       Evaluate_Accuracy.py
#       Result_Saver.py (optional, if you create one for stage_4)
#     base_class/
#       dataset.py
#       method.py
#       evaluate.py
#   data/
#     stage_4_data/
#       text_classification/
#         imdb_packed2.pkl.gz
#       text_generation/
#         data  (short-jokes.txt)
#   result/
#     stage_4_result/
#   scripts/
#     script_rnn.py (this file)

# Get the absolute path to the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root is one level up from 'scripts'
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from code.stage_4_code.Dataset_Loader import Dataset_Loader_IMDB, Dataset_Loader_Jokes # Vocab is also in here
from code.stage_4_code.Method_RNN import Method_RNN_Classifier, Method_RNN_Generator
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Text_Classification
# from code.stage_4_code.Result_Saver import Result_Saver # If you implement a Result_Saver for stage 4


# ---- Configuration ----
# Adjust these paths according to your local file structure
IMDB_PICKLE_PATH = os.path.join(project_root, "data/stage_4_data/text_classification/imdb_packed2.pkl.gz")
JOKE_TEXT_FILE_PATH = os.path.join(project_root, "data/stage_4_data/text_generation/data") # Assuming 'data' is the jokes file
RESULT_ROOT = os.path.join(project_root, "result/stage_4_result/")

# Ensure result directories exist
os.makedirs(os.path.join(RESULT_ROOT, "classification"), exist_ok=True)
os.makedirs(os.path.join(RESULT_ROOT, "generation"), exist_ok=True)


def run_classification_experiment(rnn_kind="LSTM", epochs=50, batch_size=64, lr=1e-3, eval_every=10):
    print(f"\n{'='*50}")
    print(f"Running RNN Classification Experiment ({rnn_kind})")
    print(f"{'='*50}\n")

    # 1. Load Data
    print("Loading IMDb Training Data...")
    train_imdb_loader = Dataset_Loader_IMDB(
        dName=f"IMDb Train ({rnn_kind})",
        dDescription="IMDb Movie Reviews Training Set",
        source_file_path=IMDB_PICKLE_PATH,
        split='train'
    )
    train_imdb_dataset = train_imdb_loader.load() # .load() now returns 'self' which is the dataset
    # The vocab is now an attribute of the dataset instance
    shared_vocab = train_imdb_loader.vocab

    print("Loading IMDb Testing Data...")
    test_imdb_loader = Dataset_Loader_IMDB(
        dName=f"IMDb Test ({rnn_kind})",
        dDescription="IMDb Movie Reviews Testing Set",
        source_file_path=IMDB_PICKLE_PATH,
        split='test',
        vocab=shared_vocab # Pass vocab from training set
    )
    test_imdb_dataset = test_imdb_loader.load()

    # 2. Initialize Model
    model = Method_RNN_Classifier(
        mName=f"IMDb_RNN_Classifier_{rnn_kind}",
        mDescription=f"RNN Classifier ({rnn_kind}) for IMDb Sentiment",
        vocab_size=shared_vocab.size,
        rnn_type=rnn_kind,
        # Hyperparameters can be tuned here
        embed_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.2
    )
    model.max_epoch = epochs
    model.learning_rate = lr
    model.batch_size = batch_size
    
    # Prepare data for the model's run method
    model.data = {
        'train': {'dataset': train_imdb_dataset},
        'test': {'dataset': test_imdb_dataset},
        'eval_every': eval_every
    }

    # 3. Run Training and Testing (combined in model.run())
    results = model.run() # model.run() orchestrates train_model and gets test predictions

    # 4. Evaluate
    if results and 'true_y' in results and 'pred_y' in results:
        evaluator = Evaluate_Text_Classification(
            eName="IMDb Sentiment Evaluator",
            eDescription="Evaluates accuracy, precision, recall, F1 for IMDb"
        )
        evaluator.data = {'true_y': results['true_y'], 'pred_y': results['pred_y']}
        metrics = evaluator.evaluate()
        print("\nOverall Performance on Test Set:")
        if metrics:
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name.capitalize()}: {metric_value:.4f}")
    else:
        print("Could not retrieve predictions for evaluation.")

    # 5. Plot and Save Loss
    plt.figure(figsize=(7, 5))
    plt.plot(results['train_loss_values'], label="Train Loss")
    # If you add test_loss_values to results from model.run:
    # plt.plot(results['test_loss_values'], label="Test Loss")
    plt.title(f"IMDb Classification Loss – {rnn_kind}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(RESULT_ROOT, "classification", f"loss_curve_imdb_{rnn_kind}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to {plot_path}")
    print(f"{'='*50}\n")


def run_generation_experiment(rnn_kind="LSTM", epochs=100, batch_size=64, lr=2e-3, eval_every=20):
    print(f"\n{'='*50}")
    print(f"Running RNN Generation Experiment ({rnn_kind})")
    print(f"{'='*50}\n")

    # 1. Load Data
    print("Loading Joke Data...")
    joke_loader = Dataset_Loader_Jokes(
        dName="Short Jokes",
        dDescription="Dataset of short jokes for text generation",
        source_file_path=JOKE_TEXT_FILE_PATH
    )
    joke_dataset = joke_loader.load()
    shared_vocab = joke_loader.vocab # Get vocab from the loader instance

    # 2. Initialize Model
    model = Method_RNN_Generator(
        mName=f"Joke_RNN_Generator_{rnn_kind}",
        mDescription=f"RNN Generator ({rnn_kind}) for Jokes",
        vocab_size=shared_vocab.size,
        vocab_instance=shared_vocab, # Pass vocab instance for generation method
        rnn_type=rnn_kind,
        # Hyperparameters
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    model.max_epoch = epochs
    model.learning_rate = lr
    model.batch_size = batch_size
    
    start_tokens = ["what", "did", "the"] # Example start tokens
    model.data = {
        'train': {'dataset': joke_dataset},
        'start_tokens': start_tokens,
        'eval_every': eval_every
    }

    # 3. Run Training (which includes sample generation)
    results = model.run() # model.run() for generator handles training and a final sample

    # 4. Plot and Save Loss
    plt.figure(figsize=(7, 5))
    plt.plot(results['train_loss_values'], label="Train Loss")
    plt.title(f"Joke Generation Loss – {rnn_kind}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(RESULT_ROOT, "generation", f"loss_curve_jokes_{rnn_kind}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to {plot_path}")

    print("\nFinal generated joke sample from model.run():")
    print(results.get('final_generated_sample', "No sample generated."))
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Check for CUDA
    if torch.cuda.is_available():
        print("GPU is available! Using CUDA.")
    else:
        print("GPU not available. Using CPU.")

    # ---- Run Classification Experiments ----
    # run_classification_experiment(rnn_kind="RNN", epochs=50, eval_every=10) # Vanilla RNN can be slow to train
    run_classification_experiment(rnn_kind="GRU", epochs=20, eval_every=5) # Shorter epochs for GRU example
    # run_classification_experiment(rnn_kind="LSTM", epochs=20, eval_every=5) # Shorter epochs for LSTM example

    # ---- Run Generation Experiments ----
    # run_generation_experiment(rnn_kind="RNN", epochs=100, eval_every=20)
    run_generation_experiment(rnn_kind="GRU", epochs=50, eval_every=10) # Shorter epochs
    # run_generation_experiment(rnn_kind="LSTM", epochs=50, eval_every=10) # Shorter epochs
    
    print("\nAll experiments complete.")