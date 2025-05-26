# code/stage_4_code/Evaluate_Accuracy.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from code.base_class.evaluate import evaluate # Assuming this base class exists

class Evaluate_Text_Classification(evaluate):
    def evaluate(self):
        if self.data is None or 'true_y' not in self.data or 'pred_y' not in self.data:
            print("Evaluation data not set or incomplete.")
            return None

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        if isinstance(true_y, list): true_y = np.array(true_y)
        if isinstance(pred_y, list): pred_y = np.array(pred_y)

        print('Evaluating performance...')
        try:
            accuracy = accuracy_score(true_y, pred_y)
            # Ensure target_names or labels are specified if not binary and using averages
            # For binary classification as in IMDb, default pos_label=1 is fine for precision, recall, f1
            # If it's multi-class and you want per-class, you'd need more info or use macro/weighted.
            # The original script implies binary classification (pos/neg for IMDb)
            precision = precision_score(true_y, pred_y, zero_division=0)
            recall = recall_score(true_y, pred_y, zero_division=0)
            f1 = f1_score(true_y, pred_y, zero_division=0)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            return metrics
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Please ensure true_y and pred_y are in a compatible format for scikit-learn metrics.")
            return None