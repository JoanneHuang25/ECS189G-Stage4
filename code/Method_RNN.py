# code/stage_4_code/Method_RNN.py
import torch
from torch import nn
import numpy as np
from typing import List, Tuple

from code.base_class.method import method # Assuming this base_class exists

# ------------------------------
# RNN Cell Factory
# ------------------------------
RNN_MAP = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}

def build_rnn(kind: str, input_size: int, hidden_size: int, num_layers: int, dropout_rate: float, batch_first: bool = True):
    Cell = RNN_MAP.get(kind.upper())
    if not Cell:
        raise ValueError(f"Unknown RNN type: {kind}. Available: {list(RNN_MAP.keys())}")
    return Cell(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout_rate if num_layers > 1 else 0)


class Method_RNN_Classifier(method, nn.Module):
    max_epoch = 50 # Default, can be overridden in script
    learning_rate = 1e-3
    batch_size = 64
    
    def __init__(self, mName, mDescription, vocab_size, rnn_type="LSTM", embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.2):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self) # Essential for PyTorch modules

        self.vocab_size = vocab_size
        self.rnn_type = rnn_type
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0) # Assuming 0 is PAD_IDX
        self.rnn = build_rnn(self.rnn_type, self.embed_dim, self.hidden_dim, self.num_layers, self.dropout)
        self.fc = nn.Linear(self.hidden_dim, 2) # Binary classification (e.g., pos/neg)
        
        self.optimizer = None
        self.loss_function = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x) # (batch, seq_len) -> (batch, seq_len, embed_dim)
        
        # RNN output varies: output, hidden (RNN, GRU) or output, (h_n, c_n) (LSTM)
        rnn_output, hidden_state = self.rnn(embedded)
        
        # We need the last hidden state of the last layer
        if self.rnn_type == "LSTM":
            # hidden_state is (h_n, c_n)
            # h_n is (num_layers * num_directions, batch, hidden_size)
            last_hidden = hidden_state[0][-1] # Take h_n, then last layer
        else: # RNN, GRU
            # hidden_state is (num_layers * num_directions, batch, hidden_size)
            last_hidden = hidden_state[-1] # Take last layer
            
        # last_hidden is (batch, hidden_size)
        out = self.fc(last_hidden) # (batch, hidden_size) -> (batch, num_classes)
        return out

    def _run_epoch(self, dataloader, is_train: bool):
        self.train(is_train) # Set model to train or eval mode
        total_loss = 0
        all_preds, all_trues = [], []

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            if is_train:
                self.optimizer.zero_grad()

            predictions = self.forward(x_batch)
            loss = self.loss_function(predictions, y_batch)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0) # Gradient clipping
                self.optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            
            if not is_train: # Collect predictions and true labels for evaluation
                all_preds.extend(predictions.argmax(1).cpu().tolist())
                all_trues.extend(y_batch.cpu().tolist())
        
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss, (all_preds, all_trues)


    def train_model(self, train_dataloader, test_dataloader=None, eval_every=10):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        train_losses, test_losses, test_accuracies = [], [], []
        
        print(f"Training {self.mName} on {self.device} for {self.max_epoch} epochs...")

        for epoch in range(1, self.max_epoch + 1):
            train_loss, _ = self._run_epoch(train_dataloader, is_train=True)
            train_losses.append(train_loss)

            epoch_summary = f"Epoch {epoch}/{self.max_epoch} | Train Loss: {train_loss:.4f}"

            if test_dataloader and (epoch % eval_every == 0 or epoch == 1 or epoch == self.max_epoch):
                test_loss, (preds, trues) = self._run_epoch(test_dataloader, is_train=False)
                test_losses.append(test_loss)
                # Simple accuracy for quick check during training; full eval done by script
                current_accuracy = accuracy_score(trues, preds) if trues else 0.0
                test_accuracies.append(current_accuracy)
                epoch_summary += f" | Test Loss: {test_loss:.4f} | Test Acc: {current_accuracy:.4f}"
                # For detailed metrics, the script will call evaluate on these preds/trues
            
            print(epoch_summary)
        
        # After all epochs, get final test predictions if test_dataloader is provided
        final_preds, final_trues = [], []
        if test_dataloader:
            _, (final_preds, final_trues) = self._run_epoch(test_dataloader, is_train=False)

        return {'train_losses': train_losses, 
                'test_losses': test_losses, 
                'test_accuracies_epoch': test_accuracies, # Accuracy at eval points
                'final_test_predictions': final_preds, 
                'final_test_trues': final_trues}
    
    def test_model(self, test_dataloader):
        """Simplified test method, assuming model is trained."""
        self.eval() # Set model to evaluation mode
        _, (preds, trues) = self._run_epoch(test_dataloader, is_train=False)
        return preds, trues

    def run(self):
        """ Main execution method, aligns with script_cnn.py structure """
        print(f'Running method: {self.mName}')
        if self.data is None or 'train' not in self.data or 'test' not in self.data:
            raise ValueError("Data not loaded into method or missing train/test splits.")

        train_dataset = self.data['train']['dataset']
        test_dataset = self.data['test']['dataset']
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Training
        print('--start training...')
        train_results = self.train_model(train_loader, test_loader, eval_every=self.data.get('eval_every', 10))
        print('--training finished...')

        # Testing (using predictions from the last epoch of training or a separate test run)
        print('--start testing...')
        # final_test_predictions, final_test_trues are already from train_results
        pred_y = train_results['final_test_predictions']
        true_y = train_results['final_test_trues']
        print('--testing finished...')
        
        # The evaluator object will be handled in the main script
        return {
            'pred_y': pred_y,
            'true_y': true_y,
            'train_loss_values': train_results['train_losses'],
            # 'test_loss_values': train_results['test_losses'], # Can add if needed
            # 'test_accuracy_values': train_results['test_accuracies_epoch'] # Can add if needed
        }


class Method_RNN_Generator(method, nn.Module):
    max_epoch = 100 # Default, can be overridden in script
    learning_rate = 2e-3
    batch_size = 64

    def __init__(self, mName, mDescription, vocab_size, vocab_instance, rnn_type="LSTM", embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.vocab = vocab_instance # Store vocab instance for generation
        self.rnn_type = rnn_type
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0) # Assuming 0 is PAD_IDX
        self.rnn = build_rnn(self.rnn_type, self.embed_dim, self.hidden_dim, self.num_layers, self.dropout)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        
        self.optimizer = None
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0) # Ignore PAD_IDX in loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len)
        embedded = self.embedding(x) # (batch, seq_len, embed_dim)
        
        # output: (batch, seq_len, hidden_dim)
        # hidden_state: (num_layers, batch, hidden_dim) for RNN/GRU
        # hidden_state: (h_n, c_n) where h_n, c_n are (num_layers, batch, hidden_dim) for LSTM
        output, new_hidden_state = self.rnn(embedded, hidden_state)
        
        # logits: (batch, seq_len, vocab_size)
        logits = self.fc(output)
        return logits, new_hidden_state

    def _run_epoch(self, dataloader, is_train: bool):
        self.train(is_train)
        total_loss = 0.0
        
        # For generator, hidden state is typically not carried over batches during training
        # unless it's stateful training, which is not implied by the original script.

        for x_batch, y_batch in dataloader: # x_batch is input_seq, y_batch is target_seq
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            if is_train:
                self.optimizer.zero_grad()
            
            # Initial hidden state for each batch (None will make nn.RNN initialize it)
            hidden = None 
            logits, _ = self.forward(x_batch, hidden) # logits: (batch, seq_len, vocab_size)
            
            # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size) and (batch * seq_len)
            loss = self.loss_function(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                self.optimizer.step()
            
            total_loss += loss.item() * x_batch.size(0) # x_batch.size(0) is batch_size
            
        avg_loss = total_loss / len(dataloader.dataset)
        return avg_loss


    def train_model(self, train_dataloader, start_tokens=["what", "did", "the"], eval_every=10):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        train_losses = []
        
        print(f"Training {self.mName} on {self.device} for {self.max_epoch} epochs...")

        for epoch in range(1, self.max_epoch + 1):
            train_loss = self._run_epoch(train_dataloader, is_train=True)
            train_losses.append(train_loss)

            epoch_summary = f"Epoch {epoch}/{self.max_epoch} | Train Loss: {train_loss:.4f}"
            
            if epoch % eval_every == 0 or epoch == 1 or epoch == self.max_epoch:
                generated_sample = self.generate(start_tokens=start_tokens, max_len=40, temp=0.7)
                epoch_summary += f" → Sample: \"{generated_sample}\""
            
            print(epoch_summary)
            
        return {'train_losses': train_losses}

    def generate(self, start_tokens: List[str], max_len: int = 40, temp: float = 1.0) -> str:
        self.eval() # Set model to evaluation mode
        
        # Convert start tokens to IDs
        current_ids = [self.vocab.stoi.get(t, self.vocab.stoi[self.vocab.UNK]) for t in start_tokens]
        if not current_ids or current_ids[0] != self.vocab.stoi[self.vocab.BOS]:
             # Original script adds BOS in dataset prep, generation should reflect similar sequence structure
             # However, for user-provided start_tokens, let's assume they are the actual start.
             # If BOS is always expected, it should be prepended here or ensured by user.
             # The original script's generate() does this:
             # ids=[vocab.stoi.get(t, vocab.stoi[Vocab.UNK]) for t in start]
             # inp=torch.tensor([ids], device=self.device)
             # The loop then appends to `ids`.
             pass


        generated_ids = list(current_ids) # Make a mutable copy
        inp_tensor = torch.tensor([current_ids], device=self.device) # Batch size 1
        hidden = None # Initialize hidden state

        with torch.no_grad():
            # First, process the initial sequence if it's longer than one token
            if inp_tensor.size(1) > 1:
                # Pass the whole initial sequence to get the last hidden state
                # We are interested in the *next* token, so the output for the last input token is key
                logits_seq, hidden = self.forward(inp_tensor, hidden)
                # Take logits for the last token in the input sequence
                next_token_logits = logits_seq[0, -1, :] / temp 
            elif inp_tensor.size(1) == 1: # Single start token
                logits_seq, hidden = self.forward(inp_tensor, hidden)
                next_token_logits = logits_seq[0, -1, :] / temp
            else: # No start tokens, or an issue
                return "[Error: No valid start tokens]"

            # Generate subsequent tokens
            for _ in range(max_len - len(current_ids)):
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_id = torch.multinomial(probabilities, 1).item()

                if next_id == self.vocab.stoi[self.vocab.EOS]:
                    break
                
                generated_ids.append(next_id)
                
                # Next input is just the new token
                inp_tensor = torch.tensor([[next_id]], device=self.device)
                logits_seq, hidden = self.forward(inp_tensor, hidden) # hidden state is carried over
                next_token_logits = logits_seq[0, -1, :] / temp


        # Decode IDs to tokens
        generated_tokens = self.vocab.decode(generated_ids)
        
        # Post-processing (e.g., removing BOS if it was part of generation logic)
        # Original script: if toks[0] == Vocab.BOS: toks = toks[1:]
        # This depends on how start_tokens are handled. If BOS is implicitly added, it should be removed.
        # For now, assume generated_tokens are the intended sequence.
        
        return " ".join(generated_tokens)

    def run(self):
        """ Main execution method """
        print(f'Running method: {self.mName}')
        if self.data is None or 'train' not in self.data:
             raise ValueError("Training data not loaded into method.")

        train_dataset = self.data['train']['dataset']
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training
        print('--start training...')
        train_results = self.train_model(train_dataloader=train_loader, 
                                         start_tokens=self.data.get('start_tokens', ["what", "did", "the"]),
                                         eval_every=self.data.get('eval_every', 10))
        print('--training finished...')

        # Generation (example after training)
        print('--start generation (sample)...')
        final_sample = self.generate(start_tokens=self.data.get('start_tokens', ["what", "did", "the"]))
        print(f"Final generated sample: \"{final_sample}\"")
        print('--generation finished...')
        
        return {
            'train_loss_values': train_results['train_losses'],
            'final_generated_sample': final_sample
        }