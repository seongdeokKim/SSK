from copy import deepcopy
import numpy as np
import torch

class Trainer:

    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.best_loss = np.inf
        self.best_accuracy = 0.0

    def check_best(self, model, current_loss):
        if current_loss <= self.best_loss: # If current epoch returns lower validation loss,
            self.best_loss = current_loss  # Update lowest validation loss.
            self.best_model = deepcopy(model.state_dict()) # Update best model weights.

    def train(
            self,
            model, crit, optimizer, scheduler,
            train_loader, valid_loader,
            device,
    ):

        for epoch in range(self.config.n_epochs):

            # ========================================
            #               Training
            # ========================================

            # Put the model into training mode.
            model.train()
            # Reset the total loss for this epoch.
            total_tr_loss = 0

            for step, mini_batch in enumerate(train_loader):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                # You have to reset the gradients of all model parameters
                # before to take another step in gradient descent.
                optimizer.zero_grad()

                # Take feed-forward
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss, logits = outputs[0], outputs[1]

                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # track train loss
                total_tr_loss += loss.item()
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_tr_loss = total_tr_loss / len(train_loader)
            print('Epoch {}\nTraining - loss={:.4e}'.format(
                epoch+1,
                avg_tr_loss
            ))

            # ========================================
            #               Validation
            # ========================================

            # Put the model into evaluation mode
            model.eval()
            # Reset the validation loss and accuracy for this epoch.
            total_val_loss, total_val_accuracy = 0, 0

            for step, mini_batch in enumerate(valid_loader):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                # Telling the model not to compute or store gradients,
                # saving memory and speeding up validation
                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss, logits = outputs[0], outputs[1]

                    # Calculate the accuracy for this batch of test sentences.
                    total_val_loss += loss.mean().item()

                    # Calculate accuracy only if 'y' is LongTensor,
                    # which means that 'y' is one-hot representation.
                    if isinstance(labels, torch.LongTensor) or isinstance(labels, torch.cuda.LongTensor):
                        accuracy = (torch.argmax(logits, dim=-1) == labels).sum() / float(labels.size(0))
                    else:
                        accuracy = 0

                    total_val_accuracy += float(accuracy)

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_acc = total_val_accuracy / len(valid_loader)

            self.check_best(model, avg_val_loss)

            print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4f}'.format(
                avg_val_loss,
                avg_val_acc,
                self.best_loss,
            ))

        model.load_state_dict(self.best_model)

        return model


    def test(
            self,
            model, crit, test_loader, device,
    ):

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss and accuracy for this epoch.
        total_test_loss, total_test_accuracy = 0, 0

        for step, mini_batch in enumerate(test_loader):
            input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
            input_ids, labels = input_ids.to(device), labels.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss, logits = outputs[0], outputs[1]

                # Calculate the accuracy for this batch of test sentences.
                total_test_loss += loss.mean().item()

                # Calculate accuracy only if 'y' is LongTensor,
                # which means that 'y' is one-hot representation.
                if isinstance(labels, torch.LongTensor) or isinstance(labels, torch.cuda.LongTensor):
                    accuracy = (torch.argmax(logits, dim=-1) == labels).sum() / float(labels.size(0))
                else:
                    accuracy = 0

                total_test_accuracy += float(accuracy)

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_acc = total_test_accuracy / len(test_loader)

        print('Test - loss={:.4e} accuracy={:.4f}'.format(
            avg_test_loss,
            avg_test_acc,
        ))

