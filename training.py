
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report

def eval_polarity(model, loader, device, report=False):
    """
    Evaluate the model on the polarity prediction task.
    Args:
        model: The GNN model.
        loader: DataLoader for the dataset.
        device: Device to run the model on.
        report: Whether to return a classification report.
    Returns:
        The macro F1 score and a classification report (if report=True).
    """
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pol_logits = model(batch)

            y_true_all.append(batch.edge_label.cpu())
            y_pred_all.append(pol_logits.argmax(dim=1).cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    if report:
        report = classification_report(y_true, y_pred, target_names=["Oppose", "Neutral", "Support"], digits=2)
    else:
        report = None
    return macro_f1, (y_true, y_pred), report

def train_polarity_only(model, optimizer, train_loader, val_loader, device, patience, pol_weights=None, num_epochs=50):
    """
    Train the model using the polarity loss only.
    Args:
        model: The GNN model.
        optimizer: The optimizer.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        device: Device to run the model on.
        patience: Patience for early stopping.
        pol_weights: Optional weights for polarity prediction loss.
        num_epochs: Maximum number of epochs.
    Returns:
        The trained model (with best state loaded).
    """    
    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_ex = 0.0 ,0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            y_pred = model(batch)
            y_true = batch.edge_label

            # Polarity loss
            loss_pol = F.cross_entropy(y_pred, y_true, weight=pol_weights)

            loss_pol.backward()
            optimizer.step()

            batch_size = y_true.size(0)
            total_loss += loss_pol.item() * batch_size
            total_ex += batch_size

        train_loss = total_loss / total_ex

        # Validation (polarity only)
        val_f1, _, _ = eval_polarity(model, val_loader, device, report=False)
        print(f"Epoch {epoch:02d} | Loss(avg): {train_loss:.4f} | Val macro-F1: {val_f1:.4f}")

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best Val macro-F1: {best_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Loaded best model with F1: {best_f1:.4f}")
        
    return model


def eval_hierarchical(model, loader, device, report=False):
    """
    Evaluate on a loader built over the 4-class dataset (pos+neg edges).
    We compute:
      - link prediction: NoVote vs Voted (binary)
      - final 4-class prediction: NoVote or (polarity+1) when predicted Voted
      - polarity on true-positive edges where the model also predicts Voted
    """
    model.eval()
    y_true_all, y_pred4_all, y_true_link_all, y_pred_link_all = [], [], [], []
    y_true_pol_tp, y_pred_pol_tp = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            link_logits, pol_logits = model(batch)

            # True labels
            y_true_4 = batch.edge_label  # {0,1,2,3}
            y_true_link = (y_true_4 != 0).long()

            # Predictions
            link_pred = link_logits.argmax(dim=1)            # {0,1}
            pol_pred  = pol_logits.argmax(dim=1)             # {0,1,2}

            # Final 4-class: if predicted NoVote -> 0 else (pol_pred+1)
            y_pred_4 = torch.zeros_like(y_true_4)
            voted_mask_pred = (link_pred == 1)
            y_pred_4[voted_mask_pred] = pol_pred[voted_mask_pred] + 1

            # Store global metrics
            y_true_all.append(y_true_4.cpu())
            y_pred4_all.append(y_pred_4.cpu())
            y_true_link_all.append(y_true_link.cpu())
            y_pred_link_all.append(link_pred.cpu())

            # Polarity on true-positive edges where the model also predicts Voted
            tp_mask = (y_true_link == 1) & (link_pred == 1)
            if tp_mask.any():
                y_true_pol_tp.append((y_true_4[tp_mask] - 1).cpu())  # {0,1,2}
                y_pred_pol_tp.append(pol_pred[tp_mask].cpu())

    y_true_4 = torch.cat(y_true_all).numpy()
    y_pred_4 = torch.cat(y_pred4_all).numpy()
    y_true_link = torch.cat(y_true_link_all).numpy()
    y_pred_link = torch.cat(y_pred_link_all).numpy()

    macro_f1_4 = f1_score(y_true_4, y_pred_4, average="macro")
    link_f1 = f1_score(y_true_link, y_pred_link, average="binary")

    pol_report = None
    pol_macro_f1 = 0.0
    if len(y_true_pol_tp) > 0:
        y_true_pol = torch.cat(y_true_pol_tp).numpy()
        y_pred_pol = torch.cat(y_pred_pol_tp).numpy()
        pol_macro_f1 = f1_score(y_true_pol, y_pred_pol, average="macro")
        if report:
            pol_report = classification_report(
                y_true_pol, y_pred_pol,
                target_names=["Oppose", "Neutral", "Support"],
                digits=2
            )
    else:
        pol_macro_f1 = float("nan")

    return macro_f1_4, link_f1, pol_macro_f1, (y_true_4, y_pred_4), pol_report

def train_hierarchical(model, optimizer, train_loader, val_loader, device, patience, lambda_pol, link_weights=None, pol_weights=None, num_epochs=50):
    """
    Train the model using the hierarchical loss (link + polarity).
    Args:
        model: The GNN model.
        optimizer: The optimizer.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        device: Device to run the model on.
        patience: Patience for early stopping.
        lambda_pol: Weight for polarity loss.
        link_weights: Optional weights for link prediction loss.
        pol_weights: Optional weights for polarity prediction loss.
        num_epochs: Maximum number of epochs.
    Returns:
        The trained model (with best state loaded).
    """    
    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_ex = 0.0 ,0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            link_logits, pol_logits = model(batch)

            y_true_4 = batch.edge_label                # {0,1,2,3}
            y_true_link = (y_true_4 != 0).long()       # {0,1}

            # 1) Link loss on all edges in the batch
            loss_link = F.cross_entropy(link_logits, y_true_link, weight=link_weights)

            # 2) Polarity loss only on positive (voted) edges
            pos_mask = (y_true_link == 1)
            if pos_mask.any():
                y_true_pol = (y_true_4[pos_mask] - 1).long()  # {0,1,2}
                loss_pol = F.cross_entropy(pol_logits[pos_mask], y_true_pol, weight=pol_weights)
            else:
                loss_pol = torch.tensor(0.0, device=device)

            loss = loss_link + lambda_pol * loss_pol
            loss.backward()
            optimizer.step()

            batch_size = y_true_4.size(0)
            total_loss += loss.item() * batch_size
            total_ex += batch_size

        train_loss = total_loss / total_ex

        # Validation (hierarchical 4-class + link)
        val_macro_f1_4, val_link_f1, val_pol_f1, _, _ = eval_hierarchical(model, val_loader, device, report=False)
        print(f"Epoch {epoch:02d} | Loss(avg): {train_loss:.4f} | Val macro-F1(4): {val_macro_f1_4:.4f} | Val link-F1: {val_link_f1:.4f} | Val pol-macroF1: {val_pol_f1:.4f}")

        if val_macro_f1_4 > best_f1 + 1e-4:
            best_f1 = val_macro_f1_4
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best Val macro-F1(4): {best_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Loaded best model with F1(4): {best_f1:.4f}")
        
    return model
