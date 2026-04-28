import torch

def cal_top_k_accuracy(output, target, top_k=[1]):
    output, target = output.cpu(), target.cpu()
    accuracies = []
    with torch.no_grad():
        max_k = max(top_k)
        # Get the top max_k predictions
        _, pred = output.topk(max_k, dim=1)
        pred = pred.t()
        
        if target.dim() != 1:
            _, target = target.topk(1,dim=1)
        for k in top_k:
            # Check if targets are in the top k predictions
            correct = pred[:k].eq(target.view(1, -1).expand_as(pred[:k]))
            
            # Calculate accuracy
            correct_k = correct.reshape(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(1 / target.size(0)).item()
            accuracies.append(accuracy)
    
    return accuracies