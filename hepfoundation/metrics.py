import torch

def multiclass_acc(batch_result):
    outputs = batch_result['outputs']
    labels = batch_result['labels'].data

    outputs = outputs.view(-1, outputs.shape[-1])
    labels = labels.view(-1)

    _, preds = torch.max(outputs, 1)

    results = []
    corrects = torch.sum(preds == labels)

    for ii in range(4):
        index = labels == ii
        ipreds = preds[index]
        ilabels = labels[index]

        if len(ilabels) == 0:
            results.append(1.)
        else:
            corrects = torch.sum(ipreds == ilabels)
            results.append(corrects.detach().item() / len(ilabels))

    return results
