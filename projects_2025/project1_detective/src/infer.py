import random
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import calculate_metrics2
from .OUTFOX_utils import load_OUTFOX
from .index import Indexer
from .text_embedding import TextEmbeddingModel
from .dataset import PassagesDataset

def infer(passages_dataloder, fabric, tokenizer, model):
    """
    Inference function for text embedding generation.
    Args:
        passages_dataloder (DataLoader): The data loader for passages.
        fabric (Fabric): The Fabric instance for distributed training.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (TextEmbeddingModel): The text embedding model.
    Returns:
        tuple: A tuple containing lists of all IDs, embeddings, and labels.
    """
    allids, allembeddings, alllabels = [], [], []

    model.model.eval()
    with torch.no_grad():
        for batch in passages_dataloder:
            text, label, write_model, write_model_set, ids = batch
            encoded_batch = tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                max_length=512,
                padding="max_length",
                truncation=True,
            )
            encoded_batch = {k: v.to(fabric.device) for k, v in encoded_batch.items()}
            embeddings = model(encoded_batch)
            if embeddings is None:
                print("Warning: model returned None")
                continue

            #print(f"Embedding shape: {embeddings.shape}")
            embeddings = fabric.all_gather(embeddings).view(-1, embeddings.size(1))
            label = fabric.all_gather(label).view(-1)
            ids = fabric.all_gather(ids).view(-1)

            allembeddings.append(embeddings.cuda())
            allids.extend(ids.cuda().tolist())
            alllabels.extend(label.cuda().tolist())

    if len(allembeddings) == 0:
        print("No embeddings collected — returning empty result")
        return [], [], []

    allembeddings = torch.cat(allembeddings, dim=0)
    epsilon = 1e-6
    norms = torch.norm(allembeddings, dim=1, keepdim=True) + epsilon
    allembeddings = allembeddings / norms

    emb_dict, label_dict = {}, {}
    for i in range(len(allids)):
        emb_dict[allids[i]] = allembeddings[i]
        label_dict[allids[i]] = alllabels[i]

    # Deduplicate
    allids, allembeddings, alllabels = [], [], []
    for key in emb_dict:
        allids.append(key)
        allembeddings.append(emb_dict[key])
        alllabels.append(label_dict[key])
    allembeddings = torch.stack(allembeddings, dim=0)
    return allids, allembeddings.cpu().numpy(), alllabels

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The random seed value.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def main_inference(fabric, opt):
    """
    Main function for inference.
    Args:
        fabric (Fabric): The Fabric instance for distributed training.
        opt (argparse.Namespace): The parsed command-line arguments.
    """
    model = TextEmbeddingModel(opt.model_name)
    state_dict = torch.load(opt.model_path, map_location=model.model.device)
    model.load_state_dict(state_dict)

    tokenizer = model.tokenizer
    if opt.mode=='OUTFOX':
        test_database = load_OUTFOX(opt.test_dataset_path,opt.attack)[opt.test_dataset_name]

    test_dataset = PassagesDataset(test_database, mode=opt.mode, need_ids=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True)
    test_dataloader = fabric.setup_dataloaders(test_dataloader)
    model = fabric.setup(model)

    test_ids, test_embeddings, test_labels = infer(test_dataloader, fabric, tokenizer, model)
    fabric.barrier()

    if fabric.global_rank == 0:
        index = Indexer(vector_sz=768, device='cuda')
        index.index_data(test_ids, test_embeddings)
        label_dict = dict(zip(test_ids, test_labels))
        test_labels=[str(test_labels[i]) for i in range(len(test_labels))]

        preds= {i: [] for i in range(1,opt.max_K+1)}
        if len(test_embeddings.shape) == 1:
            test_embeddings = test_embeddings.reshape(1, -1)
        top_ids_and_scores = index.search_knn(test_embeddings, opt.max_K)
        for i, (ids, scores) in enumerate(top_ids_and_scores):
            zero_num,one_num=0,0
            # Sort the scores in descending order
            sorted_scores = np.argsort(scores)
            sorted_scores = sorted_scores[::-1]
            for k in range(1,opt.max_K+1):
                id = ids[sorted_scores[k-1]]
                if label_dict[int(id)]==0:
                    zero_num+=1
                else:
                    one_num+=1
                if zero_num>one_num:
                    preds[k].append('0')
                else:
                    preds[k].append('1')
        K_values = list(range(1, opt.max_K+1))
        human_recs = []
        machine_recs = []
        avg_recs = []
        accs = []
        precisions = []
        recalls = []
        f1_scores = []

        for k in range(1, opt.max_K+1):
            human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = calculate_metrics2(test_labels, preds[k])
            #print(f"K={k}, HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}")
            human_recs.append(human_rec)
            machine_recs.append(machine_rec)
            avg_recs.append(avg_rec)
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Create subplots for illustration
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))

        # Plotting each metric in a separate subplot
        axs[0, 0].plot(K_values, human_recs, marker='o', label='Human Recognition Rate')
        axs[0, 0].set_title('Human Recognition Rate')
        axs[0, 0].grid(True)

        axs[0, 1].plot(K_values, machine_recs, marker='x', label='Machine Recognition Rate')
        axs[0, 1].set_title('Machine Recognition Rate')
        axs[0, 1].grid(True)

        axs[0, 2].plot(K_values, avg_recs, marker='^', label='Average Recognition Rate')
        axs[0, 2].set_title('Average Recognition Rate')
        axs[0, 2].grid(True)

        axs[1, 0].plot(K_values, accs, marker='s', label='Accuracy')
        axs[1, 0].set_title('Accuracy')
        axs[1, 0].grid(True)

        axs[1, 1].plot(K_values, precisions, marker='p', label='Precision')
        axs[1, 1].set_title('Precision')
        axs[1, 1].grid(True)

        axs[1, 2].plot(K_values, recalls, marker='*', label='Recall')
        axs[1, 2].set_title('Recall')
        axs[1, 2].grid(True)

        axs[2, 0].plot(K_values, f1_scores, marker='D', label='F1 Score')
        axs[2, 0].set_title('F1 Score')
        axs[2, 0].grid(True)

        # Hide empty subplots
        for i in range(2, 3):
            for j in range(1, 3):
                axs[i, j].axis('off')
        plt.tight_layout()
        plt.savefig('performance_metrics_subplot.png', dpi=300)
        max_ids=0
        for i in range(1,opt.max_K):
            if avg_recs[i]>avg_recs[max_ids]:
                max_ids=i
        print(f"Find opt.max_K is {max_ids+1}")
        print(f"HumanRec: {human_recs[max_ids]}, MachineRec: {machine_recs[max_ids]}, AvgRec: {avg_recs[max_ids]}, Acc:{accs[max_ids]}, Precision:{precisions[max_ids]}, Recall:{recalls[max_ids]}, F1:{f1_scores[max_ids]}")