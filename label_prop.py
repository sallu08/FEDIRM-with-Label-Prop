
import numpy as np
import faiss
from faiss import normalize_L2
import scipy
import torch.nn.functional as F
import torch
import scipy.stats



def update_plabels(feat,num_classes,labels,labeled_idx,k, max_iter):
        alpha=0.99
        samples, nx, ny,c = feat.shape
        feat = feat.reshape((samples,nx*ny*c))
        d = feat.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index
     
        normalize_L2(feat)
        index.add(feat) 
        N = feat.shape[0]

        D, I = index.search(feat, k + 1)

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N)) #eq 9
        W = W + W.T  #make matrix symmetric

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N,num_classes))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(num_classes):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] ==i)]
            y = np.zeros((N,))
            if cur_idx.size > 0:
                y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:,i] = f   #eq 10

        # Handle numberical errors
        Z[Z < 0] = 0 

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
        probs_l1[probs_l1 <0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(num_classes)
        weights = weights / np.max(weights)   #eq 11
        # p_labels = np.argmax(probs_l1,1)
        # Compute the accuracy of pseudolabels for statistical purposes
        # p_labels[labeled_idx] = labels[labeled_idx]
        # correct_idx = (p_labels == labels)
        # acc = correct_idx.mean()
        # print(f'Lp acc= {acc*100:>0.2f}%')
        # preds = preds.cpu()
        # preds = preds.detach()
        # preds = preds.numpy()
        # probs_l1[labeled_idx] = preds[labeled_idx]
  
        weights[labeled_idx] = 1.0

        p_weights = weights.tolist()
        # Compute the weight for each class
        # for i in range(num_classes):
        #     cur_idx = np.where(np.asarray(p_labels) == i)[0]
        #     if cur_idx.size > 0:
        #         class_weights[i] = (float(labels.shape[0]) / (num_classes)) / cur_idx.size
     

        return p_weights,probs_l1

