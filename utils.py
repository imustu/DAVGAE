import numpy as np
import torch


def get_edge_dot_products(data, model, num_dz_nodes=519):
    """
    如果学习后的 u 的嵌入与 v 的嵌入的点积的值很高, 那么我们认为模型预测 节点对 (u, v) 是连通的.
    这一函数计算并返回所有 节点对(dz_node, gene_node) 的点积.
    """
    model.eval()
    x = data.x
    z = model.encode(x, data.edge_index).detach().numpy()
    dz_z = z[:num_dz_nodes, :]
    gene_z = z[num_dz_nodes:, :]
    # dz_z 形状为 (a, i), gene_z 形状为 (b, i), 点积结果形状为 (a, b)
    dot_products = np.einsum('ai,bi->ab', dz_z, gene_z)
    return dot_products  # 大小为 (num_dz_nodes, num_gene_nodes)




# -------------------------------------------------两个模型的训练和测试方法--------------------------------------------------
def gae_train(train_data, gae_model, optimizer, device):
    gae_model.train()
    optimizer.zero_grad()
    z = gae_model.encode(train_data.x, train_data.edge_index)
    loss = gae_model.recon_loss(z, train_data.pos_edge_label_index.to(device),
                                train_data.neg_edge_label_index.to(device))
    loss.backward(retain_graph=True)
    optimizer.step()
    return float(loss)


@torch.no_grad()
def gae_test(test_data, gae_model):
    gae_model.eval()
    z = gae_model.encode(test_data.x, test_data.edge_index)

    return gae_model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)


def new_recon_loss(vgae_model, z, pos_edge_index, neg_edge_index, edge_weights):
    pos_pred = vgae_model.decoder(z, pos_edge_index, sigmoid=True)
    pos_true = edge_weights  # 此处需要将edge_weights传递给函数
    pos_loss = (edge_weights * (pos_pred - pos_true) ** 2).mean()

    neg_pred = vgae_model.decoder(z, neg_edge_index, sigmoid=True)

    neg_loss = (1 * (neg_pred - 0) ** 2).mean()

    return pos_loss + neg_loss


def vgae_train(train_data, vgae_model, optimizer, device, weight_decay, loss_type):
    vgae_model.train()
    optimizer.zero_grad()
    z = vgae_model.encode(train_data.x, train_data.edge_index)
       # Reconstruction loss + KL divergence loss
    reconstruction_loss = vgae_model.recon_loss(z, train_data.pos_edge_label_index.to(device),train_data.neg_edge_label_index.to(device))
    #reconstruction_loss = new_recon_loss(vgae_model,z, train_data.pos_edge_label_index.to(device),train_data.edge_weights,train_data.neg_edge_label_index.to(device))

    kl_loss = (1 / train_data.num_nodes) * vgae_model.kl_loss()
    if loss_type == "KL+RE+L2":
        # L2 regularization loss
        l2_loss = 0.0
        for param in vgae_model.parameters():
            l2_loss += torch.norm(param, p=2)
        loss = reconstruction_loss + kl_loss + (weight_decay * l2_loss)
        loss.backward(retain_graph=True)
        optimizer.step()
        return float(loss)
        # Total loss with L2 regularization term
    else:
        loss = reconstruction_loss + kl_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        return float(loss)

@torch.no_grad()
def vgae_test(test_data, vgae_model):
    vgae_model.eval()
    z = vgae_model.encode(test_data.x, test_data.edge_index)
    return vgae_model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
# ----------------------------------------------------------------------------------------------------------------------
