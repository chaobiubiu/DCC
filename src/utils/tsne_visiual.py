import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch.nn.functional
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

marker_types = ['.', 'o', 'v', '^', '<', '>', 's', 'p', 'x', '+']

class tSNE():
    @staticmethod
    def get_tsne_result(X, metric='euclidean', perplexity=30):
        """  Get 2D t-SNE result with sklearn

        :param X: feature with size of N x C
        :param metric: 'cosine', 'euclidean', and so on.
        :param perplexity:  the preserved local structure size
        """
        try:
            from sklearn.manifold.t_sne import TSNE
        except Exception as e:
            from sklearn.manifold._t_sne import TSNE
        tsne = TSNE(n_components=2, metric=metric, perplexity=perplexity, init='pca')
        tsne_X = tsne.fit_transform(X)
        # tsne_X = (tsne_X - tsne_X.min()) / (tsne_X.max() - tsne_X.min())
        return tsne_X

    @staticmethod
    def plot_tsne(tsne_X, labels, domain_labels=None, imgs=None, texts=None, save_name=None, figsize=(10, 10), marker_size=20, label_name=None):
        """ plot t-SNE results. All parameters are numpy format.

        Args:
            tsne_X: N x 2
            labels: N
            domain_labels: N
            imgs: N x 3 x H x W
            save_name: str
            figsize: tuple of figure size
            marker_size: size of markers
        """
        plt.figure(figsize=figsize)
        scatters = []
        if domain_labels is not None:
            # plot each domain with different shape of markers
            domains = np.unique(domain_labels)
            for d in domains:
                idx = domain_labels == d
                x_tmp = imgs[idx] if imgs is not None else None
                text_tmp = texts[idx] if texts is not None else None
                scatter = show_scatter(tsne_X[idx], labels[idx], marker_size=marker_size, marker_type=marker_types[d], imgs=x_tmp, texts=text_tmp)
                scatters.append(scatter)
        else:
            # plot simple clusters of classes with different colors
            show_scatter(tsne_X, labels, marker_size=marker_size, marker_type=marker_types[0], imgs=imgs, texts=texts)

        # plot legend
        each_labels = np.unique(labels)
        legend_elements = []
        for l in each_labels:
            if label_name is not None:
                L = label_name[l]
            else:
                L = str(l)
            if l == 5:
                l = 7
            legend_elements.append(mlines.Line2D([0], [0], marker='o', color='w', label=L, markerfacecolor=plt.cm.Set1(l), markersize=10))
        legend2 = plt.legend(handles=legend_elements, loc='upper right', fontsize=15)
        plt.gca().add_artist(legend2)

        # plt.title()
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0.5)
        if save_name is not None:
            # plt.savefig(save_name, bbox_inches='tight')
            plt.savefig(save_name, format='pdf', dpi=600)
        plt.show()

def show_scatter(tsne_X, label, marker_size=2, marker_type='o', imgs=None, ax=None, texts=None):
    # imgs : 3 x H x W
    # label : N
    # tsne_X : N x 2
    if ax is None:
        ax = plt.gca()

    label = np.array(label)

    color_num = max(np.unique(label)) + 1
    if color_num > 8:
        print('ranbow colors')
        cm = plt.get_cmap('gist_rainbow')
        colors = np.array([cm(1. * i / color_num) for i in range(color_num)])[label]
    else:
        label[label == 5] = 7  # 5太黄了
        colors = plt.cm.Set1(label)
    ret = plt.scatter(tsne_X[:, 0], tsne_X[:, 1], color=colors, s=marker_size, marker=marker_type)

    return ret

# common_latents_early = np.load('D:/code/2021-12-netease/ma_transfer/scripts/results/MPE/simple_spread/mappo+transfer/0124/run1/data/latents_6_11200.npy')
# domain_labels_early = np.load('D:/code/2021-12-netease/ma_transfer/scripts/results/MPE/simple_spread/mappo+transfer/0124/run1/data/domain_labels_6_11200.npy')
#
# common_latents_medium = np.load('D:/code/2021-12-netease/ma_transfer/scripts/results/MPE/simple_spread/mappo+transfer/0124/run1/data/latents_31_51200.npy')
# domain_labels_medium = np.load('D:/code/2021-12-netease/ma_transfer/scripts/results/MPE/simple_spread/mappo+transfer/0124/run1/data/domain_labels_31_51200.npy')
#
# common_latents_later = np.load('D:/code/2021-12-netease/ma_transfer/scripts/results/MPE/simple_spread/mappo+transfer/0124/run1/data/latents_62_100800.npy')
# domain_labels_later = np.load('D:/code/2021-12-netease/ma_transfer/scripts/results/MPE/simple_spread/mappo+transfer/0124/run1/data/domain_labels_62_100800.npy')
#
# common_latents_early = common_latents_early.reshape(4, 1600, -1)
# domain_labels_early = domain_labels_early.reshape(4, 1600, -1)
# common_latents_medium = common_latents_medium.reshape(4, 1600, -1)
# domain_labels_medium = domain_labels_medium.reshape(4, 1600, -1)
# common_latents_later = common_latents_later.reshape(4, 1600, -1)
# domain_labels_later = domain_labels_later.reshape(4, 1600, -1)
#
# common_latents_0 = torch.from_numpy(common_latents_medium[0, :, :])      # domain 0
# common_latents_1 = torch.from_numpy(common_latents_medium[1, :, :])      # domain 1
# common_latents_2 = torch.from_numpy(common_latents_medium[2, :, :])     # domain 2
# common_latents_3 = torch.from_numpy(common_latents_early[3, :, :])      # domain 3
#
# max_dis, min_dis, mean_dis = [], [], []
# for index, domain in enumerate([common_latents_0, common_latents_1, common_latents_2]):
#     common_latents_0_pair = domain.T
#     out1 = torch.matmul(common_latents_0, common_latents_0_pair)
#     out2 = torch.matmul(torch.sqrt(torch.sum(common_latents_0.pow(2), dim=-1, keepdim=True)), torch.sqrt(torch.sum(domain.pow(2), dim=-1, keepdim=True)).T)
#     cos_distance = out1 / out2
#     if index == 0:
#         cos_distance *= (1 - torch.eye(1600))
#
#     max_dis.append(torch.max(cos_distance))
#     min_dis.append(torch.min(cos_distance))
#     mean_dis.append(torch.mean(cos_distance))
# # cos_distance *= (1 - torch.eye(3200))
# print('1', max_dis, min_dis, mean_dis)

'''映射后latent余弦相似度距离可视化'''
# mapping_func = torch.from_numpy(np.random.randn(20, 5)).to(torch.float32)
# map0 = torch.mm(common_latents_0, mapping_func)
# map1 = torch.mm(common_latents_1, mapping_func)
# map2 = torch.mm(common_latents_2, mapping_func)
#
# map_0_pair = map0.T
# map_out1 = torch.matmul(map0, map_0_pair)
# map_out2 = torch.matmul(torch.sqrt(torch.sum(map0.pow(2), dim=-1, keepdim=True)), torch.sqrt(torch.sum(map0.pow(2), dim=-1, keepdim=True)).T)
# map_cos_distance = map_out1 / map_out2
# # map_cos_distance *= (1 - torch.eye(3200))
# print('1', torch.max(map_cos_distance), torch.min(map_cos_distance), torch.mean(map_cos_distance))

'''run13 之前的common_latents以及domain_labels.shape=(batch_size, num_agents, dim)'''
# torch.nn.functional.cosine_similarity()
# TSEN = tSNE()
# embedding = TSEN.get_tsne_result(common_latents_later.reshape(-1, 20))
# TSEN.plot_tsne(embedding, domain_labels_later.reshape(-1, 1).astype(int))

'''随机mapping后得到的三维可视化'''
# mapping_func = np.random.randn(20, 3)
# mapping = np.dot(common_latents_medium.reshape(-1, 20), mapping_func)
# from mpl_toolkits.mplot3d import Axes3D
# # 创建显示的figure
# fig = plt.figure()
# ax = Axes3D(fig)
# # 将数据对应坐标输入到figure中，不同标签取不同的颜色，MINIST共0-9十个手写数字
# ax.scatter(mapping[:, 0], mapping[:, 1], mapping[:, 2],
#            c=plt.cm.Set1(domain_labels_early.reshape(-1, 1).astype(int)))
#
# # 关闭了plot的坐标显示
# # plt.axis('off')
# plt.show()


# 三维可视化
# from sklearn.manifold._t_sne import TSNE
# embedded = TSNE(n_components=3).fit_transform(common_latents_early.reshape(-1, 20))
#
# # 对数据进行归一化操作
# x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
# embedded = embedded / (x_max - x_min)
#
# from mpl_toolkits.mplot3d import Axes3D
# # 创建显示的figure
# fig = plt.figure()
# ax = Axes3D(fig)
# # 将数据对应坐标输入到figure中，不同标签取不同的颜色，MINIST共0-9十个手写数字
# ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
#            c=plt.cm.Set1(domain_labels_early.reshape(-1, 1).astype(int)))
#
# # 关闭了plot的坐标显示
# # plt.axis('off')
# plt.show()
# plt.savefig('test', bbox_inches='tight')


# early
# 1 tensor(0.8777) tensor(-0.8849) tensor(0.0002)    domain 0, 0
# 1 tensor(0.8514) tensor(-0.8638) tensor(-4.3156e-05) domain 0, 1
# 1 tensor(0.8939) tensor(-0.8986) tensor(4.9735e-05) domain 0, 2

# laterly
# 1 tensor(0.8837) tensor(-0.8596) tensor(8.7813e-05) doamin 0, 0
# 1 tensor(0.8808) tensor(-0.8723) tensor(-0.0001) domain 0, 1
# 1 tensor(0.9088) tensor(-0.8697) tensor(-1.7040e-05) domain 0, 2