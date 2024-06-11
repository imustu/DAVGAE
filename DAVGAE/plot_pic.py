import numpy as np
import torch
import csv
from matplotlib import pyplot as plt
from sklearn import metrics
from chang_print import print_with_style
from data import get_mapping
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.decomposition import PCA




def plot_roc_curve(model, data):
	"""
	可视化模型预测的ROC曲线
	"""
	model.eval()
	x = data.x
	z = model.encode(x, data.edge_index)
	pos_preds = model.decode(z, data.pos_edge_label_index, sigmoid=True)
	neg_preds = model.decode(z, data.neg_edge_label_index, sigmoid=True)
	preds = torch.cat([pos_preds, neg_preds], dim=0)
	preds = preds.detach().cpu().numpy()
	labels = torch.cat((data.pos_edge_label, data.neg_edge_label), dim=0)
	labels = labels.detach().cpu().numpy()
	fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
	# 使用 J-statistic: https://en.wikipedia.org/wiki/Youden%27s_J_statistic
	J = tpr - fpr
	ix = np.argmax(J)
	best_thresh = thresholds[ix]
	print_with_style('Best Threshold=%f' % (best_thresh))
	roc_auc = metrics.roc_auc_score(labels, preds)
	plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
	plt.plot([0, 1], [0, 1], 'r--')  # 随机分类器的对角 roc 曲线
	plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best=%0.2f' % best_thresh)
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc='lower right')
	plt.title('ROC curve for model predictions')
	plt.show()

def calculate_f1_score(model, data, threshold,a_fold,output_path):
	model.eval()
	x = data.x
	z = model.encode(x, data.edge_index)
	pos_preds = model.decode(z, data.pos_edge_label_index, sigmoid=True)
	neg_preds = model.decode(z, data.neg_edge_label_index, sigmoid=True)
	preds = torch.cat([pos_preds, neg_preds], dim=0)
	preds = preds.detach().cpu().numpy()
	labels = torch.cat((data.pos_edge_label, data.neg_edge_label), dim=0)
	labels = labels.detach().cpu().numpy()
	auc1 = metrics.roc_auc_score(labels, preds)
	precision, recall, _ = precision_recall_curve(labels, preds)
	aupr = auc(recall, precision)
	# Convert probabilities to binary predictions using the threshold
	y_pred_binary = [1 if val > threshold else 0 for val in preds]


	metric = np.array((auc1,aupr))
	pos_edges = data.pos_edge_label_index
	neg_edges = data.neg_edge_label_index

#存结果
	with open(f'{output_path}result_fold{a_fold}.csv', 'w',newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['row', 'col', 'y','y_pred'])


		for i in range(len(pos_edges[0])):

			writer.writerow([pos_edges[0][i].item(), pos_edges[1][i].item(), 1,pos_preds[i].item()])

		for i in range(len(neg_edges[0])):

			writer.writerow([neg_edges[0][i].item(), pos_edges[1][i].item(), 0,neg_preds[i].item()])
	return metric
def plot_training_stats(title, losses, test_auc, test_ap, train_auc, train_ap):
	"""
	绘制训练过程中损失和性能指标的变化
	Args
	----
	losses, test_auc, test_ap, train_auc, train_ap: 训练过程的输出列表
	"""
	fig, ax = plt.subplots()
	ax2 = ax.twinx()
	ax.set_xlabel("Training Epochs")
	ax2.set_ylabel("Performance Metric")
	ax.set_ylabel("Loss")
	plt.title(title)
	p1, = ax.plot(losses, "b-", label="training loss")
	p2, = ax2.plot(test_auc, "r-", label="test AUC")
	p3, = ax2.plot(test_ap, "g-", label="test AP")
	p4, = ax2.plot(train_auc, "o-", label="train AUC")
	p5, = ax2.plot(train_ap, "v-", label="train AP")
	plt.legend(handles=[p1, p2, p3, p4, p5])
	plt.show()

def plot_loss_auc(epoch, loss_all, x_label="", y_label="", title="", message="" ,method='roc'):
	fig = plt.Figure(figsize=(12,8), dpi=300)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	handles = []
	mean_loss = None
	# 设置颜色列表
	color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'mistyrose', 'cornflowerblue', 'orange', 'purple', 'pink', 'black']
	for j in range(len(loss_all)):
		if j == 0:
			mean_loss = np.array(loss_all[j])
		else:
			mean_loss = (mean_loss + np.array(loss_all[j])) / 2
		plt.plot([a1 for a1 in range(1, epoch+1)], loss_all[j], color=color_list[j])
		handles.append(f"{message} fold {j+1} ")
	plt.plot([a1 for a1 in range(1, epoch + 1)], mean_loss.tolist(), color=color_list[10])
	handles.append(f"{message} fold mean")
	plt.legend(handles, loc ="lower right")
	plt.show()

def plot_all_roc(models, datas):

	count = 0
	x_mean = None
	y_mean = None
	mean_roc_auc = 0.1
	# 设置颜色列表
	color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'mistyrose', 'cornflowerblue', 'orange',
				  'purple', 'pink']
	for data in datas:
		model = models[count]
		model.eval()
		x = data.x
		z = model.encode(x, data.edge_index)
		pos_preds = model.decode(z, data.pos_edge_label_index, sigmoid=True)
		neg_preds = model.decode(z, data.neg_edge_label_index, sigmoid=True)
		preds = torch.cat([pos_preds, neg_preds], dim=0)
		preds = preds.detach().cpu().numpy()
		labels = torch.cat((data.pos_edge_label, data.neg_edge_label), dim=0)
		labels = labels.detach().cpu().numpy()
		fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
		if count == 0:
			x_mean = np.array(fpr)
			y_mean = np.array(tpr)
		else:
			if x_mean.shape[0] > len(fpr):
				x_mean = x_mean[:len(fpr)]
			elif x_mean.shape[0] < len(fpr):
				fpr = fpr[:x_mean.shape[0]]
			else:
				pass
			if y_mean.shape[0] > len(tpr):
				y_mean = y_mean[:len(tpr)]
			elif y_mean.shape[0] < len(tpr):
				tpr = tpr[:y_mean.shape[0]]
			else:
				pass
			x_mean =  (x_mean + np.array(fpr)) / 2
			y_mean = (y_mean + np.array(tpr)) / 2
		roc_auc = metrics.roc_auc_score(labels, preds)
		mean_roc_auc += roc_auc
		plt.plot(fpr, tpr, color=color_list[count], label='AUC = %0.4f' % roc_auc)
		count += 1
	plt.plot(x_mean, y_mean, color='none', label='Mean AUC = %0.4f' % (mean_roc_auc / 10))
	print('Mean AUC = %0.4f' % (mean_roc_auc / 10))
	plt.plot([0, 1], [0, 1], 'r--')  # 随机分类器的对角 roc 曲线
	plt.xlim(-0.05, 1.05)
	plt.ylim(-0.05, 1.05)
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc='lower right')
	plt.title('ROC curve for model predictions')
	plt.show()
from sklearn.metrics import precision_recall_curve, auc

def plot_all_pr(models, datas):

	count = 0
	precision_mean = None
	recall_mean = None
	mean_pr_auc = 0.0
	color_list = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'mistyrose', 'cornflowerblue', 'orange',
				  'purple', 'pink']

	for data in datas:
		model = models[count]
		model.eval()
		x = data.x
		z = model.encode(x, data.edge_index)
		pos_preds = model.decode(z, data.pos_edge_label_index, sigmoid=True)
		neg_preds = model.decode(z, data.neg_edge_label_index, sigmoid=True)
		preds = torch.cat([pos_preds, neg_preds], dim=0)
		preds = preds.detach().cpu().numpy()
		labels = torch.cat((data.pos_edge_label, data.neg_edge_label), dim=0)
		labels = labels.detach().cpu().numpy()

		precision, recall, _ = precision_recall_curve(labels, preds)

		if count == 0:
			precision_mean = np.array(precision)
			recall_mean = np.array(recall)
		else:
			# You can use suitable interpolation method to equalize precision and recall array lengths if needed
			pass

		pr_auc = auc(recall, precision)
		mean_pr_auc += pr_auc

		plt.plot(recall, precision, color=color_list[count], label='AUPR = %0.4f' % pr_auc)
		count += 1

	plt.plot(recall_mean, precision_mean, color='none', label='Mean AUPR = %0.4f' % (mean_pr_auc / 10))
	print('Mean AUPR = %0.4f' % (mean_pr_auc / 10))
	plt.plot([1, 0], 'r--')  # 虚线
	plt.xlim(-0.05, 1.05)
	plt.ylim(-0.05, 1.05)
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='lower right')
	plt.title('PR curve for model predictions')
	plt.show()
def visualize_tsne_embeddings(model, data, title, data_path, perplexity=30.0, labeled=False, labels=None):
	"""
	使用 t-SNE 在二维空间内可视化节点嵌入
	"""
	if labels is None:
		labels = []
	model.eval()
	x = data.x
	z = model.encode(x, data.edge_index)
	ax1, ax2 = zip(*TSNE(n_components=2, learning_rate='auto', perplexity=perplexity,
						 init='random').fit_transform(z.detach().cpu().numpy()))
	part1 = int(len(ax1) * 0.2)
	part2 = int(len(ax1) * 0.8)
	if part1 + part2 < len(ax1):
		part1 = part1 + 1
	fig = px.scatter(x=ax1, y=ax2, color=['r'] * part1 + ['g'] * part2,
					 hover_data=[get_mapping(data_path)],
					 title=title)
	if labeled:
		for i in labels:
			fig.add_annotation(x=ax1[i], y=ax2[i],
							   text=str(i), showarrow=False)
	fig.show()


def visualize_pca_embeddings(model, data, title, data_path, labeled=False, labels=None):
	"""
	使用 PCA (components=2) 在二维空间内可视化节点嵌入
	"""
	if labels is None:
		labels = []
	model.eval()
	x = data.x
	z = model.encode(x, data.edge_index)
	pca = PCA(n_components=2)
	components = pca.fit_transform(z.detach().cpu().numpy())
	part1 = int(len(components) * 0.2)
	part2 = int(len(components) * 0.8)
	if part1 + part2 < len(components):
		part1 = part1 + 1
	fig = px.scatter(components, x=0, y=1, color=['r'] * part1 + ['g'] * part2,
					 hover_data=[get_mapping(data_path)], title=title)
	if labeled:
		for i in labels:
			fig.add_annotation(x=components[:, 0][i], y=components[:, 1][i],
							   text=str(i), showarrow=False)
	fig.show()