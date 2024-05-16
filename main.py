import csv
import os
import sys
import time
import pandas as pd
import torch
import argparse
import random
import numpy as np
from colorama import init, Fore, Style
from torch_geometric.nn import VGAE, GAE
from data import initialize_data, flag_data
import torch_geometric.transforms as T
from chang_print import print_with_style
from torch_geometric.data import download_url, extract_gz
from models import GCNEncoder, VariationalGCNEncoder
from plot_pic import calculate_f1_score,plot_loss_auc,plot_all_roc,plot_all_pr
from utils import gae_train, gae_test, vgae_train, vgae_test
from RAdam.radam.radam import RAdam

def main(args):

	device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
	if args.model_type == "GAE":  # Model selection
		model = GAE(GCNEncoder(args.NUM_FEATURES, args.HIDDEN_SIZE, args.OUT_CHANNELS, dropout=args.dropout))
		model = model.to(device)
	else:
		model = VGAE(
			VariationalGCNEncoder(args.NUM_FEATURES, args.HIDDEN_SIZE, args.OUT_CHANNELS, dropout=args.dropout,heads=args.HEADS))
	model = model.to(device)
	# ------------------------------Read data and construct Data object.------------------------------------------------

	# 注意列的名字是 ["# Disease ID", "Disease Name", "Gene ID"].
	df = pd.read_csv(args.data_path, sep="\t")
	print_with_style(f"\n{df.head()}", color="cyan")
	datas, gene_mapping, dz_mapping = flag_data(args.data_path, args.NUM_FEATURES)
	print_with_style(f"Number of genes:{len(gene_mapping)}", color='cyan')
	print_with_style(f"Number of diseases:{len(dz_mapping)}", color='cyan')
	loss_all = []
	auc_all = []
	dataset_all = []
	models = []
	for a_fold in range(1, 11):
		print_with_style(f"==========fold{a_fold}===========")
		if a_fold == 1:
			print_with_style(f"\n{model}", color="cyan")
		train_data, test_data = datas[a_fold-1]
		#train_edge_weights=datas_weight[a_fold-1]
		transform_train = T.Compose([

			T.ToDevice(device),
			T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True,
							  split_labels=True, add_negative_train_samples=True),
		])
		transform_test = T.Compose([

			T.ToDevice(device),
			T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True,
							  split_labels=True, add_negative_train_samples=True),
		])
		train_dataset, _, _ = transform_train(train_data)  # Train data set selection
		test_dataset, _, _  = transform_test(test_data)  # Test data set selection
		print(f"train_dataset:\n{train_dataset}")
		print(f"test_dataset:\n{test_dataset}")
		#np.savetxt(r'sample/test_pos_edge_label_index_{}.txt'.format(a_fold), test_dataset.pos_edge_label_index.transpose(1, 0).cpu().numpy(), fmt='%d')
		#np.savetxt(r'sample/train_pos_edge_label_index_{}.txt'.format(a_fold), train_dataset.pos_edge_label_index.transpose(1, 0).cpu().numpy(), fmt='%d')
		# ------------------------------------------------------------------------------------------------------------------
		losses = []
		test_auc = []
		test_ap = []
		train_aucs = []
		train_aps = []
		cur_lr = args.learning_rate
		if args.optimizer_type == "AMSGrad":
			optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
										 amsgrad=True)
		elif args.optimizer_type == "Adam":
			optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr)
		elif args.optimizer_type == "RAdam":
			optimizer = RAdam(model.parameters())
		else:
			optimizer = torch.optim.AdamW(model.parameters(), lr=cur_lr, amsgrad=True)
		time.sleep(1)  # To avoid showing errors, sleep for 1 second
		best_loss = 9999999
		best_model = model
		for epoch in range(1, args.EPOCHS + 1):
			if args.model_type == "GAE":
				if args.adjust_lr == "True":
					cur_lr = adjust_lr(optimizer, epoch / args.EPOCHS, args.learning_rate)
					optimizer.lr = cur_lr
					loss = gae_train(train_dataset, model, optimizer, device)
					auc, ap = gae_test(test_dataset, model)
					train_auc, train_ap = gae_test(train_dataset, model)
				else:
					loss = gae_train(train_dataset, model, optimizer, device)
					auc, ap = gae_test(test_dataset, model)
					train_auc, train_ap = gae_test(train_dataset, model)

			else:
				if args.adjust_lr == "True":
					cur_lr = adjust_lr(optimizer, epoch / args.EPOCHS, args.learning_rate)
					optimizer.lr = cur_lr
					loss = vgae_train(train_dataset, model, optimizer, device, args.weight_decay, args.loss_type)
					auc, ap = vgae_test(test_dataset, model)
					train_auc, train_ap = vgae_test(train_dataset, model)
				else:
					loss = vgae_train(train_dataset, model, optimizer, device, args.weight_decay, args.loss_type)
					auc, ap = vgae_test(test_dataset, model)
					train_auc, train_ap = vgae_test(train_dataset, model)
			# Determine which round has the smallest loss
			if best_loss > loss:
				best_loss = loss

				best_model = model

			losses.append(loss)
			test_auc.append(auc)
			test_ap.append(ap)
			train_aucs.append(train_auc)
			train_aps.append(train_ap)
			loss_all.append(losses)
			auc_all.append(test_auc)
			dataset_all.append(test_dataset)
			models.append(best_model)

		text1 = 'Epoch: {:03d}, test AUC: {:.4f}, test AP: {:.4f}, loss:{:.4f}'.format(epoch, auc, ap, loss)
		print(f"{Fore.MAGENTA}{Style.BRIGHT}{text1}{Style.RESET_ALL}")



		metric = calculate_f1_score(best_model, test_dataset, 0.5, a_fold, args.output_path)
		print("第")
		print(a_fold)
		print("折各个指标auc,aupr:", metric)
		f = open(args.output_path + args.out_metrix, "a")  # 改
		f.write(",".join(map(str, metric)) + "\n")
		f.close()
		# Save the loss of each round of each fold. The i-th column is the loss of the i-th fold, and each row means one round.
		with open(f'{args.output_path}losses.csv', 'w', newline='') as f:
			writer = csv.writer(f)
			for epoch in range(args.EPOCHS):
				row = []
				for i in range(10):
					row.append(loss_all[i][epoch])
				writer.writerow(row)


	print_with_style('Program execution completed', color='cyan')

def adjust_lr(optimizer, decay_ratio, lr):
	lr_ = lr * (1 - decay_ratio)
	lr_min = lr * 0.01
	if lr_ < lr_min:
		lr_ = lr_min
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr_
	return lr_
if __name__ == "__main__":
	parse = argparse.ArgumentParser()
	parse.add_argument("--model-type", default="VGAE", type=str, help="GAE or VGAE")
	parse.add_argument("--loss-type", default="KL+RE+L2", type=str, help="KL+RE or KL+RE+L2")  # 选择VGAE的损失
	parse.add_argument("--optimizer-type", default="AdamW", type=str, help="Adam or AdamW or AMSGrad or RADAM")
	parse.add_argument("--data-path", default="data/datasetB/without_dataAugmentation.csv", type=str, help="Path to input data")
	parse.add_argument("--output-path", default="test_out/datasetB/", type=str,help="Path to output data")
	parse.add_argument("--adjust-lr", default=True, type=bool, help="select adaptive learning rate")
	parse.add_argument("--learning-rate", default=0.001, type=float, help="learning rate")
	parse.add_argument("--dropout", default=0.3, type=float, help="")
	#parse.add_argument("--show-embeddings", default=False, type=bool, help="是否显示tsne和pca编码")
	parse.add_argument("--NUM-FEATURES", default=100, type=int, help="")
	parse.add_argument("--weight_decay", default=0.0001, type=float, help="hyperparameters for L2 regularization strength")
	parse.add_argument("--HIDDEN-SIZE", default=128, type=int, help="")
	parse.add_argument("--OUT-CHANNELS", default=32, type=int, help="")
	parse.add_argument("--HEADS", default=2, type=int, help="")
	parse.add_argument("--EPOCHS", default=100, type=int, help="")
	parse.add_argument("--out_metrix", default="result_metrix_datasetB.txt", type=str, help="")
	args = parse.parse_args()

	output_path = args.output_path
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	log_file = os.path.join(output_path, 'log.txt')
	with open(log_file, 'w') as f:
		f.write(str(args))

	main(args)