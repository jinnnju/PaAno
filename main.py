import os
import time
import random
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
from model import PatchEncoder
from train import train_model
from utils.data_preprocess import *
from utils.utils import *
from utils.evaluation import *
from utils.metrics import get_metrics
warnings.simplefilter("ignore", category=UndefinedMetricWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

class AnomalyDetection:
    def __init__(self, data_dir, output_dir=None, patch_size=64,
                 num_iters=None, lr=1e-4, batch_size=512, random_seed = 2000, device=None, see_loss=False, use_revin=False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.num_iters = num_iters
        self.lr = lr
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.see_loss = see_loss
        self.use_revin = use_revin

        directory_name = os.path.basename(self.data_dir.rstrip('/'))

    def run(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)

        self.dis_aurocs = []
        self.dis_auprcs = []
        self.dis_vuspr = []
        self.dis_vusroc = []
        self.dis_f1 = []
        self.dis_Rfl = []
        self.results_by_category = defaultdict(list)
        self.summary_rows = []
        
        total_csv_files = len([f for f in os.listdir(self.data_dir) if f.endswith('.csv')])
        print(f"PaAno is running... (found {total_csv_files} files to detect)")
        current_idx = 1

    
        for file_name in sorted(os.listdir(self.data_dir)):
            if not file_name.endswith('.csv'):
                print(f"Skipping file (not a .csv file): {file_name}")
                continue

            file_path = os.path.join(self.data_dir, file_name)
            print(f"\033[1m══ Running on file ({current_idx}/{total_csv_files})\033[0m : {file_name}")
            current_idx += 1

            train_data, train_labels, test_data, test_labels = load_and_split_data(file_path)
            train_data = np.array(train_data, dtype=np.float32)
            test_data = np.array(test_data, dtype=np.float32)
            test_labels = np.array(test_labels, dtype=np.float32)

            train_mean = np.mean(train_data, axis=0, keepdims=True).astype(np.float32)
            train_std  = np.std(train_data,  axis=0, keepdims=True).astype(np.float32)
            train_std = np.where(train_std == 0.0, 1e-8, train_std)

            if self.use_revin == False: 
                train_data = (train_data - train_mean) / train_std
                test_data = (test_data - train_mean) / train_std
            
            full_data = np.concatenate([train_data, test_data], axis=0)
            full_labels = np.concatenate([train_labels, test_labels], axis = 0)

            if full_data.ndim == 1:
                sliding_input = full_data.reshape(-1, 1)
            else:
                sliding_input = full_data[:, 0].reshape(-1, 1)

            slidingWindow = find_length_rank(sliding_input, rank=1)
                       
            t0 = time.time()
            patch_creator = PatchCreator(L=self.patch_size, s=1, random_seed=self.random_seed)

            train_loader, test_loader, true_test_labels = patch_creator.create_dataloaders(
                train_data, full_data, full_labels, batch_size=self.batch_size)

            xb, _ = next(iter(train_loader))
            C = xb.shape[1]
            print(f"[init] inferred in_channels = {C}")
            model = PatchEncoder(in_channels=C, use_revin=self.use_revin).to(self.device)
            # total_params = sum(p.numel() for p in model.parameters()) 
            # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
            # print(f"Total params: {total_params:,}") 
            # print(f"Trainable params: {trainable_params:,}")
         
         
            train_model(
                model, 
                train_loader, 
                preprocess_to_patches(train_data, patch_size=self.patch_size, stride=1), 
                self.device, 
                num_iter=self.num_iters, 
                pretext_step=self.patch_size, 
                lr=self.lr, 
                see_loss=self.see_loss
            )
           
            model_path = 'trained_encoder.pth'
            torch.save(model.state_dict(), model_path)
            model.load_state_dict(torch.load(model_path))
            memory_bank, _ = create_memory_bank(model, train_loader, self.device, num_cores=0.1)
        
           
            all_scores = calculate_anomaly_scores(model, test_loader, memory_bank, top_k=3, device=self.device)
         

            dist_scores = distribute_patch_scores_to_points(all_scores, patch_size=self.patch_size, num_points=len(full_labels))    
            print(f"    >> Anomaly detection completed. Calculating the score...")      
            t1 = time.time()
          



            results = get_metrics(dist_scores, full_labels, slidingWindow=slidingWindow, pred=None, version='opt', thre=250)

            self.dis_aurocs.append(results['AUC-ROC'])
            self.dis_auprcs.append(results['AUC-PR'])
            self.dis_vuspr.append(results['VUS-PR'])
            self.dis_vusroc.append(results['VUS-ROC'])
            self.dis_f1.append(results['Standard-F1'])
            self.dis_Rfl.append(results['R-based-F1'])

            category = file_name.split('_')[1]  
            self.results_by_category[category].append(results)

            print(f"    [Anomaly Detection Results]")
            print(f"    >> AUC-ROC: {results['AUC-ROC']:.4f}, AUC-PR: {results['AUC-PR']:.4f}, "
                  f"VUS-PR: {results['VUS-PR']:.4f}, VUS-ROC: {results['VUS-ROC']:.4f}, "
                  f"BestF1: {results['Standard-F1']:.4f}, RangeF1: {results['R-based-F1']:.4f}")
            
            #summary_rows 
            if self.output_dir:
                self.summary_rows.append({
                    'file': file_name,
                    'Category': category,
                    'AUC-ROC': results['AUC-ROC'],
                    'AUC-PR': results['AUC-PR'],
                    'VUS-PR': results['VUS-PR'],
                    'VUS-ROC': results['VUS-ROC'],
                    'BestF1': results['Standard-F1'],
                    'RangeF1': results['R-based-F1']
                })

                scores_dir = os.path.join(self.output_dir, "Filewise_scores")
                os.makedirs(scores_dir, exist_ok=True)
                output_file_path = os.path.join(scores_dir, f"{file_name}_output.csv")

                df = pd.DataFrame({
                    'True Labels': full_labels,
                    'Anomaly scores': dist_scores,
                })
                df.to_csv(output_file_path, index=False)

                print(f"    >> Saved results to {output_file_path}")

        
        dist_auroc, dist_auprc, dist_vuspr, dist_vusroc, dist_F1, dist_RF1 = map(np.mean, [self.dis_aurocs, self.dis_auprcs, self.dis_vuspr, self.dis_vusroc, self.dis_f1, self.dis_Rfl])
        print(f"PaAno's Averaged Final Results: AUROC={dist_auroc:.4f}, AUPRC={dist_auprc:.4f}, VUSPR={dist_vuspr:.4f}, VUSROC={dist_vusroc:.4f}, F1-Score={dist_F1:.4f}, RangeF1 = {dist_RF1:.4f} " )

      

    
        if self.output_dir and self.summary_rows:
            summary_df = pd.DataFrame(self.summary_rows)

           
            summary_df = pd.concat([summary_df, pd.DataFrame([{
                'file': 'AVERAGE',
                'AUC-ROC': float(dist_auroc),
                'AUC-PR':  float(dist_auprc),
                'VUS-PR':  float(dist_vuspr),
                'VUS-ROC': float(dist_vusroc),
                'BestF1':  float(dist_F1),
                'RangeF1': float(dist_RF1),

              
                'TrainTime(s)': summary_df['TrainTime(s)'].mean(),
                'BankTime(s)': summary_df['BankTime(s)'].mean(),
                'ScoreTime(s)': summary_df['ScoreTime(s)'].mean(),
                'DistributeTime(s)': summary_df['DistributeTime(s)'].mean(),
                'TotalTime(s)': summary_df['TotalTime(s)'].mean(),
            }])], ignore_index=True)

            summary_path = os.path.join(self.output_dir, 'summary_metrics.csv')
            summary_df.to_csv(summary_path, index=False)

            
            cat_df = (
                summary_df.groupby('Category').agg(
                    N=('file', 'count'),
                    AUC_ROC=('AUC-ROC','mean'),
                    AUC_PR=('AUC-PR','mean'),
                    VUS_PR=('VUS-PR','mean'),
                    VUS_ROC=('VUS-ROC','mean'),
                    BestF1=('BestF1','mean'),
                    RangeF1=('RangeF1','mean'),
                    TrainTime_s=('TrainTime(s)','mean'),
                    BankTime_s=('BankTime(s)','mean'),
                    ScoreTime_s=('ScoreTime(s)','mean'),
                    DistributeTime_s=('DistributeTime(s)','mean'),
                    TotalTime_s=('TotalTime(s)','mean'),
                ).reset_index()
            )

            cat_df.to_csv(os.path.join(self.output_dir, 'categorical_metrics.csv'), index=False)
            print(f">> Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run PaAno Anomaly Detection")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--num_iters', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--see_loss', dest='see_loss', action='store_true')
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--use_revin', action='store_true', help='Use RevIN')

    args = parser.parse_args()

    experiment = AnomalyDetection(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        lr=args.lr,
        see_loss=args.see_loss,
        random_seed=args.seed,
        use_revin=args.use_revin
    )
    experiment.run()
