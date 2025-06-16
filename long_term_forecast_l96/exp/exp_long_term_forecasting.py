# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_learning_rate_new
# from utils.metrics import metric, metric_by_var
# import torch
# import torch.nn as nn
# from torch import optim
# from torch.optim import lr_scheduler 
# import os
# import time
# import warnings
# import numpy as np
# import pandas as pd

# import wandb
# # 스크립트 제일 위쪽 (import 들 아래)
# import sys, io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')




# warnings.filterwarnings('ignore')


# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)

# class Exp_Long_Term_Forecast(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Long_Term_Forecast, self).__init__(args)

#     def _build_model(self):
#         model = self.model_dict[self.args.model].Model(self.args).float()

#         if self.args.use_multi_gpu and self.args.use_gpu:
#             model = nn.DataParallel(model, device_ids=self.args.device_ids)
#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.MSELoss()
#         return criterion

#     def vali(self, vali_data, vali_loader, criterion,is_test = True):
#         total_loss = []
#         total_samples = 0
#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float()

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 # shape 확인 및 인덱싱
#                 if outputs.dim() == 3:
#                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 elif outputs.dim() == 2:
#                     outputs = outputs[:, -self.args.pred_len:]
#                     batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
#                 else:
#                     raise ValueError(f"Unexpected outputs shape: {outputs.shape}")



#                 if self.args.model == 'CARD' and is_test == False:
#                     ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
#                     ratio = torch.tensor(ratio).unsqueeze(-1).to('cuda')
#                     outputs = outputs * ratio
#                     batch_y = batch_y * ratio



#                 pred = outputs#.detach().cpu()
#                 true = batch_y#.detach().cpu()

#                 loss = criterion(pred, true)

#                 total_loss.append(loss.item()*batch_y.shape[0])
#                 total_samples += batch_y.shape[0]
#         total_loss = np.sum(total_loss) /total_samples
#         self.model.train()
#         return total_loss

#     def train(self, setting):
#         train_data, train_loader = self._get_data(flag='train')
#         vali_data, vali_loader = self._get_data(flag='val')
#         test_data, test_loader = self._get_data(flag='test0')



#         path = os.path.join(self.args.checkpoints, setting)
#         if not os.path.exists(path):
#             os.makedirs(path)

#         time_now = time.time()

#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         c = nn.L1Loss()

#         if self.args.lradj == 'TST':
#             train_steps = len(train_loader)
#             scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
#                                             steps_per_epoch = train_steps,
#                                             pct_start = self.args.pct_start,
#                                             epochs = self.args.train_epochs,
#                                             max_lr = self.args.learning_rate)
#         else:
#             scheduler = None
#         if self.args.use_amp:
#             scaler = torch.cuda.amp.GradScaler()

#         for epoch in range(self.args.train_epochs):
#             iter_count = 0
#             train_loss = []

#             self.model.train()
#             epoch_time = time.time()
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
#                 iter_count += 1
#                 model_optim.zero_grad()
#                 batch_x = batch_x.float().to(self.device)

#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

#                         f_dim = -1 if self.args.features == 'MS' else 0
#                         outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                         loss = criterion(outputs, batch_y)
#                         train_loss.append(loss.item())
#                 else:
#                     if self.args.output_attention:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
#                 # shape 확인 및 인덱싱
#                 if outputs.dim() == 3:
#                     outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 elif outputs.dim() == 2:
#                     outputs = outputs[:, -self.args.pred_len:]
#                     batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
#                 else:
#                     raise ValueError(f"Unexpected outputs shape: {outputs.shape}")




#                 if self.args.model == 'CARD':
#                     self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
#                     self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
#                     outputs = outputs *self.ratio
#                     batch_y = batch_y *self.ratio
#                     loss = c(outputs, batch_y)




#                     use_h_loss = False
#                     h_level_range = [4,8,16,24,48,96]
#                     h_loss = None
#                     if use_h_loss:
                        
#                         for h_level in h_level_range:
#                             batch,length,channel = outputs.shape
#                             # print(outputs.shape)
#                             h_outputs = outputs.transpose(-1,-2).reshape(batch,channel,-1,h_level)
#                             h_outputs = torch.mean(h_outputs,dim = -1,keepdims = True)
#                             h_batch_y = batch_y.transpose(-1,-2).reshape(batch,channel,-1,h_level)
#                             h_batch_y = torch.mean(h_batch_y,dim = -1,keepdims = True)
#                             h_ratio = self.ratio[:h_outputs.shape[-2],:]
#                             # print(h_outputs.shape,h_ratio.shape)
#                             h_ouputs_agg = torch.mean(h_outputs,dim = 1,keepdims = True)
#                             h_batch_y_agg = torch.mean(h_batch_y,dim = 1,keepdims = True)

#                             h_outputs = h_outputs*h_ratio
#                             h_batch_y = h_batch_y*h_ratio
#                             h_ouputs_agg *= h_ratio
#                             h_batch_y_agg *= h_ratio

#                             if h_loss is None:
#                                 h_loss  = c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
#                             else:
#                                 h_loss = h_loss + c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
#                         # outputs = 0


#                 else:
#                     loss = criterion(outputs, batch_y)

#                 train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#                 if self.args.use_amp:
#                     scaler.scale(loss).backward()
#                     scaler.step(model_optim)
#                     scaler.update()
#                 else:
#                     if h_loss != 0:
#                         loss = loss + h_loss * 1e-2
#                     loss.backward()
#                     model_optim.step()


#                 if self.args.lradj == 'TST':
#                     adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args, printout=False)
#                     scheduler.step()
#             print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
#             train_loss = np.average(train_loss)


#             if self.args.model == 'CARD':
#                 vali_loss = self.vali(vali_data, vali_loader, c,is_test = False)
#                 test_loss = self.vali(test_data, test_loader, nn.MSELoss(),is_test = True)
#             else:
#                 test_loss = self.vali(vali_data, vali_loader, criterion)

#             print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
#             # test_loss = self.vali(test_data, test_loader, criterion)
#             wandb.log({"Train Loss": train_loss," Vali Loss":vali_loss,"Test loss tmp": test_loss})
#             # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
#             #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
#             early_stopping(vali_loss, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break




#             if self.args.lradj != 'TST': 
#                 adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args)
#             else:
#                 print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


#         best_model_path = path + '/' + 'checkpoint.pth'
#         self.model.load_state_dict(torch.load(best_model_path))

#         return self.model

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')
#         if test:
#             print('loading model')
#             self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

#         preds = []
#         trues = []
#         attn_maps = []  # attention map 저장용 리스트 추가
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)

#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)

#                 # decoder input
#                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
#                 # encoder - decoder
#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         if self.args.output_attention:
#                             outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                         else:
#                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                             attns = None
#                 else:
#                     if self.args.output_attention:
#                         outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, output_attention=True)
#                         # outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                     else:
#                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
#                         attns = None

#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 outputs = outputs.detach().cpu().numpy()
#                 batch_y = batch_y.detach().cpu().numpy()

#                 pred = outputs
#                 true = batch_y

#                 preds.append(pred)
#                 trues.append(true)

#                 # attention map 저장
#                 if attns is not None:
#                     # 예시: 첫 번째 attention map만 저장 (필요시 전체 저장 가능)
#                     attn_map = attns[0].detach().cpu().numpy() if isinstance(attns, (list, tuple)) else attns.detach().cpu().numpy()
#                     attn_maps.append(attn_map)
#                     # 각 배치별 attention map을 파일로 저장 (원하면 주석 해제)
#                     # np.save(os.path.join(folder_path, f"attn_map_{i}.npy"), attn_map)

#                 if i % 20 == 0:
#                     input = batch_x.detach().cpu().numpy()
#                     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
#                     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
#                     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

#         preds = np.array(preds)
#         trues = np.array(trues)
#         print('test shape:', preds.shape, trues.shape)
#         preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
#         print('test shape:', preds.shape, trues.shape)

#         # attention map 전체 저장
#         if len(attn_maps) > 0:
#             attn_maps = np.array(attn_maps)
#             np.save(os.path.join(folder_path, "all_attention_maps.npy"), attn_maps)
#             print(f"Saved attention maps: {attn_maps.shape}")

#         # result save
#         folder_path = './results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         mae, mse, rmse, mape, mspe = metric(preds, trues)

#         wandb.log({"test mae": mae," test mse":mse})
#         print('mse:{}, mae:{}'.format(mse, mae))
#         f = open("result_long_term_forecast.txt", 'a')
#         f.write(setting + "  \n")
#         f.write('mse:{}, mae:{}'.format(mse, mae))
#         f.write('\n')
#         f.write('\n')
#         f.close()


from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, adjust_learning_rate_new
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

import wandb



warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion,is_test = True):
        total_loss = []
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder (tuple 안전 언패킹)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, output_attention=True)
                            if isinstance(result, tuple):
                                outputs, attns = result
                            else:
                                outputs = result
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, output_attention=True)
                        if isinstance(result, tuple):
                            outputs, attns = result
                        else:
                            outputs = result
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # print(f"[VALI DEBUG] outputs.shape after model: {outputs.shape}")
                if outputs.dim() != 3:
                    raise RuntimeError(f"Model output is not 3D! shape: {outputs.shape}")

                f_dim = -1 if self.args.features == 'MS' else 0

                if outputs.dim() == 3:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    # print(f"[VALI DEBUG] outputs.shape: {outputs.shape}, batch_y.shape: {batch_y.shape}")

                    # shape mismatch 방지: 필요시 batch_y permute
                    if outputs.shape != batch_y.shape:
                        # print(f"[VALI WARNING] Shape mismatch! outputs.shape={outputs.shape}, batch_y.shape={batch_y.shape}")
                        # print(f"[Shape Warning] outputs.shape={outputs.shape}, batch_y.shape={batch_y.shape} -> permute batch_y")
                        if outputs.shape[1] == batch_y.shape[2] and outputs.shape[2] == batch_y.shape[1]:
                            # print("[VALI INFO] Permuting batch_y to match outputs shape.")
                            batch_y = batch_y.permute(0, 2, 1)
                        else:
                            # print("[VALI ERROR] Cannot match shapes: outputs.shape={}, batch_y.shape={}".format(outputs.shape, batch_y.shape))
                            continue  # 이 배치는 loss 계산 건너뜀
                
                
                elif outputs.dim() == 2:
                    print(f"[VALI ERROR] outputs is 2D! i={i}, outputs.shape={outputs.shape}, batch_x.shape={batch_x.shape}")
                    # 강제로 batch 차원 복구
                    outputs = outputs.unsqueeze(0)
                    #batch_y = batch_y[-self.args.pred_len:, f_dim:].unsqueeze(0).to(self.device)
                    print(f"[VALI INFO] Fixed outputs.shape: {outputs.shape}, batch_y.shape: {batch_y.shape}")
                    # outputs = outputs[:, -self.args.pred_len:]
                    # batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                else:
                    raise ValueError(f"Unexpected outputs shape: {outputs.shape}")

                # loss 계산 직전
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"[NaN DETECT] step {i}, epoch {epoch}")
                    # 원하는 텐서를 전부 찍어보거나, torch.save 로 덤프
                    torch.set_printoptions(edgeitems=2, linewidth=120)
                    print(outputs[0, :5, :5])
                    raise RuntimeError("NaN in model output")

                if self.args.model == 'CARD' and is_test == False:
                    ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                    ratio = torch.tensor(ratio).unsqueeze(-1).to('cuda')
                    outputs = outputs * ratio
                    batch_y = batch_y * ratio



                pred = outputs#.detach().cpu()
                true = batch_y#.detach().cpu()

                # 디버깅
                # print(f"[VALI DEBUG] pred.shape: {pred.shape}, true.shape: {true.shape}")
                if pred.shape != true.shape:
                    # print(f"[VALI WARNING] Shape mismatch! pred.shape={pred.shape}, true.shape={true.shape}")
                    if pred.shape[1] == true.shape[2] and pred.shape[2] == true.shape[1]:
                        # print("[VALI INFO] Permuting true to match pred shape.")
                        true = true.permute(0, 2, 1)
                    else:
                        raise RuntimeError(f"Cannot match shapes: pred.shape={pred.shape}, true.shape={true.shape}")
                
                loss = criterion(pred, true)

                total_loss.append(loss.item()*batch_y.shape[0])
                total_samples += batch_y.shape[0]
        total_loss = np.sum(total_loss) /total_samples
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test0')



        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        c = nn.L1Loss()

        if self.args.lradj == 'TST':
            train_steps = len(train_loader)
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        else:
            scheduler = None
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                #print("batch_x shape:", batch_x.shape)
                if torch.isnan(batch_x).any():
                    raise RuntimeError("NaN in batch_x")
                if torch.isnan(batch_y).any():
                    raise RuntimeError("NaN in batch_y")
                
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         if self.args.output_attention:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #         else:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                #         f_dim = -1 if self.args.features == 'MS' else 0 # default는 MS
                #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
                #         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #         loss = criterion(outputs, batch_y)
                #         train_loss.append(loss.item())
                # else:
                #     if self.args.output_attention:
                #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #     else:
                #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, output_attention=self.args.output_attention)
                        if self.args.output_attention and isinstance(result, tuple):
                            outputs, attns = result
                        else:
                            outputs = result
                else:
                    result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, output_attention=self.args.output_attention)
                    if self.args.output_attention and isinstance(result, tuple):
                        outputs, attns = result
                    else:
                        outputs = result

                    f_dim = -1 if self.args.features == 'MS' else 0
                    #outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    #batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    #print("model raw outputs shape:", outputs.shape)
                    
                    if outputs.dim() == 3:
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                        # shape mismatch 방지: 필요시 batch_y permute
                        if outputs.shape != batch_y.shape:
                            print(f"[Shape Warning] outputs.shape={outputs.shape}, batch_y.shape={batch_y.shape} -> permute batch_y")
                            if outputs.shape[1] == batch_y.shape[2] and outputs.shape[2] == batch_y.shape[1]:
                                batch_y = batch_y.permute(0, 2, 1)                        
                    
                    
                    elif outputs.dim() == 2:
                        raise RuntimeError(f"outputs is 2D! shape: {outputs.shape} (batch 차원이 사라짐, Model/forward/CARDformer에서 batch 차원이 사라지는 연산이 있는지 확인 필요)")
                        # print("Warning: outputs is 2D, shape:", outputs.shape)
                        # outputs = outputs[:, -self.args.pred_len:]
                        # batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                    else:
                        raise ValueError(f"Unexpected outputs shape: {outputs.shape}")
                    
                    #print("outputs shape:", outputs.shape)
                    #print("batch_y shape:", batch_y.shape)
                    #print("self.model type:", type(self.model))
                    
                    # loss 계산 직전
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"[NaN DETECT] step {i}, epoch {epoch}")
                        # 원하는 텐서를 전부 찍어보거나, torch.save 로 덤프
                        torch.set_printoptions(edgeitems=2, linewidth=120)
                        print(outputs[0, :5, :5])
                        raise RuntimeError("NaN in model output")


                    if self.args.model == 'CARD':
                        self.ratio = np.array([max(1/np.sqrt(i+1),0.0) for i in range(self.args.pred_len)])
                        self.ratio = torch.tensor(self.ratio).unsqueeze(-1).to('cuda')
                        outputs = outputs *self.ratio
                        batch_y = batch_y *self.ratio
                        loss = c(outputs, batch_y)




                        use_h_loss = False
                        h_level_range = [4,8,16,24,48,96]
                        h_loss = None
                        if use_h_loss:
                            
                            for h_level in h_level_range:
                                batch,length,channel = outputs.shape
                                # print(outputs.shape)
                                h_outputs = outputs.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_outputs = torch.mean(h_outputs,dim = -1,keepdims = True)
                                h_batch_y = batch_y.transpose(-1,-2).reshape(batch,channel,-1,h_level)
                                h_batch_y = torch.mean(h_batch_y,dim = -1,keepdims = True)
                                h_ratio = self.ratio[:h_outputs.shape[-2],:]
                                # print(h_outputs.shape,h_ratio.shape)
                                h_ouputs_agg = torch.mean(h_outputs,dim = 1,keepdims = True)
                                h_batch_y_agg = torch.mean(h_batch_y,dim = 1,keepdims = True)


                                h_outputs = h_outputs*h_ratio
                                h_batch_y = h_batch_y*h_ratio

                                h_ouputs_agg *= h_ratio
                                h_batch_y_agg *= h_ratio

                                if h_loss is None:
                                    h_loss  = c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                                else:
                                    h_loss = h_loss + c(h_outputs, h_batch_y)*np.sqrt(h_level) /2 +c(h_ouputs_agg, h_batch_y_agg)*np.sqrt(h_level) /2
                            # outputs = 0


                    else:
                        loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    # ──── ✨ 역전파 구간을 anomaly-detect 블록으로 감싸기 ✨ ────
                with torch.autograd.set_detect_anomaly(True):        # <-- 추가 (①)
                    if self.args.use_amp:
                        scaler.scale(loss).backward()                # (AMP 사용 시)
                    else:
                        loss.backward()                              # (일반 FP32)

                # ───────────────── optimizer step ─────────────────────────
                if self.args.use_amp:
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    model_optim.step()
                # if self.args.use_amp:
                #     scaler.scale(loss).backward()
                #     scaler.step(model_optim)
                #     scaler.update()
                # else:
                #     if h_loss != 0:
                #         loss = loss #+ h_loss * 1e-2
                #     loss.backward()
                #     model_optim.step()


                if self.args.lradj == 'TST':
                    adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)


            if self.args.model == 'CARD':
                vali_loss = self.vali(vali_data, vali_loader, c,is_test = False)
                test_loss = self.vali(test_data, test_loader, nn.MSELoss(),is_test = True)
            else:
                test_loss = self.vali(vali_data, vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            # test_loss = self.vali(test_data, test_loader, criterion)
            wandb.log({"Train Loss": train_loss," Vali Loss":vali_loss,"Test loss tmp": test_loss})
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break




            if self.args.lradj != 'TST': 
                adjust_learning_rate_new(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # train 잘 되는지 확인하는 plot 코드
    def plot_train_predictions(self, setting, num_batches=1):
        print("plot_train_predictions called!") 
        train_data, train_loader = self._get_data(flag='train')
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                if outputs.dim() == 3:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if outputs.shape != batch_y.shape:
                        print(f"[Shape Warning] outputs.shape={outputs.shape}, batch_y.shape={batch_y.shape} -> permute batch_y")
                        if outputs.shape[1] == batch_y.shape[2] and outputs.shape[2] == batch_y.shape[1]:
                            batch_y = batch_y.permute(0, 2, 1)            
                elif outputs.dim() == 2:
                    outputs = outputs[:, -self.args.pred_len:]
                    batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                if i+1 >= num_batches:
                    break
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print("preds shape:", preds.shape)
        print("trues shape:", trues.shape)
        plt.figure(figsize=(10,5))
        # shape에 따라 인덱싱 다르게
        if preds.ndim == 3:
            plt.plot(trues[0,:,0], label='True')
            plt.plot(preds[0,:,0], label='Pred')
        elif preds.ndim == 2:
            plt.plot(trues[0,:], label='True')
            plt.plot(preds[0,:], label='Pred')
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")
        plt.title('Train Prediction vs True')
        plt.legend()
        plt.savefig(f'./train_pred_plot.png')
        plt.close()
        
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        attn_maps = []  # attention map 저장용 리스트 추가
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, output_attention=True)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            attns = None
                else:
                    if self.args.output_attention:
                        outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, output_attention=True)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        attns = None

                f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                if outputs.dim() == 3:
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    # shape mismatch 방지: 필요시 batch_y permute
                    if outputs.shape != batch_y.shape:
                        print(f"[Shape Warning] outputs.shape={outputs.shape}, batch_y.shape={batch_y.shape} -> permute batch_y")
                        if outputs.shape[1] == batch_y.shape[2] and outputs.shape[2] == batch_y.shape[1]:
                            batch_y = batch_y.permute(0, 2, 1)
                
                elif outputs.dim() == 2:
                    outputs = outputs[:, -self.args.pred_len:]
                    batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                else:
                    raise ValueError(f"Unexpected outputs shape: {outputs.shape}")
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                
                # --- attention map 저장 ---
                # if attns is not None:
                #     # 예시: 첫 번째 레이어의 attention map만 저장 (필요시 전체 저장)
                #     # attns shape 예시: (batch, heads, seq_len, seq_len) or (batch, heads, channels, channels)
                #     attn_map = attns[0].detach().cpu().numpy() if isinstance(attns, (list, tuple)) else attns.detach().cpu().numpy()
                #     attn_maps.append(attn_map)
                if attns is not None:
                    attn_maps_channel, attn_maps_token = attns  # tuple로 언패킹
                    # attn_maps_token이 list라면, 예를 들어 첫 번째 레이어의 token attention map만 저장
                    # attn_maps_channel -> 32개 channel 다 나옴
                    if isinstance(attn_maps_channel, list) and len(attn_maps_channel) > 0:
                        attn_map = attn_maps_channel[0].detach().cpu().numpy()
                        attn_maps.append(attn_map)
                    
                # --- pdf 저장 ---
                if i % 200 == 0:
                    input = batch_x.detach().cpu().numpy()
                    # shape이 (seq_len, feature)일 때
                    for ch in range(input.shape[2]):
                        # gt = np.concatenate((input[0, :, ch].reshape(-1), true[0, :, ch].reshape(-1)), axis=0)
                        # pd = np.concatenate((input[0, :, ch].reshape(-1), pred[0, :, ch].reshape(-1)), axis=0)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        # 컬럼명 가져오기 (없으면 ch 인덱스만 사용)
                        if hasattr(self.args, "target_columns"):
                            col_name = self.args.target_columns[ch]
                        else:
                            col_name = f"ch{ch}"
                        fname = f"{i:04d}_ch{ch:02d}_{col_name}.pdf"
                        visual(gt, pd, os.path.join(folder_path, fname))
            
            # --- attention map npy로 저장 ---
            if len(attn_maps) > 0:
                attn_maps = np.array(attn_maps)  # shape: (batch, ...)
                
                np.save(os.path.join(folder_path, "attention_maps.npy"), attn_maps)
                # 채널명 저장 (args.target_columns가 있다고 가정)
                if hasattr(self.args, "target_columns"):
                    import json
                    with open(os.path.join(folder_path, "channel_names.json"), "w", encoding="utf-8") as f:
                        json.dump(self.args.target_columns, f, ensure_ascii=False, indent=2)
                print(f"Saved attention maps: {attn_maps.shape}")

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        #wandb.log({"test mae": mae," test mse":mse})
        print('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}'.format(mse, mae, rmse, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return