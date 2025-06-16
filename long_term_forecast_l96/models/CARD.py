import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)




class Model(nn.Module):
    def __init__(self, config, **kwargs):
        
        super().__init__()    
        self.model = CARDformer(config)
        self.task_name = config.task_name
        
    def forward(self, x, *args, output_attention=False, **kwargs):
        x = x.permute(0,2,1)
        mask = args[-1]
        out = self.model(x, mask=mask, output_attention=output_attention)  # output_attention 전달
        #out = self.model(x, mask=mask)
    
        if isinstance(out, tuple):
            if len(out) == 3:
                x, attn_maps_channel, attn_maps_token = out
                attns = (attn_maps_channel, attn_maps_token)
            else:
                x, attns = out
        else:
            x = out
        
        # if self.task_name != 'classification':
        #     x = x.permute(0,2,1)
            
        # # 항상 3차원 반환 (batch, seq, feature)
        # if x.ndim == 2:
        #     x = x[None, ...]  # (seq, feature) -> (1, seq, feature)
        # elif x.ndim == 1:
        #     x = x[None, :, None]
        
        # if output_attention:
            
        #     return x, attns
        # else:
        #     return x
        
        # if self.task_name != 'classification':
        #     if x.ndim == 3:
        #         x = x.permute(0, 2, 1)  # (batch, pred_len, feature)
        #     else:
        #         raise RuntimeError(f"Model output shape {x.shape} is not 3D after CARDformer, batch dimension is missing!")

        # if output_attention:
        #     return x, attns
        # else:
        #     return x
        
        # print("Model.forward return x shape(before fix):", x.shape)
        
        # --- batch 차원 보장 ---
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (seq, feature) -> (1, seq, feature)
        elif x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.ndim != 3:
            raise RuntimeError(f"Model output shape {x.shape} is not 3D after CARDformer, batch dimension is missing!")

        # print("Model.forward return x shape(after fix):", x.shape)
        
        if output_attention:
            return x, attns
        else:
            return x
    
    
    
    
    
class CARDformer(nn.Module):
    def __init__(self, 
                 config,**kwargs):
        
        super().__init__()
        
        self.patch_len  = config.patch_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.task_name = config.task_name
        patch_num = int((config.seq_len - self.patch_len)/self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num,config.d_model)*1e-2)
        self.model_token_number = 0
        
        if self.model_token_number > 0:
            self.model_token = nn.Parameter(torch.randn(config.enc_in,self.model_token_number,config.d_model)*1e-2)
        
        
        self.total_token_number = (self.patch_num  + self.model_token_number + 1)
        config.total_token_number = self.total_token_number
             
        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)  
        self.input_dropout  = nn.Dropout(config.dropout) 
        
                
        self.use_statistic = config.use_statistic
        self.W_statistic = nn.Linear(2,config.d_model) 
        self.cls = nn.Parameter(torch.randn(1,config.d_model)*1e-2)
        
        
        
        if config.task_name == 'long_term_forecast' or config.task_name == 'short_term_forecast':
            self.W_out = nn.Linear((patch_num+1+self.model_token_number)*config.d_model, config.pred_len) 
        elif config.task_name == 'imputation' or config.task_name == 'anomaly_detection':
            self.W_out = nn.Linear((patch_num+1+self.model_token_number)*config.d_model, config.seq_len) 
        elif config.task_name == 'classification':
            self.W_out = nn.Linear(config.d_model*config.enc_in, config.num_class)

     
        
        
        self.Attentions_over_token = nn.ModuleList([Attenion(config) for i in range(config.e_layers)])
        self.Attentions_over_channel = nn.ModuleList([Attenion(config,over_hidden = True) for i in range(config.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(config.d_model,config.d_model)  for i in range(config.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(config.dropout)  for i in range(config.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2)) for i in range(config.e_layers)])       
            
        
    def forward(self, z, *args, output_attention=False, **kwargs):     
        b, c, s = z.shape
        # print(f"[DEBUG] 입력 z shape: {z.shape} (b={b}, c={c}, s={s})")  # ★ 입력 feature 개수 확인

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'anomaly_detection':
            z_mean = torch.mean(z, dim=(-1), keepdims=True)
            eps = 1e-6
            
            z_std = torch.std(z, dim=(-1), keepdims=True).clamp(min=eps)
            
            z = (z - z_mean) / (z_std + 1e-4)
        
        elif self.task_name == 'imputation':     
            mask = kwargs['mask'].permute(0, 2, 1) 
            z_mean = torch.sum(z, dim=-1) / torch.sum(mask == 1, dim=-1)
            z_mean = z_mean.unsqueeze(-1)
            z = z - z_mean
            z = z.masked_fill(mask == 0, 0)
            z_std = torch.sqrt(torch.sum(z * z, dim=-1) / torch.sum(mask == 1, dim=-1) + 1e-5)
            z_std = z_std.unsqueeze(-1)
            z /= z_std + 1e-4

        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   
        #print("zcube shape:", zcube.shape)   
        # print(f"[DEBUG] zcube shape (after unfold): {zcube.shape}")  # ★ unfold 후 shape 확인           
        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embed 
        #print("z_embed shape:", z_embed.shape)
        # print(f"[DEBUG] z_embed shape (after projection): {z_embed.shape}")  # ★ projection 후 shape 확인

        if self.use_statistic:
            
            z_stat = torch.cat((z_mean, z_std), dim=-1)
            if z_stat.shape[-2] > 1:
                z_stat = (z_stat - torch.mean(z_stat, dim=-2, keepdims=True)) / (torch.std(z_stat, dim=-2, keepdims=True) + 1e-4)
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2), z_embed), dim=-2) 
        
        else:
            cls_token = self.cls.repeat(z_embed.shape[0], z_embed.shape[1], 1, 1)
            z_embed = torch.cat((cls_token, z_embed), dim=-2) 

        inputs = z_embed
        b, c, t, h = inputs.shape  
        print(f"[DEBUG] inputs shape (최종 채널 수 c): {inputs.shape} (b={b}, c={c}, t={t}, h={h})")  # ★ 실제 attention에 들어가는 채널 수
        
        attn_maps_channel = []  # 채널 간 attention map 저장용 리스트
        attn_maps_token = []    # 토큰(patch) 간 attention map 저장용 리스트
        
        for a_2, a_1, mlp, drop, norm in zip(self.Attentions_over_token, self.Attentions_over_channel, self.Attentions_mlp, self.Attentions_dropout, self.Attentions_norm):
            # 채널 간 attention map 저장
            output_1 = a_1(inputs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            
            if hasattr(a_1, 'last_attn_map'):
                attn_maps_channel.append(a_1.last_attn_map_hidden)
                print(f"[DEBUG] attn_maps_channel shape: {a_1.last_attn_map_hidden.shape}")
            
            else:
                attn_maps_channel.append(None)
            
            # 토큰 간 attention map 저장
            output_2 = a_2(output_1)
            
            if hasattr(a_2, 'last_attn_map'):
                attn_maps_token.append(a_2.last_attn_map)
            
            else:
                attn_maps_token.append(None)
            
            outputs = drop(mlp(output_1 + output_2)) + inputs
            outputs = norm(outputs.reshape(b * c, t, -1)).reshape(b, c, t, -1) 
            inputs = outputs

        # if self.task_name != 'classification':
        #     #print("outputs before W_out:", outputs.shape)
        #     z_out = self.W_out(outputs.reshape(b, c, -1))  # (batch, feature, pred_len)
        #     #print("z_out after W_out:", z_out.shape)
        #     z = z_out * (z_std + 1e-4) + z_mean  # (batch, feature, pred_len)
        #     z = z.permute(0, 2, 1)  # (batch, pred_len, feature)
        #     #print("CARDformer return z shape:", z.shape)
        if self.task_name != 'classification':
            # print("outputs before W_out:", outputs.shape)
            # print(f"[DEBUG] outputs.shape before reshape: {outputs.shape}, b={b}, c={c}")
            z_out = self.W_out(outputs.reshape(b, c, -1).contiguous())  # (batch, feature, pred_len)
            # print(f"[DEBUG] z_out.shape after W_out: {z_out.shape}")
            # print("z_out shape:", z_out.shape)
            # print("z_mean shape:", z_mean.shape)
            # print("z_std shape:", z_std.shape)
            
            # z_mean, z_std shape 맞추기
            if z_mean.shape[-1] == 1:
                z_mean = z_mean.expand(-1, -1, z_out.shape[-1])
            if z_std.shape[-1] == 1:
                z_std = z_std.expand(-1, -1, z_out.shape[-1])
            
            # shape가 다르면 permute로 맞추기
            if z_out.shape != z_mean.shape:
                # print("Shape mismatch! z_out:", z_out.shape, "z_mean:", z_mean.shape)
                
                # 예를 들어 z_out (b, pred_len, feature), z_mean (b, feature, pred_len)일 때
                if z_out.shape[1] == z_mean.shape[2] and z_out.shape[2] == z_mean.shape[1]:
                    z_out = z_out.permute(0, 2, 1)
                    # print("Permuted z_out shape:", z_out.shape)
            z = z_out * (z_std + 1e-4) + z_mean
            z = z.permute(0, 2, 1)  # (batch, pred_len, feature)
            # print("CARDformer return z shape:", z.shape)
            
            # 추가
            if z.dim() == 2:
                z = z.unsqueeze(0)  # (1, pred_len, feature)
            elif z.dim() == 1:
                z = z.unsqueeze(0).unsqueeze(-1)
    
        else:
            z = self.W_out(torch.mean(outputs[:, :, :, :], dim=-2).reshape(b, -1))
        

        if output_attention:
            # print("CARDformer return z shape:", z.shape)
            return z, attn_maps_channel, attn_maps_token  # 두 종류 모두 반환
        else:
            return z
    

class Attenion(nn.Module):
    def __init__(self,config, over_hidden = False,trianable_smooth = False,untoken = False, *args, **kwargs):
        super().__init__()

        
        self.over_hidden = over_hidden
        self.untoken = untoken
        self.n_heads = config.n_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        
        
    
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.n_heads
        

        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear( config.d_model,  config.d_model)
        
        

        self.norm_post1  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        self.norm_post2  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)
        
        
        self.ff_1 = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                       )
        
        self.ff_2= nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                                )     
        self.merge_size = config.merge_size

        ema_size = max(config.enc_in,config.total_token_number,config.dp_rank)
        ema_matrix = torch.zeros((ema_size,ema_size))
        alpha = config.alpha
        ema_matrix[0][0] = 1
        for i in range(1,config.total_token_number):
            for j in range(i):
                ema_matrix[i][j] =  ema_matrix[i-1][j]*(1-alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer('ema_matrix',ema_matrix)
 
           

       
    def ema(self,src):
        return torch.einsum('bnhad,ga ->bnhgd',src,self.ema_matrix[:src.shape[-2],:src.shape[-2]])
        
        
    def ema_trianable(self,src):
        alpha = F.sigmoid(self.alpha)
        
        weights = alpha * (1 - alpha) ** self.arange[-src.shape[-2]:]
 

        w_f = torch.fft.rfft(weights,n = src.shape[-2]*2)
        src_f = torch.fft.rfft(src.float(),dim = -2,n = src.shape[-2]*2)    
        src_f = (src_f.permute(0,1,2,4,3)*w_f)
        src1 =torch.fft.irfft(src_f.float(),dim = -1,n=src.shape[-2]*2)[...,:src.shape[-2]].permute(0,1,2,4,3)#.half()
        return src1



    def dynamic_projection(self,src,mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp,dim = -1)
        src_dp = torch.einsum('bnhef,bnhec -> bnhcf',src,src_dp)
        return src_dp
        

        
    def forward(self, src, *args,**kwargs):

        print(f"[DEBUG][Attenion] src shape: {src.shape}")  # ★ 채널(nvars) 변화 추적
        B,nvars, H, C, = src.shape
                
        
        qkv = self.qkv(src).reshape(B,nvars, H, 3, self.n_heads, C // self.n_heads).permute(3, 0, 1,4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale_tok = self.head_dim ** -0.5
        scale_hid = q.shape[-2] ** -0.5
        
        # Token 방향 Attention
        
        if not self.over_hidden: 
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k)) * scale_tok 
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
            self.last_attn_map = attn_along_token.detach().cpu()  # <-- attention map 저장
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)
        
        else:
            v_dp,k_dp = self.dynamic_projection(v,self.dp_v) , self.dynamic_projection(k,self.dp_k)
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k_dp)) * scale_tok
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
            self.last_attn_map = attn_along_token.detach().cpu()  # <-- attention map 저장
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)

        attn_score_along_hidden = torch.einsum('bnhae,bnhaf->bnhef', q,k) * scale_hid
        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1) )    
        self.last_attn_map_hidden = attn_along_hidden.detach().cpu()  # hidden 방향도 저장
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v)
        print(f"[DEBUG][Attenion] last_attn_map_hidden shape (channel): {self.last_attn_map_hidden.shape}")


        merge_size = self.merge_size
        if not self.untoken:
            output1 = rearrange(output_along_token.reshape(B*nvars,-1,self.head_dim),
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
            output2 = rearrange(output_along_hidden.reshape(B*nvars,-1,self.head_dim),
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        output1 = self.norm_post1(output1)
        output1 = output1.reshape(B,nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B,nvars, -1, self.n_heads * self.head_dim)





        src2 =  self.ff_1(output1)+self.ff_2(output2)
        
        
        src = src + src2
        src = src.reshape(B*nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)

        src = src.reshape(B,nvars, -1, self.n_heads * self.head_dim)
        return src