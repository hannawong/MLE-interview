```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from SER_mmoe.modeling.GE2E_model import SpeakerEncoder
from SER_mmoe.modeling.audio import *
from SER_mmoe.modeling.inference import *
from SER_mmoe.modeling.DSBN import high_layers
state_fpath = "/data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/Speech-Emotion-Recognition/allosaurus+CNN/SER/modeling/encoder.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


              
            

class SER_MODEL(nn.Module):
    def __init__(self, args,audio_maxlen, num_labels, hidden_size = 128):

        ALLO_CONV_SIZE = 64
        ALLO_LSTM_SIZE = 128
        ALLO_ATTN_SIZE = 256
        ALLO_LSTM_NUM = 1

        MFCC_CONV_SIZE = 32
        MFCC_LSTM_SIZE = 64
        MFCC_LSTM_NUM = 2

        DROP_OUT = 0.1
        hidden_size_en = 256
        hidden_size_ge = 512
        hidden_size_pe = 1024
        hidden_size_fr = 512


        super(SER_MODEL, self).__init__()
        self.audio_maxlen = audio_maxlen
        self.num_labels = num_labels
        self.args = args
        #########  Allosaurus feature   ##########
        self.conv1d_1 = nn.Conv1d(in_channels=230, out_channels=ALLO_CONV_SIZE, kernel_size=3,padding = "same")
        self.conv1d_2 = nn.Conv1d(in_channels=230, out_channels=ALLO_CONV_SIZE, kernel_size=5,padding = "same")
        self.lstm = nn.LSTM(ALLO_CONV_SIZE, ALLO_LSTM_SIZE, ALLO_LSTM_NUM, bidirectional=True)
        self.Q_layer = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.K_layer = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.V_layer = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)

        ############# for MFCC ###############
        self.conv1d_1_mfcc = nn.Conv1d(in_channels=24, out_channels=MFCC_CONV_SIZE, kernel_size=3,padding = "same")
        self.conv1d_2_mfcc = nn.Conv1d(in_channels=24, out_channels=MFCC_CONV_SIZE, kernel_size=5,padding = "same")
        self.lstm_mfcc = nn.LSTM(MFCC_CONV_SIZE,MFCC_LSTM_SIZE, MFCC_LSTM_NUM, bidirectional=True)
        self.Q_layer_mfcc = nn.Linear(MFCC_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.K_layer_mfcc = nn.Linear(MFCC_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.V_layer_mfcc = nn.Linear(MFCC_LSTM_SIZE*2,ALLO_ATTN_SIZE)


        ############# for wav2vec ###############
        self.conv1d_1_wav2vec = nn.Conv1d(in_channels=512, out_channels=ALLO_CONV_SIZE, kernel_size=3,padding = "same")
        self.conv1d_2_wav2vec = nn.Conv1d(in_channels=512, out_channels=ALLO_CONV_SIZE, kernel_size=5,padding = "same")
        self.lstm_wav2vec = nn.LSTM(ALLO_CONV_SIZE,ALLO_LSTM_SIZE, ALLO_LSTM_NUM, bidirectional=True)
        self.Q_layer_wav2vec = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.K_layer_wav2vec = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)
        self.V_layer_wav2vec = nn.Linear(ALLO_LSTM_SIZE*2,ALLO_ATTN_SIZE)

        ############# for GE2E finetuning #############
        
        self.SpeakerEncoder = SpeakerEncoder('cpu','cpu')  ####small learning rate
        checkpoint = torch.load(state_fpath, map_location='cpu')
        self.SpeakerEncoder.load_state_dict(checkpoint["model_state"])
        self.SpeakerEncoder = self.SpeakerEncoder.to(DEVICE)
      
        ############### for attention ################
        
        self.W_pe = nn.Linear(256,128)
        self.W_en = nn.Linear(256,128)
        self.W_ge = nn.Linear(256,128)
        self.W_fr = nn.Linear(256,128)

        self.W_pe_1 = nn.Linear(128,64)
        self.W_en_1 = nn.Linear(128,64)
        self.W_ge_1 = nn.Linear(128,64)
        self.W_fr_1 = nn.Linear(128,64)

        self.W_pe_2 = nn.Linear(64,32)
        self.W_en_2 = nn.Linear(64,32)
        self.W_ge_2 = nn.Linear(64,32)
        self.W_fr_2 = nn.Linear(64,32)

        self.W_pe_3 = nn.Linear(32,5)
        self.W_en_3 = nn.Linear(32,5)
        self.W_ge_3 = nn.Linear(32,4)
        self.W_fr_3 = nn.Linear(32,5)

        ########### feature attention ###############

        self.high_layer = high_layers(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE)
        self.Q_layer_feat = {"ge":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda(),"en":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda(),"pe":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda()}
        self.K_layer_feat =  {"ge":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda(),"en":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda(),"pe":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda()}
        self.V_layer_feat =  {"ge":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda(),"en":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda(),"pe":nn.Linear(ALLO_ATTN_SIZE,ALLO_ATTN_SIZE).cuda()}
        

        ########## MLP ############
        self.dense1_en = nn.Linear(256+ALLO_ATTN_SIZE, hidden_size_en*2)
        self.dense1_ge = nn.Linear(256+ALLO_ATTN_SIZE, hidden_size_ge*2)
        self.dense1_pe = nn.Linear(256+ALLO_ATTN_SIZE, hidden_size_pe*2)
        self.dense1_fr = nn.Linear(256+ALLO_ATTN_SIZE, hidden_size_fr*2)

        self.bn_en = torch.nn.BatchNorm1d(hidden_size_en*2)
        self.bn_ge = torch.nn.BatchNorm1d(hidden_size_ge*2)
        self.bn_pe = torch.nn.BatchNorm1d(hidden_size_pe*2)
        self.bn_fr = torch.nn.BatchNorm1d(hidden_size_fr*2)

        self.bn1_en = torch.nn.BatchNorm1d(hidden_size_en)
        self.bn1_ge = torch.nn.BatchNorm1d(hidden_size_ge)
        self.bn1_pe = torch.nn.BatchNorm1d(hidden_size_pe)
        self.bn1_fr = torch.nn.BatchNorm1d(hidden_size_fr)

        self.bn2_en = torch.nn.BatchNorm1d(hidden_size_en//2)
        self.bn2_ge = torch.nn.BatchNorm1d(hidden_size_ge//2)
        self.bn2_pe = torch.nn.BatchNorm1d(hidden_size_pe//2)
        self.bn2_fr = torch.nn.BatchNorm1d(hidden_size_fr//2)

        self.dropout = nn.Dropout(DROP_OUT)
        self.dropout1 = nn.Dropout(0.01)

        self.dense2_en = nn.Linear(hidden_size_en*2,hidden_size_en)
        self.dense2_ge = nn.Linear(hidden_size_ge*2,hidden_size_ge)
        self.dense2_pe = nn.Linear(hidden_size_pe*2,hidden_size_pe)
        self.dense2_fr = nn.Linear(hidden_size_fr*2,hidden_size_fr)

        self.dense3_en = nn.Linear(hidden_size_en,hidden_size_en // 2)
        self.dense3_ge = nn.Linear(hidden_size_ge,hidden_size_ge // 2)
        self.dense3_pe = nn.Linear(hidden_size_pe,hidden_size_pe // 2)
        self.dense3_fr = nn.Linear(hidden_size_fr,hidden_size_fr // 2)

        self.bn3_pe = torch.nn.BatchNorm1d(hidden_size_pe//4)
        self.dense4_pe = nn.Linear(hidden_size_pe // 2,hidden_size_pe // 4)

        self.out_proj_en = nn.Linear(hidden_size_en//2, 4)
        self.out_proj_ge = nn.Linear(hidden_size_ge//2, 7)
        self.out_proj_pe = nn.Linear(hidden_size_pe//4, 6)
        self.out_proj_fr = nn.Linear(hidden_size_fr//2, 7)

        self.adaptor = nn.Linear(2048,512)
        self.adaptor1 = nn.Linear(512,ALLO_ATTN_SIZE)

        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(DEVICE)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(DEVICE)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(DEVICE)



    def forward(self, lang, feat_emb, ge2e_emb,mfcc_emb,label,length,mfcc_length,wav2vec_emb,wav2vec_length,bloy_embedding,audio_files):
        
        return self.classification_score(lang, feat_emb,ge2e_emb,mfcc_emb,label,length,mfcc_length,wav2vec_emb,wav2vec_length,bloy_embedding,audio_files)


    def get_attention_mask(self,lstm_output,length):
        length = np.array(length.detach().cpu())
        MAX_LEN = lstm_output.shape[1]
        BZ = lstm_output.shape[0]
        mask = [[1]*int(length[_])+[0]*(MAX_LEN-int(length[_])) for _ in range(BZ)]
        mask = torch.Tensor(mask)
        return mask
    

    def self_attention_layer(self,lstm_output,length):  # TODO: multi-head self-attention
        last_hidden_state = [lstm_output[i,length[i].long(),:] for i in range(lstm_output.shape[0])] ## the actual hidden state!
        last_hidden_state = torch.stack(last_hidden_state,axis = 0)
        Q_last_hidden_state = self.Q_layer(last_hidden_state) ##(batchsize,128)
        Q_last_hidden_state = torch.unsqueeze(Q_last_hidden_state,1) ##(batchsize,1,128)
        K = self.K_layer(lstm_output) ##(batchsize,seq_len,128)
        V = self.V_layer(lstm_output) ##(batchsize,seq_len,128)
        attention_scores = torch.matmul(Q_last_hidden_state,K.permute(0,2,1)) ###(batchsize,1,128)
        attention_scores = torch.multiply(attention_scores,
                                   1.0 / math.sqrt(float(K.shape[-1])))

        attention_mask = self.get_attention_mask(lstm_output,length) ##can only attend to hidden state that really exist
        adder = (1.0 - attention_mask.long()) * -10000.0  ##-infty, [batchsize,150]
        adder = torch.unsqueeze(adder,axis = 1).to(DEVICE)
        attention_scores += adder

        m = nn.Softmax(dim=2)
        attention_probs = m(attention_scores)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer.squeeze()

    def self_attention_layer_mfcc(self,lstm_output,length):  # TODO: multi-head self-attention
        last_hidden_state = [lstm_output[i,length[i].long(),:] for i in range(lstm_output.shape[0])] ## the actual hidden state!
        last_hidden_state = torch.stack(last_hidden_state,axis = 0)
        Q_last_hidden_state = self.Q_layer_mfcc(last_hidden_state) ##Query, (batchsize,128)
        Q_last_hidden_state = torch.unsqueeze(Q_last_hidden_state,1)
        K = self.K_layer_mfcc(lstm_output) ##(batchsize,max_len,128)
        V = self.V_layer_mfcc(lstm_output) ##(batchsize,max_len,128)
        attention_scores = torch.matmul(Q_last_hidden_state,K.permute(0,2,1))
        attention_scores = torch.multiply(attention_scores,
                                   1.0 / math.sqrt(float(K.shape[-1])))

        attention_mask = self.get_attention_mask(lstm_output,length) ##can only attend to hidden state that really exist
        adder = (1.0 - attention_mask.long()) * -10000.0  ##-infty, [batchsize,150]
        adder = torch.unsqueeze(adder,axis = 1).to(DEVICE)
        attention_scores += adder

        m = nn.Softmax(dim=2)
        attention_probs = m(attention_scores)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer.squeeze()

    def self_attention_layer_wav2vec(self,lstm_output,length):  # TODO: multi-head self-attention
        last_hidden_state = [lstm_output[i,length[i].long(),:] for i in range(lstm_output.shape[0])] ## the actual hidden state!
        last_hidden_state = torch.stack(last_hidden_state,axis = 0)
        Q_last_hidden_state = self.Q_layer_wav2vec(last_hidden_state) ##Query, (batchsize,128)
        Q_last_hidden_state = torch.unsqueeze(Q_last_hidden_state,1)
        K = self.K_layer_wav2vec(lstm_output) ##(batchsize,max_len,128)
        V = self.V_layer_wav2vec(lstm_output) ##(batchsize,max_len,128)
        attention_scores = torch.matmul(Q_last_hidden_state,K.permute(0,2,1))
        attention_scores = torch.multiply(attention_scores,
                                   1.0 / math.sqrt(float(K.shape[-1])+0.0001))

        attention_mask = self.get_attention_mask(lstm_output,length) ##can only attend to hidden state that really exist
        adder = (1.0 - attention_mask.long()) * -10000.0  ##-infty, [batchsize,150]
        adder = torch.unsqueeze(adder,axis = 1).to(DEVICE)
        attention_scores += adder

        m = nn.Softmax(dim=2)
        attention_probs = m(attention_scores)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer.squeeze()


    def self_attention_layer_feat(self,feature_output,langs = "ge"):  # TODO: multi-head self-attention
       
        Q = self.Q_layer_feat[langs](feature_output) ##(batchsize,4,128)
        K = self.K_layer_feat[langs](feature_output) ##(batchsize,4,128)
        V = self.V_layer_feat[langs](feature_output) ##(batchsize,4,128)
        attention_scores = torch.matmul(Q,K.permute(0,2,1))
        attention_scores = torch.multiply(attention_scores,
                                   1.0 / math.sqrt(float(Q.shape[-1])))
        m = nn.Softmax(dim=2)
        attention_probs = m(attention_scores)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer.squeeze()

    def similarity_matrix(self, embeds): ## (emotions_per_batch,utterence_per_emotion,embedding_size)
     
        emotion_per_batch, utterances_per_emotion = embeds.shape[:2]
        
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True).to(DEVICE) ##(emotions_per_batch,1,embedding_size), 类簇中心点embedding
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True)) ###类簇中心点归一化。(emotions_per_batch,1,embedding_size)

        sim_matrix = torch.zeros(emotion_per_batch, utterances_per_emotion,
                                 emotion_per_batch).to(DEVICE)
        for j in range(speakers_per_batch):
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight.to(DEVICE) + self.similarity_bias.to(DEVICE)
        return sim_matrix##(speakers_per_batch,utterances_per_speaker, speakers_per_batch)

    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(DEVICE)
        loss = self.loss_fn(sim_matrix, target).to(DEVICE)
        return loss,0

    def construct_matrix_embed(self,embeds,labels,bsize,label_num,prune = False):
      '''
      buf = [[] for _ in range(num_labels)]
      for i in range(len(labels)):
        buf[int(labels[i].item())].append(embeds[i])
      
      mmin = 10000
      for i in range(num_labels):
            buf[i] = torch.stack(buf[i])
            mmin = min(mmin,buf[i].shape[0])
      for i in range(num_labels):
            buf[i] = buf[i][:mmin]
      embedding = torch.stack(buf)
    
      return embedding 
      '''
         
      label_cnt = {}
      for i in range(len(labels)):
        if int(labels[i].item()) not in label_cnt:
          label_cnt[int(labels[i].item())] = 1
        else:
          label_cnt[int(labels[i].item())] += 1
      label_cnt_pruned = {} ###only retain those cnt >= 5
      for k in label_cnt.keys():
            if prune:
              if label_cnt[k] > 5:
                label_cnt_pruned[k] = label_cnt[k]
            else:
                label_cnt_pruned[k] = label_cnt[k]
      speaker_per_batch = len(label_cnt_pruned)
      utterances_per_speaker = np.min(list(label_cnt_pruned.values()))
      embeddings = torch.zeros((speaker_per_batch,utterances_per_speaker,embeds.shape[-1]))
      for k in range(len(label_cnt_pruned.keys())): ### emotions
        key = list(label_cnt_pruned.keys())[k]
        cnt = 0
        for i in range(len(labels)):
          if int(labels[i].item()) == key and cnt < utterances_per_speaker:
            embeddings[k][cnt] = embeds[i]
            cnt += 1
      print(embeddings.shape)
      return torch.Tensor(embeddings)

    def classification_score(self,lang,feat_emb,ge2e_emb,mfcc_emb,label,length,mfcc_length,wav2vec_emb,wav2vec_length,bloy_embedding,audio_files):
        
        ########## allosaurus features #############
        print(feat_emb.shape)
        feat_emb = feat_emb.permute(0, 2, 1)
        x = self.conv1d_1(feat_emb)
        x1 = self.conv1d_2(feat_emb)
        x = x.permute(0,2,1)
        x1 = x1.permute(0,2,1)
        x = torch.add(x,x1)
        x,_ = self.lstm(x)
        allo_hidden_state = self.self_attention_layer(x,length) ##[bz,128]

        ################## mfcc features ####################
        mfcc = mfcc_emb.permute(0, 2, 1)
        mfcc_x = self.conv1d_1_mfcc(mfcc)
        mfcc_x1 = self.conv1d_2_mfcc(mfcc)
        mfcc_x = mfcc_x.permute(0,2,1)
        mfcc_x1 = mfcc_x1.permute(0,2,1)
        mfcc_x = torch.add(mfcc_x,mfcc_x1)
        mfcc_x,_ = self.lstm_mfcc(mfcc_x)
        mfcc_hidden_state = self.self_attention_layer_mfcc(mfcc_x,mfcc_length) ##[bz,32]
        
        ################wav2vec features ##################
        wav2vec = wav2vec_emb.permute(0, 2, 1)
        wav2vec_x = self.conv1d_1_wav2vec(wav2vec)
        wav2vec_x1 = self.conv1d_2_wav2vec(wav2vec)
        wav2vec_x = wav2vec_x.permute(0,2,1)
        wav2vec_x1 = wav2vec_x1.permute(0,2,1)
        wav2vec_x = torch.add(wav2vec_x,wav2vec_x1)
        wav2vec_x,_ = self.lstm_wav2vec(wav2vec_x)
        wav2vec_hidden_state = self.self_attention_layer_wav2vec(wav2vec_x,wav2vec_length) ##[bz,32]

        ############### add ge2e #####################
        ge2e_emb = ge2e_emb.squeeze()  ### hard-encode ge2e_emb
        if lang == "ge":
        
          ge2e_emb = []
          for i in range(len(audio_files)):
            wav = preprocess_wav("/data/jiayu_xiao/IEMOCAP/path_to_wavs/"+audio_files[i].split("/")[-1])
            emb = embed_utterance(wav,self.SpeakerEncoder.to(DEVICE))
            ge2e_emb.append(torch.Tensor(emb))
          ge2e_emb = torch.stack(ge2e_emb, axis = 0)
          
        ge2e_emb = ge2e_emb.to(DEVICE)
        bloy_embedding = bloy_embedding.squeeze().to(DEVICE)


        
        inp = torch.stack([allo_hidden_state,mfcc_hidden_state,wav2vec_hidden_state])
        output = self.high_layer(inp)
        allo_hidden_state,mfcc_hidden_state,wav2vec_hidden_state = output[0],output[1],output[2]
        
        
        ############## MLP ################
        if lang == "fr":
            bloy_embedding = bloy_embedding[:,0,:] + bloy_embedding[:,1,:]
            bloy_embedding = self.adaptor(bloy_embedding)
            bloy_embedding = self.adaptor1(bloy_embedding)
            '''
            feat_input = torch.stack([allo_hidden_state,mfcc_hidden_state,wav2vec_hidden_state,bloy_embedding])
            feat_output = self.self_attention_layer_feat(feat_input,"fr")
            allo_hidden_state,mfcc_hidden_state,wav2vec_hidden_state,bloy_embedding = feat_output[0],feat_output[1],feat_output[2],feat_output[3]
            '''
            score =self.W_fr(ge2e_emb)
            score = F.relu(score)
            score = self.W_fr_1(score)
            score = F.relu(score)
            score = self.W_fr_2(score)
            score = F.relu(score)
            score = self.W_fr_3(score)
            score = nn.Softmax(dim = 1)(score)
            print(torch.mean(score,0),"******************")
            #np.save("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/SER_mmoe/pe.npy",score.cpu().detach().numpy())
            
            allo_hidden_state = torch.matmul(allo_hidden_state.T,torch.diag(score[:,0])).T
            mfcc_hidden_state = torch.matmul(mfcc_hidden_state.T,torch.diag(score[:,1])).T
            wav2vec_hidden_state = torch.matmul(wav2vec_hidden_state.T,torch.diag(score[:,2])).T
            ge2e_emb = torch.matmul(ge2e_emb.T,torch.diag(score[:,3])).T
            bloy_embedding = torch.matmul(bloy_embedding.T,torch.diag(score[:,4])).T
            
          
            mfcc_allo_hidden_state = torch.add(allo_hidden_state,mfcc_hidden_state)
            mfcc_allo_wav2vec_hidden_state = torch.add(mfcc_allo_hidden_state,wav2vec_hidden_state)
            print(bloy_embedding.shape,mfcc_allo_hidden_state.shape)
            all_hidden_state = torch.add(bloy_embedding,mfcc_allo_wav2vec_hidden_state)
            x = torch.concat((all_hidden_state, ge2e_emb),1)

            print(x.shape)
            x = self.dense1_fr(x)
            x = self.bn_fr(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.dense2_fr(x)
            x = self.bn1_fr(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.dense3_fr(x)
            x = self.bn2_fr(x)
            x = F.gelu(x)
            matrix_embedding = self.construct_matrix_embed(x,label,ge2e_emb.shape[0],7).to(DEVICE)
            contrastive_loss,err = self.SpeakerEncoder.loss(matrix_embedding)
            print("contrastive loss:",contrastive_loss,err)
            x = self.out_proj_fr(x)
            loss_fct = torch.nn.CrossEntropyLoss()
            label = label.long()

            loss = loss_fct(x.view(-1, 7), label.view(-1))
            alpha = 0.015
            total_loss = loss+alpha*contrastive_loss
            return total_loss,x

        if lang == "pe":
          
          bloy_embedding = self.adaptor(bloy_embedding)
          bloy_embedding = self.adaptor1(bloy_embedding)
          '''
          feat_input = torch.stack([allo_hidden_state,mfcc_hidden_state,wav2vec_hidden_state,bloy_embedding])
          feat_output = self.self_attention_layer_feat(feat_input,"pe")
          allo_hidden_state,mfcc_hidden_state,wav2vec_hidden_state,bloy_embedding = feat_output[0],feat_output[1],feat_output[2],feat_output[3]
          '''
          score = self.W_pe(ge2e_emb)
          score = F.relu(score)
          score = self.W_pe_1(score)
          score = F.relu(score)
          score = self.W_pe_2(score)
          score = F.relu(score)
          score = self.W_pe_3(score)
          score = nn.Softmax(dim = 1)(score)
          print(torch.mean(score,0),"******************")
          
          #np.save("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/SER_mmoe/pe.npy",score.cpu().detach().numpy())
          allo_hidden_state = torch.matmul(allo_hidden_state.T,torch.diag(score[:,0])).T
          mfcc_hidden_state = torch.matmul(mfcc_hidden_state.T,torch.diag(score[:,1])).T
          wav2vec_hidden_state = torch.matmul(wav2vec_hidden_state.T,torch.diag(score[:,2])).T
          ge2e_emb = torch.matmul(ge2e_emb.T,torch.diag(score[:,3])).T
          bloy_embedding = torch.matmul(bloy_embedding.T,torch.diag(score[:,4])).T
          
          
          mfcc_allo_hidden_state = torch.add(allo_hidden_state,mfcc_hidden_state)
          mfcc_allo_wav2vec_hidden_state = torch.add(mfcc_allo_hidden_state,wav2vec_hidden_state)
          all_hidden_state = torch.add(bloy_embedding,mfcc_allo_wav2vec_hidden_state)
          x = torch.concat((all_hidden_state, ge2e_emb),1)

          
          x = self.dense1_pe(x)
          x = self.bn_pe(x)
          m = nn.Mish()
          x = torch.tanh(x)
          x = m(x)
          #x = F.relu(x)
          x = self.dropout1(x)
          x = self.dense2_pe(x)
          x = self.bn1_pe(x)
          x = torch.tanh(x)
          x = m(x)
          x = self.dropout1(x)
          x = self.dense3_pe(x)
          x = self.bn2_pe(x)
          x = F.gelu(x)
          x = self.dropout(x)
          x = self.dense4_pe(x)
          x = self.bn3_pe(x)
          x = m(x)
          matrix_embedding = self.construct_matrix_embed(x,label,ge2e_emb.shape[0],6,True).to(DEVICE)
          contrastive_loss,err = self.loss(matrix_embedding)
          x = self.out_proj_pe(x)
          loss_fct = torch.nn.CrossEntropyLoss()
          label = label.long()
          loss = loss_fct(x.view(-1, 6), label.view(-1))
          alpha = 0.01
          total_loss = loss+alpha*contrastive_loss
          return total_loss,x

        if lang == "en":
          bloy_embedding = self.adaptor(bloy_embedding)
          bloy_embedding = self.adaptor1(bloy_embedding)
          
          
          score = self.W_en(ge2e_emb)
          score = F.relu(score)
          score = self.W_en_1(score)
          score = F.relu(score)
          score = self.W_en_2(score)
          score = F.relu(score)
          score = self.W_en_3(score)
          score = nn.Softmax(dim = 1)(score)
          print(torch.mean(score,0),"******************")
          '''
          #np.save("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/SER_mmoe/en.npy",score.cpu().detach().numpy())
          allo_hidden_state = torch.matmul(allo_hidden_state.T,torch.diag(score[:,0])).T
          mfcc_hidden_state = torch.matmul(mfcc_hidden_state.T,torch.diag(score[:,1])).T
          wav2vec_hidden_state = torch.matmul(wav2vec_hidden_state.T,torch.diag(score[:,2])).T
          ge2e_emb = torch.matmul(ge2e_emb.T,torch.diag(score[:,3])).T
          bloy_embedding = torch.matmul(bloy_embedding.T,torch.diag(score[:,4])).T
          '''
          
          mfcc_allo_hidden_state = torch.add(allo_hidden_state,mfcc_hidden_state)
          mfcc_allo_wav2vec_hidden_state = torch.add(mfcc_allo_hidden_state,wav2vec_hidden_state)
          print(bloy_embedding.shape,mfcc_allo_hidden_state.shape)
          all_hidden_state = torch.add(bloy_embedding,mfcc_allo_wav2vec_hidden_state)
          
          
          x = torch.concat((all_hidden_state, ge2e_emb),1)
          x = self.dense1_en(x)
          x = self.bn_en(x)
          m = nn.Mish()
          x = torch.tanh(x)
          x = self.dropout(x)
          
          x = self.dense2_en(x)
          x = self.bn1_en(x)
          x = F.relu(x)
          x = self.dropout(x)
          x = self.dense3_en(x)
          x = self.bn2_en(x)
          x = F.gelu(x)
          matrix_embedding = self.construct_matrix_embed(x,label,ge2e_emb.shape[0],6).to(DEVICE)
          contrastive_loss,err = self.loss(matrix_embedding)
          
          x = self.out_proj_en(x)
          loss_fct = torch.nn.CrossEntropyLoss()
          label = label.long()
          loss = loss_fct(x.view(-1, 4), label.view(-1))
          alpha = 0.1
          total_loss = loss+alpha*contrastive_loss
          return total_loss,x

        if lang == "ge":
          bloy_embedding = self.adaptor(bloy_embedding)
          bloy_embedding = self.adaptor1(bloy_embedding)
          '''
          feat_input = torch.stack([allo_hidden_state,mfcc_hidden_state,wav2vec_hidden_state])
          feat_output = self.self_attention_layer_feat(feat_input)
          allo_hidden_state,mfcc_hidden_state,wav2vec_hidden_state = feat_output[0],feat_output[1],feat_output[2]
          '''
          score = self.W_ge(ge2e_emb)
          score = F.relu(score)
          score = self.W_ge_1(score)
          score = F.relu(score)
          score = self.W_ge_2(score)
          score = F.relu(score)
          score = self.W_ge_3(score)
          score = nn.Softmax(dim = 1)(score)
          
          #np.save("/content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/SER_mmoe/ge.npy",score.cpu().detach().numpy())
          allo_hidden_state = torch.matmul(allo_hidden_state.T,torch.diag(score[:,0])).T
          mfcc_hidden_state = torch.matmul(mfcc_hidden_state.T,torch.diag(score[:,1])).T
          wav2vec_hidden_state = torch.matmul(wav2vec_hidden_state.T,torch.diag(score[:,2])).T
          ge2e_emb = torch.matmul(ge2e_emb.T,torch.diag(score[:,3])).T
          

          
          mfcc_allo_hidden_state = torch.add(allo_hidden_state,mfcc_hidden_state)
          mfcc_allo_wav2vec_hidden_state = torch.add(mfcc_allo_hidden_state,wav2vec_hidden_state)
          
          x = torch.concat((mfcc_allo_wav2vec_hidden_state, ge2e_emb),1)
          x = self.dense1_ge(x)
          x = self.bn_ge(x)
          x = torch.tanh(x)
          x = self.dropout(x)
          x = self.dense2_ge(x)
          x = self.bn1_ge(x)
          x = F.relu(x)
        
          x = self.dropout(x)
          x = self.dense3_ge(x)
          x = self.bn2_ge(x)
          x = F.gelu(x)
          matrix_embedding = self.construct_matrix_embed(x,label,ge2e_emb.shape[0],7).to(DEVICE)
          contrastive_loss,err = self.SpeakerEncoder.loss(matrix_embedding)
          x = self.out_proj_ge(x)
          loss_fct = torch.nn.CrossEntropyLoss()
          label = label.long()
          loss = loss_fct(x.view(-1, 7), label.view(-1))
          alpha = 0.01
          total_loss = loss+alpha*contrastive_loss
          return total_loss,x
```



---

GE2E:

```python

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from torch import nn
import numpy as np
import torch
from SER_mmoe.modeling.params_data import *
from SER_mmoe.modeling.params_model import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        
        # Network defition
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds
    
    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True).to(DEVICE)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds).to(DEVICE)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(DEVICE)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight.to(DEVICE) + self.similarity_bias.to(DEVICE)
        return sim_matrix
    
    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(DEVICE)
        loss = self.loss_fn(sim_matrix, target).to(DEVICE)
        print(loss,"&&&&&&&&&")
    
        
        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer
```

