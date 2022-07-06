from src.pachage_list import *

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

    
class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        return mean_max_embeddings

    
class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers+1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out
    
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model.model_name, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model.model_name, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        
        # gradient checkpointing
        if self.cfg.training.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()
            print(f"Gradient Checkpointing: {self.model.is_gradient_checkpointing}")
            
        
        # self.pooler = MeanPooling()
        
        self.bilstm = nn.LSTM(self.config.hidden_size, (self.config.hidden_size) // 2, num_layers=2, 
                              dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.cfg.model.target_size)
            # nn.Linear(256, self.cfg.target_size)
        )
        
        

    def loss(self, outputs, targets):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, targets)
        return loss
    
    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        # print(outputs)
        # print(targets)
        mll = log_loss(
            targets.cpu().detach().numpy(),
            outputs.cpu().detach().numpy(),
            labels=[0, 1, 2],
        )
        return mll
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.model(ids, mask, token_type_ids)
        else:
            transformer_out = self.model(ids, mask)
        
        # LSTM/GRU header
#         all_hidden_states = torch.stack(transformer_out[1])
#         sequence_output = self.pooler(all_hidden_states)
        
        # simple CLS
        sequence_output = transformer_out[0][:, 0, :]

        
        # Main task
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        if targets is not None:
            loss1 = self.loss(logits1, targets)
            loss2 = self.loss(logits2, targets)
            loss3 = self.loss(logits3, targets)
            loss4 = self.loss(logits4, targets)
            loss5 = self.loss(logits5, targets)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            output = nn.Softmax(dim=1)(logits)
            metric = self.monitor_metrics(output, targets)
            return logits, loss, metric

        return logits, loss, {}