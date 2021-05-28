import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ndays = config['ndays']

        self.obs_input_size = config["obs_input_size"]
        self.fcst_input_size = config["fcst_input_size"]
        self.obs_hidden_size = config["obs_hidden_size"]
        self.fcst_hidden_size = config["fcst_hidden_size"]

        self.hidden_size = config["hidden_size"]

        self.dropout_ratio = config["dropout_ratio"]

        self.lstm_cell1 = nn.LSTM(self.obs_input_size, self.obs_hidden_size, batch_first=True)
        self.lstm_cell2 = nn.LSTM(self.fcst_input_size, self.fcst_hidden_size, batch_first=True)


        self.dropout = nn.Dropout(self.dropout_ratio)
        
        self.linear1 = nn.Linear(self.fcst_hidden_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, 1)


    def forward(self, inputs, outputs=None):
        
        fcst_inputs = inputs[:, :, self.obs_input_size * self.ndays:]
        obs_inputs = inputs[:, :, :self.obs_input_size * self.ndays].reshape(-1, 24*self.ndays, self.obs_input_size)

        
        obs_out, (h_out, c_out) = self.lstm_cell1(obs_inputs)
        fcst_out, _ = self.lstm_cell2(fcst_inputs, (h_out, c_out))


        out = self.dropout(fcst_out)
        out = self.linear1(out)
        
        pred = self.linear2(out)
        pred = pred.squeeze(-1)
        
        
        return pred


