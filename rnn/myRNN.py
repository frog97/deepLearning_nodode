import torch
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden
    
    def get_hidden(self):
        return torch.zeros(1, self.hidden_size)
    


rnn_model = MyRNN(input_size=4, hidden_size=1024, output_size=2)
hidden = rnn_model.get_hidden()

# The food is good

input_tensor0 = torch.randn(1, 4) # The
input_tensor1 = torch.randn(1, 4) # food
input_tensor2 = torch.randn(1, 4) # is
input_tensor3 = torch.randn(1, 4) # good


# ouput_tensor0, hidden = rnn_model(input_tensor0, hidden)   # _ 몰라도 되는 것 들 처리
# ouput_tensor1, hidden = rnn_model(input_tensor1, hidden)
# ouput_tensor2, hidden = rnn_model(input_tensor2, hidden)
# ouput_tensor3, hidden = rnn_model(input_tensor3, hidden)


_, hidden = rnn_model.forward(input_tensor0, hidden)   # _ 몰라도 되는 것 들 처리
_, hidden = rnn_model.forward(input_tensor1, hidden)
_, hidden = rnn_model.forward(input_tensor2, hidden)
ouput_tensor3, _ = rnn_model.forward(input_tensor3, hidden)

print(ouput_tensor3)