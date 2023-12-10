import torch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=True)

    def forward(self, x):
        out = self.linear(x)
        return out

class Camera2DCalibrationNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.mlp_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1024, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(1024 , output_size)
        )

    def forward(self, x):
        logits = self.mlp_stack(x)
        return logits

# Best 2 layer 2048
class Robot2DCalibrationNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.mlp_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2048, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(2048, 2048, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(2048 , output_size)
        )

    def forward(self, x):
        logits = self.mlp_stack(x)
        return logits
    
def train_model(model, criterion, optimizer, scheduler, epochs, train_input, train_target, val_input, val_target):

    writer = SummaryWriter()
    model.to(device="cuda")
    if torch.cuda.is_available():
        val_gt = Variable(torch.from_numpy(val_target).cuda())
    else:
        val_gt = Variable(torch.from_numpy(val_target))

    for epoch in range(epochs):
    # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(train_input).cuda())
            labels = Variable(torch.from_numpy(train_target).cuda())
        else:
            inputs = Variable(torch.from_numpy(train_input))
            labels = Variable(torch.from_numpy(train_target))

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        writer.add_scalar("Loss/train", loss, epoch)

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()
        scheduler.step()

        with torch.no_grad(): # we don't need gradients in the testing phase
            if torch.cuda.is_available():
                predicted = model(Variable(torch.from_numpy(val_input).cuda()))
            else:
                predicted = model(Variable(torch.from_numpy(val_input)))
            val_loss = criterion(predicted, val_gt)
            writer.add_scalar("Loss/validation", val_loss, epoch)

        print('epoch {}, loss {}, validation loss {}'.format(epoch, loss.item(), val_loss.item()))

    writer.flush()
    writer.close()

    return model

def evaluate_model(model, input):
    with torch.no_grad(): # we don't need gradients in the testing phase
        predicted = model(Variable(torch.from_numpy(input).cuda())).cpu().data.numpy()
    return predicted