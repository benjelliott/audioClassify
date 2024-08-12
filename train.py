from tqdm import tqdm

def train1(our_net, trainloader, criterion, optimizer)

    history = []
    for epoch in tqdm(range(100)):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # compute the loss function
            outputs = our_net(inputs)
            loss = criterion(outputs, labels)

            # compute the gradient of the loss function relative to the model's parameters
            loss.backward()

            # take a step in the direction that minimizes the loss
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            history.append(loss.item())
            rounds = 100
            if (i + 1) % rounds == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / rounds:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return history