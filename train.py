# -*- coding: utf-8 -*-



def train(model, loader, optimizer, criterion, epochs=10, device='cuda'):
    model.train()
    model.to(device)

    for epoch in range(1):
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss {loss.item():.4f}")

