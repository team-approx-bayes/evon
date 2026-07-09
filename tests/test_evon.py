import torch
import torch.nn as nn
from evon import EVON

def test_evon_initialization():
    model = nn.Linear(5, 2)
    optimizer = EVON(model.parameters(), lr=0.1, ess=10.0, hess_init=1.0)
    assert optimizer is not None
    assert len(optimizer.param_groups) == 1

def test_evon_sampled_step():
    model = nn.Linear(5, 2)
    optimizer = EVON(model.parameters(), lr=0.1, ess=10.0, hess_init=1.0)
    
    x = torch.randn(4, 5)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()
    
    init_weight = model.weight.clone()
    
    # Run multiple steps because the first step initializes preconditioning stats
    # and skips the parameter update (standard Shampoo behavior).
    for step in range(2):
        for _ in range(2): # MC samples
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
        optimizer.step()
    
    # Verify that the parameters were updated
    assert not torch.equal(model.weight, init_weight)

def test_evon_deterministic_step():
    model = nn.Linear(5, 2)
    optimizer = EVON(model.parameters(), lr=0.1, ess=10.0, hess_init=1.0)
    optimizer.disable_sampling()
    
    x = torch.randn(4, 5)
    y = torch.randint(0, 2, (4,))
    criterion = nn.CrossEntropyLoss()
    
    init_weight = model.weight.clone()
    
    # Run multiple steps because the first step initializes preconditioning stats
    # and skips the parameter update (standard Shampoo behavior).
    for step in range(2):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Verify that parameters were updated in deterministic mode
    assert not torch.equal(model.weight, init_weight)
