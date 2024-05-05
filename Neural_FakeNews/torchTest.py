import torch

# Código de teste
x = torch.rand(5, 3)
print(x)

# Deve resultar em algo do tipo:
# tensor([[0.3380, 0.3845, 0.3217],
#       [0.8337, 0.9050, 0.2650],
#       [0.2979, 0.7141, 0.9069],
#       [0.1449, 0.1132, 0.1375],
#       [0.4675, 0.3947, 0.1426]])

# Testar se uma GPU Nvidia tem suporte para CUDA:
print("Disponibilidade de GPU Nvidia com CUDA:", torch.cuda.is_available())
