# torch-optional-sequential

```
from module import OptionalSequential

class A(nn.Module):
  def forward(self,x):
    print(x)
    return x

class B(nn.Module):
  def forward(self,x, mask):
    print(x, mask)
    return x

class C(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = OptionalSequential(
        A(),
        B()
    )

  def forward(self, x):
    self.layers(x, mask=3)
    return x

model = C()
model(1)

#1
#1 3
#1
```
