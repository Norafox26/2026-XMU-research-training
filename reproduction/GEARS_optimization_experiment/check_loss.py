"""独立检查新增方向损失确实提供纠正错误方向的梯度。"""
import ast
from pathlib import Path
import torch
import torch.nn.functional as F
source=ast.parse(Path(__file__).with_name('run_experiment.py').read_text(encoding='utf-8'))
node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='auxiliary')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<auxiliary>','exec'))
p=torch.tensor([[-1.,1.]],requires_grad=True)
means={'x':torch.tensor([1.,-1.])}
loss=auxiliary(p,['x'],means,torch.zeros(2),False,True)
loss.backward()
assert torch.isfinite(loss) and p.grad[0,0]<0 and p.grad[0,1]>0
# A 在预测等于训练扰动均值时应为零。
assert auxiliary(means['x'][None],['x'],means,torch.zeros(2),True,False).item()==0
print('PASS: A zero at target; B gradient corrects both wrong directions.')
