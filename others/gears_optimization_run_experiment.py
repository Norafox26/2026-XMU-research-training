"""GEARS 简单损失改进：训练、四组消融、GPU 计时及作图。"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'cache')
import sys, json, time, random, copy, argparse, shutil, hashlib, pickle
import faulthandler
faulthandler.dump_traceback_later(180, repeat=True)
sys.path.insert(0, str(ROOT / 'upstream' / 'GEARS-master'))
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gears import PertData, GEARS
from gears.utils import loss_fct
from gears.inference import evaluate, compute_metrics
import gears.utils as gears_utils

def gpu_pearson(x,y):
    # 与官方相关系数公式相同，只将大矩阵乘法放到 GPU；四组共用该预处理。
    with torch.no_grad():
        a=torch.as_tensor(x,device='cuda'); a=a-a.mean(0)
        b=a if x is y else torch.as_tensor(y,device='cuda')
        if x is not y: b=b-b.mean(0)
        denom=torch.sqrt(torch.outer((a*a).sum(0),(b*b).sum(0)))
        return ((a.T@b)/denom).clamp(-1,1).cpu().numpy()

def seed_all(seed):
    # 每个方法重新设置随机数，保证初始权重和批次顺序相同。
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def prepare():
    # 复制原始文件，所有新缓存均写入本实验目录，避免覆盖用户原实验。
    src = Path('F:/paper_reproduction/data')
    dest = ROOT / 'data'; dest.mkdir(exist_ok=True)
    for name in ['gene2go_all.pkl', 'essential_all_data_pert_genes.pkl',
                 'norman_umi_go/perturb_processed.h5ad', 'norman_umi_go/data_pyg/cell_graphs.pkl']:
        target = dest / name; target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copy2(src / name, target)
    d = PertData(str(dest), gene_set_path=str(dest / 'essential_all_data_pert_genes.pkl'))
    d.load(data_path=(dest / 'norman_umi_go').as_posix())
    d.prepare_split(split='simulation', seed=1)
    # 按扰动分组划分；检查不同集合之间没有同一扰动条件。
    sets = {k: set(v) for k,v in d.set2conditions.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not (sets[a] & sets[b]), (a,b)
    save_json(ROOT / 'split.json', d.set2conditions)
    d.get_dataloader(batch_size=32, test_batch_size=32)
    return d

def auxiliary(pred, perts, means, ctrl, use_a, use_b):
    # 只使用训练扰动的均值；不使用验证/测试集表达或 DE 排名参与训练。
    target = torch.stack([means[p] for p in perts])
    extra = pred.new_zeros(())
    if use_a:
        extra = extra + 0.1 * F.mse_loss(pred, target)
    if use_b:
        delta = target - ctrl
        # 用较大真实变化赋权，小变化不应主导方向监督；softplus 可反传。
        weight = delta.abs() / (delta.abs().mean(dim=1, keepdim=True) + 1e-6)
        extra = extra + 0.1 * (weight * F.softplus(-delta.sign() * (pred-ctrl))).mean()
    return extra

def run(args):
    assert torch.cuda.is_available(), '本任务要求 GPU，禁止静默退回 CPU'
    torch.set_num_threads(4)
    # 小矩阵验证数值一致性后启用 GPU 计算；不会改变图的定义或使用的数据。
    test=np.random.RandomState(0).normal(size=(30,8)).astype('float32')
    np.testing.assert_allclose(gpu_pearson(test,test),gears_utils.np_pearson_cor(test,test),atol=1e-6)
    gears_utils.np_pearson_cor=gpu_pearson
    d = prepare()
    save_json(ROOT / 'environment.json', {'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(),
              'source':str(sys.modules['gears'].__file__), 'args':vars(args), 'shape':d.adata.shape})
    # 平均表达由训练条件计算；每个扰动独立，标签不跨集合传播。
    obs = d.adata.obs.condition.astype(str).values
    # AnnData 过滤结果可能是 view；缓存稀疏矩阵，避免每次取 X 重复复制整个数据。
    X = d.adata.X
    print('计算训练扰动均值',type(X),len(d.set2conditions['train']),flush=True)
    means = {p:torch.tensor(np.asarray(X[obs==p].mean(0)).ravel(),
                            dtype=torch.float32, device='cuda') for p in d.set2conditions['train']}
    del X
    print('训练均值完成；加载外部GO拓扑',flush=True)
    # 只复用预训练配置中的外部 GO 拓扑；绝不加载模型或共表达图权重。
    # 该 GO 图按相同 essential gene 列表的字母顺序建立。
    with open('F:/paper_reproduction/models/checkpoint_extracted/model_ckpt/config.pkl','rb') as f:
        go_config=pickle.load(f)
    assert go_config['G_go'].max().item() < len(d.pert_names)
    rows=[]
    for seed in args.seeds:
      for name,a,b in [('GEARS',False,False),('A',True,False),('B',False,True),('AB',True,True)]:
        out = ROOT / 'results' / f'{name}_seed{seed}'; out.mkdir(parents=True,exist_ok=True)
        if (out/'metrics.json').exists():
            rows.append(json.loads((out/'metrics.json').read_text(encoding='utf-8'))); continue
        seed_all(seed)
        print('初始化',name,seed,flush=True)
        g = GEARS(d, device='cuda', weight_bias_track=False)
        print('构建当前划分的共表达图',flush=True)
        g.model_initialize(hidden_size=64, G_go=go_config['G_go'], G_go_weight=go_config['G_go_weight'])
        model=g.model.to('cuda'); opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=5e-4)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=0.5)
        best=float('inf'); history=[]; train_seconds=0; val_seconds=0; peak=0
        for epoch in range(args.epochs):
            model.train(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
            total=0; steps=0
            for batch in d.dataloader['train_loader']:
                batch=batch.to('cuda'); opt.zero_grad(); pred=model(batch)
                loss=loss_fct(pred,batch.y,batch.pert,ctrl=g.ctrl_expression,
                    dict_filter=g.dict_filter,direction_lambda=g.config['direction_lambda'])
                if a or b: loss=loss+auxiliary(pred,batch.pert,means,g.ctrl_expression,a,b)
                loss.backward(); torch.nn.utils.clip_grad_value_(model.parameters(),1.0); opt.step()
                total+=loss.item(); steps+=1
                if steps%100==0: print(name,seed,epoch+1,steps,total/steps,flush=True)
            torch.cuda.synchronize(); train_seconds+=time.perf_counter()-start
            peak=max(peak,torch.cuda.max_memory_allocated()/1024**3)
            scheduler.step(); start=time.perf_counter()
            result=evaluate(d.dataloader['val_loader'],model,False,'cuda'); metrics,_=compute_metrics(result)
            val_seconds+=time.perf_counter()-start
            history.append({'epoch':epoch+1,'loss':total/steps,**metrics})
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            if metrics['mse_de']<best:
                best=metrics['mse_de']; torch.save(model.state_dict(),out/'best.pt')
            print(name,seed,'epoch',epoch+1,'val',metrics,'train_s',train_seconds,flush=True)
        model.load_state_dict(torch.load(out/'best.pt',weights_only=True)); model.eval()
        # 预热后计时整个固定测试集，含批次搬运与预测转回 CPU，单列模型前向计时。
        batch=next(iter(d.dataloader['test_loader'])).to('cuda')
        with torch.no_grad():
            for _ in range(5): model(batch)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
        result=evaluate(d.dataloader['test_loader'],model,False,'cuda')
        torch.cuda.synchronize(); infer_seconds=time.perf_counter()-start
        infer_peak=torch.cuda.max_memory_allocated()/1024**3
        metrics,per=compute_metrics(result)
        pd.DataFrame(per).T.to_csv(out/'per_condition.csv')
        # 按论文的未见基因数量分别报告，避免总体平均掩盖困难子组退步。
        subgroup={}
        for label,conditions in d.subgroup['test_subgroup'].items():
            subgroup[label]={k:float(np.mean([per[p][k] for p in conditions if p in per]))
                             for k in ['mse','mse_de','pearson','pearson_de']}
        save_json(out/'subgroups.json',subgroup)
        # GPU 前向延迟：相同 batch，CUDA event，30 次重复。
        begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            begin.record()
            for _ in range(30): model(batch)
            end.record(); torch.cuda.synchronize()
        row={'method':name,'seed':seed,**{k:float(v) for k,v in metrics.items()},
             'train_seconds':train_seconds,'validation_seconds':val_seconds,'infer_seconds':infer_seconds,
             'forward_batch_ms':begin.elapsed_time(end)/30,'train_peak_GiB':peak,
             'infer_peak_GiB':infer_peak,'test_cells':len(result['pert_cat'])}
        save_json(out/'metrics.json',row); rows.append(row)
        pd.DataFrame(rows).to_csv(ROOT/'results'/'summary.csv',index=False)
        del g,model,opt,result; torch.cuda.empty_cache()
    plot(rows)

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.DataFrame(rows); groups=df.groupby('method').mean(numeric_only=True).reindex(['GEARS','A','B','AB'])
    deviations=df.groupby('method').std(numeric_only=True).reindex(groups.index).fillna(0)
    groups.to_csv(ROOT/'results'/'mean.csv'); deviations.to_csv(ROOT/'results'/'std.csv')
    for keys,filename in [(['mse_de','pearson_de'],'comparison'),
                          (['train_seconds','infer_seconds','train_peak_GiB'],'resources')]:
        fig,axes=plt.subplots(1,len(keys),figsize=(5*len(keys),4),squeeze=False)
        for ax,key in zip(axes.ravel(),keys):
            vals=groups[key]; ax.bar(vals.index,vals.values,yerr=deviations[key],capsize=4,
                color=['#63758b','#83b7b2','#dcba80','#467ca5'])
            ax.set_title(key); ax.grid(axis='y',alpha=.2)
            for i,v in enumerate(vals): ax.text(i,v,f'{v:.4g}',ha='center',va='bottom')
        fig.tight_layout(); fig.savefig(ROOT/'results'/f'{filename}.png',dpi=180); plt.close(fig)
    base=groups.loc['GEARS','mse_de']; gain=(base-groups.loc['AB','mse_de'])/base*100
    (ROOT/'results'/'结论.md').write_text(f'# 实测结果\n\nAB 相对 GEARS 的 Top20-DE MSE 降低 {gain:.2f}%。负值表示退步。\n\n'
      'A 为训练扰动均值监督，B 为可微方向监督。请结合 A/B 消融判断贡献，不应仅据机制猜测宣称有效。\n'
      '这是本地 Norman UMI/GO 数据的统一复现实验，不等同于论文全部数据集的结果。\n'+groups.to_string(),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]); args=parser.parse_args()
    run(args)
