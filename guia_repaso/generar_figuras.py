import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import erf
C=['#2a78d6','#eb6834','#1baf7a','#eda100']; LS=['-','--','-.',':']
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
 'axes.grid':True,'grid.color':'#e6e6e6','grid.linewidth':0.6,'axes.edgecolor':'#888','lines.linewidth':2})
out='figuras/'
# 1 activations
x=np.linspace(-4,4,400)
sig=1/(1+np.exp(-x))
f={'ReLU':(np.maximum(0,x),(x>0).astype(float)),'tanh':(np.tanh(x),1-np.tanh(x)**2),
   'sigmoide':(sig,sig*(1-sig)),'ELU':(np.where(x>0,x,np.exp(x)-1),np.where(x>0,1,np.exp(x)))}
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
for i,(n,(y,d)) in enumerate(f.items()):
    ax[0].plot(x,y,LS[i],color=C[i],label=n); ax[1].plot(x,d,LS[i],color=C[i],label=n)
ax[0].set(title='Función de activación $g(z)$',xlabel='$z$',ylim=(-1.5,3)); ax[1].set(title="Derivada $g'(z)$",xlabel='$z$',ylim=(-0.05,1.1))
ax[0].legend(frameon=False,loc='upper left'); ax[1].legend(frameon=False,loc='upper left')
fig.tight_layout(); fig.savefig(out+'activaciones.pdf'); plt.close(fig)
# 2 learning rate on f(w)=w^2 (GD)
fig,ax=plt.subplots(1,2,figsize=(9,3.3))
ws=np.linspace(-1.6,1.6,200); ax[0].plot(ws,ws**2,color='#999',lw=1.2)
for i,(lr,n) in enumerate([(0.05,'$\\eta=0.05$ (lenta)'),(0.4,'$\\eta=0.4$ (adecuada)'),(0.98,'$\\eta=0.98$ (oscila)')]):
    w=1.5; tr=[w]
    for _ in range(12): w=w-lr*2*w; tr.append(w)
    tr=np.array(tr); ax[0].plot(tr,tr**2,LS[i],marker='o',ms=4,color=C[i],label=n,lw=1.2)
    ax[1].plot(range(len(tr)),tr**2,LS[i],marker='o',ms=4,color=C[i],label=n)
ax[0].set(title='Trayectoria sobre $L(w)=w^2$',xlabel='$w$',ylabel='$L$'); ax[1].set(title='Pérdida por iteración',xlabel='iteración',ylabel='$L$')
ax[1].legend(frameon=False); fig.tight_layout(); fig.savefig(out+'learning_rate.pdf'); plt.close(fig)
# 3 neuron distribution (notebook function)
def distribuir(p,total):
    i=np.asarray(p,float); i=i/i.sum()*total; u=np.floor(i).astype(int); r=total-u.sum()
    o=np.argsort(-(i-u),kind='stable'); u[o[:r]]+=1; return u
P,W=20,64; T=P*W; cre=distribuir(np.linspace(.5,1.5,P),T)
formas={'Uniforme':[W]*P,'Creciente':cre,'Embudo':cre[::-1],'Cuello de botella':distribuir(.5+np.abs(np.linspace(-1,1,P)),T)}
fig,ax=plt.subplots(1,4,figsize=(10,2.6),sharey=True)
for i,(n,a) in enumerate(formas.items()):
    ax[i].bar(range(1,P+1),a,color=C[0],width=0.8); ax[i].set_title(n); ax[i].set_xlabel('capa'); ax[i].grid(axis='x',visible=False)
ax[0].set_ylabel('neuronas'); fig.tight_layout(); fig.savefig(out+'distribuciones.pdf'); plt.close(fig)
print({k:list(map(int,v)) for k,v in formas.items()})
# 4 threshold illustration: synthetic scores
rng=np.random.default_rng(0)
s0=rng.beta(2,9,8500); s1=rng.beta(3,7,1500)
fig,ax=plt.subplots(1,2,figsize=(9,3.3))
b=np.linspace(0,1,41)
ax[0].hist(s0,b,density=True,alpha=.55,color=C[0],label='Pagado (0)'); ax[0].hist(s1,b,density=True,alpha=.55,color=C[1],label='Impago (1)')
for t,ls in [(0.5,'-'),(0.2,'--')]: ax[0].axvline(t,color='#333',ls=ls,lw=1.2); ax[0].text(t+0.01,3.9,f'umbral {t}',fontsize=8)
ax[0].set_ylim(0,4.6); ax[0].set(title='Probabilidades predichas (ilustrativo)',xlabel='probabilidad de impago',ylabel='densidad'); ax[0].legend(frameon=False,loc='upper right')
ts=np.linspace(0.02,0.65,200); P_=[];R_=[];F_=[]
for t in ts:
    tp=(s1>=t).sum(); fp=(s0>=t).sum(); fn=(s1<t).sum()
    p=tp/max(tp+fp,1); r=tp/(tp+fn); P_.append(p);R_.append(r);F_.append(2*p*r/max(p+r,1e-9))
ax[1].plot(ts,P_,'-',color=C[0],label='precisión'); ax[1].plot(ts,R_,'--',color=C[1],label='recall'); ax[1].plot(ts,F_,'-.',color=C[2],label='F1')
tb=ts[int(np.argmax(F_))]; ax[1].axvline(tb,color='#333',ls=':',lw=1.2); ax[1].text(tb+0.01,0.92,f'máx. F1 en {tb:.2f}',fontsize=8)
ax[1].set(title='Métricas según el umbral',xlabel='umbral',ylim=(0,1)); ax[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(0.62,0.85))
fig.tight_layout(); fig.savefig(out+'umbral.pdf'); plt.close(fig)
# 5 learning curves regimes (schematic)
e=np.arange(1,101)
fig,ax=plt.subplots(1,3,figsize=(10,2.8),sharey=True)
reg=[('Subajuste',0.55-0.05*(1-np.exp(-e/30)),0.56-0.05*(1-np.exp(-e/30))),
     ('Buen ajuste',0.6*np.exp(-e/15)+0.38,0.6*np.exp(-e/15)+0.40),
     ('Sobreajuste',0.6*np.exp(-e/12)+0.05+0.30*np.exp(-e/80),0.6*np.exp(-e/12)+0.40+0.0025*np.maximum(e-20,0)**1.15)]
for a,(n,tr,va) in zip(ax,reg):
    a.plot(e,tr,'-',color=C[0],label='entrenamiento'); a.plot(e,va,'--',color=C[1],label='validación'); a.set(title=n,xlabel='época')
ax[0].set_ylabel('pérdida'); ax[2].legend(frameon=False,loc='upper right'); fig.tight_layout(); fig.savefig(out+'curvas_regimenes.pdf'); plt.close(fig)
