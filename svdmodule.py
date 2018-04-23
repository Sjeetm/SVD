# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 23:08:13 2018

@author: Subhajeet
"""

def data_clean(df):
    import numpy as np
    df=df.select_dtypes([np.number]) 
    return df

def data_norm(df):
    import numpy as np
    d_m=np.mean(df)
    d_s=np.std(df)
    d_n=(df-d_m)/d_s
    d_n=d_n.dropna(axis=1)
    return d_n

def data_eig(df):
    import numpy as np
    eig=np.linalg.eig
    dmat=np.matrix(df.T)
    dmatt=dmat.T
    u1=np.matmul(dmat,dmatt)
    v1=np.matmul(dmatt,dmat)
    ueva,uevec=eig(u1)
    veva,vevec=eig(v1)
    m=min(u1.shape)
    n=min(v1.shape)
    j=ueva.argsort()[::-1]
    ue_v=ueva[j]
    uev=uevec[:,j]
    i=veva.argsort()[::-1]
    ve_v=veva[i]
    vev=vevec[:,i]
    return ue_v,uev,ve_v,vev,m,n

def compute_df_u_e_v(df,ue_v,uev,ve_v,vev,m,n):
    import numpy as np
    if m>n:
        l=m-n
        k=n
        diag=np.diag(ve_v)
        z=np.zeros((l,k))
        e=np.concatenate((diag,z))
        eta=np.sqrt(e)
        u=np.matrix(uev)
        v=np.matrix(vev).T
        A=np.matrix(df.T)
        
    else:
        l=n-m
        k=m
        diag=np.diag(ue_v)
        z=np.zeros((l,k))
        e=np.concatenate((diag,z))
        eta=np.sqrt(e)
        u=np.matrix(vev)
        v=np.matrix(uev).T
        A=np.matrix(df.T).T
    return A,u,v,eta,k,diag

def compute_u(A,v,eta,k):
    import numpy as np
    u_e=[]
    for i in range(k):
        u=np.matmul(A,v[:,i]/np.sqrt(np.linalg.norm(v[:,i])))/np.sqrt(np.linalg.norm(eta))
        u_e.append(u)        
    U=np.matrix(np.array(u_e)).T
    return U

def plot_error2(A,U,v,diag,k):
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import norm
    from sklearn.metrics import mean_absolute_error as mae
    errr=[]
    for i in range(1,k+1):
        dmatf=np.matmul(np.matmul(U[:,:i],diag[:i,:i])/np.sqrt(norm(diag[:i,:i])),v[:i,:]/np.sqrt(norm(v[:i,:])))
        errr.append(mae(A,dmatf))
    return plt.plot(errr)

def error_svd(A,u,v,eta):
    import numpy as np
    from sklearn.metrics import mean_absolute_error as mae
    dmatf=np.matmul(np.matmul(u,eta),v)
    return mae(A,dmatf)

def svd(df):
    df=data_norm(data_clean(df))
    ue_v,uev,ve_v,vev,m,n=data_eig(df)
    A,u,v,eta,k,diag=compute_df_u_e_v(df,ue_v,uev,ve_v,vev,m,n)
    U=compute_u(A,v,eta,k)
    print(U,eta,v)
    plot_error2(A,U,v,diag,k)