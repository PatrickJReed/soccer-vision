import cv2, numpy as np, glob, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from soccer_vision.labeler.chain import load_chain
from soccer_vision.pitch.manual_anchor import build_segments, cumulative_transforms

VIDEO=os.path.expanduser("~/sv-labeler/oceanside_clip.mp4")
OUT=os.path.expanduser("~/sv-labeler/revisit_out"); os.makedirs(OUT,exist_ok=True)
npz=max(glob.glob(os.path.expanduser('~/sv-labeler/.sv_labeler_cache/*.npz')),key=os.path.getmtime)
interframe,n,size=load_chain(npz); seg=build_segments(interframe,n); tr=cumulative_transforms(interframe,seg)

# 1) pan trajectory from the chain (x-offset the cumulative transform induces at image centre)
pan=np.zeros(n)
for f in range(n):
    M=tr.get(f,np.eye(3)); v=M@np.array([0.5,0.5,1.0]); pan[f]=v[0]/v[2]-0.5
plt.figure(figsize=(11,3.2)); plt.plot(np.arange(n)/30.0, pan, lw=0.8)
plt.xlabel("t (s)"); plt.ylabel("pan proxy (norm x-offset)"); plt.title("Camera pan vs time (chain-derived; drift over long spans)")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"pan_trajectory.png"),dpi=110); plt.close()
# reversals = sweeps
d=np.diff(np.sign(np.diff(np.convolve(pan,np.ones(31)/31,mode='same'))))
sweeps=int((d!=0).sum())
print(f"pan range (norm): {pan.max()-pan.min():.3f} ; approx direction reversals (sweeps): {sweeps}")

# 2) ORB self-similarity over sampled frames
stride=25; idxs=list(range(0,n,stride)); N=len(idxs)
cap=cv2.VideoCapture(VIDEO); orb=cv2.ORB_create(1200)
desc={}; kpn={}
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,fr=cap.read()
    if not ok: desc[i]=None; kpn[i]=0; continue
    g=cv2.cvtColor(cv2.resize(fr,None,fx=0.5,fy=0.5),cv2.COLOR_BGR2GRAY)
    k,dd=orb.detectAndCompute(g,None); desc[i]=dd; kpn[i]=len(k) if k else 0
cap.release()
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
S=np.zeros((N,N))
for a in range(N):
    for b in range(a,N):
        da,db=desc[idxs[a]],desc[idxs[b]]
        if da is None or db is None or kpn[idxs[a]]<10 or kpn[idxs[b]]<10: s=0.0
        else:
            m=bf.match(da,db); good=sum(1 for x in m if x.distance<48)
            s=good/max(1,min(kpn[idxs[a]],kpn[idxs[b]]))
        S[a,b]=S[b,a]=s
np.fill_diagonal(S,S.max())
plt.figure(figsize=(6,5)); plt.imshow(S,cmap="magma",origin="lower",
    extent=[idxs[0]/30,idxs[-1]/30,idxs[0]/30,idxs[-1]/30])
plt.xlabel("t (s)"); plt.ylabel("t (s)"); plt.title("Frame self-similarity (masked ORB match fraction)\noff-diagonal bands = revisited views")
plt.colorbar(fraction=0.046); plt.tight_layout(); plt.savefig(os.path.join(OUT,"similarity_matrix.png"),dpi=120); plt.close()

# 3) distinct-view estimate + revisit density
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
D=1.0-S/max(S.max(),1e-9); np.fill_diagonal(D,0.0); D=(D+D.T)/2
Z=linkage(squareform(D,checks=False),method="average")
for thr in (0.6,0.7,0.8):
    labels=fcluster(Z,t=thr,criterion="distance"); print(f"  distinct views @ dist<{thr}: {labels.max()}")
# revisit density: for each sampled frame, # of temporally-distant (>200 frames) frames that are 'same view' (top-decile sim)
thr_s=np.quantile(S[S<S.max()],0.90)
rev=[]
for a in range(N):
    cnt=sum(1 for b in range(N) if abs(idxs[a]-idxs[b])>200 and S[a,b]>=thr_s)
    rev.append(cnt)
print(f"sampled frames: {N} (stride {stride}); similarity threshold (90th pct): {thr_s:.3f}")
print(f"mean # of temporally-distant revisits per frame: {np.mean(rev):.1f} ; frames with >=1 distant revisit: {100*np.mean(np.array(rev)>0):.0f}%")
lab=fcluster(Z,t=0.7,criterion="distance")
print(f"=> ~{lab.max()} distinct views across {n} frames  -> compression ~{n/max(lab.max(),1):.0f}x (frames per view)")
print("plots ->",OUT)
