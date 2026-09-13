import math
n_dims=64; base=1e7; n_ctx_orig=262144; bf=32.0; bs=1.0
def corr_dim(n_rot): return n_dims*math.log(n_ctx_orig/(n_rot*2*math.pi))/(2*math.log(base))
lo=max(0.0, math.floor(corr_dim(bf))); hi=min(n_dims-1.0, math.ceil(corr_dim(bs)))
print("corr_dim raw:", corr_dim(bf), corr_dim(bs), "-> corr_dims", lo, hi)
def ramp(i0): 
    y=(i0/2-lo)/max(0.001,hi-lo); return 1.0-min(1.0,max(0.0,y))
theta_scale=base**(-2.0/n_dims)
for factor in (1.1875, 2.34375, 4.0):
    fs=1.0/factor
    ms=1.0+0.1*math.log(1.0/fs)
    print(f"\n=== factor {factor}  freq_scale {fs:.6f}  mscale {ms:.5f} ===")
    print(" i  ramp   wavelen(tok)   ratio_applied   theta@262143(rad) -> scaled")
    for i in range(0,32):
        i0=2*i
        rm=ramp(i0)*1.0
        eff = fs*(1-rm) + 1.0*rm   # effective multiplier on theta
        freq = theta_scale**(i)
        wl = 2*math.pi/freq
        th = 262143*freq
        if i in (0,8,12,13,14,15,16,18,20,21,22,23,26,31):
            print(f"{i:3d} {rm:5.3f} {wl:14.0f}  {eff:13.6f}  {th:12.4f} -> {th*eff:10.4f}")
