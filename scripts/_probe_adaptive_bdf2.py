import torch, torchdae, numpy as np, time

def dydt(yv):
    v = yv if yv.dim() == 1 else yv[0]
    y_I2 = torch.clamp(v[0], min=0.0)
    y_I = torch.clamp(v[1], min=0.0)
    y_H2 = torch.clamp(v[2], min=0.0)
    v0 = 1.0 * y_I2
    v1 = 50.0 * y_H2 * y_I
    v2 = 5.0 * y_I2 * y_I
    out = torch.stack([-v0 - v2, 2 * v0 - v1 - v2, -v1, 2 * v1 + 2 * v2])
    return out if yv.dim() == 1 else out[None, :]

def F(t, y, yp):
    y1 = y if y.dim() == 1 else y[0]
    return yp - dydt(y1[None, :])[0]

y0 = torch.tensor([0.5, 0.0, 0.5, 0.0], dtype=torch.float64)[None, :]
yp0 = dydt(y0)
t_end = 1.0e6

for h in (None, 1e4, 1e3):
    t0 = time.time()
    try:
        sol = torchdae.solve_bdf2(F, (0.0, t_end), y0, h=h, yp0=yp0,
                                  max_iter_for_events=100)
        ys = sol.ys.squeeze(1).cpu().numpy()
        finite = np.all(np.isfinite(ys))
        print(f"h={h}: steps={len(sol.ts)} finite={finite} "
              f"last={np.round(ys[-1], 6)} t={time.time()-t0:.1f}s")
    except Exception as e:
        print(f"h={h}: EXC {type(e).__name__}: {str(e)[:120]}")
