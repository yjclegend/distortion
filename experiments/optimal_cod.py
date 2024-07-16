import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from common.geometry import distancePoint, distortion_state

from distortion_correction.metric.regression_dc import MetricRTMDC, MetricRTMDC1
from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid

cod_rm = np.array([[-0.00012986439042262158, -4.863773538576483], 
                    [0.0012014350033527504, -5.105613058621314], 
                    [-0.0003433666708360808, -5.372774784951883], 
                    [-4.080423581891494e-06, -5.6694314407332485], 
                    [9.493442516260924e-05, -6.0007643800657355], 
                    [-0.0003134295993513698, -6.373223734944211], 
                    [-6.83861026601275e-07, -6.794979826639685], 
                    [1.200801951626577e-06, -7.2765067597297595], 
                    [0.0008322486878570583, -7.831481355454136], 
                    [0.00010945688344912415, -8.478112523069193], 
                    [0.00014903006345257363, -9.241143585220213], 
                    [-6.795686310550849e-06, -10.155086652406396], 
                    [-0.0007543571197445731, -11.269722806557107], 
                    [-4.7562952569199426e-05, -12.659137025584595], 
                    [8.333802914843607e-05, -14.43922542127351], 
                    [-0.00022124739484700442, -16.8019857274299], 
                    [0.0008055119116170276, -20.08925530350057], 
                    [-1.9742188964026816e-06, -24.97575465088053], 
                    [-4.4997327375904885e-05, -33.004182751369726], 
                    [0.0011685109055700861, -48.637757788289555]])

cod_rtm = np.array([[7.094259909291631e-06, -9.483040400296277], 
                    [0.0002255108189272184, -9.953715442909495], 
                    [-0.00013282173328999555, -10.474817116376911], 
                    [0.00030460177688329537, -11.053361907478667], 
                    [5.7850742431468e-05, -11.699627643398436], 
                    [-7.180898659468803e-07, -12.426204339006137], 
                    [-6.286519998116588e-07, -13.247236600221298], 
                    [1.5131092219320547e-05, -14.188651068548785], 
                    [-0.0012821629809400305, -15.271520521680545], 
                    [1.6964141237810356e-06, -16.528348453282046], 
                    [0.00024797624451697374, -18.02138519718831], 
                    [2.086497257752437e-05, -19.808843604428315], 
                    [-0.00034009544088240123, -21.98699573003066], 
                    [-6.643148950673217e-05, -24.703469060893756], 
                    [8.616610458048532e-05, -28.185389390034956], 
                    [-0.00047392215657448764, -32.81518552047727], 
                    [-0.0014503732680206313, -39.26460579981072], 
                    [-9.652332363697133e-06, -48.67950657510912], 
                    [-2.148353584902757e-05, -64.34643182664419], 
                    [0.0001236946277597419, -94.75293311214966]])

def dc(m, cod, image_dt, image_u):
    m.estimate(image_dt, image_u, cod=cod)
    pred = m.undistort(image_dt)
    mse = distancePoint(image_u, pred)
    # print(np.mean(mse))
    return np.mean(mse)


RES = 2000
PS = 0.01
FOCAL = 12

ps = np.linspace(20, 2, 20)
print(ps)
rtm = MetricRTMDC1(5)
rm = MetricRTMDC1(5, tangential=False)

cods = cod_rtm
model = rtm

def precision_gain(cods, model):
    mses1 = list()
    mses2 = list()

    tmeans = list()

    ratio = list()

    for i in range(len(ps)):
        p = ps[i]/1000
        cx, cy = cods[i, 0], cods[i, 1]
        camera = CameraDistortion(RES*PS/p, FOCAL, p)

        pattern = gen_grid(camera.get_fov(extend=1.5), 100)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.05, division=False)
        image_dt = camera.distort(pattern, k1=-0.05, theta=np.radians(5/60), division=False)
        index = camera.valid_range(image_dt)
        image_dt = image_dt[index]
        image_u = image_u[index]
        image_dr = image_dr[index]
        meant, _ = distortion_state(image_dr, image_dt)
        tmeans.append(meant)
        m1 = dc(model, (0, 0), image_dt, image_u)
        m2 = dc(model, (cx, cy), image_dt, image_u)
        mses1.append(m1)
        mses2.append(m2)
        ratio.append((m1/m2 - 1)*100)
    return tmeans, np.array(mses1), np.array(mses2)

# t1, r1 = precision_gain(cod_rm, rm)
t1, m1, m2 = precision_gain(cods, model)
plt.rcParams['font.family'] = 'serif'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
print(t1)
print(m1)
print(m2)

# mpl.rcParams['figure.figsize']=7/3, 7/3

# plt.plot(tmeans, cods[:, 0], label='cx', marker='s', fillstyle='none')
# plt.plot(tmeans, cods[:, 1], label='cy', marker='^', fillstyle='none')
# plt.legend()
# plt.xlabel('mean tangential(pixels)')
# plt.ylabel('COD')
# plt.show()
fig, ax1 = plt.subplots()
plot1 = ax1.plot(t1, cods[:, 0], label=r'$COD_{nx}$', marker='o', fillstyle='none', markersize=4, color=colors[0])
plot2 = ax1.plot(t1, cods[:, 1], label=r'$COD_{ny}$', marker='^', fillstyle='none', markersize=4, color=colors[1])
ax1.set_xlabel("mean tangential(pixels)")
ax1.set_ylabel('COD')
ax1.tick_params(axis='y')

ax2 = ax1.twinx()
ax2.set_ylabel('Mean Error')

plot3 = ax2.plot(t1, m1, label=r'mean error ($COD_g$)', marker='+', color=colors[2])
plot4 = ax2.plot(t1, m2, label=r'mean error ($COD_n$)', marker='*', color=colors[3])
ax2.tick_params(axis='y')

lns = plot1 + plot2 + plot3 + plot4
labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc='center left', fontsize=6)
plt.show()
