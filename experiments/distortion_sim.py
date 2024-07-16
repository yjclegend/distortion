import numpy as np
import matplotlib.pyplot as plt
from common.geometry import distortion_amount, distortion_state
from common.plot import box, violin
from simulation.camera_distortion import CameraDistortion
from simulation.gen_sample import gen_grid, gen_rect
FOCAL = 12
RES = 2000
PS = 0.01

THETA = 5/60
BETA = 1.0001
GAMMA = 0.0001

def dist2theta():
    theta = np.linspace(0, 5/60, 20)
    tmean_dists = list()
    tmaxi_dists = list()
    for t in theta:
        camera = CameraDistortion(2000, 6, 0.01)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.001)
        image_dt = camera.distort(pattern, k1=-0.001, theta=np.radians(t))
        # index = camera.discretize(image_dr)
        index = camera.valid_range(image_dr)
        image_u = image_u[index]
        image_dr = image_dr[index]
        image_dt = image_dt[index]
        meant, maxit = distortion_state(image_dr, image_dt)
        tmean_dists.append(meant)
        tmaxi_dists.append(maxit)

    plt.plot(theta, tmaxi_dists, label='max', marker='v', markersize=4, fillstyle='none')
    plt.plot(theta, tmean_dists, label='mean', marker='s', markersize=4, fillstyle='none')
    plt.legend()
    plt.xlabel('tilt angle(\')')
    plt.ylabel('distortion amount(pixels)')
    plt.show()

def dist2ds():
    ds = np.linspace(0, 0.01, 20)
    tmean_dists = list()
    tmaxi_dists = list()
    for t in ds:
        camera = CameraDistortion(2000, 6, 0.01)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.001)
        image_dt = camera.distort(pattern, k1=-0.001, theta=np.radians(5/60), d=t)
        # index = camera.discretize(image_dr)
        index = camera.valid_range(image_dr)
        image_u = image_u[index]
        image_dr = image_dr[index]
        image_dt = image_dt[index]
        meant, maxit = distortion_state(image_dr, image_dt)
        tmean_dists.append(meant)
        tmaxi_dists.append(maxit)

    plt.plot(ds, tmaxi_dists, label='max', marker='v', markersize=4, fillstyle='none')
    plt.plot(ds, tmean_dists, label='mean', marker='s', markersize=4, fillstyle='none')
    plt.legend()
    plt.xlabel('tilt angle(\')')
    plt.ylabel('distortion amount(pixels)')
    plt.show()

def dist2focal():
    focal = np.arange(2, 50)
    rmean_dists = list()
    tmean_dists = list()
    for f in focal:
        camera = CameraDistortion(RES, f, PS)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.001)
        # validr = camera.valid_range(image_dr)
        validr = camera.discretize(image_dr)
        image_dr = image_dr[validr]
        distr = distortion_amount(image_u[validr], image_dr)
        rmean_dists.append(np.mean(distr))

        
        image_dt = camera.distort(pattern, theta=np.radians(5/60))
        validt = camera.valid_range(image_dt)
        image_dt = image_dt[validt]
        distt = distortion_amount(image_u[validt], image_dt)
        tmean_dists.append(np.mean(distt))

    plt.plot(focal, rmean_dists, label='radial', c='steelblue')
    plt.plot(focal, tmean_dists, label='tangential', c='darkorange')
    width = (50-2)/20
    focalr = np.linspace(6, 46, 5)
    focalr -= width/2
    violinr = list()
    violint = list()
    for f in focalr:
        camera = CameraDistortion(RES, f, PS)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.001)
        validr = camera.valid_range(image_dr)
        image_dr = image_dr[validr]
        distr = distortion_amount(image_u[validr], image_dr)
        rmean_dists.append(np.mean(distr))
        violinr.append(distr)

    focalt = np.linspace(6, 46, 5)
    focalt += width/2
    for f in focalt:
        camera = CameraDistortion(RES, f, PS)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        
        image_dt = camera.distort(pattern, theta=np.radians(5/60))

        validt = camera.valid_range(image_dt)
        image_dt = image_dt[validt]
        distt = distortion_amount(image_u[validt], image_dt)
        violint.append(distt)
        camera.distort(pattern, )
    
    violin(violinr, focalr, 'lightskyblue', 'steelblue', width=width)
    
    violin(violint, focalt, 'lightsalmon', 'darkorange', width=width)
    box_width = (50-2)/100
    box(violinr, focalr, 'lightskyblue', 'steelblue', width=box_width)
    box(violint, focalt, 'lightsalmon', 'darkorange', width=box_width)

    plt.legend()
    plt.xlabel('focal length(mm)')
    plt.ylabel('distortion amount(pixels)')
    ymin = 0-0.05*(6-0)
    ymax = 6+0.05*(6-0)
    plt.xticks(np.linspace(6, 46, 5))
    plt.ylim(ymin, ymax)
    plt.show()

def dist2ps():
    ps = np.linspace(2, 20, 20)
    rmean_dists = list()
    tmean_dists = list()
    for p in ps:
        camera = CameraDistortion(RES, FOCAL, p/1000)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)

        image_dr = camera.distort(pattern, k1=-0.001)
        validr = camera.valid_range(image_dr)
        image_dr = image_dr[validr]
        meanr, maxir = distortion_state(image_u[validr], image_dr)
        rmean_dists.append(meanr)

        image_dt = camera.distort(pattern, theta=np.radians(5/60))
        validt = camera.valid_range(image_dt)
        image_dt = image_dt[validt]
        meant, maxit = distortion_state(image_u[validt], image_dt)
        tmean_dists.append(meant)

    plt.plot(ps, rmean_dists, label='radial', c='steelblue')
    plt.plot(ps, tmean_dists, label='tangential', c='darkorange')
    

    psr = np.linspace(4, 18, 5)
    psr -=0.5
    violinr = list()
    violint = list()
    for p in psr:
        camera = CameraDistortion(RES, FOCAL, p/1000)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.001)
        validr = camera.valid_range(image_dr)
        # validr = camera.discretize(image_dr)
        image_dr = image_dr[validr]
        distr = distortion_amount(image_u[validr], image_dr)
        violinr.append(distr)
    pst = np.linspace(4, 18, 5)
    pst += 0.5
    for p in pst:
        camera = CameraDistortion(RES, FOCAL, p/1000)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        
        image_dt = camera.distort(pattern, theta=np.radians(5/60))
        validt = camera.valid_range(image_dt)
        # validr = camera.discretize(image_dr)
        image_dt = image_dt[validt]
        distt = distortion_amount(image_u[validt], image_dt)
        violint.append(distt)


    violin(violinr, psr, 'lightskyblue', 'steelblue', width=1)
    violin(violint, pst, 'lightsalmon', 'darkorange', width=1)

    box_width = (20-2)/100
    box(violinr, psr, 'lightskyblue', 'steelblue', width=box_width)
    box(violint, pst, 'lightsalmon', 'darkorange', width=box_width)

    box_width = (20-2)/100
    box(violinr, psr, 'lightskyblue', 'steelblue', width=box_width)
    box(violint, pst, 'lightsalmon', 'darkorange', width=box_width)

    plt.legend()
    plt.xlabel('pixel size(μm)')
    plt.ylabel('distortion amount(pixels)')
    plt.xticks(np.linspace(4, 18, 5))
    plt.show()

def distfixfov():
    ps = np.linspace(2, 20, 10)
    rmean_dists = list()
    rmaxi_dists = list()
    tmean_dists = list()
    tmaxi_dists = list()
    for p in ps:
        p = p/1000
        camera = CameraDistortion(RES*PS/p, FOCAL, p)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.001)
        validr = camera.valid_range(image_dr)
        image_dr = image_dr[validr]
        meanr, maxir = distortion_state(image_u[validr], image_dr)
        rmean_dists.append(meanr)
        rmaxi_dists.append(maxir)
        
        image_dt = camera.distort(pattern, theta=np.radians(5/60))
        validt = camera.valid_range(image_dt)
        image_dt = image_dt[validt]
        meant, maxit = distortion_state(image_u[validt], image_dt)
        tmean_dists.append(meant)
        tmaxi_dists.append(maxit)
    plt.plot(ps, rmean_dists, label='radial', c='steelblue')
    plt.plot(ps, tmean_dists, label='tangential', c='darkorange')
    
    psr = np.linspace(4, 18, 5)
    psr -= 0.5
    violinr = list()
    violint = list()
    for p in psr:
        p = p/1000
        camera = CameraDistortion(RES*PS/p, FOCAL, p)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.001)
        validr = camera.valid_range(image_dr)
        # validr = camera.discretize(image_dr)
        image_dr = image_dr[validr]
        distr = distortion_amount(image_u[validr], image_dr)
        violinr.append(distr)
    pst = np.linspace(4, 18, 5)
    pst += 0.5
    for p in pst:
        p = p/1000
        camera = CameraDistortion(RES*PS/p, FOCAL, p)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        
        image_dt = camera.distort(pattern, theta=np.radians(5/60))
        validt = camera.valid_range(image_dt)
        # validr = camera.discretize(image_dr)
        image_dt = image_dt[validt]
        distt = distortion_amount(image_u[validt], image_dt)
        violint.append(distt)

    violin(violinr, psr, 'lightskyblue', 'steelblue', width=1)
    violin(violint, pst, 'lightsalmon', 'darkorange', width=1)

    box_width = (20-2)/100
    box(violinr, psr, 'lightskyblue', 'steelblue', width=box_width)
    box(violint, pst, 'lightsalmon', 'darkorange', width=box_width)

    plt.legend()
    plt.xlabel('pixel size(μm)(fixed FOV)')
    plt.ylabel('distortion amount(pixels)')
    plt.xticks(np.linspace(4, 18, 5))
    plt.show()


def dist2res():
    res = np.linspace(500, 10000, 100)
    rmean_dists = list()
    tmean_dists = list()
    for r in res:
        camera = CameraDistortion(r, FOCAL, PS)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        image_dr = camera.distort(pattern, k1=-0.001)
        validr = camera.valid_range(image_dr)
        image_dr = image_dr[validr]
        meanr, maxir = distortion_state(image_u[validr], image_dr)
        rmean_dists.append(meanr)
        
        image_dt = camera.distort(pattern, theta=np.radians(5/60))
        validt = camera.valid_range(image_dt)
        image_dt = image_dt[validt]
        meant, maxit = distortion_state(image_u[validt], image_dt)
        tmean_dists.append(meant)
    plt.plot(res, rmean_dists, label='radial', c='steelblue')
    plt.plot(res, tmean_dists, label='tangential', c='darkorange')

    width = (10000-500)/20
    resrt = np.linspace(500+2*width, 10000-2*width, 5)
    resr = resrt + width/2
    rest = resrt - width/2
    violinr = list()
    violint = list()
    for r in resr:
        camera = CameraDistortion(r, FOCAL, PS)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)

        image_dr = camera.distort(pattern, k1=-0.001)
        validr = camera.valid_range(image_dr)
        image_dr = image_dr[validr]
        distr = distortion_amount(image_u[validr], image_dr)
        violinr.append(distr)

    for r in rest:
        camera = CameraDistortion(r, FOCAL, PS)
        pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
        image_u = camera.distortion_free(pattern)
        
        image_dt = camera.distort(pattern, theta=np.radians(5/60))
        validt = camera.valid_range(image_dt)
        image_dt = image_dt[validt]
        distt = distortion_amount(image_u[validt], image_dt)
        violint.append(distt)
    
    violin(violinr, resr, 'lightskyblue', 'steelblue', width=width)
    violin(violint, rest, 'lightsalmon', 'darkorange', width=width)
    box_width = (10000-500)/100
    box(violinr, resr, 'lightskyblue', 'steelblue', width=box_width)
    box(violint, rest, 'lightsalmon', 'darkorange', width=box_width)

    plt.legend()
    plt.xlabel('resolution(w&h)')
    plt.ylabel('distortion amount(pixels)')
    ymin = 0-0.05*(80-0)
    ymax = 80+0.05*(80-0)
    plt.ylim(ymin, ymax)
    plt.xticks(resrt)
    plt.show()

def distfixfocal():
    res = np.linspace(500, 10000, 100)
    mean_dists = list()
    max_dists = list()
    for f in res:
        ud = CameraDistortion(f, FOCAL, RES*PS/f)
        ud.set_image_i(gen_grid(ud.range_i))
        ud.gen_image_u()
        ud.gen_image_d(theta=np.radians(THETA))
        mean, max = ud.cal_dist()
        mean_dists.append(mean)
        max_dists.append(max)
    plt.plot(res, mean_dists, label='mean')
    plt.plot(res, max_dists, label='max')
    plt.legend()
    plt.show()

def distfixps():
    res = np.linspace(500, 10000, 100)
    mean_dists = list()
    max_dists = list()
    for f in res:
        ud = CameraDistortion(f, f/RES * FOCAL, PS)
        ud.set_image_i(gen_grid(ud.range_i))
        ud.gen_image_u()
        ud.gen_image_d(theta=np.radians(THETA))
        mean, max = ud.cal_dist()
        mean_dists.append(mean)
        max_dists.append(max)
    plt.plot(res, mean_dists, label='mean')
    plt.plot(res, max_dists, label='max')
    plt.legend()
    plt.show()


def tangential_3d():
    focal = np.linspace(2, 50, 10)
    ps = np.linspace(2, 20, 10)
    res = np.linspace(1000, 10000, 5)

    
    xyz = list()
    tan = list()
    ratio = list()
    for f in focal:
        for p in ps:
            for r in res:
                xyz.append([f, p, r])
                camera = CameraDistortion(r, f, p/1000)
                pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
                image_u = camera.distortion_free(pattern)
                image_dr = camera.distort(pattern, k1=-0.001)
                valid = camera.valid_range(image_dr)
                image_u1 = image_u[valid]
                image_dr = image_dr[valid]
                meanr, maxir = distortion_state(image_u1, image_dr)
                # 
                
                image_dt = camera.distort(pattern, theta=np.radians(5/60))
                valid = camera.valid_range(image_dt)
                image_u2 = image_u[valid]
                # image_dr = image_dr[valid]
                image_dt = image_dt[valid]
                meant, maxit = distortion_state(image_u2, image_dt)
                tan.append(meant)
                ratio.append(meant / meanr)

    xyz = np.array(xyz)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    my_cmap = plt.get_cmap('rainbow')
    heat = ax.scatter(x, y, z, c=np.log10(tan), cmap=my_cmap, marker='s')
    ax.set_xlabel('focal length(mm)')
    ax.set_ylabel('pixel size(μm)')
    ax.set_zlabel('resolution(pixel)')

    fig.colorbar(heat)
    plt.show()
    
    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    my_cmap = plt.get_cmap('rainbow')
    heat = ax.scatter(x, y, z, c=np.log10(ratio), cmap=my_cmap, marker='s')
    ax.set_xlabel('focal length(mm)')
    ax.set_ylabel('pixel size(μm)')
    ax.set_zlabel('resolution(pixel)')

    fig.colorbar(heat)
    plt.show()


def distsim(res, focal, ps):
    camera = CameraDistortion(res, focal, ps)
    pattern = gen_grid(camera.get_fov(extend=1.5), 1000)
    image_u = camera.distortion_free(pattern)
    image_dt = camera.distort(pattern, d=0.1, theta=np.radians(5/60))
    image_dr = camera.distort(pattern, k1=-0.001)
    
    # index = camera.discretize(image_dr)
    index = camera.valid_range(image_dt)
    image_u = image_u[index]
    image_dr = image_dr[index]
    image_dt = image_dt[index]

    statr = distortion_state(image_dr, image_u)
    statt = distortion_state(image_dt, image_u)
    print(statt)
    print(statr)
    print(statt[0] / statr[0])

    # valid = camera.valid_range(image_dr)
    # image_u = image_u[valid]
    # image_dr = image_dr[valid]
    # meanr, maxir = distortion_amount(image_u, image_dr)

    # image_dt = camera.distort(pattern, k1=-0.001, theta=np.radians(5/60))
    # valid = camera.valid_range(image_dt)
    # image_dr = image_dr[valid]
    # image_u = image_u[valid]
    # image_dt = image_dt[valid]
    # stat = distortion_amount(image_dr, image_dt)
    # print(stat)


def dist2beta():
    beta = np.linspace(1, 1.0001, 100)
    mean_dists = list()
    max_dists = list()
    for b in beta:
        ud = CameraDistortion(RES, FOCAL, PS)
        ud.gen_image_u()
        ud.gen_image_d(beta=b)
        mean, max = ud.cal_dist()
        mean_dists.append(mean)
        max_dists.append(max)
    plt.plot(beta, mean_dists, label='mean')
    plt.plot(beta, max_dists, label='max')
    plt.legend()
    plt.show()

def dist2gamma():
    gamma = np.linspace(0, 0.0001, 100)
    mean_dists = list()
    max_dists = list()
    for g in gamma:
        ud = CameraDistortion(RES, FOCAL, PS)
        ud.gen_image_u()
        ud.gen_image_d(gamma=g)
        mean, max = ud.cal_dist()
        mean_dists.append(mean)
        max_dists.append(max)
    plt.plot(gamma, mean_dists, label='mean')
    plt.plot(gamma, max_dists, label='max')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # distfixps()
    # dist2ds()
    # distfixfocal()
    # ancient
    # distsim(600, 6, 0.01)

    # hero4
    # distsim(3500, 3, 0.015)

     # hero12
    # distsim(1000, 2, 0.02)
    # exit()
    # dist2theta()


    # import matplotlib as mpl
    # mpl.rcParams['figure.figsize']=7/3, 7/3
    
    # dist2focal()
    # dist2res()
    # dist2ps()
    distfixfov()

    # tangential_3d()