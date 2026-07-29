

import numpy as np
import scipy

import ngsolve
#from ngsolve.webgui import Draw
import netgen.geom2d

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as mticker
import copy
from matplotlib.colors import  LinearSegmentedColormap, ListedColormap
#%matplotlib inline

# a set of helper routines for the 2d active membrane simulations

# This makes the stress tensor
# arguments:
#   strain: 2d strain tensor
#   mu: shear modulus
#   lam:  second Lame parameter
# returns:
#   stress tensor 2d
def Stress(strain,mu,lam):
    return 2*mu*strain + lam*ngsolve.Trace(strain)*ngsolve.Id(2)
    #https://docu.ngsolve.org/ngs24/SaS/linearelasticity.html

# create and return a bunch of useful operators (linear and bilinear forms/ngsolve)
# arguments:
#   fes: finite element system
#   mu:  shear modulus
#   lam:  second Lame parameter, sets bulk modulus
#   pi_k: spring constant
#   dt: timestep
# calls: lots of ngsolve routines, Stress()
def mkops_u(fes,mu,lam,pi_k,dt):
    u,v  = fes.TnT()

    strain_u = ngsolve.Sym(ngsolve.Grad(u))  # strain tensor created with u field , symmetrized Jacobian
    strain_v = ngsolve.Sym(ngsolve.Grad(v))
    stress_u = Stress(strain_u,mu,lam)  # stress tensor associated with u
    # note passing mu, lam Lame parameters here!
    
    a =  ngsolve.BilinearForm(fes)
    a += ngsolve.InnerProduct(stress_u, strain_v)  *ngsolve.dx
    # see https://docu.ngsolve.org/ngs24/SaS/linearelasticity.html
    a += pi_k*(u*v)*ngsolve.dx
    a.Assemble()

    m = ngsolve.BilinearForm(fes)  # to hold the mass matrix
    m += (u*v)*ngsolve.dx
    m.Assemble()
    invm = m.mat.Inverse(freedofs=fes.FreeDofs())  # inverse of mass matrix!

    astar = m.mat.CreateMatrix() # create a matrix in the form of m
    astar.AsVector().data = m.mat.AsVector() + 0.5*dt * a.mat.AsVector()
    invastar = astar.Inverse(freedofs=fes.FreeDofs())

    # linear form to hold non-linear interaction term
    f = ngsolve.LinearForm(fes)
    sp_fun = ngsolve.GridFunction(fes)
    f += sp_fun*v*ngsolve.dx

    # returning everything you need to run the time step
    return sp_fun,f,invm,a,invastar
    
# Do time stepping of PDE system!
# arguments:
#   fes: finite element space (2d)
#   gfu,gfp,gfu_vel :  grid functions, u is displacement, p is polarization, gfu_vel is velocity
#      u,p,vel are 2d vectors on fes
#      u,p are evolved, gfu_vel records du/dt
#   sp_fun, f, invm, a, invastar:  linear/bilinear forms/ops
#       except sp_fun which is a grid function that should be updated
#       sp_fun is used by the operator f
#   dt timestep, must be consistent with that used to generate a, invastar
#   AA, p0: parameters for non-linear function sp_fun (p0 is tilde p0 = p0/pi_alpha
#   t0,tend:  beginning and end times for integration
#   scount:  number of steps between outputs
#   gfu_list:  a prior run storing gfu s in a list
#   gfp_list:  a prior run storing gfp s in a list
#   gfv_list:  a prior run storing gfu_vec s  in a list
#   scene:  update display via a Drawgui object that should already be in place
#   save_list: Boolean, save lists or not
#   Wnoise:  add noise or not, if not none, then std of noise added should scale with sqrt(dt)
#     for noise to be a Wiener process, std = Wnoise*sqrt(dt)
#   alphanoise: spectrum index of added noise
# returns:
#   gfu_list,gfp_list displacement and tilt fields at different outputs but in a list
#   gfv_list velocity field at different outputs but in a list
# crank nicholson applied to stress/strain on u, op split for non-linear feedback term, p updated manually
# calls: mkinit_pink for noise
def time_stepping(fes,gfu,gfp,gfu_vel,sp_fun,f,invm,a,invastar,dt,AA,p0,\
        scene,t0=0,tend=1.0,scount=10,\
        gfu_list=[],gfp_list=[],gfv_list=[],save_list = True,Wnoise=None,alphanoise=0.1):
                  
    res_u = gfu.vec.CreateVector() #  base vector, used for storing intermediate steps of scheme
    res_f = gfu.vec.CreateVector() #
    res_p = gfu.vec.CreateVector() #
    time = t0  # initial time
    count = 0  # counts timesteps

    #gfu_list = [] # for storing a series of gfus
    
    std_noise = 0.0
    if (Wnoise != None):
        std_noise = np.sqrt(dt)*Wnoise

    while time < tend - 0.5*dt:

        res_u.data =  -dt*(a.mat * gfu.vec)  # this is -dt L u^n
 
        denom = AA/(1.0 + ngsolve.Norm(gfp)/p0 ) # is a CF
        sp_fun.Set(gfp *denom)  # set the non linear function directly into the sp_fun grid function
        # obeys Dirichlet BC
        #It is ok to use a Grid Function here, see https://docu.ngsolve.org/latest/i-tutorials/unit-3.2-navierstokes/navierstokes.html
        f.Assemble()  # create the interaction function linear form/operator
        res_f.data = dt*f.vec

        # Crank Nicolson update for elastic and damping term then op split for non-linear polarization dependent term
        gfu_vel.vec.data =  invastar*res_u  + invm*res_f   # velocity data except for factor of dt
        gfu.vec.data += gfu_vel.vec.data    # update displacement field
        res_p.data = dt*gfp.vec.data  # damping of polarization
        gfp.vec.data += gfu_vel.vec.data - res_p.data  # update polarization field

        #gfu_vel.vec.data /= dt   # velocity field , divide incremental of u by dt, maybe a bad idea to do this here
        
        # think about adding stochastic noise
        if (std_noise > 0.000001):
            noise = mkinit_pink(fes,std_noise,alphanoise,div0=False,rs=rs)
            gfp.vec.data += noise.vec.data
            # adding noise to tilt field as when I add it to displacement vorticity looks icky

        if (count % scount ==0):  # redraw fields only once in a while
            scene.Redraw()
            if (save_list == True):
                gfu_list = np.append(gfu_list,copy.copy(gfu))    # save a copy of the displacement field
                gfp_list = np.append(gfp_list,copy.copy(gfp))    # save a copy of the tilt field
                gfv_list = np.append(gfv_list,copy.copy(gfu_vel))    # save a copy of the velocity field (not corrected by dt)
            # if we don't use copy.copy you get the same gfu repeated in the list

        print("\r",time,end="")
        time += dt
        count += 1  #keep track of time steps
    return gfu_list,gfp_list,gfv_list

    
# compute vorticity from a velocity
# arguments:
#   ug: must be a 2d vector ngsolve grid function or perhaps a 2d coefficient function?
# returns:
#   coefficient function that is the curl of the 2d vector
def mycurl(ug):
    gug = ngsolve.Grad(ug) # Jacobian
    return gug[1,0] - gug[0,1]  #  this is the curl if ug is a 2d gridfunction
    # should be (partial_x u_y -partial_y u_x)  positive if counter clockwise
    # apparently the first index refers to the field and the second
    # index to the derivative?
   
# I want to compute u x p = u_x p_y - u_y p_x
# from two grid functions storing u, p
# arguments:
#  ug  gfu
#  pg  gfp
# returnds coefficient function that is u_x p_y - u_y p_x
def mk_ucrossp(ug,pg):
    ux = ug.components[0]
    uy = ug.components[1]
    px = pg.components[0]
    py = pg.components[1]
    return ux*py - uy*px  # probably a cf
    
# create a n x n real array with colored noise of spectral power p(k) propto k^-alpha (colored noise)
# for creating initial condition fields
# arguments:
#    n dimensions for square array
#    alpha: - index of power spectrum; determines color of noise in image
#      in most cases would be positive
#    sig:   standard deviation of returned image
#    rs:  random state if you want to pass it
# returns:
#  nxn real array - mean is set to zero, standard deviation scaled to sig
#  pink is 1/f which would have alpha=1, white is alpha near 0
# calls: numpy routines, scipy stats
# if you want a repeatable random sequence use this:    rs = np.random.RandomState(seed)
def mkpink(n,alpha,sig,rs=None):
    #whitenoise = np.random.uniform(0, 1, (n, n))
    # uniform random between [0,1] in an nxn grid
    whitenoise = scipy.stats.uniform.rvs(loc=0, scale=1, random_state=rs,size=(n,n))
    #rs = np.random.RandomState(seed)
    ft_arr = np.fft.fft2(whitenoise) # not using rfft2 because I need to construct square frequency array
    d=1.
    freqs = np.fft.fftfreq(n,d)  # with sample spacing of d=1
    k_x,k_y = np.meshgrid(freqs,freqs)
    #k =  np.hypot(k_x,k_y)  # sqrt(k_x^2 + k_y^2)
    k = np.sqrt(k_x**2 + k_y**2)
    jj = (k == 0 )
    k[jj] = 1e-6  # get rid of zero value
    pink_ft_arr = ft_arr / np.power(k,alpha/2)
    # fft_arr is fourier ammplitudes and amplitude is sqrt of power
    # if alpha=1 we have 1/f noise which is called pink noise
    pinknoise = np.real(np.fft.ifft2(pink_ft_arr)) # take real part
    meanval = np.mean(pinknoise)
    pinknoise -= meanval # substract off mean value
    std_pink = np.std(pinknoise)
    pinknoise *= sig/std_pink # set standard deviation  of returned image
    #print(ft_arr.shape)
    #print(k.shape)
    return pinknoise
    
# set incompressible (approximately) pinknoise initial conditions into gfu (2d)
# actually if div0=True
#   I generate a stream function that is pink then I take its derivatives
# or I generate pink noise for both components of gfu
# parameters:
#   fes: ngsolve finite element space, mesh is used from fes
#   sig: standard deviation of resulting noise
#   alpha: powerlaw spectral index of noise
#   div0:  True or False, if true generate pink and then use it as a stream function psi
#          u = -partial_y psi, partial_x psi
#          if False: return pink noise in both components ux, uy
#   rs a randomstate so you can seed the random number generator if you want
# returns: gfu with desired object which is not all that incompressible,
#      despite my efforts
#      is a vector field
# calls: mkpink(), ngsolve routines
def mkinit_pink(fes,sig,alpha,div0=True,pout=False,rs=None):

    # output fields
    gfu = ngsolve.GridFunction(fes)  # create new grid function
    u1_out = gfu.components[0]
    u2_out = gfu.components[1]
    r1  = u1_out.vec.CreateVector()  # create two vectors
    r2  = u1_out.vec.CreateVector()

    # use the bounding box to find min and max range of x and y values
    netgen_mesh = fes.mesh.ngmesh
    p1,p2 = netgen_mesh.bounding_box
    #print(p1,p2)
    xxmin = min(p1[0],p2[0])
    xxmax = max(p1[0],p2[0])
    yymin = min(p1[1],p2[1])
    yymax = max(p1[1],p2[1])  # we could use these to coordinate transform our noise
    nv = fes.mesh.nv
    dx = xxmax - xxmin
    dy = yymax - yymin
    dxymax = max(dx,dy)
    x2 = xxmin + dxymax
    y2 = yymin + dxymax
    dd = np.sqrt(dxymax**2/nv) # average distance between particles
    nft = int(2*dxymax/dd) # how big an fft we need
    if (pout==True):
        print('mkpink_init: nft=',nft)

    if (div0==True):  # try to make nearly divergence free
        pinknoise = mkpink(nft,alpha,sig,rs=rs)
        # make an fft image with appropriate noise in it
    
        paramCF = ngsolve.VoxelCoefficient(\
            start=(xxmin, yymin), end=(x2, y2),\
            values=pinknoise, linear=True).Compile()
        # returns a coefficient function which interpolates from the given grid of values

        junkuv = ngsolve.GridFunction(fes)
        ju  = junkuv.components[0]
        jv  = junkuv.components[1]
        ju.Set(paramCF)  # set to pink noise
        jv.Set(paramCF)  # set to same noise on purpose
    
        jac = ngsolve.Grad(junkuv)  # jacobian

        u1_out.Set(-1*jac[1,1])  # -d paramCF/dy
        u2_out.Set(jac[0,0])     # d paramCF/dx
        # this is supposed to give an incompressible field
        # but I find that div gfu is not all that close to zero
        
    else:  # set separate pink noises in ux and uy
        pinknoise1 = mkpink(nft,alpha,sig,rs=rs)
        pinknoise2 = mkpink(nft,alpha,sig,rs=rs)
        paramCF1 = ngsolve.VoxelCoefficient(\
            start=(xxmin, yymin), end=(x2, y2),\
            values=pinknoise1, linear=True).Compile()
        paramCF2 = ngsolve.VoxelCoefficient(\
            start=(xxmin, yymin), end=(x2, y2),\
            values=pinknoise2, linear=True).Compile()
        u1_out.Set(paramCF1)
        u2_out.Set(paramCF2) #  assume that this does not touch dirichlet boundary
        
    #gfu.Set((0.0,0.0), ngsolve.BND)
    # you can't do this as it zeros everything including boundary
    
    return gfu
    
# make some initial conditions
# calls: mkinit_pink(),  ngsolve routines
# arguments:
#   fes: finite element space
#   parm1,parm2: parameters
#   rs:  random state for random number generator
# returns gfu,gfp grid functions on fes
def mkinit(fes,parm1,parm2,ntype='pink',rs=None):
    # initial conditions
    gfu = ngsolve.GridFunction(fes)  # note making u and p separately as grid functions
    gfp = ngsolve.GridFunction(fes)
    gfu.Set((0.0,0.0))
    gfp.Set((0.0,0.0))

    dok=False; dorand=False; dopink=False;
    
    if (ntype=='pink'):
        dopink=True
    if (ntype=='dok'):
        dok=True
    if (ntype=='rand'):
        dorand=True
        
    if (dok==True):
        k = 2*np.pi/parm2
        mycf = parm1*ngsolve.cos(k*(ngsolve.x - ngsolve.y))  # incompressible sine wave
        gfu.Set((mycf,mycf))
        
    #following https://ngsolve.org/ngsolve/docs/i-tutorials/unit-2.2-eigenvalues/pinvit.html
    if (dorand==True):  # white noises from random fields, not trying to be incompressible
        r1  = gfu.vec.CreateVector()  # create some vectors
        #r1.SetRandom(); # fill with random numbers uniform distribution in [0,1]
        r1.FV().NumPy()[:] = np.random.normal(fes.ndof) *parm1 # fill with normal distn
        # scale by parm1 and set mean to be 0
        r2  = gfu.vec.CreateVector()  # create some vectors
        r2.FV().NumPy()[:] = np.random.normal(fes.ndof) *parm2
        #r2.SetRandom(); # fill with random numbers
        # scale by parm2
        #gfu.vec.data += ngsolve.Projector(fes.FreeDofs(), True) * r1  # respect Dirichlet boundary
        #gfp.vec.data += ngsolve.Projector(fes.FreeDofs(), True) * r2
        
        gfu.vec.data += r1
        gfp.vec.data += r2
        # this looks really bad! it sucks! don't do this!
        
    # should respect Dirichlet boundary
    if (dopink==True):
        alpha = parm2; sig=parm1
        gfu = mkinit_pink(fes,sig,alpha,rs=rs)
        gfp = mkinit_pink(fes,sig/100,alpha,rs=rs)
        #gfu.Set((0.0,0.0), ngsolve.BND)  # ensure that a Dirichlet boundary is not messed with
        #gfp.Set((0.0,0.0), ngsolve.BND)
        # see https://ngsolve.org/ngsolve/docs/i-tutorials/unit-1.3-dirichlet/dirichlet.html
        # BND is only for Dirichlet part of boundary
        # but this zeros the interior !!!!!!!!!!!!!!!

    return gfu,gfp


# helper routine for creating filenames in case we want to spew out many pngs
# assuming we will never go above 999
# returns a string filename
# arguments:
#   root:  string prefix
#   index: non-negative integer
def mkpngname(root,index):
    fnum = ''
    if (index < 10):
        fnum += '0'
    if  (index < 100):
        fnum += '0'
    ofile = root +  fnum + '{:d}'.format(index)   + '.png'
    print(ofile)
    if (index > 999):
        print('mkpngname: index > 999 problem!')
    return ofile

# get node points and triangles from a 2d ngsolve-netgen triangular mesh
# also return a refiner (from matplotlib.tri) which helps us make nice plots
# arguments:
#  mesh:  an NGsolve netgen mesh
# returns:
#  xv,yv mesh vertex 2d point positions
#  tri_tt: a python/matplotlib set of triangles
#  refiner: a refining function
def pts_tri(mesh):
    # find the x,y coordinates of the mesh
    xv = np.zeros(mesh.nv)
    yv = np.zeros(mesh.nv)
    k=0 # store nodes
    for v in mesh.vertices:
        #print(v,v.point)
        xv[k] = v.point[0]
        yv[k] = v.point[1]
        k+= 1
    # get a list of triangles in the mesh
    triangles = np.zeros((mesh.ne,3), dtype =int)  # number of elements
    k=0  # store triangles
    for el in mesh.Elements():
        triangles[k,:] = np.array([el.vertices[0].nr, el.vertices[1].nr, el.vertices[2].nr], dtype = int)
        k += 1
    tri_tt = tri.Triangulation(xv, yv,triangles)  # is a python triangle thing
    refiner = tri.UniformTriRefiner(tri_tt)  # refiner function
    return xv,yv,tri_tt,refiner
    
# plot displacement and tilt polarization fields with matplotlib
# we need a coarse mesh
# calls: pts_tri()
# arguments:
#  gfu,gfp:  displacement and polarization grid functions
#  mesh:  the netgen mesh we used for the grid function
#  coarsemesh: a coarser netgen mesh
#  time_label: a label to timestamp the plots
#  ofile: output filename, only makes a png if this is more than a few characters long
#  yratio: so you can adjust aspect ratio of plot
#  plot_dotprod:  If true show color map of u dot p
#  plt_crossprod:  If true show color map of u x p
#  vecscale_u, vecscale_p: to set size of quiver plot vectors
#  vlength_set_p, vlength_set_u:  for key vectors lengths
#  parmlabel: put a label on top
#  cbar_shrink: control color bar length
#  qkeyloc: control quiver key locations
def displot(gfu,gfp,mesh,coarsemesh,time_label,ofile,plt_dotprod=False,plt_crossprod=False,vecscale_u=None,\
            vlength_set_u = None, vecscale_p = None,vlength_set_p = None,yratio=1,\
            cbar_label='',cbar_shrink=0.7,parmlabel='',qkeyloc=None,fac=0.999):
    # plot displacement vectors and vorticity
    plt_pvectors = True  # plot pol vectors
    plt_uvectors = True  # plot displacement vectors

    ucolor='red'   # color choices for arrows
    pcolor='blue'
    
    if (qkeyloc == None):
        xq0 = 0.91  # quiver key location defaults
        yq0 = 0.91
        xq1 = 0.91
        yq1 = 0.82
        ldir = 'N'
    else:
        xq0 = 0.9  # quiver key backup location?
        yq0 = 1.04
        xq1 = 0.75
        yq1 = 1.04
        ldir='N'
    
    # this is the color map for u dot p
    colors = [ "xkcd:light lavender", "lightcyan","white","mistyrose","xkcd:light gold" ]
    cmapuv = LinearSegmentedColormap.from_list("mycmap", colors)
    
    ux = gfu.components[0]   # displacement components
    uy = gfu.components[1]
    px = gfp.components[0]   # pol components
    py = gfp.components[1]

    xv,yv,tri_tt,refiner = pts_tri(mesh) # is fast
    dx = np.max(xv) - np.min(xv)
    dy = np.max(yv) - np.min(yv)
    #yratio = dy/dx
    
    fig,ax= plt.subplots(1,1,figsize=(4,4*yratio),sharex=True,sharey=True,dpi=200)
    #if (qkeyloc==None):
    plt.subplots_adjust(left=0.11,right=0.92,bottom=0.1,top=0.99)
    #else:
    #    plt.subplots_adjust(left=0.11,right=0.92,bottom=0.1,top=0.97)
    ax0=ax; ax1=ax;
    ax0.set_aspect('equal'); # ax1.set_aspect('equal');

    xv_coarse,yv_coarse,tri_tt_coarse,refiner_coarse = pts_tri(coarsemesh)
    # we had too many arrows, but a coarse mesh works to give fewer arrows
    
    if (plt_dotprod==True):  # plot u dot v in color
        #cbar_shrink = 0.7
        dotprod_vals = np.zeros(mesh.nv)
        u_dot_p = ngsolve.InnerProduct(gfu,gfp)  /(ngsolve.Norm(gfu)*ngsolve.Norm(gfp) + 1e-10)
        for k in range(mesh.nv):
            dotprod_vals[k] = np.arccos(u_dot_p(mesh(xv[k],yv[k])))
        tri_refi, dotprod_refi = refiner.refine_field(dotprod_vals, subdiv=2)
        ax.tripcolor(tri_refi, dotprod_refi, cmap=cmapuv,zorder=1,vmin=0,vmax=np.pi,\
            shading='gouraud') # show field
        field0 = ax.get_children()[0]  # vertex-based temperature-colour
        fig.colorbar(field0,shrink=cbar_shrink,label=cbar_label,pad=0.02,\
                ticks=[0, np.pi/2, np.pi], format=mticker.FixedFormatter(['0',r'$\pi/2$', r'$\pi$']))
                
    if (plt_crossprod==True):  # plot u x v in color
        #cbar_shrink = 0.7
        cprod_vals = np.zeros(mesh.nv)
        u_cross_p = mk_ucrossp(gfu,gfp)
        for k in range(mesh.nv):
            cprod_vals[k] = u_cross_p(mesh(xv[k],yv[k]))
        tri_refi, cprod_refi = refiner.refine_field(cprod_vals, subdiv=2)
        cmax = 3
        ax.tripcolor(tri_refi, cprod_refi, cmap='PiYG',zorder=1,vmin=-cmax,vmax=cmax,\
            shading='gouraud') # show field
        field0 = ax.get_children()[0]  # vertex-based temperature-colour
        fig.colorbar(field0,shrink=cbar_shrink,label=cbar_label,pad=0.02,\
                ticks=[-cmax, 0, cmax], format=mticker.FixedFormatter(['-3','0', '3']))

    #fac = 0.999  # so that coarse mesh does not screw up by giving zeros on all vectors
    if (plt_uvectors==True):  # plot u vectors
        
        ux_vals = np.zeros(coarsemesh.nv)
        uy_vals = np.zeros(coarsemesh.nv)
        
        for k in range(coarsemesh.nv):
            ux_vals[k] = ux(coarsemesh(fac*xv_coarse[k],fac*yv_coarse[k]))
            uy_vals[k] = uy(coarsemesh(fac*xv_coarse[k],fac*yv_coarse[k]))
            
        if (vecscale_u==None):
            qob0 = ax0.quiver(xv_coarse,yv_coarse,ux_vals,uy_vals,color=ucolor,zorder=2,width=0.004)  # plot arrows
            vlength = np.max(uy_vals)
        else:
            qob0 = ax0.quiver(xv_coarse,yv_coarse,ux_vals,uy_vals,color=ucolor,zorder=2,scale_units='xy',\
                scale=vecscale_u,width=0.004)
            vlength = vlength_set_u
       
        label = 'u{:.1f}'.format(vlength)
        ax0.quiverkey(qob0, xq0, yq0, vlength, label=label,coordinates='axes',labelsep=0.03,labelpos=ldir)   # show an arrow key
        
    if (plt_pvectors==True):  # plot p vectors
        
        px_vals = np.zeros(coarsemesh.nv)
        py_vals = np.zeros(coarsemesh.nv)
        for k in range(coarsemesh.nv):
            px_vals[k] = px(coarsemesh(fac*xv_coarse[k],fac*yv_coarse[k]))
            py_vals[k] = py(coarsemesh(fac*xv_coarse[k],fac*yv_coarse[k]))
            
        if (vecscale_p==None):
            qob1 = ax1.quiver(xv_coarse,yv_coarse,px_vals,py_vals,color=pcolor,zorder=2,width=0.004)  # plot arrows
            vlength = np.max(py_vals)
        else:
            qob1 = ax1.quiver(xv_coarse,yv_coarse,px_vals,py_vals,color=pcolor,zorder=2,scale_units='xy',\
                scale=vecscale_p,width=0.004)
            vlength =vlength_set_p
        
        label = 'p{:.1f}'.format(vlength)
        ax1.quiverkey(qob1, xq1, yq1, vlength, label=label,coordinates='axes',labelsep=0.03,labelpos=ldir)
        # show an arrow key

    ax0.text(0.03,0.03, time_label,transform=ax0.transAxes) # label time
    if (len(parmlabel)>3):
        ax.text(0,1.03, parmlabel, transform=ax.transAxes) # label parameters

    if (len(ofile)>3):
        plt.savefig(ofile)

    plt.show()

# a helper routine to show the mesh alone as a png file
# arguments:
#    mesh:  netgen mesh
#    boundaries:  a string that is list of boundaries like 'outer|inner'
#    ofile: output filename for a png
def draw_mesh(mesh,d_boundaries,ofile,n_boundaries=''):

    xv,yv,tri_tt,refiner = pts_tri(mesh) # is fast, why not put it here
    dx = np.max(xv) - np.min(xv)
    dy = np.max(yv) - np.min(yv)
    yratio = dy/dx
    
    fig,ax = plt.subplots(1,1,figsize=(5,5*yratio),dpi=400)
    plt.subplots_adjust(left=0.01,right=0.99)
    ax.set_xlim([ 1.01*np.min(xv), 1.01*np.max(xv)])
    ax.set_ylim([ 1.01*np.min(yv), 1.01*np.max(yv)])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    ax.set_aspect(1.0)
   
    lw = 0.3

    # plot mesh
    pall = True
    if (pall==True):
        triangles = np.zeros((mesh.ne,3), dtype =int)  # number of elements
        k=0
        for el in mesh.Elements():
            triangles[k,:] = np.array([el.vertices[0].nr, el.vertices[1].nr, el.vertices[2].nr], dtype = int)
            k += 1
        for k in range(mesh.ne):
            x1 = xv[triangles[k,0]];y1 = yv[triangles[k,0]];
            x2 = xv[triangles[k,1]];y2 = yv[triangles[k,1]];
            x3 = xv[triangles[k,2]];y3 = yv[triangles[k,2]];
            ax.plot([x1,x2],[y1,y2],'k-',lw=lw)
            ax.plot([x1,x3],[y1,y3],'k-',lw=lw)
            ax.plot([x2,x3],[y2,y3],'k-',lw=lw)
            
    d_bound = False
    n_bound = False
    # only plot Dirichlet boundaries in red if boundaries argument is longer than 2
    if (len(d_boundaries) > 2):  # assuming these are Dirichlet and these in red
        d_bound = True
    if (len(n_boundaries) > 2):  # assuming these are Neumannt and these in blue
        n_bound = True

    # overplot boundary in red
    if (d_bound==True):
        edges = np.zeros((mesh.ne,2), dtype =int)  # number of elements
        k=0
        for el in mesh.Boundaries(d_boundaries).Elements():
            edges[k,0] = el.vertices[0].nr
            edges[k,1] = el.vertices[1].nr
            k +=1
        nne = k
        #print(nne)
        edges = edges[0:nne,:]
        for k in range(nne):
            x1 = xv[edges[k,0]];y1 = yv[edges[k,0]];
            x2 = xv[edges[k,1]];y2 = yv[edges[k,1]];
            ax.plot([x1,x2],[y1,y2],'r-',lw=lw*2)
            
    # overplot boundary in blue
    if (n_bound==True):
        edges = np.zeros((mesh.ne,2), dtype =int)  # number of elements
        k=0
        for el in mesh.Boundaries(n_boundaries).Elements():
            edges[k,0] = el.vertices[0].nr
            edges[k,1] = el.vertices[1].nr
            k +=1
        nne = k
        #print(nne)
        edges = edges[0:nne,:]
        for k in range(nne):
            x1 = xv[edges[k,0]];y1 = yv[edges[k,0]];
            x2 = xv[edges[k,1]];y2 = yv[edges[k,1]];
            ax.plot([x1,x2],[y1,y2],'b-',lw=lw*2)

    # make a png
    if (len(ofile) >3):
        plt.savefig(ofile)

    plt.show()
    
# make a time vector for the outputs
def mktimevec(n,scount,dt):
    timevec = (np.arange(n)+1)*scount*dt
    return timevec
    

# compute a function that is 1 if distance between points is in a certain range
# arguments:
#   x0,y0 is a position (of a vertex)
#   xv,yv are positions of a bunch of other vertices
#   r is a distance
#   dr is a width of distance shell
# returns: a list of 1s or 0s with 1 if
#     r  < |x_i - x_v | < r + dr
#     for the vertices that satisfy this condition
def S_shell(r,dr,x0,y0,xv,yv):
    dist = np.sqrt( (xv - x0)**2 + (yv - y0)**2)  # should work on vectors, is a vector
    dist1 = (dist >= r)
    dist2 = (dist <= r+dr)
    both = np.array(np.logical_and(dist1,dist2),dtype=int)  # convert to 0,1s
    #total = np.sum(both)
    #print(total)
    return both
    
# construct a spatial correlation function
# calls: above S_shell(),  pts_tri()
# arguments:
#   gfu grid function
#   mesh
#   x0,y0 position
#   dr distance range for radial shells
# returns:
#  radii of shells and spatial correlation function at those distances
def spatial_corr(gfu,mesh,x0,y0,dr):
    #x0 = 0
    #y0 = 0
    xv,yv,tri_tt,refiner = pts_tri(mesh)
    nv = len(xv)  # number of vertices
    dist = np.sqrt( (xv - x0)**2 + (yv - y0)**2)
    rmax = np.max(dist)  # compute maximum distance
    rarr = np.arange(0,rmax,dr)
    carr = rarr * 0.0
    darr = np.zeros(nv)  # storing dot products of u_i u_j

    u1 = gfu(mesh(x0,y0)) # value of field at x0,y0
    # compute an array of dot products, loop over vertices
    for k in range(nv):
        u2 = gfu(mesh(xv[k],yv[k])) # value of field at xk,yk
        darr[k] = np.dot(u1,u2)  # dot product of u(x_0), u_(x_k)

    # fill array of correlation as a function of distance, loop over radii
    for k in range(len(carr)):
        r = rarr[k]   # for each distance
        SS = S_shell(r,dr,x0,y0,xv,yv)  # tells us which vertices are in range
        # is a vector of length nv
        ns = np.sum(SS) # to normalize
        if (ns >0):
            carr[k] = np.sum(darr*SS)/ns
        # array remains 0 if there are no points that contribute
            
    return rarr, carr
        
# plot vorticity and velocity vectors with matplotlib
# we need a coarse mesh
# dt is to correct gfu_vel so it is actually a velocity
# calls: mycurl, pts_tri
# arguments:
#  gfu_vel:  velocity grid function (except off by dt)
#  mesh:  the mesh we used for grid function
#  coarsemesh: a coarser mesh
#  dt: timestep
#  time_label: a label to timestamp the plots
#  ofile: output filename, only makes a png if this is more than a few characters long
#  vecscale: for scaling sizes of vectors
#  vlength_set: for setting the arrow scale on the plot
#  vmaxx:
#  divdt: if True then divide by this so that velocity is units of velocity
#         if False then assume that you are passed the field you want to plot
#  cbar_shrink: to adjust size of colorbar
#  figy: to scale size of plot in y dims
#  qkeyloc: if you need to adjust quiverkey location
def vorplot(gfu_vel,mesh,coarsemesh,dt,time_label,ofile,vecscale=None,\
            vlength_set = None,parmlabel='',vmaxx=None,cbar_shrink=0.8,divdt=True,\
            cbar_label='',figy=1.0,qkeyloc=None,fac=0.999):

    plt_vorticity = True  # plot vorticity
    plt_uvectors = True  # plot velocity vectors
    
    if (qkeyloc == None):
        xq = 0.9  # quiver key location defaults
        yq = 0.88
        ldir = 'N'
    else:
        xq = 0.9  # quiver key backup location?
        yq = 1.04
        ldir='N'

    utx = gfu_vel.components[0]   # velocity components
    uty = gfu_vel.components[1]
    vorticity = mycurl(gfu_vel)

    xv,yv,tri_tt,refiner = pts_tri(mesh) # is fast
    #xjunk = utx(mesh(xv[0],yv[0]))

    xv_coarse,yv_coarse,tri_tt_coarse,refiner_coarse = pts_tri(coarsemesh)
    # we had too many arrows, but a coarse mesh works to give fewer arrows

    yratio = (np.max(yv) - np.min(yv))/(np.max(xv) - np.min(xv))
    
    fig,ax= plt.subplots(1,1,figsize=(4,4*figy),sharex=True,sharey=True,dpi=200)
    plt.subplots_adjust(left=0.11,right=0.92,bottom=0.1,top=0.99)
    ax.set_aspect('equal');

    if (plt_vorticity==True):
        vorticity_vals = np.zeros(mesh.nv)
        for k in range(mesh.nv):
            vorticity_vals[k] = vorticity(mesh(xv[k],yv[k]))
        if (divdt==True):
            vorticity_vals /= dt  # note dt!
        sig = np.std(vorticity_vals)
        mm = np.mean(vorticity_vals)
        tri_refi, vorticity_refi = refiner.refine_field(vorticity_vals, subdiv=2)
        if (vmaxx ==None):
            zmax = np.max(np.abs(vorticity_vals))
        else:
            zmax = vmaxx
        ax.tripcolor(tri_refi, vorticity_refi, cmap='rainbow',zorder=1  ,vmin=-zmax,vmax=zmax)  # show field on triangles
        field0 = ax.get_children()[0]  # vertex-based temperature-colour
        fig.colorbar(field0,shrink=cbar_shrink,label=cbar_label,pad=0.02)  # colorbar!
        #print("List of the child Artists of this Artist \n",\
        #      *list(ax.get_children()), sep ="\n")
        
    if (plt_uvectors==True):
        utx_vals = np.zeros(coarsemesh.nv)
        uty_vals = np.zeros(coarsemesh.nv)
        #fac = 0.999  # make sure coarse mesh points are in its own domain
        # this only works in convex domains centered at the origin for the coarsemesh
        for k in range(coarsemesh.nv):
            utx_vals[k] = utx(coarsemesh(fac*xv_coarse[k],fac*yv_coarse[k]))
            uty_vals[k] = uty(coarsemesh(fac*xv_coarse[k],fac*yv_coarse[k]))
        if (divdt==True):
            utx_vals /= dt  # correct units of velocity
            uty_vals /= dt
        if (vecscale==None):
            qob = ax.quiver(xv_coarse,yv_coarse,utx_vals,uty_vals,zorder=2)  # plot arrows, autoscale
            vlength = np.max(uty_vals)
        else:
            qob = ax.quiver(xv_coarse,yv_coarse,utx_vals,uty_vals,zorder=2,scale_units='xy', scale=vecscale)
            vlength = vlength_set
        #xq = 0.9
        #yq = 0.88
        label = '{:.1f}'.format(vlength)
        ax.quiverkey(qob, xq, yq, vlength, label=label,coordinates='axes',labelsep=0.02,labelpos=ldir)
        # show an arrow key
        
    ax.text(0.03,0.03, time_label,transform=ax.transAxes) # label time
    if (len(parmlabel)>3):
        ax.text(0,1.03, parmlabel, transform=ax.transAxes) # label parameters

    if (len(ofile)>3):
        plt.savefig(ofile)

    plt.show()


# plot a series of images of curl of field and  vectors with matplotlib
# we need a coarse mesh
# calls: mycurl, pts_tri
# arguments:
#  gfu_list:  list of grid functions (could be off by dt)
#  mesh:  the mesh we used for grid function
#  coarsemesh: a coarser mesh
#  dt: timestep
#  time_vec: a set of times
#  vecscale, vlength_set to adjust vectors
#  vmaxx hard set image range
#  parmlabel a lable for one image
#  cbar_shrink: adjust colorbar
#  divdt: Boolean/whether or not to divide by dt
#  ofile: output filename, only makes a png if this is more than a few characters long
#
#
# note: shows curl as image and vectors of whatever field you give it
def vorseries(gfv_list,mesh,coarsemesh,dt,ofile,indexlist,timevec,\
              vecscale=None,vlength_set = None,vmaxx=0.5,parmlabel='',cbar_shrink=0.3, divdt=True):

    nindex = len(indexlist)
    xv,yv,tri_tt,refiner = pts_tri(mesh) # is fast
    #xjunk = utx(mesh(xv[0],yv[0]))

    xv_coarse,yv_coarse,tri_tt_coarse,refiner_coarse = pts_tri(coarsemesh)
    # we had too many arrows, but a coarse mesh works to give fewer arrows

    xsize = 3.0
    yratio = (np.max(yv)-np.min(yv))/ (np.max(xv)-np.min(xv))

    fig,axarr= plt.subplots(nindex,1,figsize=(xsize*1.4,xsize*yratio*nindex),\
                            sharex=True,sharey=True,dpi=200)
    plt.subplots_adjust(left=0.02,right=0.94,bottom=0.03,top=0.95,wspace=0,hspace=0)
    for k in range(nindex):
        axarr[k].set_aspect('equal');
    for k in range(0,nindex-1):
        axarr[k].get_xaxis().set_visible(False)
        axarr[k].get_yaxis().set_visible(False)
        
    axarr[0].set_xlim([np.min(xv),np.max(xv)])
    axarr[0].set_ylim([np.min(yv),np.max(yv)])

    zmax = vmaxx  # hard set maximum display range for vorticity scale

    for j in range(nindex):
        gfu_vel = gfv_list[indexlist[j]]
        utx = gfu_vel.components[0]   # velocity components
        uty = gfu_vel.components[1]
        vorticity = mycurl(gfu_vel)  # compute curl
        
        vorticity_vals = np.zeros(mesh.nv)
        for k in range(mesh.nv):
            vorticity_vals[k] = vorticity(mesh(xv[k],yv[k]))
        if (divdt == True):
            vorticity_vals /= dt  # note dt!
        sig = np.std(vorticity_vals)
        mm = np.mean(vorticity_vals)
        tri_refi, vorticity_refi = refiner.refine_field(vorticity_vals, subdiv=2)
        #if (j==0):
        #    zmax = np.max(np.abs(vorticity_vals))*1.3
        axarr[j].tripcolor(tri_refi, vorticity_refi, cmap='rainbow',zorder=1  ,vmin=-zmax,vmax=zmax)
        # show field on triangles
        field0 = axarr[j].get_children()[0]  # vertex-based temperature-colour
        #fig.colorbar(field0,shrink=cbar_shrink)  # colorbar!

        utx_vals = np.zeros(coarsemesh.nv)
        uty_vals = np.zeros(coarsemesh.nv)
        fac = 0.999
        for k in range(coarsemesh.nv):
            utx_vals[k] = utx(coarsemesh(fac*xv_coarse[k],fac*yv_coarse[k]))
            uty_vals[k] = uty(coarsemesh(fac*xv_coarse[k],fac*yv_coarse[k]))
        if (divdt == True):
            utx_vals /= dt  # correct units of velocity
            uty_vals /= dt
        if (vecscale==None):
            qob = axarr[j].quiver(xv_coarse,yv_coarse,utx_vals,uty_vals,zorder=2)  # plot arrows, autoscale
            vlength = np.max(uty_vals)
        else:
            qob = axarr[j].quiver(xv_coarse,yv_coarse,utx_vals,uty_vals,zorder=2,\
                                  scale_units='xy', scale=vecscale)
            vlength = vlength_set
        xq = 0.9
        yq = 0.88
        label = '{:.1f}'.format(vlength)
        if (j==0):
            axarr[j].quiverkey(qob, xq, yq, vlength, label=label,coordinates='axes',labelsep=0.02)
            # show an arrow key

        time = timevec[indexlist[j]]
        time_label = 't={:.1f}'.format(time)
        axarr[j].text(0.03,0.03, time_label,transform=axarr[j].transAxes,fontsize=14) # label time

    field0 = axarr[0].get_children()[0]  # vertex-based temperature-colour
    fig.colorbar(field0,ax = axarr[:],shrink=cbar_shrink,pad=0.03, anchor=(0.0,0.05))  # colorbar!

    if (len(parmlabel)>3):
        axarr[0].text(0,1.03, parmlabel, transform=axarr[0].transAxes,fontsize=14) # label parameters

    if (len(ofile)>3):
        plt.savefig(ofile)

    plt.show()
    
    
# store simulation parameters
class sim_stuff():
    def __init__(self,mu,lam,pi_k,dt,p0,AA,scount,parmlabel,Wnoise=0.0):
        self.mu = mu
        self.lam = lam
        self.pi_k = pi_k
        self.dt = dt
        self.p0 = p0
        self.AA = AA
        self.scount = scount
        self.parmlabel = parmlabel
        self.Wnoise = Wnoise
    def ret_info(self):
        if (self.Wnoise == 0.0):
            return self.mu, self.lam, self.pi_k, self.dt, self.p0, self.AA, self.scount, self.parmlabel
        else:
            return self.mu, self.lam, self.pi_k, self.dt, self.p0, self.AA, self.scount, self.parmlabel, self.Wnoise
