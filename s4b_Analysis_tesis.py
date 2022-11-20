# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:00:38 2022

@author: Pablo
"""


import os
import cv2
import pickle
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

from pytracker.video_utils import MultiWormTracker
import pytracker.video_utils as vutils


# ... Filenames ... 
DIR_NAME          = './videos/Carla_EC/Carla_N2_EC_2211101415_002'
BLOB_FILENAME     = os.path.join( DIR_NAME, 'video_data_blobs.pkl')
BLOB_REF_FILENAME = os.path.join( DIR_NAME, 'video_reference_contour.pkl')
TRAJ_FILENAME     = os.path.join( DIR_NAME, 'trajectories.pkl')
NPZ_FILENAME      = os.path.join( DIR_NAME, 'trajectories.pkl.npz')
IMG_FILENAME      = os.path.join( DIR_NAME, 'trajectories.png')
BKGD_FILENAME     = os.path.join( DIR_NAME, 'video_fondo.png')
ROIS_FILENAME     = os.path.join( DIR_NAME, 'rois.pkl')


#... Parameters
AVG_WDW = 20
MIN_LENGTH    = 150 # in frames
MIN_DIST   = 5 # in mm
XY_SCALE = 40 # px/mm
T_SCALE  = 2 #fps

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def draw_circle( p0 , r0, *args, **kwargs ):
    t = np.linspace( -np.pi, np.pi, 100)
    x = p0[0] + r0*np.cos(t)
    y = p0[1] + r0*np.sin(t)
    return plt.plot( x,y,*args, **kwargs)

def rotation_matrix(theta):
    return np.array((( np.cos(theta), - np.sin(theta)), ( np.sin(theta),  np.cos(theta))))

def generate_brownian( N , steps, c, p0, s0 , speed=[1,]):
    random_part = 2 * np.random.rand( N , steps , 2 ) - 1
    X = np.zeros_like( random_part ) + p0
    
    for t in tqdm(np.arange(1,steps-1,1)):
        v_final = speed[ np.random.randint(len(speed))]
        v_source = s0-X[:,t-1,:]
        v_notnorm = random_part[:, t,:] + c*v_source
        v_norm = v_final * v_notnorm / np.linalg.norm(v_notnorm, axis=1, keepdims=True)
        X[:,t+1,:] = X[:,t,:] + v_norm
    return X

def chemotaxis_index( X ):
    pos = np.sum( X[:,:,0]>0, axis=0) + 1e-90
    neg = np.sum( X[:,:,0]<0, axis=0) + 1e-90
    return (pos - neg)/(pos+neg)


def discard_nan( array):
    idx = np.nonzero( np.logical_not (np.isnan(array) ) )
    return array[idx]

def gaussian_pdf(x, mu, s):
    return np.exp(-(x - mu)**2/(2*s**2))/(np.sqrt(2*np.pi)*s)

def time_to_goal( X , p0 , min_dist ):
    d_to_goal = np.sqrt( (X[:,:,0]- p0[0])**2 + (X[:,:,1]-p0[1])**2)
    t_to_goal = np.ones( (X.shape[0],))

    for worm_jj in range( X.shape[0] ):
        tini = np.argwhere( X[worm_jj,:,0] >-9999).min()
        tfin = np.argwhere(d_to_goal[worm_jj,:] < min_dist)
        if len(tfin)>0:
            t_to_goal[worm_jj] = (tfin.min() - tini) 
        else:
            t_to_goal[worm_jj] = X.shape[1]
    return t_to_goal






# Load and rescale data
data = np.load( NPZ_FILENAME)['data']
data = data / XY_SCALE
# ... compute number of worms and frames
n_worms, n_frames , _ = data.shape
data_t = np.array( range(n_frames)) / T_SCALE / 60

# ... repeat for rois positions
with open( ROIS_FILENAME, 'rb') as f:
    ROIS = pickle.load( f) 
    plate = ROIS[0] / XY_SCALE
    chunk = ROIS[1]/ XY_SCALE
    

# Compute speeds
v = np.sqrt( np.diff( data[:,:,0], axis=1)**2 + np.diff( data[:,:,1], axis=1)**2 )
w = np.arctan2( np.diff( data[:,:,1], axis=1), np.diff( data[:,:,0], axis=1) )

#... do the running average on coordinates and speeds
data_xy = np.nan*np.zeros( (n_worms, n_frames-AVG_WDW+1, 2))
data_vw = np.nan*np.zeros( (n_worms, n_frames-AVG_WDW, 2) )

for jj in range( data.shape[0] ):
    data_vw[jj,:,0] = moving_average( v[jj,:], AVG_WDW) * T_SCALE
    data_vw[jj,:,1] = moving_average( w[jj,:], AVG_WDW) * T_SCALE

    for kk in range(data.shape[2]):
        data_xy[jj,:,kk] = moving_average( data[jj,:,kk], AVG_WDW)     





# Select trajectories with lengths less than MIN_LENGTH
# plt.hist( traj_length, bins=np.linspace(0,1000,30) )
traj_length = np.sum( data_xy[:,:,0]>0, axis=1)
I_long = np.argwhere( traj_length > MIN_LENGTH )[:,0]


# Select trajectories that start close to the chunk
# plt.hist( dst_start, bins=np.linspace(0,20,30) )
idx_start = (data_xy[:,:,0]>0).argmax(axis=1) #... find at which idx each trajectory starts
pos_start = np.array( [ data_xy[_, idx_start[_],: ] for _ in range(n_worms) ] )
dst_start = np.sqrt( (pos_start[:,0]-chunk[0])**2 + (pos_start[:,1]-chunk[1])**2 )
I_close = np.argwhere( dst_start > MIN_DIST )[:,0]

# Select trajectories that start inside the plate
# plt.hist( dst_start, bins=np.linspace(0,20,30) )
idx_within = (data_xy[:,:,0]>0).argmax(axis=1) #... find at which idx each trajectory starts
pos_within = np.array( [ data_xy[_, idx_within[_],: ] for _ in range(n_worms) ] )
dst_within = np.sqrt( (pos_within[:,0]-plate[0])**2 + (pos_within[:,1]-plate[1])**2 )
I_within  = np.argwhere( dst_within < 0.85*plate[2] )[:,0]



######## FIGURE 1 - PANEL A ########
fondo = vutils.load_background(BKGD_FILENAME)
color = plt.cm.hsv(np.linspace(0, 1, n_worms))

plt.figure( figsize=(6,6), dpi=600 )
#... draw background image
plt.imshow(  1.2*fondo , cmap='gray', vmax=255, vmin=0)
#... draw ROIS
draw_circle( [XY_SCALE*chunk[0], XY_SCALE*chunk[1]], XY_SCALE*MIN_DIST ,'r', lw=1, alpha=0.8)
draw_circle( [XY_SCALE*plate[0], XY_SCALE*plate[1]], XY_SCALE*0.85*plate[2] ,'r', lw=1, alpha=0.8)
plt.plot( XY_SCALE*plate[0], XY_SCALE*plate[1], '+r', ms=5, zorder=1000)
#... draw all trajectories in black
plt.plot( XY_SCALE*data_xy[:,:,0].T, XY_SCALE*data_xy[:,:,1].T, lw=0.5, c='k');
#... draw only good trajectories in colorines
I_good = [ idx for idx in range(n_worms) if idx in I_long and idx in I_close and idx in I_within ]
rr = np.random.permutation(n_worms)
for ii, idx in enumerate( I_good ):
    plt.plot( XY_SCALE*data_xy[idx,:,0].T, XY_SCALE*data_xy[idx,:,1].T, lw=0.75, c=0.9*color[rr[ii],:])

plt.xticks( ticks=plt.xticks()[0], labels='')
plt.yticks( ticks=plt.yticks()[0], labels='')
plt.xlim(0, fondo.shape[1])
plt.ylim(0, fondo.shape[0])
plt.gca().invert_yaxis()



######## FIGURE 1 - PANEL B ########
col_good = color[16]
col_all = np.array([1,1,1])*0.25
density = False

plt.figure( figsize=(8,4), dpi=600)

fracs = np.array( [ len(I_good)/n_worms,                  #... fraction of good trajectories
          np.sum( data_xy[I_good,:,0]>0)/np.sum( data_xy[:,:,0]>0) #... frac of good time points
          ])
plt.subplot(2,2,1)
plt.bar( range(2), [1,1], color=col_all)
plt.bar( range(2), fracs , color=col_good)
plt.xticks( range(2), labels=['Trajectories','Points'] )
plt.ylabel('Fraction')
    
plt.subplot(2,2,2)
bbins = np.linspace(0,10,20)
plt.hist( np.clip( traj_length/T_SCALE/60         , bbins[0], bbins[-1]) , bins=bbins , color=col_all  ,density=density)
plt.hist( np.clip( traj_length[I_good]/T_SCALE/60 , bbins[0], bbins[-1]) , bins=bbins , color=col_good ,density=density)
plt.xlabel('Trajectory lengths (min)')
plt.ylabel('Number of\ntrajectories')

plt.subplot(2,2,3)
bbins = np.linspace(0,0.18,10) 
plt.hist( np.clip( np.nanmean(  data_vw[:,:,0]     , axis=1), bbins[0], bbins[-1]) , bins=bbins, color = col_all, density=density )
plt.hist( np.clip( np.nanmean(  data_vw[I_good,:,0], axis=1), bbins[0], bbins[-1]) , bins=bbins, color= col_good,density=density )
plt.xlabel('Mean trajectory speed (mm/s) ')
plt.ylabel('Number of\ntrajectories')
plt.xlim( bbins[0], bbins[-1] )

d_traveled = np.nansum( data_vw[:,:,0], axis=1)/T_SCALE
plt.subplot(2,2,4)
bbins = np.linspace(0,50,12)
plt.hist( np.clip( d_traveled        , bbins[0], bbins[-1]) , bins=bbins, color = col_all, density=density )
plt.hist( np.clip( d_traveled[I_good], bbins[0], bbins[-1]) , bins=bbins, color= col_good, density=density )
plt.xlabel('Distance traveled (mm) ')
plt.ylabel('Number of\ntrajectories')
plt.xlim( bbins[0], bbins[-1] )

plt.tight_layout()





######## FIGURE 2 - PANEL A ########
# Compute Chemotaxis index for our population of worms
#... center and rotate trajectories
R = rotation_matrix( np.arctan2( chunk[1]-plate[1] , chunk[0]-plate[0] ) )
X = np.matmul( data_xy[I_good,:,:] - plate[:2] , R)

#... ensure that worms that reach NaCl persist
for i in range(X.shape[0]):
    t_dead = np.argwhere( np.diff( 1.0*( X[i,:,0]>-9999)) ==  -1 )
    if len(t_dead)>0:
        t_ini = t_dead.max()
        X[i,t_ini:,:] = X[i, t_ini ,:]

#... compute dynamic chemotaxis index
C = chemotaxis_index( X )

#... measure the final chemotaxis index in the last 15min
x = data_t[:-(AVG_WDW-1)]
v = np.mean(C[-2000:])
vs= np.std(C[-2000:])
print('Final chemotaxis index %1.4f +/- %1.4f' % (v,vs) )

#... plot
plt.figure( figsize=(5,4), dpi=600)
plt.plot( x, C,'k', lw=0.5)
plt.plot( x[-2000:], np.zeros_like(x[-2000:])+v, 'r', lw=0.5)
plt.ylim(-1.04,1.04)
plt.xlabel('Time (min)')
plt.ylabel('Chemotaxis index C')



######## FIGURE 2 - PANEL B ########

# Compute the time to goal of the real population
t_goal = time_to_goal( data_xy[:,:,:], chunk[:2], 3.5) / T_SCALE / 60

#... fit the histogram to a mixture of gaussians
gmm = GaussianMixture(3, 
                     covariance_type='full', 
                     random_state=0).fit( np.expand_dims(t_goal,axis=-1) )


x=np.linspace(0,100,10000)
color = plt.cm.turbo(np.linspace(0, 1, 4) )

#... plot
plt.figure( figsize=(5,4), dpi=600)
H = plt.hist(t_goal[I_good], bins=np.linspace( x.min(), x.max(), 31) ,density=True, alpha=0.5, color=np.ones((3,))*0.6 )
plt.xlabel('Time to NaCl t$_{NaCl}$ (min)')
plt.ylabel('Probability density')
for ii in range(len(gmm.weights_)):
    min_dev = (H[1][1]-H[1][0])/3
    
    y = gmm.weights_[ii]*gaussian_pdf(x, gmm.means_[ii], np.clip( gmm.covariances_[ii,0], min_dev ,999))
    idx = np.argmax(y)
    plt.plot( x, y , lw = 0.5, color='k')
    print( gmm.weights_[ii], gmm.means_[ii], gmm.covariances_[ii,0] )
    
    

######## FIGURE 2 - PANEL Bb ########    
# Compute a calibration curve to estimate the attraction coefficient
# Loop over values of c to check what is the typical time_goal in brownian motion
N = 1_000 #♦typically 1000 and 10_000
steps= 1_000
speed_dist = discard_nan( data_vw[I_good,:,0].flatten() )/T_SCALE

c = 10**np.linspace( -0.5, -3.5, 15)
tG = np.zeros_like(c)
tG_std = np.zeros_like(c)

s0 = chunk[:2]
p0 = plate[:2]

for ii, _c in enumerate(c):
    X = generate_brownian( N, steps, _c, (0,0) , -s0+p0, speed=speed_dist )

    A=time_to_goal(X, -s0+p0, 3.5 )/T_SCALE/60
    A[ np.nonzero( A==A.max())] = np.nan
    
    tG[ii] = np.nanmean(A)
    tG_std[ii] = np.nanstd(A)


#... plot
plt.figure( figsize=(2,2), dpi=600)
plt.errorbar( c, tG, yerr=tG_std, fmt='k', lw=0.5 )
for ii in range(len(gmm.weights_)):
    end = np.interp( gmm.means_[ii], tG, c)
    plt.hlines(gmm.means_[ii],0, end, 'r', lw=0.5 )
    plt.vlines(end, 1, gmm.means_[ii], 'r', lw=0.5 )
    
    print('Mean tG = %1.1f \t c ~ %1.3f' % (gmm.means_[ii], end) )
plt.xlabel('$\gamma$')
plt.ylabel('t$_{NaCl}$ (min)')
plt.yscale('log')
plt.xscale('log')
plt.ylim(1,100)












######## FIGURE 3 - PANEL A ########    
# Time series of linear and angular speed
idx = np.argmax( np.sum( data_xy[:,:,0]>-9999, axis=1 ))
tini, tfin = np.argwhere( np.diff( 1.0*(data_vw[idx,:,0]>-9999)) != 0).flatten()/T_SCALE/60
dt = data_vw.shape[1]
col_gray= np.ones((3,))*0.6


fondo = vutils.load_background(BKGD_FILENAME)
color = plt.cm.hsv(np.linspace(0, 1, n_worms))

plt.figure( figsize=(3,3), dpi=600 )
plt.imshow(  1.2*fondo , cmap='gray', vmax=255, vmin=0)
plt.plot( XY_SCALE*data_xy[idx,:,0].T, XY_SCALE*data_xy[idx,:,1].T, lw=0.5, c='r');
plt.xticks( plt.xticks()[0], labels='')
plt.yticks( plt.yticks()[0], labels='')
plt.xlim(0, fondo.shape[1])
plt.ylim(0, fondo.shape[0])


plt.figure( figsize=(6,4), dpi=600 )
plt.subplot(2,1,1)
plt.plot( data_t[:dt], data_vw[idx,:,0] , color='k', lw=0.5)
# plt.plot( moving_average(data_t[:dt], WDW), moving_average(data_vw[idx,:,0],WDW) ,color='r', lw=1.5)
# plt.xticks( plt.xticks()[0], labels='')
plt.ylabel('|v| (mm/s)')
plt.xlim(tini, tfin)
ylims1 = plt.ylim()

# plt.legend(('Raw','Running avg. (1min)'))

plt.subplot(2,1,2)
plt.plot( data_t[:dt], data_vw[idx,:,1] , color= 'k', lw=0.5)
# plt.plot( moving_average(data_t[:dt], WDW), moving_average(data_vw[idx,:,1],WDW) ,'r', lw=1.5)
plt.xlabel('Time (min)')
plt.ylabel('$\omega$ (rad/s)')
plt.xlim(tini, tfin)
ylims2 = plt.ylim()

plt.tight_layout()


######## FIGURE 3 - PANEL A2 ########    
# Histograms of instantaneous linear and angular speeds
plt.figure( figsize=(3,4), dpi=600 )
plt.subplot(2,1,1)
bbins = np.linspace( np.min(ylims1), np.max(ylims1), 20 )
plt.hist( data_vw[:,:,0].flatten() ,bins=bbins, orientation='horizontal', color=col_good, density=True, lw=1.5, histtype='step')
plt.hist( data_vw[idx,:,0] ,bins=20, orientation='horizontal', color='k', alpha=0.75, density=True)
plt.yticks( plt.yticks()[0], labels='')
plt.ylim(ylims1)

plt.subplot(2,1,2)
bbins = np.linspace( np.min(ylims2), np.max(ylims2), 20 )
plt.hist( data_vw[:,:,1].flatten() ,bins=bbins, orientation='horizontal', color=col_good, density=True, lw=1.5, histtype='step')
plt.hist( data_vw[idx,:,1] ,bins=20, orientation='horizontal', color='k', alpha=0.75, density=True)
plt.yticks( plt.yticks()[0], labels='')
plt.ylim(ylims2)
plt.xlabel('Probability densities')

plt.legend(('All traj.','Only $w^1$'))

plt.tight_layout()



######## FIGURE 3 - PANEL B ########    
# Individuality matrix
v_ = np.linspace( 0 , 0.2 , 14 )
w_ = np.linspace(-2*np.pi, 2*np.pi, 12 )
Dt = 30*T_SCALE # in frames, typically 30s, 60frames

extent = [ v_[0], v_[-1],w_[0], w_[-1] ]

Indiv = []
for ii in range(n_worms):
    Indiv.append( np.histogram2d( data_vw[ii,:,0], data_vw[ii,:,1], bins=[v_,w_], density=True)[0] )
Indiv= np.array(Indiv)

M_indiv = np.reshape( Indiv, (n_worms, (len(v_)-1)*(len(w_)-1) ) )

# plt.colorbar(label='Probability density')

plt.figure( figsize=(2,2), dpi=600)
plt.imshow( Indiv[idx,:,:].T, aspect='auto',  vmin=0, vmax=5, cmap='inferno_r')
plt.xlabel('|v| (mm/s)')
plt.ylabel('$\omega$ (rad/s)')

plt.figure( figsize=(1,3.5) , dpi=600)
plt.imshow( M_indiv[idx:(idx+1),:].T, aspect='auto',vmin=0, vmax=5, cmap='inferno_r')
plt.xticks( ticks=[])
plt.ylabel('Features')

plt.tight_layout()



import umap
from scipy.spatial import ConvexHull

DRA = umap.UMAP(metric='euclidean',n_neighbors=15,random_state=9999, verbose=False, min_dist=0.1)
Xnew = DRA.fit_transform( M_indiv[:,:] )
hull = ConvexHull(Xnew)

plt.figure( figsize=(5,4), dpi=600)
plt.plot( Xnew[:,0], Xnew[:,1], '.k', ms=5, zorder=0)
plt.scatter( Xnew[I_good,0], Xnew[I_good,1], 
            c = np.nanmean( data_vw[I_good,:,0], axis=1),  
            cmap = 'turbo',
            linewidths=0.5,
            edgecolors='k')
plt.xlabel('UMAP #1')
plt.ylabel('UMAP #2')
plt.colorbar( label='Mean speed' )
for simplex in hull.simplices:
    plt.plot(Xnew[simplex, 0], Xnew[simplex, 1], '--',color=np.ones((3,))*0.75, zorder=0)

plt.tight_layout()





# New points
x_new = np.linspace(1,11,4)
y_new = np.linspace(7,12,4)
test_pts = np.array( [ [xx,yy] for xx in x_new for yy in y_new] )

new_points = DRA.inverse_transform(test_pts)

ii = 15
A = np.reshape( new_points[ii, :], ( (len(v_)-1),(len(w_)-1)  ) )
plt.imshow(A, aspect='auto',vmin=0, vmax=5, cmap='inferno_r')







Mndiv = []
Mwhere= []
t_fin = -1

while t_fin<data_vw.shape[1]:
    t_ini = t_ini+1
    t_fin = t_ini+Dt
    _M_submat =[]
    _Mw_sub   = np.zeros((n_worms))
    
    for ii in range(n_worms):
        _v = data_vw[ii,t_ini:t_fin,0]
        _w = data_vw[ii,t_ini:t_fin,1]
        
        _is_valid = not np.all(np.isnan(_v))
    
        if _is_valid:
            
            _M = np.histogram2d( _v, _w, bins=[v_,w_], density=True)[0]
            _M_submat.append( _M  )
            _Mw_sub[ii] = 1
        else:
            _M_submat.append( np.zeros_like(Indiv[0,:,:])*np.nan )
            _Mw_sub[ii] = 0
    
    Mndiv.append(_M_submat)
    Mwhere.append( _Mw_sub )
    
Mndiv = np.array(Mndiv)
Mwhere= np.array(Mwhere)

print(Mndiv.shape)


plt.figure( figsize=(5,4), dpi=600)
plt.imshow( Indiv[idx,:,:].T, aspect='auto', extent=extent, vmin=0, vmax=5, cmap='inferno_r')
plt.xlabel(' |v|')
plt.ylabel('$\omega$')
plt.colorbar(label='Probability density')

plt.tight_layout()


# plt.figure( figsize=(8,4), dpi=600)
# for jj



# plt.xticks(ticks=range(len(w_)-1), labels='')
# plt.yticks(ticks=range(len(v_)-1), labels='')




# plt.subplot(2,2,4)
# bbins = np.linspace(0,15,20)
# plt.hist( np.clip( t_to_goal, bbins[0], bbins[-1]) , bins = bbins , color= col_all )
# plt.hist( t_to_goal[I_good] , bins = bbins , color= col_good )
# plt.xlabel('Time to goal')
# plt.ylabel('Number of\ntrajectories')

# plt.tight_layout()


# H=plt.hist( t_to_goal, bins=40)

# ys = H[0]
# xs = H[1][0] + np.cumsum(np.diff(H[1]))
# xq = np.linspace( xs.min(), xs.max(), 100)
# p0 = [1, -1]
# params, cv = curve_fit(monoExp, xs, ys, p0)

# plt.figure( figsize=(5,4), dpi=300 )
# plt.bar(xs,ys)
# plt.plot( xq, monoExp(xq, *params) ,'r')
# plt.ylim(1e-3, 1.05*ys.max() )
# plt.xlabel('Time to NaCl (min)')
# plt.ylabel('Probability density')

# plt.text( 10, np.max( plt.ylim())*0.90, r"y ~ exp( t / $\tau$ )" )
# plt.text( 10, np.max( plt.ylim())*0.83, r"$\tau$ = %1.3f $\pm$ %1.3f" % (params[1], np.sqrt( cv[1,1]))  )

# # determine quality of the fit
# squaredDiffs = np.square(ys - monoExp(xs, *params) )
# squaredDiffsFromMean = np.square(ys - np.mean(ys))
# rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
# print(f"R² = {rSquared}")
# print(params)




# ####
# # Compute the 2D histograms of (v,w)
# v_ = np.linspace( 0 , 0.2 , 11 )
# w_ = np.linspace(-2*np.pi, 2*np.pi, 11 )
# t_ = 20

# indiv = np.zeros((n_worms, len(v_)-1, len(w_)-1 ) )
# #partial = np.zeros( (n_worms, n_times, len(v_)-1, len(w_)-1) )

# for worm_jj in range( n_worms ):
#     indiv[ worm_jj ] = np.histogram2d( data_vw[worm_jj,:,0], data_vw[worm_jj,:,1], bins=[v_,w_], density=True)[0]


# plt.figure( figsize=(16,16), dpi=300)
# for jj in range(5):
#     for kk in range(5):
#         idx = kk+5*jj
#         plt.subplot(5,5, idx+1 )
#         plt.imshow(indiv[idx,:,:], aspect='auto')
#         plt.xticks(ticks=range(len(w_)-1), labels='')
#         plt.yticks(ticks=range(len(v_)-1), labels='')
# plt.tight_layout()



# v_indiv = [ indiv[_,:,:].flatten() for _ in range(n_worms) ]
# v_indiv = np.array( v_indiv )


# from sklearn.manifold import TSNE
# # plt.figure(dpi=300)
# # for jj in range(9):
# #     perp = jj*2+1
# #     X_embedded = TSNE(n_components=2, learning_rate='auto',
# #                       init='random', perplexity=perp).fit_transform( v_indiv)
    

# #     plt.subplot(3,3,jj+1)    
# #     plt.scatter(X_embedded[:,0], X_embedded[:,1], c= np.nanmean(data_vw[:,:,0], axis=1))
# #     plt.xticks( ticks= (), labels='')
# #     plt.yticks( ticks= (), labels='')
# #     plt.colorbar()
# #     plt.title(perp)
# # plt.tight_layout()


# plt.figure(dpi=300)
# X_embedded = TSNE(n_components=2, learning_rate='auto',
#                   init='pca', perplexity=10).fit_transform( v_indiv)
# plt.scatter(X_embedded[:,0], X_embedded[:,1], c= np.nanmean(data_vw[:,:,0], axis=1), s=28-t_to_goal)
# plt.xticks( ticks= (), labels='')
# plt.yticks( ticks= (), labels='')
# plt.colorbar(label=' $v_0$ (mm/s)')










# # # Distance traveled by each worm, and mean speed of each one
# # plt.subplot(1,2,1)
# # plt.hist( np.nansum( data_vw[:,:,0], axis=1) , bins=10)
# # plt.xlabel('Distance traveled (mm)')

# # plt.subplot(1,2,2)
# # # # plt.hist( np.nanmean( data_vw[:,:,0], axis=1 ))
# # # # plt.hist( np.nanmax( data_vw[:,:,0], axis=1 ), bins=np.linspace(0,1,20), alpha=0.5, zorder=0)
# # # # plt.xlabel('Speed (mm/s)')
# # # # plt.legend(('Mean','Max'))

# # # idx=3
# # plt.plot( data_xy[:,:,0].T, data_xy[:,:,1].T,'.-', ms=1, lw=0.5)
# # plt.scatter( chunk[0], chunk[1], s=1000, alpha=0.5)
# # plt.scatter( chunk[0], chunk[1], s=10)
# # plt.scatter( plate[0], plate[1], s=50000, c='r',alpha=0.2)
# # plt.scatter( plate[0], plate[1], s=10,c='r')
# # plt.scatter( x_goal, y_goal, s=10,c='b')
# # plt.xlim(0,60)
# # plt.ylim(0,60)




