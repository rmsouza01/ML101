import matplotlib.pylab as plt
import numpy as np
import os
import natsort as ns
import dicom
from scipy.ndimage import morphological_gradient,sobel
from scipy.ndimage.morphology import binary_erosion
from skimage import feature
import nibabel as nib

def dice_coefficient(A,B):
    return 2.0*np.logical_and(A,B).sum()/(A.sum()+B.sum())

def save_seg(seg,affine,file_name):
    #seg = segp.uint8)        
    nii1 = nib.Nifti1Image(seg, affine)
    nib.save(nii1,file_name)
    
    
def bin_circle(img_shape,center,radius,nangles = 50):
    circle_img = np.zeros(img_shape,dtype = bool)
    angles = np.linspace(0,2*np.pi,nangles)
    xc = np.zeros(nangles)
    yc = xc.copy()
    xc = np.around(center[0] + radius*np.cos(angles)).astype(int)
    yc = np.around(center[1] + radius*np.sin(angles)).astype(int)
    circle_img[xc,yc] = True
    return circle_img

def mosaic(f,N):
    f = np.asarray(f)
    d,h,w = f.shape
    N = int(N)
    nLines = int(np.ceil(float(d)/N))
    nCells = nLines*N

    # Add black slices to match the exact number of mosaic cells
    fullf = np.resize(f, (nCells,h,w))
    fullf[d:nCells,:,:] = 0        
    Y,X = np.indices((nLines*h,N*w))
    Pts = np.array([
               (np.floor(Y/h)*N + np.floor(X/w)).ravel(),
                np.mod(Y,h).ravel(),
                np.mod(X,w).ravel() ]).astype(int).reshape((3,int(nLines*h),int(N*w)))
    g = fullf[Pts[0],Pts[1],Pts[2]]
    return g

def mosaic_color(f,N):
    d,h,w,c = f.shape
      
    #linhas e colunas da imagem
    nLines = int(np.ceil(float(d)/N))
    nCells = nLines*N
       
    # Add black slices to match the exact number of mosaic cells
    mosaico = np.zeros((3,h*nLines, w*N))
    for i in xrange(d):
        j = i/N
        k = i%N
        slice_ = f[i].transpose(2,0,1) 
        mosaico[:,j*h:(j+1)*h,k*w:(k+1)*w] = slice_
        return mosaico

    
    
def gshow(X, X1=[], X2=[], X3=[],X4=[]):
    X_new = np.array([X,X,X])
    if X1!= []:
        X_new[0,X1] = 255
        X_new[1,X1]= 0
        X_new[2,X1] = 0
    if X2!= []:
        X_new[0,X2] = 0
        X_new[1,X2]= 255
        X_new[2,X2] = 0
    if X3!= []:
        X_new[0,X3] = 0
        X_new[1,X3]= 0
        X_new[2,X3] = 255
    if X4!= []:
        X_new[0,X4] = 255
        X_new[1,X4]= 0
        X_new[2,X4] = 255
    return X_new.transpose(1,2,0)

#discrete_cmap is not my code all credit goes to @jakevdp 
#https://gist.github.com/jakevdp/91077b0cae40f8f8244a

def discrete_cmap(N, base_cmap=None):
   """Create an N-bin discrete colormap from the specified input map"""

   # Note that if base_cmap is a string or None, you can simply do
   #    return plt.cm.get_cmap(base_cmap, N)
   # The following works for string, None, or a colormap instance:

   base = plt.cm.get_cmap(base_cmap)
   color_list = base(np.linspace(0, 1, N))
   cmap_name = base.name + str(N)
   return base.from_list(cmap_name, color_list, N)

def intensity_normalization(img,lower = 0.02, upper = 0.98,NEW_MAX = 800):
   """
   Intensity normalization. The image pixels intensities are rescaled to
   fit between minimum (lower intensity percentile) and maximum (upper 
   intensity percentile). Values below and above these percentiles are clipped
   to the lower and upper percentiles accordingly.
   Input:
   img -> Input image.
   lower -> Lower percentile.
   upper -> Upper percentile.
   NEW_MAX -> new maximum value of the image
   Output:
   img_norm -> Normalized image. 
   """

   img_max = img.max()
   img_min = img.min()

   hist, bin_edges = np.histogram(img.ravel(), bins=int(img_max + 1))
   cum_hist = 1.0*hist.cumsum()/img.size
   
   #Computing the percentiles
   t_lower = np.where(cum_hist >= lower)[0][0]
   t_upper = np.where(cum_hist >= upper)[0][0] 

   #Normalization step
   img_float = img.astype(float)
   img_norm = (img_float - t_lower)/(t_upper - t_lower)*(NEW_MAX)
   img_norm = np.clip(img_norm,0 , (NEW_MAX))
   img_norm = np.floor(img_norm).astype(np.uint16)
   return img_norm

def crop3D(img):
   """
   img -> Bin image to be croped
   (xmin,xmax,ymin,ymax,zmin,zmax) -> coordinates to crop the image
   """
   temp = np.nonzero(img)
   xmin,xmax = temp[0].min(),temp[0].max()+1
   ymin,ymax = temp[1].min(),temp[1].max()+1
   zmin,zmax = temp[2].min(),temp[2].max()+1
   return xmin,xmax,ymin,ymax,zmin,zmax 

def lbp3D(f):
   """
   3D Local Binary Pattern implementation. It uses a 6-neighborhood scheme and
   a 10 groups clustering
   Input:
   f -> Input image.
   Output:
   lbp -> LBP 3D 
   Encoding table:
   Code    Card  Condition 
   1        0
   2        1 
   3        2    opposite voxels
   4        2    bend voxels
   5        3    voxels on the same plane
   6        3    voxels on different planes
   7        4    voxels on the same plane
   8        4    voxels on different planes
   9        5
   10       6 
   """
   
   H,W,Z = f.shape               
   p1 = f[0:-2,1:-1,1:-1] >= f[1:-1,1:-1,1:-1] #x-up
   p2 = f[2:,1:-1,1:-1] >= f[1:-1,1:-1,1:-1] #x-down
   p3 = f[1:-1,0:-2,1:-1] >= f[1:-1,1:-1,1:-1] #y-up
   p4 = f[1:-1,2:,1:-1] >= f[1:-1,1:-1,1:-1] #y-down
   p5 = f[1:-1,1:-1,0:-2] >= f[1:-1,1:-1,1:-1] #z-up
   p6 = f[1:-1,1:-1,2:] >= f[1:-1,1:-1,1:-1] #z-down
   accum = p1.astype(np.uint8) + p2.astype(np.uint8) + p3.astype(np.uint8)\
         + p4.astype(np.uint8) + p5.astype(np.uint8) + p6.astype(np.uint8)
   lbp = np.ones((H-2,W-2,Z-2), dtype = np.uint8)
   lbp[accum == 1] = 2
   
   # card = 2 and opposite voxels
   aux1 = (p1 & p2)
   aux2 = (p3 & p4)
   aux3 = (p5 & p6)
   indexes = (accum == 2) & (aux1 | aux2 | aux3)
   lbp[indexes] = 3
   
   # card = 2 and bend voxels
   indexes = (accum == 2) & (~aux1) & (~aux2) & (~aux3)
   lbp[indexes] = 4
   
   # card = 3 and voxels on the same plane
   aux1 = (p1 + p2 + p5 + p6)
   aux2 = (p1 + p2 + p3 + p4)
   aux3 = (p3 + p4 + p5 + p6)
   pxz = (aux1 == 3) 
   pxy = (aux2 == 3) 
   pyz = (aux3 == 3) 
   indexes = (accum == 3) & (pxz | pxy | pyz)
   lbp[indexes] = 5
   
   # card = 3 and voxels on different planes
   indexes = (accum == 3) & (~pxz) & (~pxy) & (~pyz)
   lbp[indexes] = 6
   
   # card = 4 and voxels on the same plane
   pxz = (aux1 == 4) 
   pxy = (aux2 == 4) 
   pyz = (aux3 == 4) 
   indexes = (accum == 4) & (pxz | pxy | pyz)
   lbp[indexes] = 7
   
   # card = 4 and voxels on different planes
   indexes = (accum == 4) & (~pxz) & (~pxy) & (~pyz)
   lbp[indexes] = 8
   
   # card = 5
   lbp[accum == 5] = 9
   
   #card = 6
   lbp[accum == 6] = 10
   lbp2 = np.zeros((H,W,Z), dtype = np.uint8)
   lbp2[1:-1,1:-1,1:-1] = lbp
   return lbp2


def ws_lines(seg):
   """
   This function receives as input a 2D segmentation mask and returns
   the segmentation lines.
   """
   p1 = seg[0:-2,1:-1] > seg[1:-1,1:-1] #x-up
   p2 = seg[2:,1:-1] > seg[1:-1,1:-1] #x-down
   p3 = seg[1:-1,0:-2] > seg[1:-1,1:-1] #y-up
   p4 = seg[1:-1,2:] > seg[1:-1,1:-1] #y-down
   seg2 = np.zeros_like(seg)
   indexes = p1 | p2 | p3 | p4
   seg2[1:-1,1:-1][indexes] = 1#seg[1:-1,1:-1][indexes]
   return seg2


def read_dicom_volume(dcm_path):
   """
   This function reads all dicom volumes in  a folder as a volume.
   """
   dcm_files = [ f for f in os.listdir(dcm_path) if f.endswith('.dcm')]
   dcm_files = ns.natsorted(dcm_files, alg=ns.IGNORECASE) 
   Z = len(dcm_files)
   reference = dicom.read_file(os.path.join(dcm_path,dcm_files[0]))
   H,W = reference.pixel_array.shape
   type = reference.pixel_array.dtype   
   volume = np.zeros((H,W,Z), dtype = type)
   for (ii,dcm_slice) in enumerate(dcm_files):
      volume[:,:,ii] = dicom.read_file(os.path.join(dcm_path,dcm_files[ii])).pixel_array 
   return volume 


class Tracker: #(object):
   def __init__(self, ax, X):
      self.ax = ax
      #ax.set_title('use scroll wheel to navigate images')
      self.seeds = []
      self.X = X
      rows, cols, self.slices = X.shape
      self.ind = self.slices//2

      self.im = ax.imshow(self.X[:, :, self.ind],cmap = 'gray')
      ax.axis('off')
      ax.set_title('slice %s' % self.ind)
      self.update()

   def key_event(self, event):
      if event.key == 'right':
         self.ind = (self.ind + 1) % self.slices
      elif event.key == "left":
         self.ind = (self.ind - 1) % self.slices
      elif event.key == "d":
         self.seeds = []                
      else:
         return
      self.update()
   def onclick(self,event):
      self.seeds.append((event.xdata, event.ydata,self.ind))
        
   def update(self):
      self.im.set_data(self.X[:, :, self.ind])
      self.ax.set_title('slice %s' % self.ind)
      self.im.axes.figure.canvas.draw()

def lbp(f):
   lbp_img = np.zeros(f.shape, dtype = np.uint8) 
   lbp_img[1:-1,1:-1] = np.power(2,0) * (f[0:-2,0:-2] >= f[1:-1,1:-1]) + \
       np.power(2,1) * (f[0:-2,1:-1] >= f[1:-1,1:-1]) + \
       np.power(2,2) * (f[0:-2,2:] >= f[1:-1,1:-1]) + \
       np.power(2,3) * (f[1:-1,0:-2] >= f[1:-1,1:-1]) + \
       np.power(2,4) * (f[1:-1,2:] >= f[1:-1,1:-1]) + \
       np.power(2,5) * (f[2:,0:-2] >= f[1:-1,1:-1]) + \
       np.power(2,6) * (f[2:,1:-1] >= f[1:-1,1:-1]) + \
       np.power(2,7) * (f[2:,2:] >= f[1:-1,1:-1])
   return lbp_img
     


    

  
                
