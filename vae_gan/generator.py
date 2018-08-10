import os
from scipy.misc import imsave

class Generator(object):

    def generate_and_save_images(self,num_samples,directory,iter):
        '''Generate the images using model and save them in directory'''
        imgs=self.sess.run(self.sampled_tensor)
        for k in range(imgs.shape[0]):
            imgs_folder=os.path.join(directory,f'imgs_{iter}')
            if not os.path.exists(imgs_folder):
                os.makedirs(imgs_folder)

            imsave(os.path.join(imgs_folder,'%d.png') %k, imgs[k].reshape(28,28))
