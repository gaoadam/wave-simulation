#import libraries
import wavetorch
import unittest
import numpy as np
import torch

class TestClass(unittest.TestCase):

    def test_step_2d(self):
        """
        tests step_2d from wavetorch
        """
        #create torch tensor object
        u1_t0 = torch.zeros((17,12))

        #test step_2d on u1_t0 for 1 timestep
        u1_t1 = wavetorch.step_2d(dt=0.1, dx=0.1, dy=0.1, c=1, u_prev1=u1_t0, u=u1_t0, boundary='None')
        #validate u1_t0 and u1_t1 have same shape without boundary condition
        self.assertTrue(u1_t1.shape == u1_t0.shape)

        #test step_2d on u1_t0 for 1 timestep, with boundary condition
        u1_t1_boundary = wavetorch.step_2d(dt=0.1, dx=0.1, dy=0.1, c=1, u_prev1=u1_t1,  u=u1_t0, boundary='t0')
        #validate u1_t0 and u1_t1_boundary have same shape without boundary condition
        self.assertTrue(u1_t1_boundary.shape == u1_t0.shape)

    def test_wave_eq(self):
        """
        tests wave_eq from wavetorch
        """
        #create torch tensor object
        u1_t0 = torch.zeros((17,12))

        #define wave source functions
        def g1(x):
            return np.sin(10*np.pi*x)
        
        def g2(x):
            return x**2 + 3*x**3 + 10
        
        #test wave_eq on u1_t0
        N_t=5
        data = wavetorch.wave_eq(u_t0=u1_t0, g_r=[{'coordinate':(0,0), 'function':g1},
                                                  {'coordinate':(5,9), 'function':g2}],
                                                  wave_meta={'dx':0.1, 'dy':0.1, 'dt':0.1, 'c':1, 'N_t':N_t})
        u_tensor = data['u']
        #validate shape of output wave simulation tensor is as expected
        print('shape of torch tensor output: {}'.format(str(u_tensor.shape)))
        self.assertTrue(u_tensor.shape[0]==N_t+2)
        self.assertTrue(u_tensor.shape[1]==u1_t0.shape[0])
        self.assertTrue(u_tensor.shape[2]==u1_t0.shape[1])

#run unittest
unittest.main()