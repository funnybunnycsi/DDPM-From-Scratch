import unittest
import torch
from unet import UNet

import torch.nn as nn

class TestUNet(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            'img_channels': 1,
            'down_channels': [32, 64, 128, 256],
            'mid_channels': [256, 256, 128],
            'down_sample': [True, True, False],
            'time_emb_dim': 128,
            'num_down_layers': 2,
            'num_mid_layers': 2,
            'num_up_layers': 2,
            'num_heads': 4
        }
        self.batch_size = 4
        self.img_size = 28
        
    def test_init(self):
        model = UNet(self.valid_config)
        self.assertIsInstance(model, nn.Module)
        
    def test_forward_pass(self):
        model = UNet(self.valid_config)
        x = torch.randn(self.batch_size, self.valid_config['img_channels'], 
                       self.img_size, self.img_size)
        t = torch.randint(0, 1000, (self.batch_size,))
        
        output = model(x, t)
        self.assertEqual(output.shape, x.shape)
        
    def test_shape_consistency(self):
        model = UNet(self.valid_config)
        x = torch.randn(self.batch_size, self.valid_config['img_channels'], 
                       self.img_size, self.img_size)
        t = torch.randint(0, 1000, (self.batch_size,))
        
        # Check output maintains input dimensions
        output = model(x, t)
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.valid_config['img_channels'])
        self.assertEqual(output.shape[2], self.img_size)
        self.assertEqual(output.shape[3], self.img_size)
        
    def test_device_placement(self):
        if torch.cuda.is_available():
            model = UNet(self.valid_config).cuda()
            x = torch.randn(self.batch_size, self.valid_config['img_channels'], 
                          self.img_size, self.img_size).cuda()
            t = torch.randint(0, 1000, (self.batch_size,)).cuda()
            
            output = model(x, t)
            self.assertTrue(output.is_cuda)
            
    def test_invalid_config(self):
        invalid_config = self.valid_config.copy()
        invalid_config['mid_channels'] = [128, 128, 64]  # Invalid mid channels
        
        with self.assertRaises(AssertionError):
            UNet(invalid_config)

if __name__ == '__main__':
    unittest.main()