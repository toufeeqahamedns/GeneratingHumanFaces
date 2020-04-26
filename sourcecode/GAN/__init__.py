""" Package has implementation of Teacher project
    The gan architecture supplies gradients to the generator
    at different scales for image Generation
"""
from GAN import GAN
from GAN import CustomLayers
from GAN import Losses
from GAN import Infersent
from GAN import TextEncoder
from GAN import ConditionAugmentation