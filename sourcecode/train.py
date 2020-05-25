""" script for training the MSG-GAN on given dataset """

import argparse

import numpy as np
import torch as th
import yaml
from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# enable fast training
cudnn.benchmark = True

# set seed = 3
th.manual_seed(seed=3)

def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", action="store", type=str,
                        default="/content/GeneratingHumanFaces/sourcecode/configs/colab.conf",
                        help="default configuration for the Network")

    # =======================================================================================
    # AUGMENTOR RELATED ARGUMENTS ... :)
    # =======================================================================================

    parser.add_argument("--ca_file", action="store", type=str, default=None,
                        help="pretrained conditioning augmentor file (compatible with my code)")

    parser.add_argument("--ca_optim_file", action="store", type=str, default=None,
                        help="saved state for conditioning augmentor")

    # =======================================================================================
    # GAN RELATED ARGUMENTS ... :)
    # =======================================================================================

    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained weights file for generator")

    parser.add_argument("--generator_optim_file", action="store", type=str, default=None,
                        help="saved state for generator optimizer")

    parser.add_argument("--shadow_generator_file", action="store", type=str, default=None,
                        help="pretrained weights file for the shadow generator")

    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained_weights file for discriminator")

    parser.add_argument("--discriminator_optim_file", action="store", type=str, default=None,
                        help="saved state for discriminator optimizer")

    # =======================================================================================
    # FID RELATED ARGUMENTS ... :)
    # =======================================================================================

    parser.add_argument("--fid_real_stats", action="store", type=str,
                        default=None,
                        help="Path to the precomputed fid real statistics file (.npz)")

    # ========================================================================================

    args = parser.parse_args()

    return args


def get_config(conf_file):
    """
    parse and load the provided configuration
    :param conf_file: configuration file
    :return: conf => parsed configuration
    """
    from easydict import EasyDict as edict

    with open(conf_file, "r") as file_descriptor:
        data = yaml.load(file_descriptor)

    # convert the data into an easyDictionary
    return edict(data)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """
    from GAN.GAN import ConditionalGAN
    from data_processing.DataLoader import get_transform, get_data_loader, \
        RawTextFace2TextDataset
    from GAN import Losses as lses
    from GAN.TextEncoder import PretrainedEncoder
    from GAN.ConditionAugmentation import ConditionAugmentor

    config = get_config(args.config)
    print("Current Configuration:", config)

    # transformation routine:
    res = int(np.power(2, config.depth + 1))
    img_transform = get_transform(
        (res, res), flip_horizontal=config.flip_augment)

    # create a data source:
    dataset = RawTextFace2TextDataset(
        annots_file=config.annotations_file,
        img_dir=config.images_dir,
        img_transform=img_transform,
    )

    # create a new session object for the pretrained encoder:
    text_encoder = PretrainedEncoder(
        model_file=config.encoder_file,
        embedding_file=config.embedding_file,
        device=device,
    )
    encoder_optim = None

    data = get_data_loader(dataset, config.batch_size, config.num_workers)
    print("Total number of images in the dataset:", len(dataset))

    # create a gan from these
    gan = ConditionalGAN(depth=config.depth,
                         latent_size=config.latent_size,
                         ca_hidden_size=config.ca_hidden_size,
                         ca_out_size=config.ca_out_size,
                         loss_fn=config.loss_function,
                         use_eql=config.use_eql,
                         use_ema=config.use_ema,
                         ema_decay=config.ema_decay,
                         device=device)

    if args.ca_file is not None:
        print("loading conditioning augmenter from:", args.ca_file)
        gan.ca.load_state_dict(th.load(args.ca_file))

    print("Augmentor Configuration: ")
    print(gan.ca)

    if args.generator_file is not None:
        # load the weights into generator
        print("loading generator_weights from:", args.generator_file)
        gan.gen.load_state_dict(th.load(args.generator_file))

    print("Generator Configuration: ")
    print(gan.gen)

    if args.shadow_generator_file is not None:
        # load the weights into generator
        print("loading shadow_generator_weights from:",
              args.shadow_generator_file)
        gan.gen_shadow.load_state_dict(
            th.load(args.shadow_generator_file))

    if args.discriminator_file is not None:
        # load the weights into discriminator
        print("loading discriminator_weights from:", args.discriminator_file)
        gan.dis.load_state_dict(th.load(args.discriminator_file))

    print("Discriminator Configuration: ")
    print(gan.dis)

    # create the optimizer for Condition Augmenter separately
    ca_optim = th.optim.Adam(gan.ca.parameters(),
                             lr=config.a_lr,
                             betas=[config.adam_beta1, config.adam_beta2])

    # create optimizer forImportError: No module named 'networks.pro_gan_pytorch' generator:
    gen_optim = th.optim.Adam(gan.gen.parameters(), config.g_lr,
                              [config.adam_beta1, config.adam_beta2])

    dis_optim = th.optim.Adam(gan.dis.parameters(), config.d_lr,
                              [config.adam_beta1, config.adam_beta2])

    if args.generator_optim_file is not None:
        print("loading gen_optim_state from:", args.generator_optim_file)
        gen_optim.load_state_dict(th.load(args.generator_optim_file))

    if args.discriminator_optim_file is not None:
        print("loading dis_optim_state from:", args.discriminator_optim_file)
        dis_optim.load_state_dict(th.load(args.discriminator_optim_file))

    # train the GAN
    gan.train(
        data,
        gen_optim,
        dis_optim,
        ca_optim,
        text_encoder,
        encoder_optim,
        num_epochs=config.num_epochs,
        checkpoint_factor=config.checkpoint_factor,
        data_percentage=config.data_percentage,
        feedback_factor=config.feedback_factor,
        num_samples=config.num_samples,
        sample_dir=config.sample_dir,
        save_dir=config.model_dir,
        log_dir=config.model_dir,
        start=config.start,
        spoofing_factor=config.spoofing_factor,
        log_fid_values=config.log_fid_values,
        num_fid_images=config.num_fid_images,
        fid_temp_folder=config.fid_temp_folder,
        fid_real_stats=args.fid_real_stats,
        fid_batch_size=config.fid_batch_size
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
