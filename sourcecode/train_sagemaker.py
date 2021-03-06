""" script for training the MSG-GAN on given dataset """

# Set to True if using in SageMaker
from torch.backends import cudnn
import yaml
import os
import torch as th
import numpy as np
import argparse

USE_SAGEMAKER = True


# sagemaker_containers required to access SageMaker environment (SM_CHANNEL_TRAINING, etc.)
# See https://github.com/aws/sagemaker-containers
if USE_SAGEMAKER:
    import sagemaker_containers

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

    # =======================================================================================
    # AUGMENTOR RELATED ARGUMENTS ... :)
    # =======================================================================================

    parser.add_argument("--ca_file", action="store", type=str, default=None,
                        help="pretrained conditioning augmentor file (compatible with my code)")

    parser.add_argument("--ca_optim_file", action="store", type=str, default=None,
                        help="saved state for conditioning augmentor")

    parser.add_argument("--ca_hidden_size", action="store", type=int, default=4096,
                        help="output size of the pretrained encoder")

    parser.add_argument("--ca_out_size", action="store", type=int, default=256,
                        help="output size of the conditioning augmentor")

    parser.add_argument("--compressed_latent_size", action="store", type=int, default=128,
                        help="output size of the compressed latents")

    parser.add_argument("--a_lr", action="store", type=int, default=0.003,
                        help="learning rate for augmentor")

    # =======================================================================================
    # ENCODER RELATED ARGUMENTS ... :)
    # =======================================================================================
    
    ## These have been moved down

    #parser.add_argument("--annotations_file", action="store", type=str, default="/opt/ml/input/face2text_v1.0/raw.json",
    #                    help="pretrained weights file for annotations")

    #parser.add_argument("--encoder_file", action="store", type=str, default="/opt/ml/input/infersent2/infersent2.pkl",
    #                    help="pretrained encoder file (compatible with my code)")

    #parser.add_argument("--embedding_file", action="store", type=str, default="/opt/ml/input/fasttext/crawl-300d-2M.vec",
    #                    help="pretrained embedding file")

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

    ## This has been moved down
    
    #parser.add_argument("--images_dir", action="store", type=str,
    #                    default="/opt/ml/input/data/",
    #                    help="path for the images directory")

    parser.add_argument("--flip_augment", action="store", type=bool,
                        default=True,
                        help="whether to randomly mirror the images during training")

    parser.add_argument("--sample_dir", action="store", type=str,
                         default=os.environ['SM_MODEL_DIR'],
                        help="path for the generated samples directory")

    parser.add_argument("--model_dir", action="store", type=str,
                         default=os.environ['SM_MODEL_DIR'],
                        help="path for saved models directory")

    parser.add_argument("--loss_function", action="store", type=str,
                        default="relativistic-hinge-cond",
                        help="loss function to be used: " +
                             "standard-gan, wgan-gp, lsgan," +
                             "lsgan-sigmoid," +
                             "hinge, relativistic-hinge")

    parser.add_argument("--depth", action="store", type=int,
                        default=7,
                        help="Depth of the GAN")

    parser.add_argument("--latent_size", action="store", type=int,
                        default=512,
                        help="latent size for the generator")

    parser.add_argument("--batch_size", action="store", type=int,
                        default=20,
                        help="batch_size for training")

    parser.add_argument("--spoofing_factor", action="store", type=int,
                        default=16,
                        help="number of passes done (gradient accumulation) " +
                             "before making an update step")

    parser.add_argument("--start", action="store", type=int,
                        default=1,
                        help="starting epoch number")

    parser.add_argument("--num_epochs", action="store", type=int,
                        default=1000,
                        help="number of epochs for training")

    parser.add_argument("--feedback_factor", action="store", type=int,
                        default=10,
                        help="number of logs to generate per epoch")

    parser.add_argument("--num_samples", action="store", type=int,
                        default=36,
                        help="number of samples to generate for creating the grid" +
                             " should be a square number preferably")

    parser.add_argument("--checkpoint_factor", action="store", type=int,
                        default=10,
                        help="save model per n epochs")

    parser.add_argument("--g_lr", action="store", type=float,
                        default=0.003,
                        help="learning rate for generator")

    parser.add_argument("--d_lr", action="store", type=float,
                        default=0.003,
                        help="learning rate for discriminator")

    parser.add_argument("--adam_beta1", action="store", type=float,
                        default=0,
                        help="value of beta_1 for adam optimizer")

    parser.add_argument("--adam_beta2", action="store", type=float,
                        default=0.99,
                        help="value of beta_2 for adam optimizer")

    parser.add_argument("--use_eql", action="store", type=bool,
                        default=True,
                        help="Whether to use equalized learning rate or not")

    parser.add_argument("--use_ema", action="store", type=bool,
                        default=True,
                        help="Whether to use exponential moving averages or not")

    parser.add_argument("--ema_decay", action="store", type=float,
                        default=0.999,
                        help="decay value for the ema")

    parser.add_argument("--data_percentage", action="store", type=float,
                        default=100,
                        help="percentage of data to use")

    parser.add_argument("--num_workers", action="store", type=int,
                        default=3,
                        help="number of parallel workers for reading files")

    # =======================================================================================
    # FID RELATED ARGUMENTS ... :)
    # =======================================================================================

    parser.add_argument("--fid_real_stats", action="store", type=str,
                        default=None,
                        help="Path to the precomputed fid real statistics file (.npz)")

    parser.add_argument("--log_fid_values", action="store", type=bool,
                        default=False,
                        help="Whether to log the fid values during training." +
                             " Following args are used only if this is true")

    parser.add_argument("--num_fid_images", action="store", type=int,
                        default=1000,
                        help="number of images used for calculating fid. Default: 50K")

    parser.add_argument("--fid_temp_folder", action="store", type=str,
                        default=None,
                        help="folder to store the temporary generated fid images")


    parser.add_argument("--fid_batch_size", action="store", type=int,
                        default=64,
                        help="Batch size used for the fid computation" +
                             "(Both image generation and fid calculation)")
    # ========================================================================================

    args = parser.parse_args()

    return args

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
    
    base_dir = os.environ["SM_CHANNEL_TRAINING"]
    
    images_dir_path = os.path.join(base_dir, "data")
    annotation_file = os.path.join(base_dir, "face2text_v1.0/raw.json")
    encoder_file = os.path.join(base_dir, "infersent2/infersent2.pkl")
    embedding_file = os.path.join(base_dir, "fasttext/crawl-300d-2M.vec")

    # transformation routine:
    res = int(np.power(2, args.depth + 1))
    img_transform = get_transform(
        (res, res), flip_horizontal=args.flip_augment)

    # create a data source:
    dataset = RawTextFace2TextDataset(
        annots_file=annotation_file,
        img_dir=images_dir_path,
        img_transform=img_transform,
    )

    # create a new session object for the pretrained encoder:
    text_encoder = PretrainedEncoder(
        model_file=encoder_file,
        embedding_file=embedding_file,
        device=device,
    )
    encoder_optim = None

    data = get_data_loader(dataset, args.batch_size, args.num_workers)
    print("Total number of images in the dataset:", len(dataset))

    # create a gan from these
    gan = ConditionalGAN(depth=args.depth,
                         latent_size=args.latent_size,
                         ca_hidden_size=args.ca_hidden_size,
                         ca_out_size=args.ca_out_size,
                         loss_fn=args.loss_function,
                         use_eql=args.use_eql,
                         use_ema=args.use_ema,
                         ema_decay=args.ema_decay,
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
                             lr=args.a_lr,
                             betas=[args.adam_beta1, args.adam_beta2])

    # create optimizer forImportError: No module named 'networks.pro_gan_pytorch' generator:
    gen_optim = th.optim.Adam(gan.gen.parameters(), args.g_lr,
                              [args.adam_beta1, args.adam_beta2])

    dis_optim = th.optim.Adam(gan.dis.parameters(), args.d_lr,
                              [args.adam_beta1, args.adam_beta2])

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
        num_epochs=args.num_epochs,
        checkpoint_factor=args.checkpoint_factor,
        data_percentage=args.data_percentage,
        feedback_factor=args.feedback_factor,
        num_samples=args.num_samples,
        sample_dir=args.sample_dir,
        save_dir=args.model_dir,
        log_dir=args.model_dir,
        start=args.start,
        spoofing_factor=args.spoofing_factor,
        log_fid_values=args.log_fid_values,
        num_fid_images=args.num_fid_images,
        fid_temp_folder=args.fid_temp_folder,
        fid_real_stats=args.fid_real_stats,
        fid_batch_size=args.fid_batch_size
    )


if __name__ == '__main__':
    
    # install necessary packages
    os.system('pip install imageio')
    os.system('pip install tensorboardX')
    
    # invoke the main function of the script
    main(parse_arguments())