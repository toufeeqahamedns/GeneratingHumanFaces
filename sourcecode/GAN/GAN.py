""" Module implementing GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""
import datetime
import os
import time
import timeit
import copy
import numpy as np
import torch as th

from GAN.ConditionAugmentation import ConditionAugmentor


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, Conv2d
        from GAN.CustomLayers import GenGeneralConvBlock, \
            GenInitialBlock, _equalized_conv2d

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(
                2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the Generator Below ...
        # create the ToRGB layers for various outputs:
        if self.use_eql:
            def to_rgb(in_channels):
                return _equalized_conv2d(in_channels, 3, (1, 1), bias=True)
        else:
            def to_rgb(in_channels):
                return Conv2d(in_channels, 3, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList(
            [GenInitialBlock(self.latent_size, use_eql=self.use_eql)])
        self.rgb_converters = ModuleList([to_rgb(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            use_eql=self.use_eql)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

    def forward(self, x):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        outputs = []  # initialize to empty list

        y = x  # start the computational pipeline
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(converter(y))

        return outputs

    @staticmethod
    def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
        """
        adjust the dynamic colour range of the given input data
        :param data: input image data
        :param drange_in: original range of input
        :param drange_out: required range of output
        :return: img => colour range adjusted images
        """
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) -
                    np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return th.clamp(data, min=0, max=1)


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512, embedding_size=4096, compressed_latent_size=128,
                 use_eql=True, gpu_parallelize=False):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param embedding_size: size of embedding for discriminator
        :param compressed_latent_size: size of compressed version
        :param use_eql: whether to use the equalized learning rate or not
        :param gpu_parallelize: whether to use DataParallel on the discriminator
                                Note that the Last block contains StdDev layer
                                So, it is not parallelized.
        """
        from torch.nn import ModuleList
        from GAN.CustomLayers import DisGeneralConvBlock, \
            DisFinalBlock, _equalized_conv2d
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.gpu_parallelize = gpu_parallelize
        self.use_eql = use_eql
        self.depth = depth
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.compressed_latent_size = compressed_latent_size

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            def from_rgb(out_channels):
                return _equalized_conv2d(3, out_channels, (1, 1), bias=True)
        else:
            def from_rgb(out_channels):
                return Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList()
        self.final_converter = from_rgb(self.feature_size)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList()
        self.final_block = DisFinalBlock(
            self.feature_size * 2, self.embedding_size, self.compressed_latent_size, use_eql=self.use_eql)

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 3)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = DisGeneralConvBlock(self.feature_size * 2, self.feature_size,
                                            use_eql=self.use_eql)
                rgb = from_rgb(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # handle the case where the depth is less than or equal to 4
        if self.depth > 4:
            self.rgb_to_features[self.depth - 2] = \
                from_rgb(self.feature_size // np.power(2, i - 3))
        else:
            self.rgb_to_features[self.depth - 2] = \
                from_rgb(self.feature_size * 2)

        # parallelize the modules from the module-lists if asked to:
        if self.gpu_parallelize:
            for i in range(len(self.layers)):
                self.layers[i] = th.nn.DataParallel(self.layers[i])
                self.rgb_to_features[i] = th.nn.DataParallel(
                    self.rgb_to_features[i])

        # Note that since the FinalBlock contains the StdDev layer,
        # it cannot be parallelized so easily. It will have to be parallelized
        # from the Lower level (from CustomLayers). This much parallelism
        # seems enough for me.

    def forward(self, inputs, latent_vector):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :param latent_vector: latent vector required for discriminator
        :return: out => raw prediction values
        """

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 2](inputs[self.depth - 1])
        y = self.layers[self.depth - 2](y)
        for x, block, converter in \
                zip(reversed(inputs[1:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = th.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block

        # calculate the final block:
        input_part = self.final_converter(inputs[0])
        y = th.cat((input_part, y), dim=1)
        y = self.final_block(y, latent_vector)

        # return calculated y
        return y


class GAN:
    """ Unconditional MSG-GAN

        args:
            depth: depth of the GAN (will be used for each generator and discriminator)
            latent_size: latent size of the manifold used by the GAN
            ca_hidden_size: output size of pretrained encoder
            ca_out_size: output size of augmentor
            use_eql: whether to use the equalized learning rate
            use_ema: whether to use exponential moving averages.
            ema_decay: value of ema decay. Used only if use_ema is True
            device: device to run the GAN on (GPU / CPU)
    """

    def __init__(self, depth=7, latent_size=512, ca_hidden_size=4096, ca_out_size=256,
                 use_eql=True, use_ema=True, ema_decay=0.999,
                 device=th.device("cpu")):
        """ constructor for the class """
        from torch.nn import DataParallel

        self.ca = ConditionAugmentor(ca_hidden_size,
                                     ca_out_size,
                                     use_eql, device)

        self.gen = Generator(depth, latent_size, use_eql=use_eql).to(device)

        # Parallelize them if required:
        if device == th.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = Discriminator(depth, latent_size, ca_hidden_size, ca_out_size,
                                     use_eql=use_eql, gpu_parallelize=True).to(device)
        else:
            self.dis = Discriminator(
                depth, latent_size, ca_hidden_size, ca_out_size, use_eql=True).to(device)

        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.use_eql = use_eql
        self.latent_size = latent_size
        self.ca_hidden_size = ca_hidden_size,
        self.ca_out_size = ca_out_size,
        self.depth = depth
        self.device = device

        if self.use_ema:
            from GAN.CustomLayers import update_average

            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)

            # updater function:
            self.ema_updater = update_average

            # initialize the gen_shadow weights equal to the
            # weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()
        if self.use_ema:
            self.gen_shadow.eval()

    def generate_samples(self, num_samples):
        """
        generate samples using this gan
        :param num_samples: number of samples to be generated
        :return: generated samples tensor: list[ Tensor(B x H x W x C)]
        """
        noise = th.randn(num_samples, self.latent_size).to(self.device)
        generated_images = self.gen(noise)

        # reshape the generated images
        generated_images = list(map(lambda x: (x.detach().permute(0, 2, 3, 1) / 2) + 0.5,
                                    generated_images))

        return generated_images

    def optimize_discriminator(self, dis_optim, noise,
                               real_batch, loss_fn,
                               accumulate=False, zero_grad=True,
                               num_accumulations=1):
        """
        performs one step of weight update on discriminator using the batch of data
        :param dis_optim: discriminator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :param accumulate: whether to accumulate or make a step
        :param zero_grad: used to control the behaviour of grad buffers
        :param num_accumulations: number of accumulation steps performed
                                  (required to scale the loss function)
        :return: current loss (accumulation scaled value)
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)
        fake_samples = list(map(lambda x: x.detach(), fake_samples))

        # scale the loss by the number of accumulation steps performed
        # (if not performed, it is 1)
        loss = loss_fn.dis_loss(real_batch, fake_samples) / num_accumulations

        # optimize discriminator according to the accumulation dynamics
        # zero the grad of the discriminator weights if required
        if zero_grad:
            dis_optim.zero_grad()

        # perform the backward pass to accumulate the gradients
        loss.backward()

        # if not running in accumulate mode, (make a step)
        if not accumulate:
            dis_optim.step()

        return loss.item()

    def optimize_generator(self, gen_optim, noise,
                           real_batch, loss_fn,
                           accumulate=False, zero_grad=True,
                           num_accumulations=1):
        """
        performs one step of weight update on generator using the batch of data
        :param gen_optim: generator optimizer
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
                           should contain a list of tensors at different scales
        :param loss_fn: loss function to be used (object of GANLoss)
        :param accumulate: whether to accumulate or make a step
        :param zero_grad: used to control the behaviour of grad buffers
        :param num_accumulations: number of accumulation steps performed
                                  (required to scale the loss function)
        :return: current loss (accumulation scaled value)
        """

        # generate a batch of samples
        fake_samples = self.gen(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples) / num_accumulations

        # optimize the generator according the accumulation dynamics
        if zero_grad:
            gen_optim.zero_grad()

        # perform backward pass for gradient accumulation
        loss.backward()

        # perform an update step if not running in the accumulate mode
        if not accumulate:
            gen_optim.step()

            # if self.use_ema is true, apply the moving average here:
            # Note that ema update will also be done only during the update
            # pass of the function (not during accumulation).
            if self.use_ema:
                self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        return loss.item()

    def create_grid(self, samples, img_files,
                    sum_writer=None, reses=None, step=None):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing list[Tensors]
        :param img_files: list of names of files to write
        :param sum_writer: summary writer object
        :param reses: resolution strings (used only if sum_writer is not None)
        :param step: global step (used only if sum_writer is not None)
        :return: None (saves multiple files)
        """
        from torchvision.utils import save_image, make_grid
        from torch.nn.functional import interpolate
        from numpy import sqrt, power, ceil

        # dynamically adjust the colour range of the images:
        samples = [Generator.adjust_dynamic_range(
            sample) for sample in samples]

        # resize the samples to have same resolution:
        for i in range(len(samples)):
            samples[i] = interpolate(samples[i],
                                     scale_factor=power(2, self.depth - 1 - i))

        # save the images:
        for sample, img_file in zip(samples, img_files):
            save_image(sample, img_file, nrow=int(sqrt(sample.shape[0])),
                       normalize=True, scale_each=True, padding=0)

        if sum_writer is not None:
            for sample, res in zip(samples, reses):
                image = make_grid(sample, nrow=int(ceil(sqrt(sample.shape[0]))),
                                  normalize=True, scale_each=True, padding=0)
                sum_writer.add_image(res, image, step)

    def create_grid_fixed(self, samples, scale_factor, img_file, real_imgs=False):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :param real_imgs: turn off the scaling of images
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        samples = th.clamp((samples / 2) + 0.5, min=0, max=1)

        # upsample the image
        if not real_imgs and scale_factor > 1:
            samples = interpolate(samples,
                                  scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))))

    def create_descriptions_file(self, file, captions, dataset):
        """
        utility function to create a file for storing the captions
        :param file: file for storing the captions
        :param captions: encoded_captions or raw captions
        :param dataset: the dataset object for transforming captions
        :return: None (saves a file)
        """
        from functools import reduce

        # transform the captions to text:
        if isinstance(captions, th.Tensor):
            captions = list(map(lambda x: dataset.get_english_caption(x.cpu()),
                                [captions[i] for i in range(captions.shape[0])]))

            with open(file, "w") as filler:
                for caption in captions:
                    filler.write(reduce(lambda x, y: x + " " + y, caption, ""))
                    filler.write("\n\n")
        else:
            with open(file, "w") as filler:
                for caption in captions:
                    filler.write(caption)
                    filler.write("\n\n")

    def _downsampled_images(self, images):
        """
        private utility function to compute list of downsampled images
        :param images: Original sized images
        :return: images => list of downsampled images
        """
        from torch.nn.functional import avg_pool2d
        # create a list of downsampled images from the real images:
        images = [images] + [avg_pool2d(images, int(np.power(2, i)))
                             for i in range(1, self.depth)]
        images = list(reversed(images))

        return images

    def _get_images_and_latents(self, data_store, encoder, normalize_latents):
        """
        private utility function to obtain random latent_points and
        downsampled images from the datastore
        :param data_store: object containing the data
        :param normalize_latents: boolean for hyper-sphere normalization
        :return: images, latents => images and random latent points
        """
        # extract current batch of data for training
        batch_captions, batch_images = next(data_store)
        captions = batch_captions
        images = batch_images.to(self.device)

        # list of downsampled versions of images
        images = self._downsampled_images(images)

        embeddings = encoder(captions)
        embeddings = th.from_numpy(embeddings).to(self.device)

        c_not_hats, mus, sigmas = self.ca(embeddings)

        noise = th.randn(
            captions.shape[0] if isinstance(
                captions, th.Tensor) else len(captions),
            self.latent_size - c_not_hats.shape[-1]).to(self.device)

        gan_input = th.cat((c_not_hats, noise), dim=-1)

        # normalize them if asked
        if normalize_latents:
            gan_input = ((gan_input
                          / gan_input.norm(dim=-1, keepdim=True))
                         * (self.latent_size ** 0.5))

        return captions, images, gan_input, mus, sigmas

    def train(self, data, gen_optim, dis_optim, ca_optim, text_encoder, encoder_optim, loss_fn, normalize_latents=True,
              start=1, num_epochs=12, spoofing_factor=1,
              feedback_factor=10, checkpoint_factor=1,
              data_percentage=100, num_samples=36,
              log_dir=None, sample_dir="./samples",
              log_fid_values=False, num_fid_images=50000,
              save_dir="./models", fid_temp_folder="./samples/fid_imgs/",
              fid_real_stats=None, fid_batch_size=64):
        """
        Method for training the network
        :param data: pytorch dataloader which iterates over images
        :param gen_optim: Optimizer for generator.
                          please wrap this inside a Scheduler if you want to
        :param dis_optim: Optimizer for discriminator.
                          please wrap this inside a Scheduler if you want to
        :param ca_optim: Optimizer for augmentor.
        :param loss_fn: Object of GANLoss
        :param normalize_latents: whether to normalize the latent vectors during training
        :param start: starting epoch number
        :param num_epochs: total number of epochs to run for (ending epoch number)
                           note this is absolute and not relative to start
        :param spoofing_factor: number of actual batches used to spoof a bigger batch
                                for instance, actual batch size is 16 and
                                spoofing factor is 4, then virtual batch_size is 64
        :param feedback_factor: number of logs generated and samples generated
                                during training per epoch
        :param checkpoint_factor: save model after these many epochs
        :param data_percentage: amount of data to be used
        :param num_samples: number of samples to be drawn for feedback grid
        :param log_dir: path to directory for saving the loss.log file
        :param sample_dir: path to directory for saving generated samples' grids
        :param log_fid_values: boolean for whether to log fid values during training or not
        :param num_fid_images: number of images to generate for calculating the FID
        :param save_dir: path to directory for saving the trained models
        :param fid_temp_folder: path to save the generated images
        :param fid_real_stats: path to the npz stats file for real images
        :param fid_batch_size: batch size used for generating fid images
                               Same will be used for calculating the fid too.
        :return: None (writes multiple files to disk)
        """
        from tensorboardX import SummaryWriter
        from shutil import rmtree
        from tqdm import tqdm
        from imageio import imwrite
        from GAN.FID import fid_score
        from math import ceil
        from GAN.utils.iter_utils import hn_wrapper

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        self.ca.train()

        assert isinstance(gen_optim, th.optim.Optimizer), \
            "gen_optim is not an Optimizer"
        assert isinstance(dis_optim, th.optim.Optimizer), \
            "dis_optim is not an Optimizer"
        assert isinstance(ca_optim, th.optim.Optimizer), \
            "ca_optim is not an Optimizer"

        print("Starting the training process ... ")

        # create the summary writer
        sum_writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))

        # create a grid of samples and save it
        reses = [str(int(np.power(2, dep))) + "_x_"
                 + str(int(np.power(2, dep)))
                 for dep in range(2, self.depth + 2)]

        # create fixed_input for debugging
        real_data_store = iter(hn_wrapper(data))
        fixed_captions, fixed_real_images, fixed_input, _, _ = self._get_images_and_latents(
            real_data_store, text_encoder, normalize_latents)

        # save the fixed images once
        fixed_save_dir = os.path.join(sample_dir, "__Real_Info__")
        os.makedirs(fixed_save_dir, exist_ok=True)
        self.create_grid_fixed(fixed_real_images, None,  # scale factor is not required here
                               os.path.join(fixed_save_dir, "real_samples.png"), real_imgs=True)
        self.create_descriptions_file(os.path.join(fixed_save_dir, "real_captions.txt"),
                                      fixed_captions,
                                      real_data_store)

        viter_samples = 2 * data.batch_size * spoofing_factor
        total_imgs = len(data.dataset)
        total_batches = int(total_imgs / viter_samples)
        limit = int(ceil((data_percentage / 100) * total_batches))

        # create a global time counter
        global_time = time.time()
        global_step = 0

        for epoch in range(start, num_epochs + 1):
            start_time = timeit.default_timer()  # record time at the start of epoch
            print("\nEpoch: %d" % epoch)

            # setup the dataloader (where the real images are sampled from)
            real_data_store = iter(hn_wrapper(data))
            print(real_data_store)
            batch_counter = 0  # counter for number of batches completed
            # this includes the two Generator passes and spoofing adjusted
            # batch_sizes

            # Note that this might lose the last batch (samples less than batch_size)
            # but the subsequent epochs perform random shuffling prior to starting the
            # training for that epoch
            while real_data_store.hasnext() and batch_counter < limit:

                # perform batch spoofing via gradient accumulation
                dis_loss, gen_loss = 0, 0  # losses initialized to zeros

                # =================================================================
                # discriminator iterations:
                # =================================================================
                for spoofing_iter in range(spoofing_factor - 1):

                    # =============================================================
                    # Discriminator spoofing pass
                    # =============================================================

                    # sample images and latents for discriminator pass
                    _, images, gan_input, _, _ = self._get_images_and_latents(
                        real_data_store, text_encoder, normalize_latents)

                    # accumulate gradients in the discriminator:
                    dis_loss += self.optimize_discriminator(
                        dis_optim, gan_input,
                        images, loss_fn,
                        accumulate=True,
                        zero_grad=(spoofing_iter == 0),
                        num_accumulations=spoofing_factor)

                # =============================================================
                # Discriminator update pass
                # (note values for accumulate and zero_grad)
                # =============================================================

                # sample images and latents for discriminator pass
                _, images, gan_input, _, _ = self._get_images_and_latents(
                    real_data_store, text_encoder, normalize_latents)

                # accumulate final gradients in the discriminator and make a step:
                dis_loss += self.optimize_discriminator(
                    dis_optim, gan_input,
                    images, loss_fn,
                    accumulate=False,  # perform update
                    # make gradient buffers zero if spoofing_factor is 1
                    zero_grad=spoofing_factor == 1,
                    num_accumulations=spoofing_factor)

                # =================================================================

                # =================================================================
                # generator iterations:
                # =================================================================
                for spoofing_iter in range(spoofing_factor - 1):

                    # =============================================================
                    # Generator spoofing pass
                    # =============================================================

                    # re-sample images and latents for generator pass
                    _, images, gan_input, mus, sigmas = self._get_images_and_latents(
                        real_data_store, text_encoder, normalize_latents)

                    if encoder_optim is not None:
                        encoder_optim.zero_grad()

                    ca_optim.zero_grad()
                    # accumulate gradients in the generator
                    gen_loss += self.optimize_generator(
                        gen_optim, gan_input,
                        images, loss_fn,
                        accumulate=True,
                        zero_grad=(spoofing_iter == 0),
                        num_accumulations=spoofing_factor)

                    kl_loss = th.mean(0.5 * th.sum((mus ** 2) + (sigmas ** 2)
                                                   - th.log((sigmas ** 2)) - 1, dim=1))
                    kl_loss.backward()
                    ca_optim.step()
                    if encoder_optim is not None:
                        encoder_optim.step()
                # =============================================================
                # Generator update pass
                # (note values for accumulate and zero_grad)
                # =============================================================

                # sample images and latents for generator pass
                _, images, gan_input, _, _ = self._get_images_and_latents(
                    real_data_store, text_encoder, normalize_latents)

                if encoder_optim is not None:
                    encoder_optim.zero_grad()

                ca_optim.zero_grad()
                # accumulate final gradients in the generator and make a step:
                gen_loss += self.optimize_generator(
                    gen_optim, gan_input,
                    images, loss_fn,
                    accumulate=False,  # perform update
                    # make gradient buffers zero if spoofing_factor is 1
                    zero_grad=spoofing_factor == 1,
                    num_accumulations=spoofing_factor)

                kl_loss = th.mean(0.5 * th.sum((mus ** 2) + (sigmas ** 2)
                                               - th.log((sigmas ** 2)) - 1, dim=1))
                kl_loss.backward()
                ca_optim.step()
                if encoder_optim is not None:
                    encoder_optim.step()
                # =================================================================

                # increment the global_step and the batch_counter:
                global_step += 1
                batch_counter += 1

                # provide a loss feedback
                if batch_counter % int(limit / feedback_factor) == 0 or \
                        batch_counter == 1:
                    elapsed = time.time() - global_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    print("Elapsed [%s] batch: %d  d_loss: %f  g_loss: %f kl_loss: %f"
                          % (elapsed, batch_counter, dis_loss, gen_loss, kl_loss.item()))

                    # add summary of the losses
                    sum_writer.add_scalar("dis_loss", dis_loss, global_step)
                    sum_writer.add_scalar("gen_loss", gen_loss, global_step)
                    sum_writer.add_scalar("kl_loss", kl_loss.item(), global_step)
                    
                    # also write the losses to the log file:
                    if log_dir is not None:
                        log_file = os.path.join(log_dir, "loss.log")
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)
                        with open(log_file, "a") as log:
                            log.write(str(global_step) + "\t" + str(dis_loss) +
                                      "\t" + str(gen_loss) + "\t" + str(kl_loss.item()) + "\n")

                    # create a grid of samples and save it
                    gen_img_files = [os.path.join(sample_dir, res, "gen_" +
                                                  str(epoch) + "_" +
                                                  str(batch_counter) + ".png")
                                     for res in reses]

                    # Make sure all the required directories exist
                    # otherwise make them
                    os.makedirs(sample_dir, exist_ok=True)
                    for gen_img_file in gen_img_files:
                        os.makedirs(os.path.dirname(
                            gen_img_file), exist_ok=True)

                    # following zero_grads are required to allow pytorch
                    # adjust buffers properly on the GPU.
                    # This causes lesser GPU memory consumption
                    dis_optim.zero_grad()
                    gen_optim.zero_grad()
                    with th.no_grad():  # this makes the training faster.
                        self.create_grid(
                            self.gen(fixed_input) if not self.use_ema
                            else self.gen_shadow(fixed_input),
                            gen_img_files,
                            sum_writer,
                            reses,
                            global_step)

            # calculate the time required for the epoch
            stop_time = timeit.default_timer()
            print("Time taken for epoch: %.3f secs" % (stop_time - start_time))

            if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == num_epochs:
                os.makedirs(save_dir, exist_ok=True)
                ca_save_file = os.path.join(save_dir, "CA_" + str(epoch) + ".pth")
                gen_save_file = os.path.join(
                    save_dir, "GAN_GEN_" + str(epoch) + ".pth")
                dis_save_file = os.path.join(
                    save_dir, "GAN_DIS_" + str(epoch) + ".pth")
                ca_optim_save_file = os.path.join(save_dir, "CA_OPTIM_" + str(epoch) + ".pth")
                gen_optim_save_file = os.path.join(save_dir,
                                                   "GAN_GEN_OPTIM_" + str(epoch) + ".pth")
                dis_optim_save_file = os.path.join(save_dir,
                                                   "GAN_DIS_OPTIM_" + str(epoch) + ".pth")

                th.save(self.ca.state_dict(), ca_save_file)
                th.save(self.gen.state_dict(), gen_save_file)
                th.save(self.dis.state_dict(), dis_save_file)
                th.save(ca_optim.state_dict(), ca_optim_save_file)
                th.save(gen_optim.state_dict(), gen_optim_save_file)
                th.save(dis_optim.state_dict(), dis_optim_save_file)

                if self.use_ema:
                    gen_shadow_save_file = os.path.join(save_dir, "GAN_GEN_SHADOW_"
                                                        + str(epoch) + ".pth")
                    th.save(self.gen_shadow.state_dict(), gen_shadow_save_file)

                print("log_fid_values:", log_fid_values)
                if log_fid_values:  # perform the following fid calculations during training
                    # if the boolean is set to true
                    # ==================================================================
                    # Perform the FID calculation during training for estimating
                    # the quality of the training
                    # ==================================================================

                    # setup the directory for generating the images
                    if os.path.isdir(fid_temp_folder):
                        rmtree(fid_temp_folder)
                    os.makedirs(fid_temp_folder, exist_ok=True)

                    # generate the images:
                    print("generating images for fid calculation ...")
                    pbar = tqdm(total=num_fid_images)
                    generated_images = 0

                    while generated_images < num_fid_images:
                        b_size = min(fid_batch_size,
                                     num_fid_images - generated_images)
                        points = th.randn(
                            b_size, self.latent_size).to(self.device)
                        if normalize_latents:
                            points = (points / points.norm(dim=1, keepdim=True)) \
                                * (self.latent_size ** 0.5)
                            imgs = self.gen(points)[-1].detach()
                        for i in range(len(imgs)):
                            imgs[i] = Generator.adjust_dynamic_range(imgs[i])
                        pbar.update(b_size)
                        for img in imgs:
                            imwrite(os.path.join(fid_temp_folder,
                                                 str(generated_images) + ".jpg"),
                                    img.permute(1, 2, 0).cpu())
                            generated_images += 1
                    pbar.close()

                    # compute the fid now:
                    fid = fid_score.calculate_fid_given_paths(
                        (fid_real_stats, fid_temp_folder),
                        fid_batch_size,
                        True if self.device == th.device("cuda") else False,
                        2048  # using he default value
                    )

                    # print the compute fid value:
                    print("FID at epoch %d: %.6f" % (epoch, fid))

                    # log the fid value in tensorboard:
                    sum_writer.add_scalar("FID", fid, epoch)
                    # note that for fid value, the global step is the epoch number.
                    # it is not the global step. This makes the fid graph more informative

                    # ==================================================================

        print("Training completed ...")

        # return the generator and discriminator back to eval mode
        self.gen.eval()
        self.dis.eval()
