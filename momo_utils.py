from momo_imports import *
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import animation, rc
from IPython.display import HTML
USE_CUDA = torch.cuda.is_available()

class CentraliseImage(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        size = (self.output_size,self.output_size)
        layer = Image.new
        layer = Image.new("RGB", size, (255,255,255))
        layer.paste(img, tuple(map(lambda x: int((x[0]-x[1])/2), zip(size,img.size))))
        return layer

class PositionImage(object):
    def __init__(self, output_size, pos):
        self.output_size = output_size
        self.pos = pos

    def __call__(self, img):
        layer = Image.new
        layer = Image.new("RGB", (self.output_size, self.output_size), (255,255,255))
        layer.paste(img, self.pos)
        return layer

class LogoDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.logo_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.logo_df)

    def __getitem__(self, idx):
        image = Image.open(self.logo_df.iloc[idx]["path"])
        if self.transform:
            image = self.transform(image)
        item = {"image": image, "idx": idx}
        return item

def save_batch_grid(batch, labels=None, title=None, path="debug"):
    bs = len(batch)
    size = batch.size(2)
    # make grid
    grid = utils.make_grid(batch)
    f = plt.figure(figsize = (20,20))
    plt.imshow(grid.numpy().transpose((1,2,0)))
    if labels is not None:
        side = int(np.sqrt(bs))
        label_pos_x = size * np.arange(1,side+1) + np.arange(1,side+1) * 2 - 5
        label_pos_y = size * np.arange(0,side) + np.arange(1,side+1) * 2 + 10
        labels = labels.reshape(side,side)
        X,Y = np.meshgrid(label_pos_x,label_pos_y)
        ax = plt.gca()
        for i in range(side):
            for j in range(side):
                ax.text(X[i,j],Y[i,j],labels[i,j],ha="right",size=10)
    plt.axis("off")
    if title:
        plt.title(title)
    else:
        plt.title("Batch from Dataloader")
    fn = f"{path}/{title}.png"
    plt.savefig(fn, format="png")
    tqdm.write(f"Saved {fn}")
    plt.close()

def imscatter(x,y,imgs,ax=None,zoom=1):
    if ax is None:
        ax = plt.gca()
    artists = []
    for i in tqdm(range(len(x))):
        im = OffsetImage(plt.imread(imgs[i]), zoom=zoom)
        ab = AnnotationBbox(im,xy=(x[i],y[i]),frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

class Style2StyleIconDataset(Dataset):
    def __init__(self, csv_file, from_style, to_style, transform=None, training=False, out_size=128, debug=False):
        t_df = pd.read_csv(csv_file)
        self.logo_df = pd.merge(t_df[t_df["style"]==from_style],t_df[t_df["style"]==to_style], on=["content","category"],suffixes=("_from","_to"))
        self.transform = transform
        self.from_style = from_style
        self.to_style = to_style
        self.out_size = out_size
        self.debug = debug
        self.training = training
        print(f"==> Loaded {'TRAINING' if training else 'VALID'} for '{self.from_style}' to '{self.to_style}'.")

    def __len__(self):
        return len(self.logo_df)

    def __getitem__(self, idx):
        from_img = Image.open(self.logo_df.iloc[idx]["path_from"])
        from_img_uid = self.logo_df.iloc[idx]["uid_from"]
        to_img = Image.open(self.logo_df.iloc[idx]["path_to"])
        to_img_uid = self.logo_df.iloc[idx]["uid_to"]

        # data augmentation with probability 80%
        if self.training:
            if np.random.rand() > 0.8:

                # rescale
                _s = np.random.randint(20,int(self.out_size/5*4))
                if self.debug:
                    print(f"rescaling image to {_s}")
                scale_random = transforms.Scale(_s)
                from_img = scale_random(from_img)
                to_img = scale_random(to_img)

                # move from and target images with probability 0.5
                if np.random.rand() > 0.6:
                    if self.debug:
                        print("centralised")
                    pos_transform = CentraliseImage(self.out_size)
                else:
                    padding = 0
                    _min = padding
                    _max = self.out_size - _s - padding
                    pos = tuple(np.random.randint(_min,_max, size=2))
                    if self.debug:
                        print(f"positioned at {pos}")
                    pos_transform = PositionImage(self.out_size, pos)

                from_img = pos_transform(from_img)
                to_img = pos_transform(to_img)

                # change from_image sharpness
                enhancer = ImageEnhance.Sharpness(from_img)
                blurriness = np.random.normal(1,1)
                from_img = enhancer.enhance(blurriness)
                if self.debug:
                    print(f"Sharpened by {blurriness}")

                # add noise to from image
                from_img_arr = np.asarray(from_img)
                noise_factor = 0.5 * np.random.rand()
                noise = np.random.normal(0, noise_factor, from_img_arr.shape).astype(from_img_arr.dtype)
                from_img = Image.fromarray(from_img_arr + noise)
                if self.debug:
                    print(f"Noise factor {noise_factor}")

                # flip image upside down with probability 0.5 ==> don't do it yet
        #             if np.random.rand() > 0.5:
        #                 if self.debug:
        #                     print("flipped upside down")
        #                 from_img = from_img.transpose(Image.FLIP_TOP_BOTTOM)
        #                 to_img = to_img.transpose(Image.FLIP_TOP_BOTTOM)

        #             # flip image on its side with probability 0.5
        #             if np.random.rand() > 0.5:
        #                 if self.debug:
        #                     print("transposed")
        #                 from_img = from_img.transpose(Image.TRANSPOSE)
        #                 to_img = to_img.transpose(Image.TRANSPOSE)

            else:
                scale_128 = transforms.Scale(self.out_size)
                from_img = scale_128(from_img)
                to_img = scale_128(to_img)

        # convert to origin image to greyscale
        if self.from_style == "Outline":
            if self.debug:
                print("converted to greyscale")
            from_img = from_img.convert("L")

        if self.transform:
            from_img = self.transform(from_img)
            to_img = self.transform(to_img)

        item = {"from": {"img":from_img, "idx": from_img_uid}, "to": {"img": to_img, "idx": to_img_uid}}
        return item

class CustomRandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation

    @staticmethod
    def get_params(img):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img1, img2):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly cropped and resize image.
        """
        i, j, h, w = self.get_params(img1)
        return F.resized_crop(img1, i, j, h, w, self.size, self.interpolation), F.resized_crop(img2, i, j, h, w, self.size, self.interpolation)

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.bn(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output

class SRResnet(nn.Module):
    def __init__(self):
        super(SRResnet, self).__init__()
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.residual = self._make_layer(block=ResidualBlock, num_layers=16)
        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn_mid = nn.BatchNorm2d(num_features=64)
        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.orthogonal(m.weight, math.sqrt(2))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.relu(self.conv_input(x))
        residual = output
        output = self.bn_mid(self.conv_mid(self.residual(output)))
        output = torch.add(output, residual)
        output = self.conv_output(output)
        return output

def save_checkpoint(model, optimizer, loss, epoch, model_pth):
    model_out_path = model_pth
    state = {"model": model, "loss": loss,"epoch":epoch,"optimizer":optimizer}
    if not os.path.exists("model/"):
        os.makedirs("model/")
    torch.save(state, model_out_path)
    tqdm.write(f"Saved {model_out_path}")

def train_model(exp_name, max_epochs=50, bs=32, lr=1e-4, im_sz=128, checkpoint=None):
    if not os.path.exists(f"model/{exp_name}"):
        os.makedirs(f"model/{exp_name}")
        tqdm.write(f"Made directory model/{exp_name}")
    use_cuda = torch.cuda.is_available()
    gen_net = SRResnet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(gen_net.parameters(),lr=lr)
    from_imgs = Variable(torch.FloatTensor(bs, 3, im_sz, im_sz), requires_grad=False)
    to_imgs = Variable(torch.FloatTensor(bs, 3, im_sz, im_sz), requires_grad=False)
    newT = torch.FloatTensor(64,3,128,128)
    if use_cuda:
        gen_net = gen_net.cuda()
        criterion = criterion.cuda()
        from_imgs = from_imgs.cuda()
        to_imgs = to_imgs.cuda()
    debug_batch = torch.load("debug/debug_batch.pth")
    print("===> Loaded debug_batch")

    # load model checkpoint if provided
    start_epoch = 0
    if checkpoint:
        gen_net.load_state_dict(checkpoint["model"].state_dict())
        start_epoch = checkpoint["epoch"] + 1

    print(f"===> Beginning training for {exp_name}")
    with tqdm(range(start_epoch, max_epochs)) as t:
        for epoch_i in t:
            gen_net.train()
            from_imgs.data.copy_(debug_batch["from"])
            to_imgs.data.copy_(debug_batch["to"])
            output = gen_net(from_imgs)
            loss = criterion(output, to_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch_i}/{max_epochs}. Loss = {loss.data[0]:.6f}")
            # save model and produce an image
            if (epoch_i + 1) % 50 == 0:
                save_checkpoint(model=gen_net, epoch=epoch_i, loss=loss.data[0],
                                   model_pth=f"model/{exp_name}/model_epoch_{epoch_i}.pth")

def generate_image_from_models(checkpoint_pths, exp_name):
    bs = 32
    im_sz = 128
    use_cuda = torch.cuda.is_available()
    gen_net = SRResnet()
    newT = torch.FloatTensor(64,3,128,128)
    from_imgs = Variable(torch.FloatTensor(bs, 3, im_sz, im_sz), requires_grad=False)
    if use_cuda:
        gen_net = gen_net.cuda()
        from_imgs = from_imgs.cuda()

    debug_batch = torch.load("debug/debug_batch.pth")
    from_imgs.data.copy_(debug_batch["from"])
    print("===> Loaded debug_batch")

    for pth in checkpoint_pths:
        checkpoint = torch.load(pth)
        gen_net.load_state_dict(checkpoint["model"].state_dict())
        epoch_i = checkpoint["epoch"]
        gen_net.eval()
        output = gen_net(from_imgs)
        for i in range(32):
            newT[2*i] = debug_batch["from"][i]
            newT[2*i+1] = output.data[i]
        save_batch_grid(newT,title=f"{exp_name}_output_epoch_{epoch_i}",path=f"model/{exp_name}")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.c0=nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.c1=nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.c2=nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.c3=nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.c4=nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.c5=nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.c6=nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.c7=nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.c8=nn.Conv2d(512, 512, 3, 1, 1, bias=False)

        self.dc8=nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.dc7=nn.Conv2d(512, 256, 3, 1, 1, bias=False)
        self.dc6=nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.dc5=nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.dc4=nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.dc3=nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.dc2=nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.dc1=nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.dc0=nn.Conv2d(64, 3, 3, 1, 1)

        self.bnc0=nn.BatchNorm2d(32)
        self.bnc1=nn.BatchNorm2d(64)
        self.bnc2=nn.BatchNorm2d(64)
        self.bnc3=nn.BatchNorm2d(128)
        self.bnc4=nn.BatchNorm2d(128)
        self.bnc5=nn.BatchNorm2d(256)
        self.bnc6=nn.BatchNorm2d(256)
        self.bnc7=nn.BatchNorm2d(512)
        self.bnc8=nn.BatchNorm2d(512)

        self.bnd8=nn.BatchNorm2d(512)
        self.bnd7=nn.BatchNorm2d(256)
        self.bnd6=nn.BatchNorm2d(256)
        self.bnd5=nn.BatchNorm2d(128)
        self.bnd4=nn.BatchNorm2d(128)
        self.bnd3=nn.BatchNorm2d(64)
        self.bnd2=nn.BatchNorm2d(64)
        self.bnd1=nn.BatchNorm2d(32)

    def forward(self, x):
        e0 = F.relu(self.bnc0(self.c0(x)))
        e1 = F.relu(self.bnc1(self.c1(e0)))
        e2 = F.relu(self.bnc2(self.c2(e1)))
        del e1
        e3 = F.relu(self.bnc3(self.c3(e2)))
        e4 = F.relu(self.bnc4(self.c4(e3)))
        del e3
        e5 = F.relu(self.bnc5(self.c5(e4)))
        e6 = F.relu(self.bnc6(self.c6(e5)))
        del e5
        e7 = F.relu(self.bnc7(self.c7(e6)))
        e8 = F.relu(self.bnc8(self.c8(e7)))

        d8 = F.relu(self.bnd8(self.dc8(torch.cat([e7, e8],dim=1))))
        del e7, e8
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        del d8
        d6 = F.relu(self.bnd6(self.dc6(torch.cat([e6, d7],dim=1))))
        del d7, e6
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        del d6
        d4 = F.relu(self.bnd4(self.dc4(torch.cat([e4, d5],dim=1))))
        del d5, e4
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        del d4
        d2 = F.relu(self.bnd2(self.dc2(torch.cat([e2, d3],dim=1))))
        del d3, e2
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        del d2
        d0 = self.dc0(torch.cat([e0, d1],dim=1))
        output = F.sigmoid(d0)
        del d0

        return output

class UNetLeaky(nn.Module):
    def __init__(self):
        super(UNetLeaky, self).__init__()

        self.c0=nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.c1=nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.c2=nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.c3=nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.c4=nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.c5=nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.c6=nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.c7=nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.c8=nn.Conv2d(512, 512, 3, 1, 1, bias=False)

        self.dc8=nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.dc7=nn.Conv2d(512, 256, 3, 1, 1, bias=False)
        self.dc6=nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.dc5=nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.dc4=nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.dc3=nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.dc2=nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.dc1=nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.dc0=nn.Conv2d(64, 3, 3, 1, 1)

        self.bnc0=nn.BatchNorm2d(32)
        self.bnc1=nn.BatchNorm2d(64)
        self.bnc2=nn.BatchNorm2d(64)
        self.bnc3=nn.BatchNorm2d(128)
        self.bnc4=nn.BatchNorm2d(128)
        self.bnc5=nn.BatchNorm2d(256)
        self.bnc6=nn.BatchNorm2d(256)
        self.bnc7=nn.BatchNorm2d(512)
        self.bnc8=nn.BatchNorm2d(512)

        self.bnd8=nn.BatchNorm2d(512)
        self.bnd7=nn.BatchNorm2d(256)
        self.bnd6=nn.BatchNorm2d(256)
        self.bnd5=nn.BatchNorm2d(128)
        self.bnd4=nn.BatchNorm2d(128)
        self.bnd3=nn.BatchNorm2d(64)
        self.bnd2=nn.BatchNorm2d(64)
        self.bnd1=nn.BatchNorm2d(32)

    def forward(self, x):
        e0 = F.leaky_relu(self.bnc0(self.c0(x)))
        e1 = F.leaky_relu(self.bnc1(self.c1(e0)))
        e2 = F.leaky_relu(self.bnc2(self.c2(e1)))
        del e1
        e3 = F.leaky_relu(self.bnc3(self.c3(e2)))
        e4 = F.leaky_relu(self.bnc4(self.c4(e3)))
        del e3
        e5 = F.leaky_relu(self.bnc5(self.c5(e4)))
        e6 = F.leaky_relu(self.bnc6(self.c6(e5)))
        del e5
        e7 = F.leaky_relu(self.bnc7(self.c7(e6)))
        e8 = F.leaky_relu(self.bnc8(self.c8(e7)))

        d8 = F.leaky_relu(self.bnd8(self.dc8(torch.cat([e7, e8],dim=1))))
        del e7, e8
        d7 = F.leaky_relu(self.bnd7(self.dc7(d8)))
        del d8
        d6 = F.leaky_relu(self.bnd6(self.dc6(torch.cat([e6, d7],dim=1))))
        del d7, e6
        d5 = F.leaky_relu(self.bnd5(self.dc5(d6)))
        del d6
        d4 = F.leaky_relu(self.bnd4(self.dc4(torch.cat([e4, d5],dim=1))))
        del d5, e4
        d3 = F.leaky_relu(self.bnd3(self.dc3(d4)))
        del d4
        d2 = F.leaky_relu(self.bnd2(self.dc2(torch.cat([e2, d3],dim=1))))
        del d3, e2
        d1 = F.leaky_relu(self.bnd1(self.dc1(d2)))
        del d2
        d0 = self.dc0(torch.cat([e0, d1],dim=1))
        output = F.sigmoid(d0)
        del d0

        return output

class UNetGAN(nn.Module):
    def __init__(self):
        super(UNetGAN, self).__init__()

        self.c0=nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.c1=nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.c2=nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.c3=nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.c4=nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.c5=nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.c6=nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.c7=nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.c8=nn.Conv2d(512, 512, 3, 1, 1, bias=False)

        self.dc8=nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.dc7=nn.Conv2d(512, 256, 3, 1, 1, bias=False)
        self.dc6=nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.dc5=nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.dc4=nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.dc3=nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        self.dc2=nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.dc1=nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.dc0=nn.Conv2d(64, 3, 3, 1, 1)

        self.bnc0=nn.BatchNorm2d(32)
        self.bnc1=nn.BatchNorm2d(64)
        self.bnc2=nn.BatchNorm2d(64)
        self.bnc3=nn.BatchNorm2d(128)
        self.bnc4=nn.BatchNorm2d(128)
        self.bnc5=nn.BatchNorm2d(256)
        self.bnc6=nn.BatchNorm2d(256)
        self.bnc7=nn.BatchNorm2d(512)
        self.bnc8=nn.BatchNorm2d(512)

        self.bnd8=nn.BatchNorm2d(512)
        self.bnd7=nn.BatchNorm2d(256)
        self.bnd6=nn.BatchNorm2d(256)
        self.bnd5=nn.BatchNorm2d(128)
        self.bnd4=nn.BatchNorm2d(128)
        self.bnd3=nn.BatchNorm2d(64)
        self.bnd2=nn.BatchNorm2d(64)
        self.bnd1=nn.BatchNorm2d(32)

    def forward(self, x):
        e0 = F.leaky_relu(self.bnc0(self.c0(x)),negative_slope=0.2)
        e1 = F.leaky_relu(self.bnc1(self.c1(e0)),negative_slope=0.2)
        e2 = F.leaky_relu(self.bnc2(self.c2(e1)),negative_slope=0.2)
        del e1
        e3 = F.leaky_relu(self.bnc3(self.c3(e2)),negative_slope=0.2)
        e4 = F.leaky_relu(self.bnc4(self.c4(e3)),negative_slope=0.2)
        del e3
        e5 = F.leaky_relu(self.bnc5(self.c5(e4)),negative_slope=0.2)
        e6 = F.leaky_relu(self.bnc6(self.c6(e5)),negative_slope=0.2)
        del e5
        e7 = F.leaky_relu(self.bnc7(self.c7(e6)),negative_slope=0.2)
        e8 = F.leaky_relu(self.bnc8(self.c8(e7)),negative_slope=0.2)

        d8 = F.relu(self.bnd8(self.dc8(torch.cat([e7, e8],dim=1))))
        del e7, e8
        d7 = F.relu(self.bnd7(self.dc7(d8)))
        del d8
        d6 = F.relu(self.bnd6(self.dc6(torch.cat([e6, d7],dim=1))))
        del d7, e6
        d5 = F.relu(self.bnd5(self.dc5(d6)))
        del d6
        d4 = F.relu(self.bnd4(self.dc4(torch.cat([e4, d5],dim=1))))
        del d5, e4
        d3 = F.relu(self.bnd3(self.dc3(d4)))
        del d4
        d2 = F.relu(self.bnd2(self.dc2(torch.cat([e2, d3],dim=1))))
        del d3, e2
        d1 = F.relu(self.bnd1(self.dc1(d2)))
        del d2
        d0 = self.dc0(torch.cat([e0, d1],dim=1))
        output = F.sigmoid(d0)
        del d0

        return output

class SubsetSequentialSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def xavier_init(model):
    for param in model.parameters():
        if len(param.size()) == 2:
            nn.init.xavier_normal(param)

def initialise_model(model_class, checkpoint=None):
    m = model_class()
    # load weights from checkpoint if provided, otherwise
    # initialise using xavier_init
    if checkpoint:
        print(f"==> Initialised {model_class} from checkpoint from epoch {checkpoint['epoch']}. Its prev loss was {checkpoint['loss']}")
        m.load_state_dict(checkpoint["model"])
    else:
        print(f"==> Initialised {model_class} with xavier_init")
        xavier_init(m)
    # move to GPU
    if USE_CUDA:
        print("==> Loaded to GPU")
        m = m.cuda()
    return m

def get_logo_dataloaders(csv_file, from_style, to_style, transform=None, random_seed=21, valid_size=256, input_img_size=128, batch_size=32):
    print(f"==> Using random seed {random_seed}, valid size {valid_size}, input size {input_img_size} and batch size {batch_size}")
    if transform:
        tfms = transform
    else:
        tfms = transforms.Compose([transforms.Scale(input_img_size),transforms.ToTensor()])
    train_dataset = Style2StyleIconDataset(csv_file,from_style,to_style,transform=tfms, training=True)
    valid_dataset = Style2StyleIconDataset(csv_file,from_style,to_style,transform=tfms, training=False)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split = valid_size
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetSequentialSampler(valid_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              sampler=valid_sampler, drop_last=False)
    return train_loader, valid_loader

def prepare_directory(exp_name):
    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists(f"model/{exp_name}"):
        os.makedirs(f"model/{exp_name}")
        tqdm.write(f"Made directory model/{exp_name}")
    if not os.path.exists(f"model/{exp_name}/valid_images"):
        os.makedirs(f"model/{exp_name}/valid_images")
    if not os.path.exists(f"model/{exp_name}/train_loss_history"):
        os.makedirs(f"model/{exp_name}/train_loss_history")
    if not os.path.exists(f"model/{exp_name}/val_loss_history"):
        os.makedirs(f"model/{exp_name}/val_loss_history")
    if not os.path.exists(f"model/{exp_name}/accuracy_history"):
        os.makedirs(f"model/{exp_name}/accuracy_history")
    print(f"==> {exp_name} directories prepped.")

class DIS(nn.Module):

    def __init__(self):
        super(DIS, self).__init__()
        self.c1=nn.Conv2d(4, 32, 4, 2, 1, bias=False)
        self.c2=nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.c3=nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.c4=nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.c5=nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.c6=nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.c7=nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.l8l=nn.Conv2d(256, 1, 8, 1, 0, bias=False)

        self.bnc1=nn.BatchNorm2d(32)
        self.bnc2=nn.BatchNorm2d(32)
        self.bnc3=nn.BatchNorm2d(64)
        self.bnc4=nn.BatchNorm2d(64)
        self.bnc5=nn.BatchNorm2d(128)
        self.bnc6=nn.BatchNorm2d(128)
        self.bnc7=nn.BatchNorm2d(256)

    def forward(self, x):
        h = F.leaky_relu(self.bnc1(self.c1(x)),negative_slope=0.2)
        h = F.leaky_relu(self.bnc2(self.c2(h)),negative_slope=0.2)
        h = F.leaky_relu(self.bnc3(self.c3(h)),negative_slope=0.2)
        h = F.leaky_relu(self.bnc4(self.c4(h)),negative_slope=0.2)
        h = F.leaky_relu(self.bnc5(self.c5(h)),negative_slope=0.2)
        h = F.leaky_relu(self.bnc6(self.c6(h)),negative_slope=0.2)
        h = F.leaky_relu(self.bnc7(self.c7(h)),negative_slope=0.2)
        output = self.l8l(h).view(-1, 1).squeeze(1)
        # we don't add in a sigmoid layer since the criterion BCEwithLogitsLoss does it for us
        return output

def pred_single_image(img_path, model_class, checkpoint_path, img_out="assets/output.jpg", debug=False):
    # image pre-processing
    rgba_im = Image.open(img_path).convert("RGBA")
    bg = Image.new("RGB",rgba_im.size,(255,255,255))
    bg.paste(rgba_im,(0,0),rgba_im)
    if np.max(rgba_im.size) > 200:
        bg = transforms.Scale(200)(bg)
    rgb_im = CentraliseImage(300)(bg)
    grayscale_im = rgb_im.convert("L")
    tfms = transforms.Compose([
        transforms.Scale(128),
        transforms.ToTensor()
    ])
    preproc_original_image = Variable(tfms(grayscale_im).unsqueeze(0), requires_grad=False)
    print("==> Initializing model...")
    generator = model_class()
    checkpoint = torch.load(checkpoint_path)
    if debug:
        print(f"Using {model_class} from epoch {checkpoint['epoch']} with loss {checkpoint['loss']}")
    generator.load_state_dict(checkpoint["model"])
    generator.eval()
    print("==> Coloring icon...")
    colored_image = generator(preproc_original_image)
    print("==> Creating JPG...")
    combined = torch.cat([tfms(rgb_im).unsqueeze(0),colored_image.data],dim=0)
    utils.save_image(combined, img_out)
    print(f"==> Done! Check it out at {img_out}.")
    return Image.open(img_out)

# get epoch numbers
def visualise_experiment_loss(exp_name, plots=[1,2,3,4,5]):
    epoch_idxs = sorted([int(fn[fn.find("epoch_")+6:fn.find("(")]) for fn in os.listdir(f"model/{exp_name}/valid_images/")])
    loss_hist_train_content = []
    loss_hist_train_adv = []
    loss_hist_train_d = []
    loss_hist_val_adv = []
    loss_hist_val_content = []
    real_acc_hist = []
    fake_acc_hist = []
    fool_success_hist = []

    for i in epoch_idxs:
        if 1 in plots:
            with open(f"model/{exp_name}/train_loss_history/loss_hist_train_adv_epoch_{i}.p", "rb") as pf:
                loss = pickle.load(pf)
                loss_hist_train_adv.append(np.mean(loss))


        if 2 in plots:
            with open(f"model/{exp_name}/train_loss_history/loss_hist_train_content_epoch_{i}.p", "rb") as pf:
                loss = pickle.load(pf)
                loss_hist_train_content.append(np.mean(loss))

        if 3 in plots:
            with open(f"model/{exp_name}/train_loss_history/loss_hist_train_d_epoch_{i}.p", "rb") as pf:
                loss = pickle.load(pf)
                loss_hist_train_d.append(np.mean(loss))


        if 4 in plots:
            with open(f"model/{exp_name}/val_loss_history/loss_hist_val_adv_epoch_{i}.p", "rb") as pf:
                loss = pickle.load(pf)
                loss_hist_val_adv.append(np.mean(loss))

        if 5 in plots:
            with open(f"model/{exp_name}/val_loss_history/loss_hist_val_content_epoch_{i}.p", "rb") as pf:
                loss = pickle.load(pf)
                loss_hist_val_content.append(np.mean(loss))

        if 6 in plots:
            with open(f"model/{exp_name}/accuracy_history/real_acc_hist_epoch_{i}.p", "rb") as pf:
                acc = pickle.load(pf)
                real_acc_hist.append(np.mean(acc))

        if 7 in plots:
            with open(f"model/{exp_name}/accuracy_history/fake_acc_hist_epoch_{i}.p", "rb") as pf:
                acc = pickle.load(pf)
                fake_acc_hist.append(np.mean(acc))

        if 8 in plots:
            with open(f"model/{exp_name}/accuracy_history/fool_success_hist_epoch_{i}.p", "rb") as pf:
                rate = pickle.load(pf)
                fool_success_hist.append(np.mean(rate))

    if 1 in plots:
        plt.plot(epoch_idxs, loss_hist_train_adv, label="1. gen adv loss (train)")
    if 2 in plots:
        plt.plot(epoch_idxs, loss_hist_train_content, label="2. gen content loss (train)")
    if 3 in plots:
        plt.plot(epoch_idxs, loss_hist_train_d, label="3. dis loss (train)")
    if 4 in plots:
        plt.plot(epoch_idxs, loss_hist_val_adv, label="4. gen adv loss (val)")
    if 5 in plots:
        plt.plot(epoch_idxs, loss_hist_val_content, label="5. gen content loss (val)")
    if 6 in plots:
        plt.plot(epoch_idxs, real_acc_hist, label="6. dis real acc")
    if 7 in plots:
        plt.plot(epoch_idxs, fake_acc_hist, label="7. dis fake acc")
    if 8 in plots:
        plt.plot(epoch_idxs, fool_success_hist, label="8. gen fool success")
    plt.legend()
    plt.title(f"{exp_name}")
    plt.ylabel("Metrics")
    plt.xlabel("Epochs")

def make_valid_images_animation(exp_name):
    valid_imgs_sorted = sorted([fn for fn in os.listdir(f"model/{exp_name}/valid_images/")], key=lambda x: int(x[x.find("epoch_")+6:x.find("(")]))
    fig = plt.figure()
    im = plt.imshow(Image.open(f"model/{exp_name}/valid_images/{valid_imgs_sorted[0]}"), animated=True)
    def updateFig(img_path):
        im.set_data(Image.open(img_path))
        return im
    frames = [f"model/{exp_name}/valid_images/{valid_imgs_sorted[i]}" for i in range(1,len(valid_imgs_sorted))]
    ani = animation.FuncAnimation(fig, updateFig, frames=frames)
    HTML(ani.to_html5_video())
