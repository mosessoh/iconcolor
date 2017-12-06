
# coding: utf-8

# # U-Net GAN

# In[2]:
import matplotlib
matplotlib.use("Agg")
from momo_imports import *
import momo_utils as mu


# ## Globals

# In[3]:

batch_size = 32
img_size = 128
max_epochs = 120
save_every = 1
from_style = "Outline"
to_style = "Yellow"
debug = False
checkpoint = None
lowest_loss = np.inf
random_seed = 21
valid_size = 256
start_epoch = 0
lambda_adv = 1e-2
exp_name = f"unet_cgan_L1_lambda{lambda_adv}_{from_style}2{to_style}"
USE_CUDA = torch.cuda.is_available()

# In[4]:


mu.prepare_directory(exp_name)


# ## Preparation of data and architecture

# ### Load data

# In[5]:


train_loader, valid_loader = mu.get_logo_dataloaders("data/dataset.csv",from_style,to_style,
                                                     input_img_size=img_size,batch_size=batch_size)


# In[6]:


deproc = transforms.Compose([
    transforms.ToPILImage(),
    lambda x: x.convert("RGB"),
    transforms.ToTensor()
])


# ### Load models

# In[7]:

# use a pre-trained generator
generator = mu.initialise_model(mu.UNetGAN, checkpoint=torch.load("outline2yellow_generator"))


# In[8]:


discriminator = mu.initialise_model(mu.DIS)


# ### Choose optimizers and criterion

# In[9]:


opt_g = optim.Adam(generator.parameters(),lr=1e-4,weight_decay=1e-5)
opt_d = optim.Adam(discriminator.parameters(),lr=1e-4,weight_decay=1e-5)
print(f"==> opt_g and opt_d initialised.")


# ### Choose criterion

# In[11]:


criterion_content = nn.L1Loss()
criterion_d = nn.BCEWithLogitsLoss()
if USE_CUDA:
    print("==> Loaded criterions to GPU")
    criterion_content.cuda()
    criterion_d.cuda()


# ###  Load pre-allocated memory for common variables

# In[13]:


from_imgs = Variable(torch.FloatTensor(batch_size,1,img_size,img_size), requires_grad=False)
to_imgs = Variable(torch.FloatTensor(batch_size,3,img_size,img_size), requires_grad=False)
newT = torch.FloatTensor(64,3,128,128)
labels = Variable(torch.FloatTensor(batch_size))
print(f"==> from_imgs: {from_imgs.size()}")
print(f"==> to_imgs: {to_imgs.size()}")
print(f"==> newT: {newT.size()}")
print(f"==> labels: {labels.size()}")
if USE_CUDA:
    print("==> Loaded variables to GPU")
    from_imgs = from_imgs.cuda()
    to_imgs = to_imgs.cuda()
    labels = labels.cuda()


# ## Model training

# In[18]:

tqdm.write(f"==> Starting training from epoch {start_epoch} to {start_epoch+max_epochs} with lowest val loss = {lowest_loss}.")
for epoch_i in range(start_epoch,start_epoch+max_epochs):
    with tqdm(train_loader) as t:
        loss_hist_train_content = []
        loss_hist_train_adv = []
        loss_hist_train_d = []
        real_acc_hist = []
        fake_acc_hist = []

        ##############################
        # Training loop
        ##############################

        for batch_i, batch in enumerate(t):
            from_imgs.data.copy_(batch["from"]["img"])
            to_imgs.data.copy_(batch["to"]["img"])

            ##############################
            # Train discriminator
            ##############################

            # train with real
            discriminator.train()
            discriminator.zero_grad()
            pred_real = discriminator(torch.cat([to_imgs, from_imgs],1))
            labels.data.fill_(1.0)
            loss_d_real = criterion_d(pred_real, labels)
            real_acc = torch.sum(F.sigmoid(pred_real.data) > 0.5).data[0] / len(pred_real)
            real_acc_hist.append(real_acc)
            loss_d_real.backward()
            # calculate accuracy

            imgs_generated = generator(from_imgs)
            # train with fake
            pred_fake = discriminator(torch.cat([imgs_generated.detach(),from_imgs],1))
            labels.data.fill_(0.0)
            loss_d_fake = criterion_d(pred_fake, labels)
            fake_acc = torch.sum(F.sigmoid(pred_fake.data) < 0.5).data[0] / len(pred_fake)
            fake_acc_hist.append(fake_acc)
            loss_d_fake.backward()
            loss_d = loss_d_real + loss_d_fake
            opt_d.step()

            ##############################
            # Train generator
            ##############################

            generator.train()
            generator.zero_grad()
            labels.data.fill_(1.0)
            loss_content = criterion_content(imgs_generated, to_imgs)
            loss_adv = criterion_d(discriminator(torch.cat([imgs_generated, from_imgs],1)), labels) * lambda_adv
            loss_g = loss_content + loss_adv
            loss_g.backward()
            opt_g.step()

            t.set_description(f"[TRAINING] epoch {epoch_i} | loss_d = {loss_d.data[0]:.4f} | loss_content = {loss_content.data[0]:.4f} | loss_adv = {loss_adv.data[0]:.4f} | real_acc = {real_acc:.1%} | fake_acc = {fake_acc:.1%}")
            loss_hist_train_content.append(loss_content.data[0])
            loss_hist_train_adv.append(loss_adv.data[0])
            loss_hist_train_d.append(loss_d.data[0])

        ##############################
        # Save losses and models after each training epoch
        ##############################

        # print and save train loss
        tqdm.write(f"--------------- SUMMARY for EPOCH {epoch_i} ---------------")
        tqdm.write(f"==> [TRAINING AVERAGES] epoch {epoch_i} | loss_d = {np.mean(loss_hist_train_d):.4f} | loss_content = {np.mean(loss_hist_train_content):.4f} | loss_adv = {np.mean(loss_hist_train_adv):.4f} | real_acc = {np.mean(real_acc_hist):.1%} | fake_acc = {np.mean(fake_acc_hist):.1%}")
        with open(f"model/{exp_name}/train_loss_history/loss_hist_train_content_epoch_{epoch_i}.p","wb") as pf:
            pickle.dump(loss_hist_train_content, pf)
        with open(f"model/{exp_name}/train_loss_history/loss_hist_train_adv_epoch_{epoch_i}.p","wb") as pf:
            pickle.dump(loss_hist_train_adv, pf)
        with open(f"model/{exp_name}/train_loss_history/loss_hist_train_d_epoch_{epoch_i}.p","wb") as pf:
            pickle.dump(loss_hist_train_d, pf)
        with open(f"model/{exp_name}/accuracy_history/real_acc_hist_epoch_{epoch_i}.p","wb") as pf:
            pickle.dump(real_acc_hist, pf)
        with open(f"model/{exp_name}/accuracy_history/fake_acc_hist_epoch_{epoch_i}.p","wb") as pf:
            pickle.dump(fake_acc_hist, pf)

        # save latest generator and discriminator from training
        mu.save_checkpoint(model=generator.state_dict(),
                           optimizer=opt_g.state_dict(),
                           epoch=epoch_i,
                           loss=np.mean(loss_hist_train_content),
                           model_pth=f"model/{exp_name}/best_generator_fulldataset_train.pth")

        mu.save_checkpoint(model=discriminator.state_dict(),
                           optimizer=opt_d.state_dict(),
                           epoch=epoch_i,
                           loss=np.mean(loss_hist_train_adv),
                           model_pth=f"model/{exp_name}/best_discriminator_fulldataset_train.pth")

        ##############################
        # Evaluate model on validation set
        ##############################

        generator.eval()
        discriminator.eval()

        loss_hist_val_content = []
        loss_hist_val_adv = []
        fool_success_hist = []

        for valid_batch_i, valid_batch in enumerate(valid_loader):
            from_imgs.data.copy_(valid_batch["from"]["img"])
            to_imgs.data.copy_(valid_batch["to"]["img"])

            # get content loss
            imgs_generated = generator(from_imgs)
            valid_loss_content = criterion_content(imgs_generated, to_imgs)

            # get adversarial loss
            labels.data.fill_(1.0)
            pred = discriminator(torch.cat([imgs_generated, from_imgs],1))
            fool_success_rate = torch.sum(F.sigmoid(pred.data) > 0.5).data[0] / len(pred)
            fool_success_hist.append(fool_success_rate)
            valid_loss_adv = criterion_d(pred, labels) * lambda_adv

            # save each batch's loss
            loss_hist_val_content.append(valid_loss_content.data[0])
            loss_hist_val_adv.append(valid_loss_adv.data[0])

            if valid_batch_i == 0:
                for i in range(batch_size):
                    newT[2*i] = deproc(valid_batch["from"]["img"][i])
                    newT[2*i+1] = imgs_generated.data[i]
                mu.save_batch_grid(newT,
                                   title=f"{exp_name}_fulldataset_epoch_{epoch_i}(valid_batch {valid_batch_i})",
                                   path=f"model/{exp_name}/valid_images")

        tqdm.write(f"==> [VALIDATION AVERAGES] epoch {epoch_i} | loss_content = {np.mean(loss_hist_val_content):.4f} | loss_adv = {np.mean(loss_hist_val_adv):.4f} | fool_success_rate = {np.mean(fool_success_hist):.1%}")

        # save loss histories for this validation epoch to disk
        with open(f"model/{exp_name}/val_loss_history/loss_hist_val_content_epoch_{epoch_i}.p","wb") as pf:
            pickle.dump(loss_hist_val_content, pf)
        with open(f"model/{exp_name}/val_loss_history/loss_hist_val_adv_epoch_{epoch_i}.p","wb") as pf:
            pickle.dump(loss_hist_val_adv, pf)
        with open(f"model/{exp_name}/accuracy_history/fool_success_hist_epoch_{epoch_i}.p","wb") as pf:
            pickle.dump(fool_success_hist, pf)

        avg_valid_loss = np.mean(loss_hist_val_content) + np.mean(loss_hist_val_adv)
        if avg_valid_loss < lowest_loss:
            tqdm.write(f"==> [WOOHOO] Found new best model with valid_loss = {avg_valid_loss}")

            mu.save_checkpoint(model=generator.state_dict(),
                               optimizer=opt_g.state_dict(),
                               epoch=epoch_i,
                               loss=np.mean(loss_hist_val_content) ,
                               model_pth=f"model/{exp_name}/best_generator_fulldataset_valid.pth")

            mu.save_checkpoint(model=discriminator.state_dict(),
                               optimizer=opt_d.state_dict(),
                               epoch=epoch_i,
                               loss=np.mean(loss_hist_val_adv),
                               model_pth=f"model/{exp_name}/best_discriminator_fulldataset_valid.pth")

            lowest_loss = avg_valid_loss
