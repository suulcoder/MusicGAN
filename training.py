import numpy as np
import matplotlib.pyplot as plt
import torch
import pypianoroll
from pypianoroll import Multitrack, Track, BinaryTrack, write, to_pretty_midi
from tqdm import tqdm
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from settings import *
from data import *
from Generator import *
from Discriminator import *

from IPython.display import clear_output
from ipywidgets import interact, IntSlider

#First we load the data
data_loader = load_data()
history_samples = {}

# Create discriminator
discriminator = Discriminator()


# Create generator
generator = Generator()

print("{} parameters in Generator".format(
    sum(p.numel() for p in generator.parameters() if p.requires_grad)))
print("{} parameters in Discriminator".format(
    sum(p.numel() for p in discriminator.parameters() if p.requires_grad)))

#Now lets create some optimizers
d_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.001,  betas=(0.5, 0.9))
g_optimizer = torch.optim.Adam(
    generator.parameters(), lr=0.001, betas=(0.5, 0.9))

sample_latent = torch.randn(number_of_samples, latent_dim)

# Transfer to GPU
if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
    sample_latent = sample_latent.cuda()

liveloss = PlotLosses(outputs=[MatplotlibPlot(cell_size=(6,2))])


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_samples.size(0), 1).cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_one_step(d_optimizer, g_optimizer, real_samples):
    # Sample from the lantent distribution
    latent = torch.randn(batch, latent_dim)

    # Transfer to GPU
    if torch.cuda.is_available():
        real_samples = real_samples.cuda()
        latent = latent.cuda()
    
    # Train the discriminator

    d_optimizer.zero_grad()
    # Get discriminator outputs for the real samples
    prediction_real = discriminator(real_samples)
    d_loss_real = -torch.mean(prediction_real)
    d_loss_real.backward()
    
    # Generate fake samples
    fake_samples = generator(latent)
    
    # Discriminator checks the fake samples
    prediction_fake_d = discriminator(fake_samples.detach())
    
    d_loss_fake = torch.mean(prediction_fake_d)
    
    d_loss_fake.backward()

    # Compute gradient penalty
    gradient_penalty = 10.0 * compute_gradient_penalty(
        discriminator, real_samples.data, fake_samples.data)
    gradient_penalty.backward()

    d_optimizer.step()
    
    # Train the generator
    g_optimizer.zero_grad()
    prediction_fake_g = discriminator(fake_samples)
    g_loss = -torch.mean(prediction_fake_g)
    g_loss.backward()
    g_optimizer.step()

    return d_loss_real + d_loss_fake, g_loss
    

step = 0
progress_bar = tqdm(total=number_steps, initial=step, ncols=80, mininterval=1)


while step < number_steps + 1:
    for real_samples in data_loader:
        generator.train()
        d_loss, g_loss = train_one_step(d_optimizer, g_optimizer, real_samples[0])

        if step > 0:
            running_d_loss = 0.05 * d_loss + 0.95 * running_d_loss
            running_g_loss = 0.05 * g_loss + 0.95 * running_g_loss
        else:
            running_d_loss, running_g_loss = 0.0, 0.0
        liveloss.update({'negative_critic_loss': -running_d_loss})
        
        progress_bar.set_description_str(
            "(d_loss={: 8.6f}, g_loss={: 8.6f})".format(d_loss, g_loss))
        
        if step % interval == 0:
            generator.eval()
            samples = generator(sample_latent).cpu().detach().numpy()
            history_samples[step] = samples

            """
            if we want to see the discriminator loss we should uncomment this lines
            if step > 0:
                liveloss.send()
            """
            
            samples = samples.transpose(1, 0, 2, 3).reshape(number_of_trakcs, -1, number_of_pitches)
            tracks = []
            for selected, (program, is_drum, track_name) in enumerate(
                zip(programs, is_drums, track_names)
            ):
                pianoroll = np.pad(
                    samples[selected] > 0.5,
                    ((0, 0), (lowest_pitch, 128 - lowest_pitch - number_of_pitches)),
                    mode='constant'
                )
                tracks.append(
                    BinaryTrack(
                        name=track_name,
                        program=program,
                        is_drum=is_drum,
                        pianoroll=pianoroll
                    )
                )
            m = Multitrack(
                tracks=tracks,
                tempo=tempo_array,
                resolution=beat
            )

            to_pretty_midi(m).write(str('./results/results_' + str(step) + '.mid'))
            
            my_axis = m.plot()
            plt.gcf().set_size_inches((16, 8))
            for ax in my_axis:
                for x in range(
                    measure_resolution,
                    4 * measure_resolution * number_of_measures,
                    measure_resolution
                ):
                    if x % (measure_resolution * 4) == 0:
                        ax.axvline(x - 0.5, color='k')
                    else:
                        ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
            plt.savefig(str('./results/results_' + str(step) + '.png'))
            
        step += 1
        progress_bar.update(1)
        if step >= number_steps:
            break