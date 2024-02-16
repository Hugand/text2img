
# SamplingPipeline
- model
    - model
    - first_stage_model/decoder
- conditioner

# Samplers
## Sampler Params:

__ETA__: How much noise is added back to the latent image after each step. This influences how much the output image changes with the increase of sampling steps.

__sigma churn__: The higher the sigma churn, the simple the image gets, but at the cost of it looking fuzzier.

__sigma noise__: Modifies the impact of sigma churn. The smaller the sigma noise, the higher impact sigma churn has. 

## Scheduler/Discretizer Params:

__tmin__: Min value 0.03. The higher it is, the blurrier and messy the image gets.

__tmax__: Higher values results in darker images, and lower values in brighter images. 

__rho__: Not all schedulers are impacted by it, but messing with the values might create blurry and messy images.

Euler is the simplest, and thus one of the fastest. It and Heun are classics in terms of solving ODEs.

Euler & Heun are closely related. Heun is an 'improvement' on Euler in terms of accuracy, but it runs at about half the speed (which makes sense - it has to calculate the normal Euler term, then do it again to get the final output).

LMS and PLMS are their cousins - they use a related, but slightly different approach (averaging out a couple of steps in the past to improve accuracy). As I understand it, PLMS is effectively LMS (a classical method) adapted to better deal with the weirdness in neural network structure.

DDIM is a neural network method. It's quite fast per step, but relatively inefficient in that it takes a bunch of steps to get a good result.

DPM2 is a fancy method designed for diffusion models explicitly aiming to improve on DDIM in terms of taking less steps to get a good output. It needs to run the denoising twice per step, so once again - it's about twice as slow.

The Ancestral samplers are deceptively much further away from the corresponding non-Ancestral samplers and closer to each other. The corresponding algorithms are used - hence the names - but in a different context.

They can add a bunch of noise per step, so they are more chaotic and diverge heavily from non-Ancestral samplers in terms of the output images. As per the normal-flavored samplers, DPM2-A is about half as fast as Euler-A.

Weirdly, in some comparisons DPM2-A generates very similar images as Euler-A... on the previous seed. Might be due to it being a second-order method vs first-order, might be an experiment muck-up. 