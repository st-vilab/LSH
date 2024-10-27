from inversion import *
from prompttoprompt import *

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    cond_emb=False
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    # torch.save(text_embeddings,"/home/prompt-to-prompt/objectremoval/components/cat_text_emb.pt")
    null_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
    null_embeddings = model.text_encoder(null_input.input_ids.to(model.device))[0]
    
    # torch.save(null_embeddings,"/home/prompt-to-prompt/objectremoval/components/null_emb.pt")
    neg_input = model.tokenizer(
        ["lead"],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    neg_embeddings = model.text_encoder(neg_input.input_ids.to(model.device))[0]
    # torch.save(neg_embeddings,"/home/prompt-to-prompt/objectremoval/components/lead_emb.pt")
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    e_store = []
    e_unc_store = []
    e_txt_store = []

    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            if cond_emb:
                context = torch.cat([null_embeddings,text_embeddings])
                ctx_w_neg=torch.cat([null_embeddings, uncond_embeddings[i].expand(*text_embeddings.shape)])
            else:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
            
        if i<60:
            latents,e,e_unc,e_txt = ptp_utils.diffusion_step(model, controller, latents, ctx_w_neg, t, guidance_scale, low_resource=False,i=i,neg_path=None)
        else:
            latents,e,e_unc,e_txt = ptp_utils.diffusion_step(model, controller, latents, ctx_w_neg, t, guidance_scale, low_resource=False)
            e_store.append(e)
            e_unc_store.append(e_unc)
            e_txt_store.append(e_txt)

    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent, e_store,e_unc_store,e_txt_store


def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, name=None, cond_emb = False):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    if cond_emb:
        images, x_t, e_store,e_unc_store,e_txt_store = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings, cond_emb=cond_emb)
    else :    
        images, x_t, e_store,e_unc_store,e_txt_store = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images, name=name)
    return images, x_t,e_store,e_unc_store,e_txt_store



(image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,200,0), verbose=True, with_txt=True)
# # print("Modify or remove offsets according to your image!")
torch.save(uncond_embeddings,"/home/prompt-to-prompt/objectremoval/components/c_opt_emb.pt")
torch.save(x_t,"/home/prompt-to-prompt/objectremoval/components/c_xt.pt")
controller = AttentionStore()
image_inv, x_t, e_store,e_unc_store,e_txt_store = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False, name='cond_rec', cond_emb=True)

print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
ptp_utils.view_images([image_inv[0]])
show_cross_attention(controller, 16, ["up", "down"]) 