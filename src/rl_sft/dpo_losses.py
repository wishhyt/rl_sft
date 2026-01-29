import torch
import torch.nn.functional as F

def get_adaptive_lose_l_scale(model_pred, ref_pred, target, config):
    """Adaptive scaling for the loser branch in SDPO."""
    model_diff = model_pred - target
    ref_diff = ref_pred - target
    cosine = F.cosine_similarity(model_diff.view(model_diff.shape[0], -1), 
                                 ref_diff.view(ref_diff.shape[0], -1), dim=1)
    scale = (cosine.unsqueeze(1).unsqueeze(2).unsqueeze(3) * config.dpo.sdpo_alpha).clamp(min=0.0)
    return scale

def loss_diffusion_dpo(model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config):
    return -1 * F.logsigmoid(inside_term).mean()

def loss_dspo(model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config):
    pred2, _ = (model_pred - ref_pred).chunk(2)
    model_diff_w, _ = (model_pred - target).chunk(2)
    # Using 1 - sigmoid(beta * (rw - rl)) as weight
    loss = (model_diff_w - config.dpo.beta_dpo * (1 - F.sigmoid(inside_term)[:, None, None, None]) * pred2).pow(2).mean(dim=[1, 2, 3]).mean()
    return loss

def loss_dmpo(model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config):
    ut = inside_term
    p = torch.sigmoid(ut)
    eps = 1e-6
    alpha = 0.0
    q = p.new(data=(1.0 - alpha,)).reshape(-1)
    q = q.clamp(eps, 1 - eps)
    log_p = -F.softplus(-ut)
    log_1mp = -F.softplus(ut)
    loss = p * (log_p - torch.log(q)) + (1 - p) * (log_1mp - torch.log1p(-q))
    return loss.mean()

def loss_sdpo(model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config):
    # For now falling back to standard DPO logic but intended to use adaptive scaling implies we might need 
    # to re-structure 'inside_term' or how it is used. 
    # Current placeholders suggest using standard logSigmoid on inside_term effectively.
    # If we wanted to STRICTLY apply SDPO as described in some contexts:
    # We would scale the loser's contribution to relative divergence.
    # Since we can't fully rewrite inside_term here without recomputing it:
    return -1 * F.logsigmoid(inside_term).mean()

def loss_kto(model_losses, ref_losses, kto_labels, config):
    # val = r_model - r_ref
    # We define success r = -MSE
    # so r_model - r_ref = -(model_MSE - ref_MSE) = ref_MSE - model_MSE
    val = (ref_losses - model_losses).mean(dim=[1,2,3])
    v = config.dpo.beta_dpo * val
    
    weights = torch.where(kto_labels == 1, config.dpo.kto_lambda_d, config.dpo.kto_lambda_u)
    target_sign = torch.where(kto_labels == 1, 1.0, -1.0)
    
    # KTO Loss
    loss = (weights * (1 - F.sigmoid(target_sign * v))).mean()
    return loss

# Dispatch Map
LOSS_FUNCTIONS = {
    "diffusion-dpo": loss_diffusion_dpo,
    "dspo": loss_dspo,
    "dmpo": loss_dmpo,
    "sdpo": loss_sdpo,
    "kto": loss_kto, # KTO has different signature, handled separately in trainer
}
