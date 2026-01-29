import torch
import torch.nn.functional as F

def get_adaptive_lose_l_scale(model_pred, ref_pred, target, config):
    """Adaptive scaling for the loser branch in SDPO.
    
    Returns λ based on winner-preserving condition:
    Constraint: (-∇A + λ∇B) · ∇A ≤ -μ ||∇A||^2
    
    Args:
        model_pred: Model predictions (concat of winner and loser)
        ref_pred: Reference predictions (concat of winner and loser)
        target: Ground truth noise
        config: Configuration object with dpo.sdpo_alpha and dpo.winner_preserving_mu
    
    Returns:
        Adaptive scaling factor for loser branch
    """
    # Split into win/lose along the batch dimension
    pred_w, pred_l = model_pred.detach().requires_grad_(True).chunk(2, dim=0)
    true_w, true_l = target.detach().chunk(2, dim=0)
    
    # Use noise-prediction MSE as scalar proxy
    A = ((pred_w - true_w) ** 2).mean()
    B = ((pred_l - true_l) ** 2).mean()
    
    # Output-space proxy gradient
    gA = torch.autograd.grad(A, pred_w, create_graph=False, retain_graph=False)[0]
    gB = torch.autograd.grad(B, pred_l, create_graph=False, retain_graph=False)[0]
    
    # Scalars: ||gA||^2 and gB·gA
    num = (gA.flatten().pow(2).sum())  # ||gA||^2
    den = (gB.flatten() * gA.flatten()).sum()  # gB·gA
    
    eps = 1e-9
    mu = getattr(config.dpo, 'winner_preserving_mu', 0.0)
    max_lambda = 1.0
    
    # When den > 0 (gradients point in same direction), apply upper bound
    lam_cap = ((1.0 - float(mu)) * num) / (den + eps)
    lam = torch.where(
        den > 0,
        lam_cap,
        torch.tensor(max_lambda, device=model_pred.device, dtype=model_pred.dtype)
    )
    
    # Numerical safeguard
    lam = lam.clamp(min=0.0, max=max_lambda).detach().to(dtype=model_pred.dtype)
    return lam

def loss_sdpo(model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config):
    """SDPO loss with winner-preserving constraint.
    
    Implements the SDPO algorithm from Diffusion-SDPO repository.
    Uses adaptive scaling on the loser branch to preserve winner gradients.
    
    Note: This requires model_diff to be pre-computed with adaptive scaling
    in the trainer if use_winner_preserving is enabled.
    """
    # Check if winner-preserving is enabled
    use_wp = getattr(config.dpo, 'use_winner_preserving', False)
    
    if use_wp:
        # Winner-preserving mode: model_diff should already be scaled in trainer
        # We just use standard DPO loss
        loss = -1 * F.logsigmoid(inside_term).mean()
    else:
        # Standard DPO without winner-preserving
        loss = -1 * F.logsigmoid(inside_term).mean()
    
    return loss

def loss_diffusion_dpo(model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config):
    """Standard Diffusion-DPO loss."""
    return -1 * F.logsigmoid(inside_term).mean()

def loss_dspo(model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config):
    """DSPO loss from DSPO repository.
    
    SD1.5 version that combines preference learning with MSE optimization.
    """
    pred2, _ = (model_pred - ref_pred).chunk(2)
    model_diff_w, _ = (model_pred - target).chunk(2)
    # Using 1 - sigmoid(beta * (rw - rl)) as weight
    loss = (model_diff_w - config.dpo.beta_dpo * (1 - F.sigmoid(inside_term)[:, None, None, None]) * pred2).pow(2).mean(dim=[1, 2, 3]).mean()
    return loss

def loss_dmpo(model_diff, ref_diff, model_pred, ref_pred, target, inside_term, config):
    """DMPO loss - variational DPO objective."""
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
