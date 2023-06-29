
import re
import torch
from e3nn import o3


        

def replace_parameters(model, trained_model, model_config):
    # assumes that the added species are appended.
    num_original_species = trained_model.atomic_numbers.size(0)
    num_new_species = model.atomic_numbers.size(0)
    num_added_species = num_new_species - num_original_species
    norm_factor = (num_new_species/num_original_species)**0.5

    special_parameter_patterns = [
        "node_embedding\.linear\.weight",
        "interactions\..\.skip_tp\.weight",
        "products\..\.symmetric_contractions.contractions\..\.weights_max",
        "products\..\.symmetric_contractions.contractions\..\.weights\.."
    ]

    # do all the species independent weights
    for (name, param), (tname, tparam) in zip(model.named_parameters(), trained_model.named_parameters()):
        # cannot change paramerters without turning off grad first. 
        param.requires_grad = False

        assert name == tname, "missmatch between parameter name in current model and pretrained model. Check model specification."
        skip = False
        for pattern in special_parameter_patterns:
            if re.fullmatch(pattern, name):
                skip = True
        if skip: 
            continue

        assert param.shape == tparam.shape, "missmatch between parameter shape in current model and pretrained model. Check model specification."
        param[:] = tparam.detach().clone()
    
    # node embedding
    hidden_irreps = model_config["hidden_irreps"]
    node_feats_irreps_first = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
    for param, tparam in zip(
        model.node_embedding.linear.parameters(),
        trained_model.node_embedding.linear.parameters()
    ):
        if num_added_species == 0:
            param[:] = tparam.detach().clone() * norm_factor
        else:
            param[:-node_feats_irreps_first.dim * num_added_species] = tparam.detach().clone() * norm_factor

    # interaction
    for inter, tinter in zip(model.interactions, trained_model.interactions):
        for i in range(len(inter.skip_tp.instructions)):
            weights = inter.skip_tp.weight_view_for_instruction(i)
            weights[:,:num_original_species,:] = tinter.skip_tp.weight_view_for_instruction(i).detach().clone() * norm_factor
        
        # and finally... (this is not a buffer)
        inter.avg_num_neighbors = tinter.avg_num_neighbors #.detach().clone()

    # product
    for product, trained_product in zip(
        model.products, 
        trained_model.products
    ):
        for contraction, trained_contraction in zip(
            product.symmetric_contractions.contractions,
            trained_product.symmetric_contractions.contractions
        ):
            for weight, trained_weight in zip(
                contraction.weights,
                trained_contraction.weights
            ):
                weight[:num_original_species,...] = trained_weight.detach().clone()
            contraction.weights_max[:num_original_species,...] = trained_contraction.weights_max.detach().clone()

    # cannot change paramerters without turning off grad first. 
    for (name, param), (name_, tparam) in zip(model.named_parameters(), trained_model.named_parameters()):
        param.requires_grad = True

    # now do buffers
    new_buffers = [
        "atomic_numbers",
        "atomic_energies_fn.atomic_energies"
    ]
    check_buffers = [
        "r_max"
    ]

    for (name, buffer), (name_, tbuffer) in zip(model.named_buffers(), trained_model.named_buffers()):
        assert name == name_, "missmatch between buffer name in current model and pretrained model. Check model specification."
        if name in check_buffers:
            assert buffer == tbuffer, f"missmatch buffer value in curent model and pretrained model. Values for {name} are {buffer} and {tbuffer}"
        if name in new_buffers:
            continue

        assert buffer.shape == tbuffer.shape, "missmatch between buffer shape in current model and pretrained model. Check model specification."
        buffer.data = tbuffer.detach().clone()
        

    for (name, param), (tname, tparam) in zip(model.named_parameters(), trained_model.named_parameters()):
        assert name == tname, "missmatch between parameter name in current model and pretrained model. Check model specification."
        skip = False
        for pattern in special_parameter_patterns:
            if re.fullmatch(pattern, name):
                skip = True
        if skip: 
            continue

        assert torch.allclose(param, tparam)


def train_species_dep_weights(model, flag):
    species_dep_parameter_patterns = [
        "node_embedding\.linear\.weight",
        "interactions\..\.skip_tp\.weight",
        "products\..\.symmetric_contractions.contractions\..\.weights_max",
        "products\..\.symmetric_contractions.contractions\..\.weights\.."
    ]

    for name, param in model.named_parameters():
        for pattern in species_dep_parameter_patterns:
            if re.fullmatch(pattern, name):
                param.requires_grad = flag