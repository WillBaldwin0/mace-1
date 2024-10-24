###########################################################################################
# Atomic Data Class for handling molecules as graphs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from copy import deepcopy
from typing import Optional, Sequence

import torch.utils.data

from mace.tools import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
    to_one_hot,
    torch_geometric,
    voigt_to_matrix,
)

from .neighborhood import get_neighborhood
from .utils import Configuration


class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    charges: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        head: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        stress_weight: Optional[torch.Tensor],  # [,]
        virials_weight: Optional[torch.Tensor],  # [,]
        dipole_weight: Optional[torch.Tensor],  # [,]
        charges_weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        stress: Optional[torch.Tensor],  # [1,3,3]
        virials: Optional[torch.Tensor],  # [1,3,3]
        dipole: Optional[torch.Tensor],  # [, 3]
        charges: Optional[torch.Tensor],  # [n_nodes, ]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]

        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert len(node_attrs.shape) == 2
        assert weight is None or len(weight.shape) == 0
        assert head is None or len(head.shape) == 0
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert stress_weight is None or len(stress_weight.shape) == 0
        assert virials_weight is None or len(virials_weight.shape) == 0
        assert dipole_weight is None or dipole_weight.shape == (1, 3), dipole_weight
        assert charges_weight is None or len(charges_weight.shape) == 0
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert dipole is None or dipole.shape[-1] == 3
        assert charges is None or charges.shape == (num_nodes,)
        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "cell": cell,
            "node_attrs": node_attrs,
            "weight": weight,
            "head": head,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "virials_weight": virials_weight,
            "dipole_weight": dipole_weight,
            "charges_weight": charges_weight,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "dipole": dipole,
            "charges": charges,
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        z_table: AtomicNumberTable,
        cutoff: float,
        heads: Optional[list] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> "AtomicData":
        if heads is None:
            heads = ["Default"]
        edge_index, shifts, unit_shifts, cell = get_neighborhood(
            positions=config.positions,
            cutoff=cutoff,
            pbc=deepcopy(config.pbc),
            cell=deepcopy(config.cell),
        )
        indices = atomic_numbers_to_indices(config.atomic_numbers, z_table=z_table)
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        try:
            head = torch.tensor(heads.index(config.head), dtype=torch.long)
        except ValueError:
            head = torch.tensor(len(heads) - 1, dtype=torch.long)
        default_dtype = torch.get_default_dtype()

        cell = (
            torch.tensor(cell, dtype=default_dtype)
            if cell is not None
            else torch.tensor(3 * [0.0, 0.0, 0.0], dtype=default_dtype).view(3, 3)
        )

        data = {
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "positions": torch.tensor(config.positions, dtype=default_dtype),
            "shifts": torch.tensor(shifts, dtype=default_dtype),
            "unit_shifts": torch.tensor(unit_shifts, dtype=default_dtype),
            "cell": cell,
            "node_attrs": one_hot,
            "head": head,
        }

        required_weights = [
            "weight",
            "energy",
            "forces",
            "stress",
            "virials",
            "dipole",
            "charges",
        ]
        weight_dict = {}
        for req_weight in required_weights:
            if req_weight == "weight":
                value = config.weight
                key = "weight"
            else:
                value = config.property_weights.get(req_weight)
                key = f"{req_weight}_weight"
            if value is not None:
                weight_tensor = torch.tensor(value, dtype=default_dtype)
            else:
                weight_tensor = torch.tensor(1.0, dtype=default_dtype)
            weight_dict[key] = weight_tensor
            
        # Adjust dipole weight shape
        dipole_weight = weight_dict["dipole_weight"]
        if len(dipole_weight.shape) == 0:
            dipole_weight = dipole_weight * torch.tensor(
                [[1.0, 1.0, 1.0]], dtype=default_dtype
            )
        elif len(dipole_weight.shape) == 1:
            dipole_weight = dipole_weight.unsqueeze(0)
        weight_dict["dipole_weight"] = dipole_weight

        data.update(weight_dict)

        per_atom_props = [("forces", 3), ("charges", None)] # (property_name, property_dim)
        voigt_props = ["stress", "virials"]
        scalar_props = ["energy"]
        vector_props = ["dipole"]
        prop_dict = {}

        num_atoms = len(config.atomic_numbers)
        for prop, prop_dim in per_atom_props:
            value = config.properties.get(prop)
            if value is not None:
                prop_tensor = torch.tensor(value, dtype=default_dtype)
            else:
                if prop_dim is not None:
                    prop_tensor = torch.zeros(num_atoms, prop_dim, dtype=default_dtype)
                else:
                    prop_tensor = torch.zeros(num_atoms, dtype=default_dtype)
            prop_dict[prop] = prop_tensor

        for prop in scalar_props:
            value = config.properties.get(prop)
            if value is not None:
                prop_tensor = torch.tensor(value, dtype=default_dtype)
            else:
                prop_tensor = torch.tensor(0.0, dtype=default_dtype)
            prop_dict[prop] = prop_tensor

        for prop in voigt_props:
            value = config.properties.get(prop)
            if value is not None:
                prop_tensor = voigt_to_matrix(
                    torch.tensor(value, dtype=default_dtype)
                ).unsqueeze(0)
            else:
                prop_tensor = torch.zeros(1, 3, 3, dtype=default_dtype)
            prop_dict[prop] = prop_tensor

        for prop in vector_props:
            value = config.properties.get(prop)
            if value is not None:
                prop_tensor = torch.tensor(value, dtype=default_dtype).unsqueeze(0)
            else:
                prop_tensor = torch.zeros(1, 3, dtype=default_dtype)
            prop_dict[prop] = prop_tensor

        data.update(prop_dict)

        return cls(
            **data,
        )


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
